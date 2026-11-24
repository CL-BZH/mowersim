"""Headland passes: rings that follow the border of the lawn and of its holes.

Ring k is a closed path of the middle of the rear axle at the distance
c_k = c_1 + (k - 1) s from the border (s: spacing between passes). It is
built in three steps (course, chapter 6):

1. offset the lawn inwards by c_k (Shapely buffer with mitred joins, so that a
   corner stays a single vertex);
2. at each vertex where the polyline bends gently, a circular *fillet* of
   radius rho = min(l_prev, l_next) / (2 tan(phi / 2)), as long as rho is at
   least the minimum turning radius: a curved border becomes a chain of arcs;
3. every other vertex is a *corner*: the mower stops following the ring at a
   distance b before it and joins the ring again at b after it with the
   fastest Reeds-Shepp manoeuvre that keeps the whole mower on the lawn; b
   grows from 0 until such a manoeuvre exists.

The clearance c_1 of the first ring is the half-width of the mower, plus a
safety margin, plus the *swing-out* of the front corner in the tightest left
fillet (a car's front corner swings outwards in a turn).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LinearRing
from shapely.geometry.polygon import orient

from ..geometry import Lawn, Pose
from ..kinematics import Segment, Trajectory
from ..mower import Mower
from .common import (Checker, connect, deck_front, lateral_half_width, lawn_polygon,
                     polygons, unit)

CORNER_DEG = 20.0          # a vertex turning more than this is a corner, not a bend


def swing_out(mower: Mower, radius: float) -> float:
    """How much farther than in a straight line the mower sticks out sideways, on the
    outside of a left arc of the given radius (Proposition "swing-out").

    In the mower frame the centre of rotation is C = (0, R). A point (a, b) of the
    outline moves on a circle of radius sqrt(a^2 + (R - b)^2) about C, while in a
    straight line the mower reaches R + h from C on that side (h: half-width). The
    swing-out is the largest excess over the outline."""
    pts = mower.footprint_points(0.01)
    h = lateral_half_width(mower)
    return float(np.max(np.hypot(pts[:, 0], radius - pts[:, 1]))) - (radius + h)


@dataclass
class Vertex:
    p: np.ndarray              # position
    phi: float                 # signed deflection (> 0: left turn), radians
    kind: str = "bend"         # "straight", "bend" (fillet) or "corner"
    radius: float = math.inf   # fillet radius
    t_in: float = 0.0          # distance before the vertex where the straight part ends
    t_out: float = 0.0         # distance after the vertex where it starts again
    move: list = field(default_factory=list)   # segments of the corner manoeuvre


def ring_vertices(coords: np.ndarray, min_edge: float = 0.01) -> list[np.ndarray]:
    pts = [np.asarray(p, dtype=float) for p in coords]
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    out = []
    for p in pts:
        if not out or np.linalg.norm(p - out[-1]) > min_edge:
            out.append(p)
    if len(out) > 2 and np.linalg.norm(out[0] - out[-1]) <= min_edge:
        out.pop()
    # drop collinear vertices
    clean = []
    n = len(out)
    for i in range(n):
        a, b, c = out[i - 1], out[i], out[(i + 1) % n]
        u, v = b - a, c - b
        if abs(u[0] * v[1] - u[1] * v[0]) > 1e-9 * np.linalg.norm(u) * np.linalg.norm(v) \
                or np.dot(u, v) < 0:
            clean.append(b)
    return clean


def merge_close_corners(pts: list[np.ndarray], mower: Mower, min_edge: float = 2.0,
                        max_move: float = 4.0) -> list[np.ndarray]:
    """Two corners closer than ``min_edge`` leave no room for a manoeuvre between
    them: replace them by one corner where the edges before and after them meet."""
    changed = True
    while changed and len(pts) > 3:
        changed = False
        vs = classify(pts, mower)
        n = len(pts)
        for i in range(n):
            j = (i + 1) % n
            if vs[i].kind != "corner" or vs[j].kind != "corner":
                continue
            if np.linalg.norm(pts[j] - pts[i]) >= min_edge:
                continue
            a, b = pts[i - 1], pts[i]                 # incoming edge
            c, d = pts[j], pts[(j + 1) % n]           # outgoing edge
            u, v = b - a, d - c
            m = np.array([u, -v]).T
            if abs(np.linalg.det(m)) < 1e-9:
                continue
            lam, _ = np.linalg.solve(m, c - a)
            x = a + lam * u
            if np.linalg.norm(x - b) > max_move or np.linalg.norm(x - c) > max_move:
                continue
            new = [p for k, p in enumerate(pts) if k not in (i, j)]
            new.insert(i if j > i else i - 1, x)
            pts, changed = new, True
            break
    return pts


def classify(pts: list[np.ndarray], mower: Mower) -> list[Vertex]:
    """Deflection at each vertex, and whether it is a bend (fillet) or a corner."""
    n = len(pts)
    vs = []
    for i in range(n):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        u, v = b - a, c - b
        phi = math.atan2(u[0] * v[1] - u[1] * v[0], float(np.dot(u, v)))
        lp, ln = float(np.linalg.norm(u)), float(np.linalg.norm(v))
        vx = Vertex(b, phi)
        if abs(phi) < 1e-9:
            vx.kind = "straight"
        else:
            rho = min(lp, ln) / (2 * math.tan(abs(phi) / 2))
            if abs(phi) <= math.radians(CORNER_DEG) and rho >= mower.min_radius:
                vx.kind, vx.radius = "bend", rho
                vx.t_in = vx.t_out = rho * math.tan(abs(phi) / 2)
            else:
                vx.kind = "corner"
        vs.append(vx)
    return vs


def solve_corner(vx: Vertex, heading_in: float, heading_out: float, room_in: float,
                 room_out: float, mower: Mower, checker: Checker, mow_reverse: bool,
                 step: float = 0.2) -> bool:
    """Smallest backoff b (in steps) with a collision-free manoeuvre around the corner."""
    b = 0.0
    while b <= min(room_in, room_out) + 1e-9:
        s = Pose(*(vx.p - b * unit(heading_in)), heading_in)
        g = Pose(*(vx.p + b * unit(heading_out)), heading_out)
        c = connect(s, g, mower, checker, mow=True, mow_reverse=mow_reverse)
        if c is not None:
            vx.t_in = vx.t_out = b
            vx.move = c.segments
            return True
        b += step
    return False


@dataclass
class Ring:
    pieces: list               # [(start pose, segments)]: one closed loop, or open pieces
    offset: float
    corners: int
    failed: int                # corners with no collision-free path: the ring is cut there

    @property
    def closed(self) -> bool:
        return self.failed == 0

    @property
    def start(self) -> Pose:
        return self.pieces[0][0]

    @property
    def segments(self) -> list[Segment]:
        return self.pieces[0][1]

    @property
    def trajectory(self) -> Trajectory:
        return Trajectory(self.start, list(self.segments))


def push_tight_bends(pts: list[np.ndarray], mower: Mower, clearance: float,
                     need: float, smooth: float = 1.5) -> list[np.ndarray]:
    """Move the vertices of left bends inwards where the front corner's swing-out needs
    more clearance than the ring has (``need`` is the clearance in a straight line).

    Each vertex i needs the clearance need + swing_out(rho_i); it is moved along its
    inward normal by the shortfall, taken as the largest shortfall within ``smooth``
    metres along the ring so that the pushed ring stays smooth."""
    n = len(pts)
    vs = classify(pts, mower)
    short = np.zeros(n)
    for i, vx in enumerate(vs):
        if vx.kind == "bend" and vx.phi > 0:
            # 5 cm of tolerance: the fillets of neighbouring vertices interact
            short[i] = max(0.0, need + swing_out(mower, vx.radius) + 0.05 - clearance)
    if not short.any():
        return pts
    # largest shortfall within `smooth` metres along the ring
    cum = np.concatenate([[0], np.cumsum([np.linalg.norm(pts[(i + 1) % n] - pts[i])
                                          for i in range(n)])])
    total = cum[-1]
    pushed = []
    for i in range(n):
        d = np.abs(cum[:n] - cum[i])
        d = np.minimum(d, total - d)
        delta = short[d <= smooth].max()
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % n]
        t = (c - a) / max(np.linalg.norm(c - a), 1e-12)
        normal = np.array([-t[1], t[0]])                  # left = inside the lawn
        pushed.append(b + delta * normal)
    return pushed


def shift_arcs(delta: float, radius: float, left: bool) -> tuple[list[Segment], float]:
    """An S-curve moving the mower sideways by ``delta`` without changing its heading:
    two opposite arcs of the same radius and angle alpha, with
    delta = 2 R (1 - cos alpha). Returns the segments and the distance covered along
    the original direction, 2 R sin alpha."""
    if delta <= 0:
        return [], 0.0
    alpha = math.acos(max(-1.0, 1.0 - delta / (2 * radius)))
    k = 1.0 / radius
    first, second = (k, -k) if left else (-k, k)
    return ([Segment(first, radius * alpha), Segment(second, radius * alpha)],
            2 * radius * math.sin(alpha))


def corner_path(p: np.ndarray, heading_in: float, heading_out: float, phi: float,
                delta: float, rho: float, shift_radius: float):
    """Path around the corner p: shift inwards by delta, turn with radius rho on the
    shifted lines, shift back. Returns (segments, b_in, b_out): the path starts on the
    incoming line b_in before p and ends on the outgoing line b_out after it."""
    u1, u2 = unit(heading_in), unit(heading_out)
    n1, n2 = np.array([-u1[1], u1[0]]), np.array([-u2[1], u2[0]])   # left normals
    # corner of the two lines shifted by delta to the left (inwards)
    a = p + delta * n1
    b = p + delta * n2
    m = np.array([u1, -u2]).T
    lam, _ = np.linalg.solve(m, b - a)
    corner = a + lam * u1
    t = rho * math.tan(abs(phi) / 2)
    shift_in, d_shift = shift_arcs(delta, shift_radius, left=True)
    shift_out, _ = shift_arcs(delta, shift_radius, left=False)
    # along u1: the arc starts at corner - t u1, i.e. at lam - t from p; the shift
    # starts d_shift earlier, on the original line
    b_in = -(lam - t - d_shift)
    lam2 = float(np.dot(corner - p, u2))
    b_out = lam2 + t + d_shift
    segs = [*shift_in, Segment(math.copysign(1.0 / rho, phi), rho * abs(phi)), *shift_out]
    return segs, b_in, b_out


def solve_corner_shift(vx: Vertex, heading_in: float, heading_out: float, room_in: float,
                       room_out: float, mower: Mower, checker: Checker,
                       max_delta: float = 1.5) -> bool:
    """Smallest inward shift delta, then smallest radius rho, whose corner path keeps
    the mower on the lawn."""
    R = mower.min_radius
    for delta in np.arange(0.0, max_delta + 1e-9, 0.1):
        for rho in (R, R + 0.5, R + 1.0, R + 2.0):
            for shift_radius in (2 * R, 4 * R, 8 * R):
                segs, b_in, b_out = corner_path(vx.p, heading_in, heading_out, vx.phi,
                                                float(delta), rho, shift_radius)
                if b_in < 0 or b_out < 0 or b_in > room_in or b_out > room_out:
                    continue
                start = Pose(*(vx.p - b_in * unit(heading_in)), heading_in)
                if checker.free(start, segs):
                    vx.t_in, vx.t_out, vx.move = b_in, b_out, segs
                    vx.radius = rho
                    return True
                if delta == 0:
                    break                       # without a shift, the S-curve is unused
    return False


def build_ring(coords: np.ndarray, offset: float, mower: Mower, checker: Checker,
               mow_reverse: bool = False, need: float | None = None) -> Ring | None:
    """A closed headland path along one ring of the offset lawn (region on the left)."""
    pts = ring_vertices(coords)
    if len(pts) < 3:
        return None
    if need is not None:
        for _ in range(2):
            pts = push_tight_bends(pts, mower, offset, need)
    pts = merge_close_corners(pts, mower)
    vs = classify(pts, mower)
    n = len(vs)
    heading = [math.atan2(*(pts[(i + 1) % n] - pts[i])[::-1]) for i in range(n)]
    length = [float(np.linalg.norm(pts[(i + 1) % n] - pts[i])) for i in range(n)]
    failed = 0
    for i, vx in enumerate(vs):
        if vx.kind != "corner":
            continue
        prev_c = vs[i - 1].kind == "corner"
        next_c = vs[(i + 1) % n].kind == "corner"
        room_in = length[i - 1] / 2 if prev_c else length[i - 1] - vs[i - 1].t_out
        room_out = length[i] / 2 if next_c else length[i] - vs[(i + 1) % n].t_in
        if solve_corner_shift(vx, heading[i - 1], heading[i], room_in, room_out, mower,
                              checker):
            continue
        if solve_corner(vx, heading[i - 1], heading[i], room_in, room_out, mower,
                        checker, mow_reverse):
            continue
        # nothing keeps the mower on the lawn around this corner: the ring is cut
        # here, and stops (and starts again) a little before (after) the corner
        failed += 1
        vx.kind = "break"
        vx.t_in = min(1.0, room_in)
        vx.t_out = min(1.0, room_out)
    free = [length[i] - vs[i].t_out - vs[(i + 1) % n].t_in for i in range(n)]
    if min(free) < -1e-6:
        return None
    breaks = [i for i, v in enumerate(vs) if v.kind == "break"]
    if not breaks:
        # closed ring: start in the middle of the longest straight piece
        i0 = int(np.argmax(free))
        s0 = vs[i0].t_out + free[i0] / 2
        start = Pose(*(pts[i0] + s0 * unit(heading[i0])), heading[i0])
        segs = walk(pts, vs, heading, length, i0, s0, n)
        segs.append(Segment(0.0, s0 - vs[i0].t_out))
        pieces = [(start, [sg for sg in segs if sg.length > 1e-9])]
    else:
        # open pieces from just after one break to just before the next
        pieces = []
        for k, b0 in enumerate(breaks):
            b1 = breaks[(k + 1) % len(breaks)]
            steps = (b1 - b0) % n or n
            start = Pose(*(pts[b0] + vs[b0].t_out * unit(heading[b0])), heading[b0])
            segs = walk(pts, vs, heading, length, b0, vs[b0].t_out, steps)
            start, segs = trim_piece(start, [sg for sg in segs if sg.length > 1e-9], checker)
            if segs:
                pieces.append((start, segs))
    corners = sum(v.kind in ("corner", "break") for v in vs)
    return Ring(pieces, offset, corners, failed)


def cut_segments(start: Pose, segs: list[Segment], s_a: float, s_b: float):
    """The part of a path between the distances s_a and s_b from its start."""
    from ..kinematics import advance
    out, pose, s0 = [], start, 0.0
    new_start = None
    for sg in segs:
        s1 = s0 + sg.length
        a, b = max(s_a, s0), min(s_b, s1)
        if b > a + 1e-9:
            if new_start is None:
                new_start = advance(pose, sg.curvature, sg.signed_length * (a - s0) / sg.length)
            out.append(Segment(sg.curvature, b - a, sg.reverse, sg.mow))
        pose = advance(pose, sg.curvature, sg.signed_length)
        s0 = s1
    return (new_start or start), out


def trim_piece(start: Pose, segs: list[Segment], checker: Checker, step: float = 0.05):
    """Keep the longest collision-free stretch of an open piece: at a break the
    piece may end (or start) too close to the corner for the mower's outline."""
    smp = Trajectory(start, list(segs)).sample(step)
    ok = checker.free_poses(smp.x, smp.y, smp.theta)
    if ok.all():
        return start, segs
    best, run0 = (0, 0), None
    for i, f in enumerate(list(ok) + [False]):
        if f and run0 is None:
            run0 = i
        elif not f and run0 is not None:
            if i - run0 > best[1] - best[0]:
                best = (run0, i)
            run0 = None
    if best[1] - best[0] < 2:
        return start, []
    return cut_segments(start, segs, float(smp.s[best[0]]), float(smp.s[best[1] - 1]))


def walk(pts, vs, heading, length, i0: int, s0: float, steps: int) -> list[Segment]:
    """Segments from the point s0 along edge i0, over ``steps`` edges: straight parts,
    fillets at bends, manoeuvres at corners; stops before a break vertex."""
    n = len(pts)
    segs: list[Segment] = []
    for j in range(steps):
        i = (i0 + j) % n
        nxt = (i + 1) % n
        begin = s0 if j == 0 else vs[i].t_out
        segs.append(Segment(0.0, max(length[i] - vs[nxt].t_in - begin, 0.0)))
        vx = vs[nxt]
        if vx.kind == "break":
            break
        if vx.kind == "bend":
            segs.append(Segment(math.copysign(1.0 / vx.radius, vx.phi),
                                vx.radius * abs(vx.phi)))
        elif vx.kind == "corner":
            segs.extend(vx.move)
    return segs


@dataclass
class Headland:
    rings: list[Ring]
    first_offset: float
    spacing: float
    passes: int
    interior: object            # Shapely geometry still to mow (inside the headland)
    edge_strip: float           # width along the border no blade can reach (m)


def orient_ring(ring, poly) -> list:
    """Coordinates of a ring of ``poly`` walked with the lawn on the left."""
    p = orient(poly, sign=1.0)                 # exterior CCW, holes CW
    if LinearRing(ring.coords).equals(p.exterior):
        return list(p.exterior.coords)
    for i in p.interiors:
        if LinearRing(ring.coords).equals(i):
            return list(i.coords)
    return list(ring.coords)


def headland_passes_needed(mower: Mower, turn_ahead: float, c1: float, spacing: float,
                           margin: float) -> int:
    """Rings needed so that the cut of the headland meets the end of the passes.

    A pass must end ``turn_ahead + margin`` from the border (the turn sticks out
    ``turn_ahead`` beyond the rear axle); its blades cut up to ``deck_front`` ahead of
    the rear axle. Ring k cuts up to c1 + (k - 1) s + w / 2 from the border."""
    need = turn_ahead + margin - deck_front(mower)          # uncut depth to fill
    reach1 = c1 + mower.cutting_width / 2
    return max(1, math.ceil((need - reach1) / spacing) + 1)


def rings_at(lawn: Lawn, offset: float, mower: Mower, checker: Checker, need: float,
             mow_reverse: bool) -> list[Ring]:
    out = []
    for poly in polygons(lawn_polygon(lawn).buffer(-offset, join_style="mitre",
                                                    mitre_limit=10)):
        for ring in [poly.exterior, *poly.interiors]:
            r = build_ring(np.asarray(orient_ring(ring, poly)), offset, mower, checker,
                           mow_reverse, need=need)
            if r is not None:
                out.append(r)
    return out


def plan_headland(lawn: Lawn, mower: Mower, passes: int = 2, overlap: float = 0.05,
                  margin: float = 0.05, mow_reverse: bool = False,
                  checker: Checker | None = None, max_slack: float = 0.5,
                  break_cost: float = 4.0) -> Headland:
    """Headland rings. The first ring keeps the clearance h + margin + e from the
    border. The extra slack e (by steps of 5 cm) is the one that minimises
    e P + break_cost F, where P is the length of the border and F the number of
    corners left without a collision-free path (each one cuts its ring) plus
    the number of pieces that still collide (the outline grazes the margin when
    e = 0): the
    slack leaves a strip of area about e P uncut, a break a patch of about
    ``break_cost`` square metres."""
    checker = checker or Checker(lawn, mower, margin=margin)
    s = mower.cutting_width - overlap
    base = lateral_half_width(mower) + margin
    border = lawn_polygon(lawn).boundary.length
    best = None
    for slack in np.arange(0.0, max_slack + 1e-9, 0.05):
        rings = rings_at(lawn, base + float(slack), mower, checker, base + float(slack),
                         mow_reverse)
        failed = sum(r.failed + sum(not checker.free(*pc) for pc in r.pieces)
                     for r in rings) if rings else 10 ** 6
        score = (slack * border + break_cost * failed, failed)
        if best is None or score < best[0]:
            best = (score, base + float(slack), rings)
        if failed == 0:
            break
    _, c1, first = best
    rings = list(first)
    for k in range(1, passes):
        rings += rings_at(lawn, c1 + k * s, mower, checker, c1, mow_reverse)
    inner = c1 + (passes - 1) * s + mower.cutting_width / 2 - overlap
    interior = lawn_polygon(lawn).buffer(-inner, join_style="mitre", mitre_limit=10)
    strip = c1 - mower.cutting_width / 2
    return Headland(rings, c1, s, passes, interior, strip)


def reverse_piece(start: Pose, segs: list[Segment]) -> tuple[Pose, list[Segment]]:
    """The same forward path driven the other way: start at its end, heading back,
    segments in reverse order with opposite curvatures."""
    end = Trajectory(start, list(segs)).end
    back = [Segment(-sg.curvature, sg.length, sg.reverse, sg.mow) for sg in reversed(segs)]
    return Pose(end.x, end.y, end.theta + math.pi), back


def link_rings(rings: list[Ring], mower: Mower, checker: Checker, start: Pose | None = None,
               mow_reverse: bool = False) -> Trajectory:
    """Drive the rings one after the other (outer rings first), joining them with the
    fastest collision-free Reeds-Shepp connection (blades off). A closed ring may be
    entered anywhere on a straight part; an open piece at either end."""
    traj = None
    for ring in rings:
        for piece in ring.pieces:
            if traj is None and start is None:
                traj = Trajectory(piece[0], list(piece[1]))
                continue
            if traj is None:
                traj = Trajectory(start)
            here = traj.end
            if ring.closed:
                entry, conn = best_entry(here, ring, mower, checker, mow_reverse)
                options = [] if conn is None else [(conn, entry)]
            else:
                options = []
                for p0, segs in (piece, reverse_piece(*piece)):
                    if not checker.free(p0, segs):   # driven backwards the mower faces
                        continue                     # the other way: check it again
                    conn = connect(here, p0, mower, checker, mow=False)
                    if conn is not None:
                        options.append((conn, segs))
            if not options:                # no direct path: search around obstacles
                from .search import hybrid_astar
                from ..paths import path_time
                cands = [piece] if ring.closed else [
                    c for c in (piece, reverse_piece(*piece)) if checker.free(*c)]
                for p0, segs in cands:
                    path = hybrid_astar(here, p0, mower, checker)
                    if path is not None:
                        t = path_time(path, mower.v_mow, mower.v_turn, mower.v_reverse,
                                      mower.t_shift)
                        from .common import Connection
                        options.append((Connection(path, t), segs))
            if not options:
                continue                   # unreachable: left uncut, reported
            conn, segs = min(options, key=lambda o: o[0].time)
            traj.extend(conn.segments)
            traj.extend(segs)
    return traj


def rotate_ring(ring: Ring, at: float) -> tuple[Pose, list[Segment]]:
    """The ring started at a distance ``at`` along it (on a straight segment)."""
    from ..kinematics import advance
    pose, acc = ring.start, 0.0
    for i, seg in enumerate(ring.segments):
        if acc + seg.length >= at and seg.curvature == 0 and not seg.reverse:
            d = max(at - acc, 0.0)
            p = advance(pose, 0.0, d)
            first = Segment(0.0, seg.length - d)
            rest = ring.segments[i + 1:] + ring.segments[:i] + [Segment(0.0, d)]
            return p, [first, *rest]
        pose = advance(pose, seg.curvature, seg.signed_length)
        acc += seg.length
    return ring.start, list(ring.segments)


def best_entry(here: Pose, ring: Ring, mower: Mower, checker: Checker,
               mow_reverse: bool, tries: int = 12):
    """Try several entry points along the ring (closest first); keep the fastest
    collision-free connection."""
    total = sum(s.length for s in ring.segments)
    cand = []
    for at in np.linspace(0, total, 60, endpoint=False):
        p, segs = rotate_ring(ring, float(at))
        if segs and segs[0].curvature == 0:
            cand.append((math.hypot(p.x - here.x, p.y - here.y), p, segs))
    cand.sort(key=lambda c: c[0])
    best = (None, None)
    for _, p, segs in cand[:tries]:
        conn = connect(here, p, mower, checker, mow=False, mow_reverse=False)
        if conn is not None and (best[1] is None or conn.time < best[1].time):
            best = (segs, conn)
    return best
