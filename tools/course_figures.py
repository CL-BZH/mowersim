"""Draw the figures of the course that come from the code (course/figures/*.pdf).

    python tools/course_figures.py

The schematic drawings (the mower's dimensions, the bicycle model, the
construction of the paths) are TikZ, in the course itself.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mowersim import Lawn, Mower, Pose, Trajectory, evaluate, example_lawns  # noqa: E402
from mowersim.coverage import Raster, effective_width  # noqa: E402
from mowersim.demo import back_and_forth  # noqa: E402
from mowersim.geometry import rectangle  # noqa: E402
from mowersim.paths import candidates, path_time  # noqa: E402
from mowersim.viz import C, draw_coverage, draw_lawn, draw_mower, draw_trajectory  # noqa: E402

OUT = ROOT / "course" / "figures"
M = Mower()
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9,
                     "pdf.fonttype": 42,
                     "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    print("wrote", OUT / name)


def fig_lawns():
    lawns = example_lawns()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6),
                             gridspec_kw={"width_ratios": [20, 24, 34]})
    for ax, name in zip(axes, ("rectangle", "l_shape", "garden")):
        lawn = lawns[name]
        draw_lawn(ax, lawn)
        draw_mower(ax, M, lawn.start)
        ax.set_title(f"{name}: {lawn.area:.0f} m²")
        ax.tick_params(labelsize=7)
    save(fig, "lawns.pdf")


def fig_raster():
    lawn = example_lawns()["garden"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    # (a) cells of a curved border, at a coarse resolution so that they are visible
    res = 0.25
    r = Raster.of(lawn, res)
    ax = axes[0]
    ny, nx = r.shape
    ax.imshow(np.where(r.mask, 1, 0), origin="lower", cmap="Greens", vmin=-0.3, vmax=1.6,
              extent=(r.x0, r.x0 + nx * res, r.y0, r.y0 + ny * res), interpolation="nearest")
    ax.plot(*np.r_[lawn.outer, lawn.outer[:1]].T, color="#263238", lw=1.2)
    ax.set_xlim(0, 6)
    ax.set_ylim(7, 13)
    ax.set_xticks(np.arange(0, 6.01, 1))
    ax.set_aspect("equal")
    ax.set_title(f"(a) a cell is lawn if its centre is\n"
                 f"(cells of {res * 100:.0f} cm for visibility)")
    # (b), (c) a straight band on the grid: aligned and at 17 degrees
    for ax, deg, tag in ((axes[1], 0, "(b)"), (axes[2], 17, "(c)")):
        res = 0.1
        sq = Lawn.make(rectangle(-1, -1.5, 3, 1.5))
        t = Trajectory(Pose(0, 0, math.radians(deg))).then(0, 2.0)
        result = evaluate(sq, M, t, res=res, check_collisions=False)
        draw_coverage(ax, result)
        a = math.radians(deg)
        for side in (-1, 1):                              # exact edges of the band
            off = side * M.cutting_width / 2 * np.array([-math.sin(a), math.cos(a)])
            p0 = off + M.deck_offset * np.array([math.cos(a), math.sin(a)])
            p1 = p0 + 2.0 * np.array([math.cos(a), math.sin(a)])
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="black", lw=1)
        ax.set_xlim(0, 3.2)
        ax.set_ylim(-1, 1.6)
        band = M.cutting_width * 2.0
        ax.set_title(f"{tag} band at {deg}°, {res * 100:.0f} cm cells\n"
                     f"cut area {result.cut_area:.2f} m² (exact {band + 0.389:.2f} m²)")
        ax.tick_params(labelsize=7)
    save(fig, "raster.pdf")


def fig_turning_circle():
    R = M.min_radius
    lawn = Lawn.make(rectangle(-3.5, -1.5, 3.5, 5))
    t = Trajectory(Pose(0, 0, 0)).then(M.max_curvature, 2 * math.pi * R)
    res = evaluate(lawn, M, t, check_collisions=False)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    draw_coverage(ax, res)
    icr = (0.0, R)
    c = M.blade_centres
    d = np.linalg.norm(c - np.array(icr), axis=1)
    rho_in, rho_out = d.min() - M.blade_radius, d.max() + M.blade_radius
    for rad, style, label in ((R, "-", f"rear axle, R = {R:.2f} m"),
                              (rho_in, "--", f"ρ_in = {rho_in:.2f} m"),
                              (rho_out, ":", f"ρ_out = {rho_out:.2f} m")):
        ax.add_patch(Circle(icr, rad, fill=False, ls=style, lw=1.2, ec="black", label=label))
    ax.plot(*icr, "k+", ms=10)
    ax.annotate("C", icr, xytext=(4, 4), textcoords="offset points")
    for a in (0, 2 * math.pi / 3, 4 * math.pi / 3):
        th = a
        pose = Pose(R * math.sin(th), R - R * math.cos(th), th)
        draw_mower(ax, M, pose, curvature=M.max_curvature, alpha=0.85)
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-1.1, 4.1)
    ax.set_aspect("equal")
    ax.legend(loc="lower center", fontsize=7, ncol=1, framealpha=0.9, bbox_to_anchor=(0.5, -0.36))
    ax.set_title(f"full lock to the left: the blades sweep an annulus\n"
                 f"measured {res.cut_area:.2f} m², π(ρ_out² − ρ_in²) = "
                 f"{math.pi * (rho_out ** 2 - rho_in ** 2):.2f} m²")
    save(fig, "turning_circle.pdf")


def fig_effective_width():
    ks = np.linspace(0, M.max_curvature, 100)
    fig, ax = plt.subplots(figsize=(4.0, 2.4))
    ax.plot(ks, [effective_width(M, k) for k in ks], color=C["forward"])
    ax.axhline(M.cutting_width, color="grey", ls="--", lw=1)
    ax.text(0.02, M.cutting_width + 0.002, "w (straight)", fontsize=7, va="bottom")
    ax.set_xlabel("curvature κ (1/m)")
    ax.set_ylabel("w_eff(κ) (m)")
    ax.set_title("area swept per metre driven by the rear axle")
    ax.grid(alpha=0.3)
    save(fig, "effective_width.pdf")


def _arrow(ax, pose, color, label=None):
    L = 0.6
    ax.annotate("", xy=(pose.x + L * math.cos(pose.theta), pose.y + L * math.sin(pose.theta)),
                xytext=(pose.x, pose.y),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6), zorder=8)
    ax.plot(pose.x, pose.y, "o", color=color, ms=4, zorder=8)
    if label:
        ax.annotate(label, (pose.x, pose.y), xytext=(-10, -10), textcoords="offset points",
                    fontsize=7, color=color)


def _draw_path(ax, segs, start, goal, title, mowers=True):
    """Rear-axle path (blue forward, red dashed reverse), start and goal as heading
    arrows, and the mower drawn where it changes direction (black triangle)."""
    from mowersim.kinematics import advance
    segs = [type(s)(s.curvature, s.length, s.reverse, True) for s in segs]
    traj = Trajectory(start).extend(segs)
    draw_trajectory(ax, traj, lw=1.8)
    _arrow(ax, start, "#2e7d32", "start")
    _arrow(ax, goal, "#6a1b9a", "goal")
    pose = start
    for i, seg in enumerate(segs[:-1]):
        pose = advance(pose, seg.curvature, seg.signed_length)
        if seg.reverse != segs[i + 1].reverse:
            if mowers:
                draw_mower(ax, M, pose, curvature=seg.curvature, alpha=0.3, blades=False,
                           ref_point=False, zorder=2)
            ax.plot(pose.x, pose.y, "k^", ms=5, zorder=7)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.margins(0.12)
    ax.set_title(title)
    ax.tick_params(labelsize=7)


def fig_next_pass():
    R = M.min_radius
    spacing = M.cutting_width - 0.05
    a, b = Pose(0, 0, 0), Pose(0, spacing, math.pi)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, rev, name in ((axes[0], False, "Dubins (forward only)"),
                          (axes[1], True, "Reeds–Shepp (with reverse)")):
        w = candidates(a, b, R, reverse=rev)[0]
        segs = w.segments(R)
        t = path_time(segs, M.v_mow, M.v_turn, M.v_reverse, M.t_shift)
        _draw_path(ax, segs, a, b, f"{name}: {w.letters}\n{w.length * R:.2f} m, {t:.1f} s")
    save(fig, "next_pass.pdf")


def fig_ties():
    R = M.min_radius
    spacing = M.cutting_width - 0.05
    a, b = Pose(0, 0, 0), Pose(0, spacing, math.pi)
    words = [w for w in candidates(a, b, R) if abs(w.length - candidates(a, b, R)[0].length)
             < 1e-6]
    fig, axes = plt.subplots(1, len(words), figsize=(7.2, 2.4))
    for ax, w in zip(axes, words):
        segs = w.segments(R)
        t = path_time(segs, M.v_mow, M.v_turn, M.v_reverse, M.t_shift)
        sig = " ".join(f"{c}{'+' if x > 0 else '−'}" for c, x in zip(w.letters, w.lengths))
        _draw_path(ax, segs, a, b, f"{sig}\n{w.length * R:.2f} m, {t:.2f} s")
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-2.3, 3.2)
    save(fig, "ties.pdf")


def fig_rs_examples():
    rng = np.random.default_rng(7)
    R = M.min_radius
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    wanted = ["LSL", "LSR", "LRL", "LRLR", "LRSL", "LRSLR"]
    found = {}
    while len(found) < len(wanted):
        a = Pose(0, 0, 0)
        b = Pose(*rng.uniform(-5, 5, 2), rng.uniform(-math.pi, math.pi))
        w = candidates(a, b, R)[0]
        key = w.letters if w.letters in wanted else w.letters.translate(str.maketrans("LR", "RL"))
        visible = min(abs(t) for t in w.lengths) > 0.35 and w.lengths[0] > 0
        # every segment visible, and starting forward (the time-flipped words are the
        # same shapes driven backwards)
        if key in wanted and key not in found and 2.0 < math.hypot(b.x, b.y) and visible:
            found[key] = (b, w)
    for ax, key in zip(axes.ravel(), wanted):
        b, w = found[key]
        sig = " ".join(f"{c}{'+' if x > 0 else '−'}" for c, x in zip(w.letters, w.lengths))
        _draw_path(ax, w.segments(R), Pose(0, 0, 0), b, f"{sig}   ({w.length * R:.2f} m)",
                   mowers=False)
    fig.tight_layout()
    save(fig, "rs_examples.pdf")


def fig_demo():
    lawn = example_lawns()["rectangle"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    for row, room in enumerate((False, True)):
        traj = back_and_forth(lawn, M, room_for_turns=room)
        res = evaluate(lawn, M, traj)
        d = res.as_dict()
        draw_lawn(axes[row, 0], lawn)
        draw_trajectory(axes[row, 0], traj, lw=1)
        draw_mower(axes[row, 0], M, traj.start)
        draw_coverage(axes[row, 1], res, lawn)
        what = "passes to the edge" if not room else "room left for the turns"
        axes[row, 0].set_title(f"{what}: trajectory")
        axes[row, 1].set_title(f"coverage {d['coverage_%']}%, {d['collisions']} collision "
                               f"samples\n{d['length_m']} m, {d['time_min']} min")
        for ax in axes[row]:
            ax.tick_params(labelsize=7)
    fig.tight_layout(h_pad=2.0)
    fig.savefig(OUT / "demo.png", dpi=110)          # for the README
    save(fig, "demo.pdf")


# ------------------------------------------------------------- chapter 6
def _headland(name):
    from mowersim.planning.common import Checker
    from mowersim.planning.headland import link_rings, plan_headland
    lawn = example_lawns()[name]
    ck = Checker(lawn, M, 0.05)
    h = plan_headland(lawn, M, passes=2, checker=ck)
    return lawn, h, link_rings(h.rings, M, ck)


def fig_offsets():
    from shapely.geometry import Polygon as SPolygon  # noqa: F401
    from mowersim.planning.common import lawn_polygon, polygons
    lawn = example_lawns()["garden"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, c in zip(axes, (0.6, 1.8, 3.0)):
        draw_lawn(ax, lawn, alpha=0.6)
        for poly in polygons(lawn_polygon(lawn).buffer(-c, join_style="mitre",
                                                        mitre_limit=10)):
            for ring in [poly.exterior, *poly.interiors]:
                xy = np.asarray(ring.coords)
                ax.plot(xy[:, 0], xy[:, 1], color=C["forward"], lw=1.4)
        ax.set_title(f"offset by c = {c} m")
        ax.tick_params(labelsize=7)
    save(fig, "offsets.pdf")


def fig_swing():
    from mowersim.kinematics import advance
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    R = M.min_radius
    start = Pose(0, 0, 0)
    pts = M.footprint_points(0.02)
    t = Trajectory(start).then(0, 1.5).then(M.max_curvature, R * math.pi / 2)
    smp = t.sample(0.02)
    for x, y, th in zip(smp.x[::4], smp.y[::4], smp.theta[::4]):
        w = np.c_[x + np.cos(th) * pts[:, 0] - np.sin(th) * pts[:, 1],
                  y + np.sin(th) * pts[:, 0] + np.cos(th) * pts[:, 1]]
        ax.plot(w[:, 0], w[:, 1], ",", color="#d84315", alpha=0.25)
    ax.plot(smp.x, smp.y, color=C["forward"], lw=1.8, label="rear axle")
    h = 0.515
    ax.axvline(1.5 + R + h, color="black", ls="--", lw=1)
    ax.text(1.5 + R + h + 0.05, 0.3, "half-width\nfrom the end\nof the turn",
            fontsize=7, va="bottom")
    # the farthest point reached to the right
    xs = []
    for x, y, th in zip(smp.x, smp.y, smp.theta):
        xs.append((x + np.cos(th) * pts[:, 0] - np.sin(th) * pts[:, 1]).max())
    ax.axvline(max(xs), color="#c62828", lw=1)
    ax.annotate("", xy=(max(xs), 1.6), xytext=(1.5 + R + h, 1.6),
                arrowprops=dict(arrowstyle="<->", color="#c62828"))
    ax.text((max(xs) + 1.5 + R + h) / 2, 1.7, f"{max(xs) - (1.5 + R + h):.2f} m",
            color="#c62828", ha="center", fontsize=8)
    draw_mower(ax, M, advance(advance(start, 0, 1.5), M.max_curvature, R * 0.6),
               curvature=M.max_curvature, alpha=0.9)
    ax.set_aspect("equal")
    ax.set_xlim(-0.8, 4.6)
    ax.set_ylim(-0.8, 3.2)
    ax.set_title("quarter turn at full lock: the front corner swings out")
    ax.tick_params(labelsize=7)
    save(fig, "swing.pdf")


def fig_corner():
    from mowersim.planning.common import Checker
    from mowersim.planning.headland import corner_path, unit
    lawn = example_lawns()["rectangle"]
    ck = Checker(lawn, M, 0.05)
    c1 = 0.615
    V = np.array([20 - c1, c1])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    cases = ((0.0, M.min_radius, 2 * M.min_radius, "no shift: δ = 0, ρ = 1.43 m"),
             None)
    # the solution the planner finds
    from mowersim.planning.headland import Vertex, solve_corner, solve_corner_shift
    vx = Vertex(V, math.pi / 2)
    rooms = ((20 - 2 * c1) / 2, (12 - 2 * c1) / 2)
    how = "shift then turn"
    if not solve_corner_shift(vx, 0.0, math.pi / 2, *rooms, M, ck):
        solve_corner(vx, 0.0, math.pi / 2, *rooms, M, ck, False)
        how = f"manoeuvre with reverse, b = {vx.t_in:.1f} m"
    found = vx
    for ax, case in zip(axes, (cases[0], "found")):
        draw_lawn(ax, lawn)
        if case == "found":
            segs, b_in = found.move, found.t_in
            title = f"planner's choice: {how}"
        else:
            delta, rho, rs, title = case
            segs, b_in, _ = corner_path(V, 0.0, math.pi / 2, math.pi / 2, delta, rho, rs)
        start = Pose(*(V - b_in * unit(0.0)), 0.0)
        traj = Trajectory(start).extend(segs)
        smp = traj.sample(0.02)
        ok = ck.free_poses(smp.x, smp.y, smp.theta)
        draw_trajectory(ax, Trajectory(start, [type(sg)(sg.curvature, sg.length, sg.reverse,
                                                        True) for sg in segs]), lw=1.6)
        if (~ok).any():
            ax.plot(smp.x[~ok], smp.y[~ok], ".", color="#c62828", ms=2)
        for f in (0.0, 0.35, 0.7, 1.0):
            i = min(int(f * (len(smp.x) - 1)), len(smp.x) - 1)
            draw_mower(ax, M, Pose(smp.x[i], smp.y[i], smp.theta[i]), smp.curvature[i],
                       alpha=0.55, blades=False, ref_point=False)
        ax.plot([V[0] - 9, V[0]], [V[1], V[1]], ":", color="black", lw=1)
        ax.plot([V[0], V[0]], [V[1], V[1] + 9], ":", color="black", lw=1)
        ax.set_xlim(11.5, 20.3)
        ax.set_ylim(-0.3, 8.2)
        ax.set_title(title + ("\ncollision (red)" if (~ok).any() else
                              "\nthe mower stays on the lawn"))
        ax.tick_params(labelsize=7)
    save(fig, "corner.pdf")


def fig_headlands():
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.6),
                             gridspec_kw={"height_ratios": [12, 20, 20]})
    for row, name in enumerate(("rectangle", "l_shape", "garden")):
        lawn, h, traj = _headland(name)
        res = evaluate(lawn, M, traj)
        d = res.as_dict()
        draw_lawn(axes[row, 0], lawn)
        draw_trajectory(axes[row, 0], traj, lw=1)
        draw_coverage(axes[row, 1], res, lawn)
        axes[row, 0].set_title(f"{name}: two headland passes, c₁ = {h.first_offset:.2f} m")
        axes[row, 1].set_title(f"cut {d['coverage_%']}%, {d['collisions']} collisions, "
                               f"{d['time_min']} min")
        for ax in axes[row]:
            ax.tick_params(labelsize=7)
    fig.tight_layout(h_pad=1.5)
    save(fig, "headlands.pdf")


# ----------------------------------------------------------------- chapter 7

_PLANS = {}


def _plan(name, pattern):
    from mowersim.planning.swaths import plan_simple
    if (name, pattern) not in _PLANS:
        lawn = example_lawns()[name]
        traj, info = plan_simple(lawn, M, pattern=pattern)
        _PLANS[name, pattern] = (lawn, traj, info, evaluate(lawn, M, traj))
    return _PLANS[name, pattern]


def _outline_reach(segs, start):
    from mowersim.planning.common import footprint_samples
    smp = Trajectory(start).extend(segs).sample(0.02)
    pts = footprint_samples(M, 0.05)
    return max(float((x + math.cos(t) * pts[:, 0] - math.sin(t) * pts[:, 1]).max())
               for x, t in zip(smp.x, smp.theta))


def fig_shift():
    """End of a pass and start of the next: the rear axle moves out by 2d."""
    from mowersim.paths import fastest
    d, s = M.deck_offset, M.cutting_width - 0.05
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4),
                             gridspec_kw={"width_ratios": [1, 1]})
    for ax, jump, title in ((axes[0], 1, "next pass: three-point turn"),
                            (axes[1], 4, "four passes over: forward turn")):
        a, b = Pose(-d, 0.0, 0.0), Pose(d, jump * s, math.pi)
        segs = fastest(a, b, M)
        t = path_time(segs, M.v_mow, M.v_turn, M.v_reverse, M.t_shift)
        for q in (0.0, jump * s):
            ax.plot([-4.5, 0], [q, q], color="#1565c0", lw=1.0, ls=":")
            ax.add_patch(plt.Rectangle((-4.5, q - M.cutting_width / 2), 4.5, M.cutting_width,
                                       color="#a5d6a7", alpha=0.5, lw=0, zorder=0))
        ax.axvline(0, color="#2e7d32", lw=1.2)
        reach = _outline_reach(segs, a)
        ax.axvline(reach, color="#c62828", lw=1.0, ls="--")
        ax.annotate("", xy=(reach, -1.05), xytext=(a.x, -1.05),
                    arrowprops=dict(arrowstyle="<->", color="#c62828", lw=0.9))
        label = rf"$a_{{\mathrm{{turn}}}}$ = {reach - a.x:.2f} m"
        ax.text((a.x + reach) / 2, -1.25, label, color="#c62828", ha="center", va="top",
                fontsize=7)
        _draw_path(ax, segs, a, b, f"{title}\n{t:.2f} s", mowers=False)
        draw_mower(ax, M, a, alpha=0.3, ref_point=False, zorder=3)
        draw_mower(ax, M, b, alpha=0.3, ref_point=False, zorder=3)
        ax.annotate("", xy=(b.x, jump * s + 0.75), xytext=(a.x, jump * s + 0.75),
                    arrowprops=dict(arrowstyle="<->", color="k", lw=0.8))
        ax.text(0, jump * s + 0.85, "$2d$", ha="center", va="bottom", fontsize=8)
        top = jump * s + 1.5
        ax.set_xlim(-4.5, reach + 0.6)
        ax.set_ylim(-1.6, top)
    fig.tight_layout()
    save(fig, "shift.pdf")


def fig_turn_times():
    from mowersim.planning.swaths import turn_ahead, turn_time
    s = M.cutting_width - 0.05
    j = np.arange(1, 11)
    rs = [turn_time(M, k * s) for k in j]
    db = [turn_time(M, k * s, reverse=False) for k in j]
    ah = [turn_ahead(M, k * s) for k in j]
    ahd = [turn_ahead(M, k * s, reverse=False) for k in j]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    axes[0].plot(j, db, "o-", color="#ef6c00", ms=3.5, label="forward only (Dubins)")
    axes[0].plot(j, rs, "s-", color="#1565c0", ms=3.5, label="with reverse (Reeds–Shepp)")
    axes[0].set_ylabel("time of the turn (s)")
    axes[1].plot(j, ahd, "o-", color="#ef6c00", ms=3.5, label="forward only (Dubins)")
    axes[1].plot(j, ah, "s-", color="#1565c0", ms=3.5, label="with reverse (Reeds–Shepp)")
    axes[1].set_ylabel(r"$a_{\mathrm{turn}}$: reach beyond the axle (m)")
    for ax in axes:
        ax.set_xlabel("jump (passes of s = 0.92 m)")
        ax.set_xticks(j)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "turn_times.pdf")


def fig_direction():
    from mowersim.planning.swaths import estimate_time, turn_time
    s = M.cutting_width - 0.05
    turn = turn_time(M, s)
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    th = np.radians(np.arange(0, 180.1, 2.5))
    for name, color in (("rectangle", "#1565c0"), ("l_shape", "#ef6c00")):
        lawn, h, _ = _headland(name)
        tt = [estimate_time(h.interior, x, M, 0.05, turn) / 60 for x in th]
        ax.plot(np.degrees(th), tt, color=color, label=f"{name} (interior)")
        k = int(np.argmin(tt))
        ax.plot(np.degrees(th[k]), tt[k], "o", color=color, ms=4)
    ax.set_xlabel("direction of the passes θ (degrees)")
    ax.set_ylabel("estimated time (min)")
    ax.set_xticks(range(0, 181, 15))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "direction.pdf")


def fig_patterns():
    rows = (("rectangle", "adjacent"), ("rectangle", "skip"), ("l_shape", "adjacent"))
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 9.0),
                             gridspec_kw={"height_ratios": [12, 12, 20]})
    for row, (name, pattern) in enumerate(rows):
        lawn, traj, info, res = _plan(name, pattern)
        d = res.as_dict()
        pp = info["passes_plan"]
        draw_lawn(axes[row, 0], lawn)
        draw_trajectory(axes[row, 0], traj, lw=0.9)
        for rank, j in enumerate(pp.order):
            m = (pp.swaths[j].a + pp.swaths[j].b) / 2
            axes[row, 0].text(m[0] + (rank % 2) * 1.2, m[1], str(rank + 1), fontsize=6,
                              ha="center", va="center", zorder=9,
                              bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
        draw_coverage(axes[row, 1], res, lawn)
        axes[row, 0].set_title(f"{name}, {pattern}: {info['headland_passes']} headland "
                               f"+ {info['passes']} passes")
        axes[row, 1].set_title(f"cut {d['coverage_%']}%, {d['direction_changes']} changes "
                               f"of direction, {d['time_min']} min")
        for ax in axes[row]:
            ax.tick_params(labelsize=7)
    fig.tight_layout(h_pad=1.5)
    save(fig, "patterns.pdf")


# ----------------------------------------------------------------- chapter 8

def _pond():
    from shapely.geometry import Polygon, box
    return box(0, 0, 20, 14).difference(Polygon([(7, 5), (12, 4), (13, 9), (8, 10)]))


def _fill_cells(ax, cells, labels=True, swath_lines=False):
    from mowersim.planning.common import polygons
    from mowersim.planning.swaths import swaths
    colors = ["#90caf9", "#ffcc80", "#a5d6a7", "#ef9a9a", "#ce93d8", "#fff59d", "#80cbc4"]
    for k, c in enumerate(cells):
        for p in polygons(c.polygon):
            x, y = p.exterior.xy
            ax.fill(x, y, color=colors[k % len(colors)], alpha=0.75, lw=0)
            ax.plot(x, y, color="k", lw=0.7)
            for h in p.interiors:
                hx, hy = h.xy
                ax.fill(hx, hy, color="white")
                ax.plot(hx, hy, color="k", lw=0.7)
        if swath_lines:
            for sw in swaths(c.polygon, c.theta, M):
                ax.plot([sw.a[0], sw.b[0]], [sw.a[1], sw.b[1]], color="#1565c0", lw=0.6)
        if labels:
            q = c.polygon.representative_point()
            txt = f"{k + 1}" + (f"\n{math.degrees(c.theta):.0f}°" if swath_lines else "")
            ax.text(q.x, q.y, txt, ha="center", va="center", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)


def fig_decomposition():
    from mowersim.planning.cells import best_decomposition, decompose, slices
    pond = _pond()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.4))
    # strips
    ax = axes[0, 0]
    class _C:
        def __init__(self, p):
            self.polygon, self.theta = p, 0.0
    pieces = [_C(p) for strip in slices(pond, 0.0) for p in strip]
    _fill_cells(ax, pieces, labels=False)
    ys = sorted({y for _, y in pond.exterior.coords} | {y for _, y in pond.interiors[0].coords})
    for y in ys:
        ax.axhline(y, color="#c62828", lw=0.6, ls="--")
    ax.set_title(f"cut at every vertex: {len(pieces)} pieces")
    # merged
    ax = axes[0, 1]
    cells = decompose(pond, 0.0)
    _fill_cells(ax, cells)
    hole = np.asarray(pond.interiors[0].coords)[:-1]
    for k in (int(np.argmin(hole[:, 1])), int(np.argmax(hole[:, 1]))):
        ax.plot(*hole[k], "o", color="#c62828", ms=5, zorder=5)
    ax.set_title(f"merged one to one: {len(cells)} cells")
    # L-shape interior
    _, h, _ = _headland("l_shape")
    ax = axes[1, 0]
    one = decompose(h.interior, 0.0, max_jump=None)
    two = decompose(h.interior, 0.0)
    _fill_cells(ax, two)
    ax.set_title(f"L-shape, θ = 0°: {len(one)} cell one to one,\n"
                 f"{len(two)} with the rule on the ends")
    ax = axes[1, 1]
    best = best_decomposition(h.interior, M)
    _fill_cells(ax, best, swath_lines=True)
    ax.set_title(f"L-shape, best: {len(best)} cells, own directions")
    fig.tight_layout()
    save(fig, "decomposition.pdf")


def fig_garden_cells():
    from mowersim.planning.cells import best_decomposition
    from mowersim.planning.common import Checker
    from mowersim.planning.headland import plan_headland
    lawn = example_lawns()["garden"]
    h = plan_headland(lawn, M, passes=2, checker=Checker(lawn, M, 0.05))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for ax, own in ((axes[0], False), (axes[1], True)):
        draw_lawn(ax, lawn, alpha=0.5)
        cells = best_decomposition(h.interior, M, own_directions=own)
        _fill_cells(ax, cells, swath_lines=True)
        ax.set_title(("one direction" if not own else "a direction per cell")
                     + f": {len(cells)} cells")
    fig.tight_layout()
    save(fig, "garden_cells.pdf")


_CELLPLANS = {}


def _cell_plan(name, pattern):
    from mowersim.planning.cells import plan_cells
    if (name, pattern) not in _CELLPLANS:
        lawn = example_lawns()[name]
        traj, info = plan_cells(lawn, M, pattern=pattern)
        _CELLPLANS[name, pattern] = (lawn, traj, info, evaluate(lawn, M, traj))
    return _CELLPLANS[name, pattern]


def fig_cell_plans():
    rows = (("l_shape", "skip"), ("garden", "adjacent"))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6), gridspec_kw={"height_ratios": [20, 21]})
    for row, (name, pattern) in enumerate(rows):
        lawn, traj, info, res = _cell_plan(name, pattern)
        d = res.as_dict()
        draw_lawn(axes[row, 0], lawn)
        draw_trajectory(axes[row, 0], traj, lw=0.8)
        draw_coverage(axes[row, 1], res, lawn)
        axes[row, 0].set_title(f"{name}, {pattern}: {len(info['cells'])} cells, "
                               f"{info['passes']} passes")
        axes[row, 1].set_title(f"cut {d['coverage_%']}%, {d['collisions']} collisions, "
                               f"{d['time_min']} min")
        for ax in axes[row]:
            ax.tick_params(labelsize=7)
    fig.tight_layout(h_pad=1.5)
    save(fig, "cell_plans.pdf")


# ----------------------------------------------------------------- chapter 9

def fig_hybrid():
    from mowersim.planning.common import Checker
    from mowersim.planning.search import grid_distance, hybrid_astar
    from mowersim.geometry import rectangle as rect
    lawn = Lawn.make(rect(0, 0, 16, 10), holes=[rect(7.5, 0.0001, 8.5, 6.5)])
    ck = Checker(lawn, M, 0.05)
    a, b = Pose(3, 2, math.pi / 2), Pose(13, 2, -math.pi / 2)
    trace = []
    segs = hybrid_astar(a, b, M, ck, trace=trace)
    dist, k = grid_distance(ck, b)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    r = ck.raster
    cell = r.res * k
    ext = (r.x0, r.x0 + dist.shape[1] * cell, r.y0, r.y0 + dist.shape[0] * cell)
    d = np.where(np.isfinite(dist), dist, np.nan)
    im = axes[0].imshow(d, origin="lower", extent=ext, cmap="viridis")
    fig.colorbar(im, ax=axes[0], fraction=0.035, pad=0.02)
    axes[0].set_title("heuristic: grid distance to the goal (m)")
    draw_lawn(axes[1], lawn)
    xs = [p.x for p in trace]
    ys = [p.y for p in trace]
    axes[1].plot(xs, ys, ".", color="#8d6e63", ms=1.5, alpha=0.6)
    _draw_path(axes[1], segs, a, b, f"Hybrid A*: {len(trace)} poses expanded, "
               f"{path_time(segs, M.v_mow, M.v_turn, M.v_reverse, M.t_shift):.1f} s",
               mowers=False)
    for ax in axes:
        ax.add_patch(plt.Rectangle((7.5, 0), 1, 6.5, color="#5d4037", zorder=3))
        ax.plot(*zip(*[(a.x, a.y)]), "o", color="#2e7d32", ms=4, zorder=6)
        ax.plot(*zip(*[(b.x, b.y)]), "o", color="#6a1b9a", ms=4, zorder=6)
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    save(fig, "hybrid.pdf")


_FINAL = {}


def _final(name):
    from mowersim.planning.planner import plan
    if name not in _FINAL:
        lawn = example_lawns()[name]
        traj, info = plan(lawn, M)
        _FINAL[name] = (lawn, traj, info, evaluate(lawn, M, traj))
    return _FINAL[name]


def fig_final():
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 9.4),
                             gridspec_kw={"height_ratios": [12, 20, 21]})
    for row, name in enumerate(("rectangle", "l_shape", "garden")):
        lawn, traj, info, res = _final(name)
        d = res.as_dict()
        draw_lawn(axes[row, 0], lawn)
        draw_trajectory(axes[row, 0], traj, lw=0.8)
        for rank, (c, _) in enumerate(info["plans"]):
            q = c.polygon.representative_point()
            axes[row, 0].text(q.x, q.y, f"{rank + 1}", fontsize=8, ha="center", va="center",
                              color="white", zorder=9, fontweight="bold",
                              bbox=dict(boxstyle="circle,pad=0.2", fc="#37474f", ec="none"))
        draw_coverage(axes[row, 1], res, lawn)
        axes[row, 0].set_title(f"{name}: {info['pattern']}, {len(info['cells'])} cell"
                               f"{'s' if len(info['cells']) > 1 else ''}")
        axes[row, 1].set_title(f"cut {d['coverage_%']}%, {d['collisions']} collisions, "
                               f"{d['time_min']} min, η = {d['efficiency']}")
        for ax in axes[row]:
            ax.tick_params(labelsize=7)
    fig.tight_layout(h_pad=1.5)
    save(fig, "final.pdf")


def fig_breakdown():
    from mowersim.planning.planner import time_breakdown
    names = ("rectangle", "l_shape", "garden")
    parts = (("passes", "straight, blades on", "#43a047"),
             ("turns", "arcs and reverse, blades on", "#fb8c00"),
             ("transfers", "blades off", "#9e9e9e"),
             ("stops", "changes of direction", "#e53935"))
    fig, ax = plt.subplots(figsize=(7.2, 2.4))
    for y, name in enumerate(names):
        lawn, traj, info, res = _final(name)
        b = time_breakdown(traj, M)
        left = 0.0
        for key, label, color in parts:
            w = b[key] / 60
            ax.barh(y, w, left=left, color=color, label=label if y == 0 else None)
            if w > 0.6:
                ax.text(left + w / 2, y, f"{w:.1f}", ha="center", va="center", fontsize=7,
                        color="white")
            left += w
        lb = res.as_dict()["lower_bound_m"] / M.v_mow / 60
        ax.plot([lb, lb], [y - 0.4, y + 0.4], color="k", lw=1.5,
                label="lower bound" if y == 0 else None)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("time (min)")
    ax.legend(fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              frameon=False, handlelength=1.2, columnspacing=1.0)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save(fig, "breakdown.pdf")


FIGURES = {
    "ch2-5": [fig_lawns, fig_raster, fig_turning_circle, fig_effective_width, fig_next_pass,
              fig_ties, fig_rs_examples, fig_demo],
    "ch6": [fig_offsets, fig_swing, fig_corner, fig_headlands],
    "ch7": [fig_shift, fig_turn_times, fig_direction, fig_patterns],
    "ch8": [fig_decomposition, fig_garden_cells, fig_cell_plans],
    "ch9": [fig_hybrid, fig_final, fig_breakdown],
}


def main(argv=None):
    """``python tools/course_figures.py [chapters...]``, e.g. ``ch6``; all by default."""
    names = (argv if argv is not None else sys.argv[1:]) or list(FIGURES)
    for name in names:
        for f in FIGURES[name]:
            f()


if __name__ == "__main__":
    main()
