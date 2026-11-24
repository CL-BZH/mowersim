import math
import random

import numpy as np
import pytest

from mowersim import (Lawn, Mower, Pose, Trajectory, advance, candidates, effective_width,
                      evaluate, example_lawns, length_lower_bound)
from mowersim.demo import back_and_forth
from mowersim.geometry import rectangle, signed_area
from mowersim.paths import dubins_words, fastest, path_time, pify, shortest

M = Mower()


def random_pose(rng, span=5.0):
    return Pose(rng.uniform(-span, span), rng.uniform(-span, span), rng.uniform(-math.pi, math.pi))


def euler(start, segments, h=1e-4):
    """Independent integration of the kinematic model (midpoint rule, tiny steps)."""
    x, y, th = start
    for s in segments:
        n = int(s.length / h) + 1
        ds = s.length / n * (-1 if s.reverse else 1)
        for _ in range(n):
            x += ds * math.cos(th + 0.5 * s.curvature * ds)
            y += ds * math.sin(th + 0.5 * s.curvature * ds)
            th += s.curvature * ds
    return x, y, th


# ------------------------------------------------------------------- geometry
def test_area_orientation_and_holes():
    sq = rectangle(0, 0, 4, 3)
    assert signed_area(np.array(sq)) == pytest.approx(12)
    lawn = Lawn.make(sq[::-1], holes=[rectangle(1, 1, 2, 2)])   # given clockwise
    assert lawn.area == pytest.approx(11)
    assert lawn.contains([[0.5, 0.5], [1.5, 1.5], [5, 5]]).tolist() == [True, False, False]


def test_example_lawns_round_trip(tmp_path):
    for name, lawn in example_lawns().items():
        p = tmp_path / f"{name}.json"
        import json
        p.write_text(json.dumps(lawn.to_dict()))
        again = Lawn.from_file(p)
        assert again.area == pytest.approx(lawn.area, rel=1e-6)
        assert again.contains([[again.start.x, again.start.y]])[0]


# --------------------------------------------------------------------- mower
def test_mower_derived_quantities():
    assert M.min_radius == pytest.approx(M.wheelbase / math.tan(math.radians(M.max_steer_deg)))
    assert M.cutting_width == pytest.approx(0.97)
    left, right = M.ackermann_angles(M.max_curvature)
    # inner (left) wheel turns more; the bicycle angle lies between the two
    assert left > M.steering_angle(M.max_curvature) > right > 0


# ---------------------------------------------------------------- kinematics
def test_arc_is_exact():
    rng = random.Random(0)
    for _ in range(20):
        p = random_pose(rng)
        k = rng.uniform(-M.max_curvature, M.max_curvature)
        s = rng.uniform(-3, 3)
        seg = Trajectory(p).then(k, abs(s), reverse=s < 0).segments
        x, y, th = euler(p, seg)
        q = advance(p, k, s)
        assert math.hypot(q.x - x, q.y - y) < 1e-6 and abs(pify(q.theta - th)) < 1e-6


def test_full_circle_returns_to_start():
    t = Trajectory(Pose(1, 2, 0.3)).then(M.max_curvature, 2 * math.pi * M.min_radius)
    e = t.end
    assert (e.x, e.y) == pytest.approx((1, 2), abs=1e-9)


def test_trajectory_counts_direction_changes():
    t = Trajectory(Pose(0, 0, 0)).then(0, 1).then(0.5, 1, reverse=True).then(0, 1)
    assert t.direction_changes() == 2 and t.length == 3


# --------------------------------------------------------------------- paths
def test_reeds_shepp_reaches_the_goal_independently():
    rng = random.Random(1)
    for _ in range(200):
        a, b = random_pose(rng), random_pose(rng)
        segs = shortest(a, b, M.min_radius)
        x, y, th = euler(a, segs)
        assert math.hypot(x - b.x, y - b.y) < 1e-6 and abs(pify(th - b.theta)) < 1e-6
        assert all(abs(s.curvature) <= M.max_curvature + 1e-12 for s in segs)


def test_reeds_shepp_all_48_words_occur():
    rng = random.Random(2)
    seen = set()
    for _ in range(3000):
        for w in candidates(random_pose(rng), random_pose(rng), 1.0):
            seen.add((w.letters, tuple(t > 0 for t in w.lengths)))
    assert len(seen) == 48


def test_reeds_shepp_is_a_symmetric_distance_with_triangle_inequality():
    rng = random.Random(3)
    d = lambda a, b: candidates(a, b, 1.0)[0].length  # noqa: E731
    for _ in range(300):
        a, b, c = random_pose(rng), random_pose(rng), random_pose(rng)
        assert d(a, b) == pytest.approx(d(b, a), abs=1e-6)
        assert d(a, c) <= d(a, b) + d(b, c) + 1e-6
        assert d(a, b) >= math.hypot(a.x - b.x, a.y - b.y) - 1e-9


def test_reeds_shepp_never_longer_than_an_independent_implementation():
    rsplan = pytest.importorskip("rsplan")
    rng = random.Random(4)
    for _ in range(300):
        a, b = random_pose(rng), random_pose(rng)
        ref = rsplan.planner.path((a.x, a.y, a.theta), (b.x, b.y, b.theta), 1.0, 0, 0.05)
        assert candidates(a, b, 1.0)[0].length <= ref.total_length + 1e-6


def test_dubins_matches_reeds_shepp_when_forward_is_optimal():
    rng = random.Random(5)
    checked = 0
    for _ in range(2000):
        a, b = random_pose(rng), random_pose(rng)
        rs = candidates(a, b, 1.0)[0]
        dub = candidates(a, b, 1.0, reverse=False)
        assert dub and dub[0].length >= rs.length - 1e-9
        if all(t >= 0 for t in rs.lengths):
            checked += 1
            assert dub[0].length == pytest.approx(rs.length, abs=1e-6)
    assert checked > 100
    assert len(dubins_words(3.0, 1.0, 0.5)) >= 4


def test_turn_to_next_pass_reverse_beats_forward_only():
    """The case every mowing pattern meets: the next pass is one cutting width away,
    less than twice the turning radius."""
    a, b = Pose(0, 0, 0), Pose(0, M.cutting_width, math.pi)
    rs = candidates(a, b, M.min_radius)[0]
    dub = candidates(a, b, M.min_radius, reverse=False)[0]
    assert 2 * M.min_radius > M.cutting_width
    assert rs.length < 0.6 * dub.length
    fast = fastest(a, b, M)
    t = path_time(fast, M.v_mow, M.v_turn, M.v_reverse, M.t_shift)
    assert t <= path_time(dub.segments(M.min_radius), M.v_mow, M.v_turn, M.v_reverse,
                          M.t_shift) + 1e-9


# ------------------------------------------------------------------ coverage
def test_full_circle_sweeps_the_annulus():
    lawn = Lawn.make(rectangle(-6, -6, 6, 8))
    t = Trajectory(Pose(0, 0, 0)).then(M.max_curvature, 2 * math.pi * M.min_radius)
    r = evaluate(lawn, M, t, check_collisions=False)
    annulus = 2 * math.pi * M.min_radius * effective_width(M, M.max_curvature)
    assert r.cut_area == pytest.approx(annulus, rel=0.01)


def test_straight_pass_area_at_an_angle():
    lawn = Lawn.make(rectangle(-3, -3, 14, 10))
    t = Trajectory(Pose(0, 0, math.radians(17))).then(0, 10)
    r = evaluate(lawn, M, t, check_collisions=False)
    # band w x 10, plus the front and back halves of the two blade discs, minus their lens
    d, rr = 0.47, M.blade_radius
    lens = 2 * rr * rr * math.acos(d / (2 * rr)) - d / 2 * math.sqrt(4 * rr * rr - d * d)
    exact = M.cutting_width * 10 + 2 * math.pi * rr ** 2 - lens
    assert r.cut_area == pytest.approx(exact, rel=0.005)
    assert r.overlap_area == 0


def test_blades_off_cut_nothing_and_second_pass_counts():
    lawn = Lawn.make(rectangle(-3, -3, 14, 3))
    off = Trajectory(Pose(0, 0, 0)).then(0, 5, mow=False)
    assert evaluate(lawn, M, off, check_collisions=False).cut_area == 0
    back = Trajectory(Pose(0, 0, 0)).then(0, 5).then(0, 5, reverse=True)
    r = evaluate(lawn, M, back, check_collisions=False)
    assert r.overlap_area > 0.8 * M.cutting_width * 4


def test_collisions_and_lower_bound():
    lawn = Lawn.make(rectangle(0, 0, 10, 5))
    inside = Trajectory(Pose(2, 2.5, 0)).then(0, 3)
    outside = Trajectory(Pose(2, 2.5, 0)).then(0, 8)
    assert evaluate(lawn, M, inside).collisions == 0
    assert evaluate(lawn, M, outside).collisions > 0
    assert length_lower_bound(lawn, M) < lawn.area / M.cutting_width
    assert effective_width(M, M.max_curvature) > M.cutting_width   # the deck is ahead


def test_demo_room_for_turns_stays_on_the_lawn():
    lawn = example_lawns()["rectangle"]
    with_room = evaluate(lawn, M, back_and_forth(lawn, M, room_for_turns=True))
    to_edge = evaluate(lawn, M, back_and_forth(lawn, M, room_for_turns=False))
    assert with_room.collisions == 0 and to_edge.collisions > 0
    assert to_edge.coverage > with_room.coverage
    assert with_room.efficiency <= 1


def test_cli(tmp_path, capsys):
    from mowersim.__main__ import main
    main(["path", "0", "0.92", "180", "--compare", "--png", str(tmp_path / "p.png")])
    main(["demo", "lawns/rectangle.json", "--png", str(tmp_path / "d.png"), "--save",
          str(tmp_path / "t.json")])
    main(["evaluate", "lawns/rectangle.json", str(tmp_path / "t.json")])
    out = capsys.readouterr().out
    assert "Reeds-Shepp" in out and "coverage_%" in out
    assert (tmp_path / "d.png").stat().st_size > 0


# ---------------------------------------------------------------- headlands
def test_swing_out_is_the_front_corner_circle():
    from mowersim.planning.headland import swing_out
    for R in (1.43, 3.0, 10.0):
        # front corner of the body (1.55, -0.475); half-width 0.515 (deck housing)
        assert swing_out(M, R) == pytest.approx(math.hypot(1.55, R + 0.475) - (R + 0.515),
                                                abs=2e-3)
    assert swing_out(M, 2.0) > swing_out(M, 10.0) > 0


def test_shift_and_corner_paths_land_where_they_should():
    from mowersim.planning.headland import corner_path, shift_arcs, unit
    segs, d = shift_arcs(0.4, 3.0, left=True)
    end = Trajectory(Pose(0, 0, 0), segs).end
    assert (end.x, end.y, end.theta) == pytest.approx((d, 0.4, 0.0), abs=1e-9)
    p = np.array([5.0, 1.0])
    for phi in (math.pi / 2, -math.pi / 2, 2.0):
        segs, b_in, b_out = corner_path(p, 0.0, phi, phi, 0.5, 2.0, 4.0)
        end = Trajectory(Pose(*(p - b_in * unit(0.0)), 0.0), segs).end
        target = p + b_out * unit(phi)
        assert (end.x, end.y) == pytest.approx(tuple(target), abs=1e-9)
        assert pify(end.theta - phi) == pytest.approx(0, abs=1e-9)


def test_headland_on_the_rectangle():
    from mowersim.planning.common import Checker
    from mowersim.planning.headland import link_rings, plan_headland
    lawn = example_lawns()["rectangle"]
    ck = Checker(lawn, M, 0.05)
    h = plan_headland(lawn, M, passes=2, checker=ck)
    assert all(r.closed for r in h.rings) and len(h.rings) == 2
    r = evaluate(lawn, M, link_rings(h.rings, M, ck))
    assert r.collisions == 0 and r.coverage > 0.4
    assert h.interior.area < lawn.area


def test_hybrid_astar_goes_around_an_obstacle():
    from mowersim.planning.common import Checker
    from mowersim.planning.search import hybrid_astar
    # a wall across the middle of a 16 x 10 lawn, with a passage at the top
    lawn = Lawn.make(rectangle(0, 0, 16, 10), holes=[rectangle(7.5, 0.0001, 8.5, 6.5)])
    ck = Checker(lawn, M, 0.05)
    a, b = Pose(3, 2, math.pi / 2), Pose(13, 2, -math.pi / 2)
    assert candidates(a, b, M.min_radius)            # a direct path exists...
    segs = hybrid_astar(a, b, M, ck)
    assert segs is not None and ck.free(a, segs)     # ...but this one avoids the wall
    end = Trajectory(a, segs).end
    assert math.hypot(end.x - b.x, end.y - b.y) < 1e-5


# ------------------------------------------------------------ swaths, turns
def test_swath_count_and_spacing():
    from shapely.geometry import box
    from mowersim.planning.swaths import swaths
    region = box(2, 2, 18, 10)                       # W = 8 across theta = 0
    sw = swaths(region, 0.0, M, overlap=0.05)
    s, w, o = M.cutting_width - 0.05, M.cutting_width, 0.05
    assert len(sw) == 1 + math.ceil((8 - w + 2 * o) / s)
    q = [x.q for x in sw]
    assert q[0] == pytest.approx(2 + w / 2 - o) and q[-1] == pytest.approx(10 - w / 2 + o)
    assert np.diff(q).max() == pytest.approx(s)
    assert sw[0].a[0] == pytest.approx(2 + M.blade_radius)   # stops r before the edge


def test_forward_turn_matches_the_lsl_formula():
    from mowersim.planning.swaths import min_forward_jump, turn_time
    R, d = M.min_radius, M.deck_offset
    jump = 3.68
    straight = math.hypot(2 * d, jump - 2 * R)
    assert turn_time(M, jump) == pytest.approx(math.pi * R / M.v_turn + straight / M.v_mow,
                                               abs=1e-6)
    assert min_forward_jump(M, 0.92) == 3


def test_pass_order_is_a_permutation_and_saves_time():
    from mowersim.planning.swaths import order_cost, pass_order, turn_time
    n, s = 12, 0.92
    order = pass_order(n, "skip", M, s)
    assert sorted(order) == list(range(n))
    cost = {j: turn_time(M, j * s) for j in range(1, n)}
    assert order_cost(order, cost) < 0.8 * order_cost(list(range(n)), cost)


def test_passes_on_a_rectangular_region():
    from shapely.geometry import box
    from mowersim.planning.common import Checker
    from mowersim.planning.swaths import plan_passes
    lawn = example_lawns()["rectangle"]
    ck = Checker(lawn, M, 0.05)
    region = box(3.5, 3, 16.5, 9)
    for pattern in ("adjacent", "skip"):
        pp = plan_passes(region, 0.0, M, ck, Pose(10, 1.5, math.pi), pattern)
        assert pp.turn_failures == 0 and sorted(pp.order) == list(range(pp.passes))
        assert ck.free(Pose(10, 1.5, math.pi), pp.trajectory.segments)
