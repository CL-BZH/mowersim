"""Command line.

    python -m mowersim show lawns/garden.json                # lawn + mower at the start
    python -m mowersim mower                                 # top view of the mower
    python -m mowersim path 3 2 180                          # shortest path to (3 m, 2 m, 180 deg)
    python -m mowersim demo lawns/rectangle.json --png out.png --gif out.gif
    python -m mowersim evaluate lawns/rectangle.json traj.json
    python -m mowersim plan lawns/rectangle.json --pattern skip --png plan.png

Every command accepts ``--mower config.json`` to use your own dimensions.
"""
from __future__ import annotations

import argparse
import json
import math

import matplotlib

from .coverage import evaluate
from .demo import back_and_forth
from .geometry import Lawn, Pose
from .kinematics import Trajectory
from .mower import Mower
from .paths import candidates, path_time


def _mower(a) -> Mower:
    return Mower.from_file(a.mower) if a.mower else Mower()


def _save_or_show(fig, path):
    import matplotlib.pyplot as plt
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print("wrote", path)
    else:
        plt.show()


def cmd_show(a):
    import matplotlib.pyplot as plt
    from .viz import draw_lawn, draw_mower
    lawn, m = Lawn.from_file(a.lawn), _mower(a)
    fig, ax = plt.subplots(figsize=(8, 6))
    draw_lawn(ax, lawn)
    draw_mower(ax, m, lawn.start or Pose(*lawn.outer.mean(axis=0), 0.0))
    ax.set_title(f"{lawn.name}: {lawn.area:.1f} m² to mow")
    _save_or_show(fig, a.png)


def cmd_mower(a):
    import matplotlib.pyplot as plt
    from .viz import draw_mower
    m = _mower(a)
    print(json.dumps(m.to_dict(), indent=1))
    print(f"minimum turning radius {m.min_radius:.3f} m, cutting width {m.cutting_width:.3f} m")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, k, title in ((axes[0], 0.0, "straight"), (axes[1], m.max_curvature, "full lock left")):
        draw_mower(ax, m, Pose(0, 0, 0), curvature=k)
        ax.set_xlim(-0.8, 2.0)
        ax.set_ylim(-0.9, 0.9)
        ax.set_aspect("equal")
        ax.set_title(title)
    _save_or_show(fig, a.png)


def cmd_path(a):
    import matplotlib.pyplot as plt
    from .viz import draw_mower, draw_trajectory
    m = _mower(a)
    start, goal = Pose(0, 0, 0), Pose(a.x, a.y, math.radians(a.deg))
    for rev in ((True, False) if a.compare else (not a.forward_only,)):
        words = candidates(start, goal, m.min_radius, reverse=rev)
        w = words[0]
        segs = w.segments(m.min_radius, mow=False)
        t = path_time(segs, m.v_mow, m.v_turn, m.v_reverse, m.t_shift)
        name = "Reeds-Shepp" if rev else "Dubins"
        print(f"{name:12s} {w.letters:6s} length {w.length * m.min_radius:6.2f} m, "
              f"time {t:5.1f} s, lengths {[round(x * m.min_radius, 2) for x in w.lengths]}")
    fig, ax = plt.subplots(figsize=(7, 6))
    traj = Trajectory(start).extend(segs)
    draw_trajectory(ax, traj, lw=2)
    draw_mower(ax, m, start, alpha=0.5)
    draw_mower(ax, m, goal, alpha=0.5)
    ax.set_aspect("equal")
    ax.autoscale()
    _save_or_show(fig, a.png)


def cmd_demo(a):
    from .viz import animate, plot_result
    lawn, m = Lawn.from_file(a.lawn), _mower(a)
    traj = back_and_forth(lawn, m, room_for_turns=not a.no_room)
    res = evaluate(lawn, m, traj, res=a.res)
    print(json.dumps(res.as_dict(), indent=1))
    if a.save:
        traj.save(a.save)
    if a.png or not a.gif:
        matplotlib.use("Agg") if a.png else None
        fig = plot_result(lawn, m, traj, res)
        _save_or_show(fig, a.png)
    if a.gif:
        animate(lawn, m, traj, a.gif)
        print("wrote", a.gif)


def cmd_evaluate(a):
    from .viz import plot_result
    lawn, m = Lawn.from_file(a.lawn), _mower(a)
    traj = Trajectory.load(a.trajectory)
    res = evaluate(lawn, m, traj, res=a.res)
    print(json.dumps(res.as_dict(), indent=1))
    if a.png:
        _save_or_show(plot_result(lawn, m, traj, res), a.png)


def cmd_plan(a):
    from .planning.planner import plan, time_breakdown
    from .viz import animate, plot_result
    lawn, m = Lawn.from_file(a.lawn), _mower(a)
    traj, info = plan(lawn, m, pattern=a.pattern, order=a.order)
    res = evaluate(lawn, m, traj, res=a.res)
    dirs = ", ".join(f"{math.degrees(c.theta):.0f}" for c in info["cells"])
    print(f"order of the passes: {info['pattern']}; {info['headland_passes']} headland passes; "
          f"{len(info['cells'])} cells (directions {dirs} deg); {info['passes']} passes")
    print("time (s):", {k: round(v, 1) for k, v in time_breakdown(traj, m).items()})
    print(json.dumps(res.as_dict(), indent=1))
    if a.save:
        traj.save(a.save)
    if a.png:
        _save_or_show(plot_result(lawn, m, traj, res), a.png)
    if a.gif:
        animate(lawn, m, traj, a.gif)
        print("wrote", a.gif)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m mowersim")
    ap.add_argument("--mower", help="JSON file with the mower's dimensions and speeds")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show")
    s.add_argument("lawn")
    s.add_argument("--png")
    s.set_defaults(func=cmd_show)
    s = sub.add_parser("mower")
    s.add_argument("--png")
    s.set_defaults(func=cmd_mower)
    s = sub.add_parser("path", help="shortest path from (0, 0, 0) to a pose")
    s.add_argument("x", type=float)
    s.add_argument("y", type=float)
    s.add_argument("deg", type=float)
    s.add_argument("--forward-only", action="store_true", help="Dubins instead of Reeds-Shepp")
    s.add_argument("--compare", action="store_true", help="print both")
    s.add_argument("--png")
    s.set_defaults(func=cmd_path)
    s = sub.add_parser("demo", help="back and forth on the lawn's bounding box (not a planner)")
    s.add_argument("lawn")
    s.add_argument("--no-room", action="store_true", help="run the passes to the edge")
    s.add_argument("--res", type=float, default=0.05)
    s.add_argument("--save", help="save the trajectory (JSON)")
    s.add_argument("--png")
    s.add_argument("--gif")
    s.set_defaults(func=cmd_demo)
    s = sub.add_parser("evaluate")
    s.add_argument("lawn")
    s.add_argument("trajectory")
    s.add_argument("--res", type=float, default=0.05)
    s.add_argument("--png")
    s.set_defaults(func=cmd_evaluate)
    s = sub.add_parser("plan", help="plan a path: headland, cells, passes (course, ch. 9)")
    s.add_argument("lawn")
    s.add_argument("--pattern", choices=("best", "adjacent", "skip"), default="best",
                   help="order of the passes (course, chapter 7); best tries both")
    s.add_argument("--order", choices=("optimal", "greedy"), default="optimal",
                   help="order of the cells (course, chapter 9)")
    s.add_argument("--res", type=float, default=0.05)
    s.add_argument("--save", help="save the trajectory (JSON)")
    s.add_argument("--png")
    s.add_argument("--gif")
    s.set_defaults(func=cmd_plan)
    a = ap.parse_args(argv)
    if getattr(a, "png", None):
        matplotlib.use("Agg")
    a.func(a)


if __name__ == "__main__":
    main()
