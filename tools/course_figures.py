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


def main():
    fig_lawns()
    fig_raster()
    fig_turning_circle()
    fig_effective_width()
    fig_next_pass()
    fig_ties()
    fig_rs_examples()
    fig_demo()


if __name__ == "__main__":
    main()
