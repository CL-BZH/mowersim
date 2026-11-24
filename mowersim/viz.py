"""Top-view drawings: the lawn, the mower, trajectories and coverage maps."""
from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Polygon

from .geometry import Lawn, Pose
from .kinematics import Trajectory
from .mower import Mower, to_world

C = {"lawn": "#b7dd9b", "outside": "#f2f2f2", "hole": "#8d6e63", "border": "#33691e",
     "body": "#d84315", "wheel": "#212121", "deck": "#455a64", "blade": "#eceff1",
     "forward": "#1565c0", "reverse": "#c62828", "transit": "#9e9e9e",
     "missed": "#e53935", "once": "#43a047", "twice": "#fb8c00"}


def draw_lawn(ax, lawn: Lawn, alpha: float = 1.0) -> None:
    ax.add_patch(Polygon(lawn.outer, closed=True, fc=C["lawn"], ec=C["border"], lw=1.5,
                         alpha=alpha, zorder=0))
    for h in lawn.holes:
        ax.add_patch(Polygon(h, closed=True, fc=C["hole"], ec="#4e342e", lw=1, zorder=1))
    x0, y0, x1, y1 = lawn.bounds
    m = 0.05 * max(x1 - x0, y1 - y0)
    ax.set_xlim(x0 - m, x1 + m)
    ax.set_ylim(y0 - m, y1 + m)
    ax.set_aspect("equal")
    ax.set_facecolor(C["outside"])


def draw_mower(ax, mower: Mower, pose: Pose, curvature: float = 0.0, alpha: float = 1.0,
               blades: bool = True, ref_point: bool = True, zorder: int = 5) -> None:
    """The mower seen from above: chassis, four wheels (front ones steered with
    Ackermann angles), deck housing and blade circles, and the reference point."""
    x, y, th = pose

    def poly(pts, **kw):
        ax.add_patch(Polygon(to_world(pts, x, y, th), closed=True, alpha=alpha,
                             zorder=zorder, **kw))

    poly(mower.body_outline(), fc=C["body"], ec="#5d1a07", lw=1)
    poly(mower.deck_outline(), fc=C["deck"], ec="#263238", lw=1)
    if blades:
        for c in mower.blade_centres:
            cw = to_world(c[None, :], x, y, th)[0]
            ax.add_patch(Circle(cw, mower.blade_radius, fc=C["blade"], ec="#263238", lw=0.8,
                                alpha=0.9 * alpha, zorder=zorder))
            d = mower.blade_radius * np.array([math.cos(th + 0.4), math.sin(th + 0.4)])
            ax.plot(*np.c_[cw - d, cw + d], color="#263238", lw=1.6, alpha=alpha,
                    zorder=zorder)                                   # the blade bar
    hw, hl = mower.wheel_width / 2, mower.wheel_diameter / 2
    wheel = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    left, right = mower.ackermann_angles(curvature)
    for wx, wy, steer in ((0, mower.track / 2, 0), (0, -mower.track / 2, 0),
                          (mower.wheelbase, mower.track / 2, left),
                          (mower.wheelbase, -mower.track / 2, right)):
        c, s = math.cos(steer), math.sin(steer)
        pts = np.c_[wx + c * wheel[:, 0] - s * wheel[:, 1], wy + s * wheel[:, 0] + c * wheel[:, 1]]
        poly(pts, fc=C["wheel"], ec="black", lw=0.5)
    if ref_point:
        ax.plot(x, y, "o", ms=4, color="white", mec="black", zorder=zorder + 1, alpha=alpha)


def draw_trajectory(ax, traj: Trajectory, step: float = 0.05, lw: float = 1.4) -> None:
    """Rear-axle path: blue forward, red dashed in reverse, grey forward with the
    blades off."""
    smp = traj.sample(step)
    pts = np.c_[smp.x, smp.y]
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    colors = np.where(smp.reverse[:-1], C["reverse"],
                      np.where(smp.mow[:-1], C["forward"], C["transit"]))
    styles = ["--" if r else "-" for r in smp.reverse[:-1]]
    for style in ("-", "--"):
        sel = np.array([s == style for s in styles])
        if sel.any():
            ax.add_collection(LineCollection(segs[sel], colors=colors[sel], lw=lw,
                                             linestyles=style, zorder=3))


def draw_coverage(ax, result, lawn: Lawn | None = None) -> None:
    """Missed lawn in red, cut once in green, cut twice or more in orange."""
    r = result.raster
    img = np.full(r.shape, 0, dtype=int)                     # 0 = not lawn
    img[r.mask & (result.passes == 0)] = 1
    img[r.mask & (result.passes == 1)] = 2
    img[r.mask & (result.passes >= 2)] = 3
    cmap = ListedColormap([C["outside"], C["missed"], C["once"], C["twice"]])
    ny, nx = r.shape
    ax.imshow(img, origin="lower", cmap=cmap, vmin=0, vmax=3, interpolation="nearest",
              extent=(r.x0, r.x0 + nx * r.res, r.y0, r.y0 + ny * r.res), zorder=0)
    if lawn is not None:
        for poly in (lawn.outer, *lawn.holes):
            ax.add_patch(Polygon(poly, closed=True, fill=False, ec="#263238", lw=1, zorder=2))
        x0, y0, x1, y1 = lawn.bounds
        m = 0.05 * max(x1 - x0, y1 - y0)
        ax.set_xlim(x0 - m, x1 + m)
        ax.set_ylim(y0 - m, y1 + m)
    ax.set_aspect("equal")


def plot_result(lawn: Lawn, mower: Mower, traj: Trajectory, result, path=None, title=""):
    """Two panels: the trajectory on the lawn, and the coverage map."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    draw_lawn(axes[0], lawn)
    draw_trajectory(axes[0], traj)
    draw_mower(axes[0], mower, traj.start)
    draw_coverage(axes[1], result, lawn)
    d = result.as_dict()
    axes[0].set_title(title or f"{lawn.name}: trajectory (blue forward, red dashed reverse)")
    axes[1].set_title(f"coverage {d['coverage_%']}%, missed {d['missed_m2']} m², "
                      f"cut twice {d['overlap_m2']} m²\n{d['length_m']} m, "
                      f"{d['direction_changes']} changes of direction, {d['time_min']} min, "
                      f"{d['collisions']} collision samples")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        plt.close(fig)
    return fig


def animate(lawn: Lawn, mower: Mower, traj: Trajectory, path, fps: int = 20,
            seconds_per_frame: float = 1.0) -> None:
    """GIF of the mower driving the trajectory (needs pillow)."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    smp = traj.sample(0.02)
    # one frame per ``seconds_per_frame`` of driving, using the speeds of the mower
    speed = np.where(smp.reverse, mower.v_reverse,
                     np.where(smp.curvature == 0, mower.v_mow, mower.v_turn))
    t = np.concatenate([[0], np.cumsum(np.diff(smp.s) / speed[:-1])])
    frames = np.searchsorted(t, np.arange(0, t[-1], seconds_per_frame))
    fig, ax = plt.subplots(figsize=(7, 7 * (lawn.bounds[3] - lawn.bounds[1] + 2)
                                    / (lawn.bounds[2] - lawn.bounds[0] + 2)))

    def update(i):
        ax.clear()
        draw_lawn(ax, lawn)
        j = frames[i]
        pts = np.c_[smp.x[:j + 1], smp.y[:j + 1]]
        if len(pts) > 1:
            ax.plot(pts[:, 0], pts[:, 1], color=C["forward"], lw=1)
        draw_mower(ax, mower, Pose(smp.x[j], smp.y[j], smp.theta[j]), smp.curvature[j])
        ax.set_title(f"{lawn.name}  t = {t[j] / 60:.1f} min")
        return []

    anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
