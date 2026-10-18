"""
workspace_table_map.py  -  Sections 5 and 6: task and dexterous workspace at
the table's working heights, for a VERTICAL (top-down) gripper, with the table
and tray drawn on top.

For every (x, y) cell and each of 8 gripper yaws (0, 45, ..., 315 deg) the
planners from compare_ik.py (DLS, then QP, from two seeds) solve IK for the
TCP (fingertip centre) at
  grasp height : table top + cube half-size   (TCP at the cube centre)
  lift height  : grasp + 0.15 m
A yaw counts as feasible if both converge (1.5 mm, 0.01 rad).

  task workspace      : >= 1 yaw feasible   (the cube can be grasped somehow)
  dexterous workspace : all 8 yaws feasible (any cube yaw can be grasped)

  python3 workspace_table_map.py            -> workspace_table_map.png / .npz
"""

import os as _os
SCENE_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "scene")
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import compare_ik as C

ap = argparse.ArgumentParser()
ap.add_argument("--xml", default=_os.path.join(SCENE_DIR, "pick_place_cube_scene.xml"))
ap.add_argument("--step", type=float, default=0.04)
ap.add_argument("--out", default="workspace_table_map")
args = ap.parse_args()

C.PLAN_MAX_ITERS = 150                     # unreachable cells give up sooner
S = C.Scene(args.xml)
z_grasp = S.table_top + S.cube_h
z_lift = z_grasp + 0.15
xs = np.arange(-0.10, 0.90 + 1e-9, args.step)
ys = np.arange(-0.50, 0.50 + 1e-9, args.step)
yaws = np.radians(np.arange(0, 360, 45))
count = np.full((len(ys), len(xs)), np.nan)
seed2 = S.q_home.copy()
seed2[6] = np.clip(seed2[6] - np.pi, S.q_lo_hw[6] + 0.05, S.q_hi_hw[6] - 0.05)
SEEDS = (S.q_home, seed2)

for i, y in enumerate(ys):
    for j, x in enumerate(xs):
        if np.hypot(x, y) < 0.15:          # inside the robot base
            continue
        n = 0
        if np.hypot(x, y) > 0.95:          # far outside the arm's reach
            count[i, j] = 0
            continue
        for yaw in yaws:
            R = C.grasp_rotation(yaw, 0.0)
            # Feasible = SOME IK solution exists. Single-start planners can stall
            # at joint 7's limit (DLS) or in a local minimum (QP), so try both,
            # from the home pose and from a seed with the wrist turned by -pi.
            ok = False
            for seed in SEEDS:
                for meth in ("dls", "qp"):
                    ok, _, _, _, q = C.plan_ik(S, meth, np.array([x, y, z_grasp]), R, seed)
                    if ok:
                        ok, *_ = C.plan_ik(S, meth, np.array([x, y, z_lift]), R, q)
                    if ok:
                        break
                if ok:
                    break
            n += int(ok)
        count[i, j] = n
    print(f"row {i + 1}/{len(ys)}", flush=True)

np.savez(args.out + ".npz", xs=xs, ys=ys, count=count, z_grasp=z_grasp, z_lift=z_lift)

# ---------- table coverage ----------
x0, x1, y0, y1 = S.table_xy
X, Y = np.meshgrid(xs, ys)
on_table = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
task = np.nansum((count >= 1) & on_table) / on_table.sum()
dex = np.nansum((count == len(yaws)) & on_table) / on_table.sum()
far = X[(count == len(yaws)) & on_table & (np.abs(Y) < 0.05)]
print(f"Table area in task workspace: {task:.0%}, in dexterous workspace: {dex:.0%}")
if far.size:
    print(f"Dexterous reach along y = 0: x up to {far.max():.2f} m")

# ---------- figure ----------
fig, ax = plt.subplots(figsize=(7.5, 6.6))
cmap = plt.get_cmap("viridis", len(yaws) + 1)
pm = ax.pcolormesh(xs, ys, count, cmap=cmap, vmin=-0.5, vmax=len(yaws) + 0.5, shading="nearest")
cb = fig.colorbar(pm, ax=ax, ticks=range(len(yaws) + 1), shrink=0.85)
cb.set_label("feasible gripper yaws (of 8), vertical gripper, grasp + lift")
c = np.nan_to_num(count, nan=-1)
ax.contour(xs, ys, c, levels=[0.5], colors="w", linewidths=1.6, linestyles="--")
ax.contour(xs, ys, c, levels=[len(yaws) - 0.5], colors="w", linewidths=2.0)
ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#ff9f1c", lw=2.5))
ax.text(x0 + 0.01, y1 + 0.015, f"table {x1 - x0:.2f} x {y1 - y0:.2f} m, top z = {S.table_top:.2f} m",
        color="#ff9f1c", fontsize=9, fontweight="bold")
bp, bh = S.box_pos, S.box_half_outer
ax.add_patch(plt.Rectangle((bp[0] - bh, bp[1] - bh), 2 * bh, 2 * bh, fill=False, ec="#e63946", lw=2))
ax.text(bp[0], bp[1], "tray", color="#e63946", ha="center", va="center", fontsize=9,
        fontweight="bold")
ax.plot(0, 0, "ws", ms=10, mec="k")
ax.text(0.02, -0.05, "robot base", color="w", fontsize=9)
ax.plot([], [], "w--", label="task workspace boundary (>= 1 yaw)")
ax.plot([], [], "w-", lw=2, label="dexterous workspace boundary (8/8 yaws)")
ax.legend(loc="lower left", fontsize=8, facecolor="#333", labelcolor="w", framealpha=0.8)
ax.set_aspect("equal")
ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.set_title(f"Top-down workspace at table height (TCP z = {z_grasp:.3f} and {z_lift:.3f} m)\n"
             f"table area: {task:.0%} task workspace, {dex:.0%} dexterous", fontsize=10)
plt.tight_layout()
plt.savefig(args.out + ".png", dpi=150)
print(f"Saved {args.out}.png")
