"""
Franka Panda cube pick and place with CLOSED-LOOP Quadratic-Programming IK
with joint-limit constraints (no mink; uses qpsolvers + DAQP).

  python3 pick_place_qp.py                     # with viewer
  python3 pick_place_qp.py --headless
  python3 pick_place_qp.py --cube-x 0.72 --cube-y 0.0 --headless   # near limits

Every CONTROL_DT = 10 ms, from the MEASURED robot state, solve

    min_qdot   || J qdot - xdot ||^2_W  +  lam^2 ||qdot||^2  +  mu || qdot - qdot_post ||^2
    s.t.       lb <= qdot <= ub

    xdot      = [ v_des + Kp (p_des - p) ;  w_des + Ko e_rot(R_des, R) ]
    W         = diag(task weights)                     position vs orientation
    qdot_post = Kn (q_home - q)                        weak posture preference
    lb, ub    = intersection of
                  joint speed limits          |qdot| <= QDOT_MAX
                  joint POSITION limits as a velocity damper:
                  K_LIM (q_lo - q) <= qdot <= K_LIM (q_hi - q)
                  (the arm can approach a limit only exponentially, so it
                   never crosses it, whatever the task asks for)

Unlike DLS, the limits are part of the optimisation: when a joint saturates,
the QP redistributes the motion over the remaining joints instead of
clipping the solution afterwards (which would distort the fingertip motion).
"""

import os as _os
SCENE_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "scene")
import argparse
import time

import mujoco
import numpy as np
from qpsolvers import solve_qp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--xml", default=_os.path.join(SCENE_DIR, "pick_place_cube_scene.xml"))
parser.add_argument("--cube-x", type=float, default=None)
parser.add_argument("--cube-y", type=float, default=None)
parser.add_argument("--cube-yaw", type=float, default=0.0, help="degrees")
parser.add_argument("--joint-limit", nargs=3, action="append", default=[],
                    metavar=("J", "LO", "HI"),
                    help="software limit for joint J (1-7) in rad, tighter than the "
                         "hardware range, e.g. --joint-limit 4 -3.0 -2.3")
parser.add_argument("--grasp-yaw-offset", type=float, default=0.0,
                    help="extra gripper yaw in degrees (multiples of 90 are valid "
                         "cube grasps); e.g. 180 drives joint 7 into its limit")
args = parser.parse_args()

if not args.headless:
    import mujoco.viewer

# ============================================================
# PARAMETERS
# ============================================================

TCP_OFFSET = 0.103            # hand origin -> fingertip pad centre (m)
GRIPPER_OPEN, GRIPPER_CLOSED = 255.0, 0.0
CONTROL_DT = 0.01             # control period (s)

KP_POS = 10.0                 # position feedback gain (1/s)
KP_ORI = 8.0                  # orientation feedback gain (1/s)
W_POS, W_ORI = 1.0, 0.3       # task weights (orientation in rad vs position in m)
LAMBDA = 0.02                 # damping (regularisation) on qdot
MU_POSTURE = 1e-3             # weight of the posture preference
K_NULL = 1.0                  # posture gain (1/s)
QDOT_MAX = 1.5                # joint speed limit (rad/s)
K_LIM = 5.0                   # joint-limit velocity damper gain (1/s)
LIMIT_MARGIN = 0.02           # constraints use limits shrunk by this (rad)
MAX_TASK_STEP = 0.5           # clamp on feedback position term (m/s)
QP_SOLVER = "daqp"

# ============================================================
# MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(args.xml)
data = mujoco.MjData(model)
KEY = model.key("pick_place_home").id
mujoco.mj_resetDataKeyframe(model, data, KEY)

hand_id = model.body("hand").id
cube_id = model.body("cube").id
cube_qadr = model.jnt_qposadr[model.joint("cube_joint").id]
cube_h = model.geom("cube_geom").size[0]
box_pos = model.body("box").pos.copy()
wall_top = box_pos[2] + 0.08
table_top_z = model.body("table").pos[2] + model.geom("tabletop").size[2]
sim_per_ctrl = max(1, int(round(CONTROL_DT / model.opt.timestep)))

arm_dofs = np.arange(7)                       # arm joints are dofs 0..6
q_lo = model.jnt_range[:7, 0] + LIMIT_MARGIN
q_hi = model.jnt_range[:7, 1] - LIMIT_MARGIN
q_home = model.key_qpos[KEY][:7].copy()
for j, lo, hi in args.joint_limit:                  # optional software limits
    j = int(j) - 1
    q_lo[j] = max(q_lo[j], float(lo))
    q_hi[j] = min(q_hi[j], float(hi))
SOFT_LO, SOFT_HI = q_lo.copy(), q_hi.copy()

# Cube start pose (keyframe position unless overridden)
cube_start = model.key_qpos[KEY][cube_qadr:cube_qadr + 3].copy()
if args.cube_x is not None:
    cube_start[0] = args.cube_x
if args.cube_y is not None:
    cube_start[1] = args.cube_y
cube_start[2] = table_top_z + cube_h
cube_yaw = np.radians(args.cube_yaw)
data.qpos[cube_qadr:cube_qadr + 3] = cube_start
data.qpos[cube_qadr + 3:cube_qadr + 7] = [np.cos(cube_yaw / 2), 0, 0, np.sin(cube_yaw / 2)]
mujoco.mj_forward(model, data)

# ============================================================
# KINEMATICS HELPERS
# ============================================================

def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


R_FLIP = np.diag([1.0, -1.0, -1.0])           # hand z-axis pointing down
HOME_YAW = np.arctan2(data.xmat[hand_id][3], data.xmat[hand_id][0])


def grasp_yaw_for(cy):
    """Fingers close along hand y: align them with the cube faces, choosing
    the 90-deg-symmetric option closest to the home yaw."""
    k = np.round(wrap(HOME_YAW - cy) / (np.pi / 2))
    return wrap(cy + k * np.pi / 2)


R_GRASP = Rz(grasp_yaw_for(cube_yaw) + np.radians(args.grasp_yaw_offset)) @ R_FLIP


def tcp_pose():
    R = data.xmat[hand_id].reshape(3, 3)
    return data.xpos[hand_id] + R[:, 2] * TCP_OFFSET, R


def tcp_jacobian(p_tcp):
    """6x7 geometric Jacobian of the TCP point (linear; angular)."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, data, jacp, jacr, p_tcp, hand_id)
    return np.vstack([jacp[:, arm_dofs], jacr[:, arm_dofs]])


def rotation_error(R_des, R):
    """World-frame rotation vector taking R to R_des (small-angle form,
    exact direction, robust for errors up to ~90 deg)."""
    return 0.5 * (np.cross(R[:, 0], R_des[:, 0]) + np.cross(R[:, 1], R_des[:, 1])
                  + np.cross(R[:, 2], R_des[:, 2]))


def qp_step(p_des, R_des, v_des, w_des, stats):
    """One closed-loop QP-IK update from the measured state. Returns qdot."""
    p, R = tcp_pose()
    q = data.qpos[:7].copy()

    e_p = p_des - p
    e_r = rotation_error(R_des, R)
    fb_p = KP_POS * e_p
    n = np.linalg.norm(fb_p)
    if n > MAX_TASK_STEP:
        fb_p *= MAX_TASK_STEP / n
    xdot = np.concatenate([v_des + fb_p, w_des + KP_ORI * e_r])

    J = tcp_jacobian(p)
    W = np.diag([W_POS] * 3 + [W_ORI] * 3)
    qdot_post = K_NULL * (q_home - q)

    # 0.5 x^T P x + c^T x
    P = J.T @ W @ J + (LAMBDA ** 2 + MU_POSTURE) * np.eye(7)
    c = -(J.T @ W @ xdot + MU_POSTURE * qdot_post)

    # Box constraints: speed limits AND joint-position limits (velocity damper)
    lb = np.maximum(-QDOT_MAX, K_LIM * (q_lo - q))
    ub = np.minimum(QDOT_MAX, K_LIM * (q_hi - q))
    ub = np.maximum(ub, lb)            # if already past a limit: allow only moving back

    qdot = solve_qp(P, c, lb=lb, ub=ub, solver=QP_SOLVER)
    if qdot is None:                   # should not happen for a box-constrained QP
        stats["qp_fail"] += 1
        qdot = np.zeros(7)

    at_lb, at_ub = np.abs(qdot - lb) < 1e-6, np.abs(qdot - ub) < 1e-6
    pos_lb = K_LIM * (q_lo - q) > -QDOT_MAX          # position limit is the binding bound
    pos_ub = K_LIM * (q_hi - q) < QDOT_MAX
    act_pos = (at_lb & pos_lb) | (at_ub & pos_ub)
    act_spd = (at_lb & ~pos_lb) | (at_ub & ~pos_ub)
    stats["pos_limit_steps"] += int(act_pos.any())
    stats["speed_limit_steps"] += int(act_spd.any())
    for j in np.where(act_pos)[0]:
        stats["pos_limit_joints"].add(int(j) + 1)
    active = int(act_pos.sum() + act_spd.sum())
    margin = np.minimum(q - SOFT_LO, SOFT_HI - q)      # vs. the limits in force
    stats["solves"] += 1
    stats["active_steps"] += int(active > 0)
    stats["max_active"] = max(stats["max_active"], int(active))
    if margin.min() < stats["min_margin"]:
        stats["min_margin"] = float(margin.min())
        stats["min_margin_joint"] = int(np.argmin(margin)) + 1
    stats["pos_err"].append(np.linalg.norm(e_p))
    stats["ori_err"].append(np.linalg.norm(e_r))
    return qdot


# ============================================================
# TASK
# ============================================================

drop = np.array([box_pos[0], box_pos[1]])
PHASES = [
    # name,       kind,  target,                                           duration
    ("pre_grasp", "move", cube_start + [0, 0, 0.12],                        2.5),
    ("descend",   "move", cube_start.copy(),                                1.5),
    ("close",     "grip", GRIPPER_CLOSED,                                   1.0),
    ("lift",      "move", cube_start + [0, 0, 0.15],                        1.5),
    ("above_box", "move", np.array([*drop, wall_top + cube_h + 0.05]),      2.5),
    ("lower",     "move", np.array([*drop, wall_top + cube_h + 0.01]),      1.0),
    ("release",   "grip", GRIPPER_OPEN,                                     1.0),
    ("retreat",   "move", np.array([*drop, wall_top + cube_h + 0.12]),      1.5),
]


def quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def slerp(R0, R1, a):
    q0, q1 = quat(R0), quat(R1)
    if q0 @ q1 < 0:
        q1 = -q1
    d = np.clip(q0 @ q1, -1, 1)
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)
    q /= np.linalg.norm(q)
    M = np.zeros(9)
    mujoco.mju_quat2Mat(M, q)
    return M.reshape(3, 3)


def smooth(t):
    """smoothstep and its derivative."""
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t), 6 * t * (1 - t)


# ============================================================
# RUN
# ============================================================

q_cmd = data.qpos[:7].copy()
grip = GRIPPER_OPEN


def physics_step():
    data.ctrl[:7] = q_cmd
    data.ctrl[7] = grip
    data.qfrc_applied[:7] = data.qfrc_bias[:7]        # gravity compensation
    mujoco.mj_step(model, data)
    # refresh kinematics so the next Jacobian/pose uses the new state
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)


def run(viewer=None):
    global q_cmd, grip
    print("=" * 66)
    print("FRANKA PANDA PICK AND PLACE  -  closed-loop QP IK (joint-limit constraints)")
    print("=" * 66)
    print(f"Cube start : {cube_start.round(3)}  yaw {args.cube_yaw:.0f} deg "
          f"-> grasp yaw {np.degrees(grasp_yaw_for(cube_yaw)):.0f} deg")
    print(f"Gains: Kp={KP_POS}, Ko={KP_ORI}, lambda={LAMBDA}, K_lim={K_LIM}, "
          f"qdot_max={QDOT_MAX} rad/s, solver={QP_SOLVER}\n")
    print(f"{'phase':<10} {'end pos err':>12} {'end ori err':>12} {'max pos err':>12} "
          f"{'solves':>7} {'us/solve':>9} {'jl-act':>6} {'spd-act':>7} {'min jl margin':>15}")

    p_cmd, R_cmd = tcp_pose()
    p_cmd, R_cmd = p_cmd.copy(), R_cmd.copy()
    totals = {"solves": 0, "time": 0.0}

    for name, kind, target, duration in PHASES:
        stats = {"solves": 0, "qp_fail": 0, "active_steps": 0, "max_active": 0,
                 "pos_limit_steps": 0, "speed_limit_steps": 0, "pos_limit_joints": set(),
                 "min_margin": np.inf, "min_margin_joint": 0,
                 "pos_err": [], "ori_err": []}
        p0, R0 = p_cmd.copy(), R_cmd.copy()
        n = int(duration / CONTROL_DT)
        t_solve = 0.0
        for k in range(n):
            tic = time.perf_counter()
            s, ds = smooth((k + 1) / n)
            if kind == "move":
                p_cmd = p0 + s * (target - p0)
                v_des = (ds / duration) * (target - p0)    # feedforward velocity
            else:
                grip = target
                v_des = np.zeros(3)
            R_cmd = slerp(R0, R_GRASP, s)
            w_des = np.zeros(3)                            # rotations here are slow

            t0 = time.perf_counter()
            qdot = qp_step(p_cmd, R_cmd, v_des, w_des, stats)
            t_solve += time.perf_counter() - t0

            q_cmd = q_cmd + qdot * CONTROL_DT
            # keep the reference close to the measured joints (anti wind-up).
            # No clipping to joint limits is needed: the QP already keeps qdot
            # inside them (checked below via the min joint-limit margin).
            q_cmd = data.qpos[:7] + np.clip(q_cmd - data.qpos[:7], -0.1, 0.1)

            for _ in range(sim_per_ctrl):
                physics_step()

            if viewer is not None:
                if not viewer.is_running():
                    return False
                viewer.sync()
                rest = CONTROL_DT - (time.perf_counter() - tic)
                if rest > 0:
                    time.sleep(rest)

        p, R = tcp_pose()
        end_pe = np.linalg.norm(p_cmd - p) * 1000
        end_oe = np.degrees(np.linalg.norm(rotation_error(R_cmd, R)))
        totals["solves"] += stats["solves"]
        totals["time"] += t_solve
        totals["jl"] = totals.get("jl", 0) + stats["pos_limit_steps"]
        totals["spd"] = totals.get("spd", 0) + stats["speed_limit_steps"]
        totals["jl_joints"] = totals.get("jl_joints", set()) | stats["pos_limit_joints"]
        totals["qp_fail"] = totals.get("qp_fail", 0) + stats["qp_fail"]
        if stats["min_margin"] < totals.get("min_margin", (np.inf, 0))[0]:
            totals["min_margin"] = (stats["min_margin"], stats["min_margin_joint"])
        print(f"{name:<10} {end_pe:9.2f} mm {end_oe:9.2f} deg "
              f"{max(stats['pos_err']) * 1000:9.2f} mm {stats['solves']:7d} "
              f"{t_solve / stats['solves'] * 1e6:9.1f} {stats['pos_limit_steps']:6d} "
              f"{stats['speed_limit_steps']:7d} "
              f"{stats['min_margin']:8.3f} rad j{stats['min_margin_joint']}")

    for _ in range(int(1.0 / model.opt.timestep)):       # let the cube settle
        physics_step()
    print(f"\nTotal QP solves: {totals['solves']}  "
          f"(mean {totals['time'] / totals['solves'] * 1e6:.1f} us per solve), "
          f"QP failures: {totals['qp_fail']}")
    print(f"Steps with an active joint-POSITION limit: {totals['jl']}"
          f"{' (joints ' + str(sorted(totals['jl_joints'])) + ')' if totals['jl_joints'] else ''}; "
          f"with an active SPEED limit: {totals['spd']}")
    m, j = totals["min_margin"]
    print(f"Closest approach to any joint limit: {m:.3f} rad (joint {j})"
          f"{'  -- LIMIT VIOLATED' if m < 0 else ''}")
    return True


def report():
    R = data.xmat[cube_id].reshape(3, 3)
    s = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    C = data.xpos[cube_id] + (s * cube_h) @ R.T
    inside = np.all(np.abs(C[:, :2] - box_pos[:2]) <= 0.06 + 0.002)
    ok = bool(inside and C[:, 2].max() < wall_top)
    print("\n" + "=" * 66)
    print(f"Final cube position: {data.xpos[cube_id].round(3)}")
    print("RESULT:", "SUCCESS - cube is in the box." if ok
          else "FAILED - cube is not in the box.")
    print("=" * 66)


if args.headless:
    run()
    report()
else:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.45, 0.0, 0.45]
        viewer.cam.distance = 1.6
        viewer.cam.azimuth = 145
        viewer.cam.elevation = -25
        if run(viewer):
            report()
            print("\nClose the viewer window to exit.")
            while viewer.is_running():
                physics_step()
                viewer.sync()
                time.sleep(model.opt.timestep)
