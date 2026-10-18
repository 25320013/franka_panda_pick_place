"""
Franka Panda pick and place (CUBE version): randomized multi-episode benchmark.

Each episode places the cube at a random position AND yaw on the table, aligns
the gripper with the cube faces, plans the task
with mink IK, executes it in MuJoCo physics and records what happened.

  python3 pick_place_batch_cube.py                       # 25 episodes, with viewer
  python3 pick_place_batch_cube.py --headless            # no viewer (much faster)
  python3 pick_place_batch_cube.py --episodes 50 --seed 7 --headless

Ball positions are sampled over the WHOLE table surface (not just the verified
safe spot), so some episodes fail for real reasons: target out of reach,
the hand hitting the box when the cube spawns next to it, etc. No failure is
injected artificially. By default 30 % of spawns are drawn from a 6 cm ring
around the box (--near-box-frac) because uniform sampling rarely lands there.

Outputs (in ./batch_results/):
  episodes.csv          one row per episode (flat, spreadsheet friendly)
  episodes.jsonl        one JSON object per episode, incl. per-waypoint IK data
  summary.txt           success rate, failure breakdown, timing statistics
  cube_positions.png    top view of the table, each start pose coloured by outcome

Failure reasons:
  ik_no_solution           a waypoint's IK did not converge (out of reach)
  planned_pose_collision   IK converged but the arm pose penetrates table/box
  collision                robot touched table/box/floor during execution
  cube_disturbed           fingers pushed/rotated the cube before grasping
  grasp_failed             cube not lifted after closing the gripper
  dropped_in_transport     cube left the gripper on the way to the box
  missed_box               cube released but did not end up inside the box
"""

import os as _os
SCENE_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "scene")
import argparse
import csv
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from mink import Configuration, FrameTask, PostureTask, SE3, SO3, solve_ik

# ============================================================
# SETTINGS
# ============================================================

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=25)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--near-box-frac", type=float, default=0.3,
                    help="fraction of spawns placed in a 6 cm ring around the box")
parser.add_argument("--headless", action="store_true",
                    help="run without the MuJoCo viewer")
parser.add_argument("--xml", default=_os.path.join(SCENE_DIR, "pick_place_cube_scene.xml"))
parser.add_argument("--out", default="batch_results_cube")
args = parser.parse_args()

TCP_OFFSET = 0.103              # hand origin -> fingertip pad centre (m)
GRIPPER_OPEN, GRIPPER_CLOSED = 255.0, 0.0

CONTROL_DT = 0.01
EXEC_IK_ITERS = 5               # IK iterations per control step (execution)

PLAN_MAX_ITERS = 500            # IK iteration budget per waypoint (planning)
PLAN_POS_TOL = 0.0015           # 1.5 mm
PLAN_ORI_TOL = 0.010            # rad

MARGIN_RANGE = 0.20             # distances beyond 20 cm are reported as 0.20
CLEARANCE_EVERY = 3             # measure clearance every N control steps

# Cube spawn region: table surface minus a 4 cm edge band (a rotated 5 cm cube
# reaches 3.5 cm from its centre). Table top spans x 0.20..0.80, y -0.25..0.25.
SPAWN_X = (0.24, 0.76)
SPAWN_Y = (-0.21, 0.21)

USE_VIEWER = not args.headless
if USE_VIEWER:
    import mujoco.viewer

os.makedirs(args.out, exist_ok=True)

# ============================================================
# MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(args.xml)
data = mujoco.MjData(model)
KEY = model.key("pick_place_home").id
mujoco.mj_resetDataKeyframe(model, data, KEY)
mujoco.mj_forward(model, data)

sim_per_ctrl = max(1, int(round(CONTROL_DT / model.opt.timestep)))
hand_id = model.body("hand").id
cube_id = model.body("cube").id
cube_qadr = model.jnt_qposadr[model.joint("cube_joint").id]
cube_h = model.geom("cube_geom").size[0]          # half edge length
cube_reach = cube_h * np.sqrt(2)                  # centre -> vertical edge
box_pos = data.body("box").xpos.copy()
box_half_outer = 0.07
box_inner_half = 0.06
wall_top = box_pos[2] + 0.08
table_top_z = data.body("table").xpos[2] + model.geom("tabletop").size[2]

# Robot collision geoms (everything attached under link0 that can collide)
root = model.body_rootid[model.body("link0").id]
robot_geoms = [g for g in range(model.ngeom)
               if model.body_rootid[model.geom_bodyid[g]] == root
               and (model.geom_contype[g] or model.geom_conaffinity[g])]
robot_geom_set = set(robot_geoms)

table_geoms = [model.geom(n).id for n in ("tabletop", "table_leg_1", "table_leg_2",
                                          "table_leg_3", "table_leg_4")]
box_geoms = [model.geom(n).id for n in ("box_floor", "box_wall_px", "box_wall_nx",
                                        "box_wall_py", "box_wall_ny")]
floor_geom = model.geom("floor").id
env_name = {**{g: "table" for g in table_geoms},
            **{g: "box" for g in box_geoms}, floor_geom: "floor"}
link0_geoms = {g for g in robot_geoms if model.geom_bodyid[g] == model.body("link0").id}

arm_lo = model.jnt_range[:7, 0]
arm_hi = model.jnt_range[:7, 1]
home_q = model.key_qpos[KEY].copy()

# ============================================================
# MINK
# ============================================================

configuration = Configuration(model)
hand_task = FrameTask("hand", "body", position_cost=1.0,
                      orientation_cost=1.0, lm_damping=1e-3)
posture_task = PostureTask(model, cost=1e-3)
posture_task.set_target(home_q)
tasks = [hand_task, posture_task]

# Top-down orientation. The fingers close along the hand y-axis, so for a
# gripper yaw psi the closing direction is at angle psi - 90 deg. To close
# across two opposite cube faces we need psi = cube_yaw + k * 90 deg; of the
# four choices we take the one closest to the home yaw (least wrist rotation).
mujoco.mj_forward(model, data)
R0 = data.xmat[hand_id].reshape(3, 3)
HOME_YAW = np.arctan2(R0[1, 0], R0[0, 0])


def down_rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.diag([1.0, -1.0, -1.0])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def grasp_yaw_for(cube_yaw):
    k = np.round(wrap(HOME_YAW - cube_yaw) / (np.pi / 2))
    return wrap(cube_yaw + k * np.pi / 2)


R_DOWN = down_rotation(HOME_YAW)         # updated every episode


def set_tcp_target(tcp):
    hand_task.set_target(SE3.from_rotation_and_translation(
        SO3.from_matrix(R_DOWN), tcp - R_DOWN[:, 2] * TCP_OFFSET))


def ik_step(n_iters):
    for _ in range(n_iters):
        v = solve_ik(configuration, tasks, CONTROL_DT, "daqp", damping=1e-3)
        configuration.integrate_inplace(v, CONTROL_DT)
        q = configuration.q.copy()
        q[:7] = np.clip(q[:7], arm_lo, arm_hi)
        configuration.update(q)


def plan_waypoint(tcp):
    """Iterate IK until converged. Returns (ok, iters, pos_err, ori_err, q)."""
    set_tcp_target(tcp)
    for it in range(1, PLAN_MAX_ITERS + 1):
        ik_step(1)
        err = hand_task.compute_error(configuration)
        pe, oe = np.linalg.norm(err[:3]), np.linalg.norm(err[3:])
        if pe < PLAN_POS_TOL and oe < PLAN_ORI_TOL:
            return True, it, pe, oe, configuration.q.copy()
    return False, PLAN_MAX_ITERS, pe, oe, configuration.q.copy()


# ============================================================
# MEASUREMENTS
# ============================================================

class BoxClearance:
    """Version-independent signed distance from robot collision geoms to box obstacles.
    Robot geoms are represented by points sampled on their surfaces; obstacles are
    boxes, for which point-to-box signed distance is exact."""
    def __init__(self, model, robot_geoms, box_geoms, pts_per_geom=300, seed=0):
        rng = np.random.default_rng(seed)
        self.model, self.box_geoms = model, list(box_geoms)
        local, owner = [], []
        for g in robot_geoms:
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[g]
                v0, nv = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                f0, nf = model.mesh_faceadr[mid], model.mesh_facenum[mid]
                V = model.mesh_vert[v0:v0 + nv]
                F = model.mesh_face[f0:f0 + nf]
                tri = V[F]                                   # (nf,3,3)
                area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
                k = rng.choice(nf, size=pts_per_geom, p=area / area.sum())
                r1, r2 = rng.random(pts_per_geom), rng.random(pts_per_geom)
                s = np.sqrt(r1)
                P = ((1 - s)[:, None] * tri[k, 0] + (s * (1 - r2))[:, None] * tri[k, 1]
                     + (s * r2)[:, None] * tri[k, 2])
                P = np.vstack([V, P])
            elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                h = model.geom_size[g]
                g1 = np.linspace(-1, 1, 7)
                P = np.array([[a, b, c] for a in g1 for b in g1 for c in g1
                              if max(abs(a), abs(b), abs(c)) == 1]) * h
            else:
                continue
            local.append(P); owner.append(np.full(len(P), g))
        self.local = local
        self.geoms = [o[0] for o in owner]
        self.owner = np.concatenate(owner)

    def world_points(self, data):
        return np.concatenate([data.geom_xpos[g] + P @ data.geom_xmat[g].reshape(3, 3).T
                               for g, P in zip(self.geoms, self.local)])

    def query(self, data, box_geoms=None, W=None):
        """Returns (min signed distance [m], robot geom id) over the obstacle boxes."""
        if W is None:
            W = self.world_points(data)
        best, who = np.inf, -1
        for e in (self.box_geoms if box_geoms is None else box_geoms):
            R = data.geom_xmat[e].reshape(3, 3)
            q = np.abs((W - data.geom_xpos[e]) @ R) - self.model.geom_size[e]
            sd = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
            i = int(np.argmin(sd))
            if sd[i] < best:
                best, who = sd[i], self.owner[i]
        return best, who


# Version-independent clearance. mj_geomDistance on mesh geoms gives different
# answers across MuJoCo versions, so robot meshes are sampled into surface
# points and their exact signed distance to the box obstacles is computed.
# Validated against mj_geomDistance (MuJoCo 3.16): mean error 0.2 mm,
# 95 % within 1 mm, rare worst case ~10 mm (box corner against a mesh face).
clearance = BoxClearance(model, robot_geoms, table_geoms + box_geoms)


def env_clearance():
    """Min signed distance robot -> table, robot -> box (m), with link names."""
    W = clearance.world_points(data)
    out = {}
    for key, geoms in (("table", table_geoms), ("box", box_geoms)):
        d, g = clearance.query(data, geoms, W)
        out[key] = (min(float(d), MARGIN_RANGE), model.body(model.geom_bodyid[g]).name)
    return out


def joint_limit_margin(q):
    m = np.minimum(q[:7] - arm_lo, arm_hi - q[:7])
    j = int(np.argmin(m))
    return float(m[j]), f"joint{j + 1}"


def robot_env_contact():
    """First contact between robot and table/box/floor, or None."""
    for c in data.contact[:data.ncon]:
        g1, g2 = c.geom1, c.geom2
        for r, e in ((g1, g2), (g2, g1)):
            if r in robot_geom_set and e in env_name:
                if e == floor_geom and r in link0_geoms:
                    continue                       # base standing on floor
                return model.body(model.geom_bodyid[r]).name, env_name[e]
    return None


def tcp_pos():
    R = data.xmat[hand_id].reshape(3, 3)
    return data.xpos[hand_id] + R[:, 2] * TCP_OFFSET


def cube_corners():
    R = data.xmat[cube_id].reshape(3, 3)
    signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    return data.xpos[cube_id] + (signs * cube_h) @ R.T


def cube_yaw_now():
    R = data.xmat[cube_id].reshape(3, 3)
    return np.arctan2(R[1, 0], R[0, 0])


def cube_in_box():
    """All 8 corners inside the cavity and below the wall top."""
    C = cube_corners()
    inside = np.all(np.abs(C[:, :2] - box_pos[:2]) <= box_inner_half + 0.002)
    below = C[:, 2].max() < wall_top
    resting = data.xpos[cube_id][2] < box_pos[2] + 0.01 + cube_h * np.sqrt(3) + 0.005
    return bool(inside and below and resting)


# ============================================================
# EPISODE
# ============================================================

def sample_cube_pose(rng):
    """Uniform on the table (or in a ring around the box), never overlapping
    the box footprint. Random yaw in [-45, 45) deg (a cube repeats every 90).
    Returns (xy, yaw, region)."""
    near = rng.random() < args.near_box_frac
    clear = box_half_outer + cube_reach + 0.003
    while True:
        if near:
            x = rng.uniform(box_pos[0] - clear - 0.06, box_pos[0] + clear + 0.06)
            y = rng.uniform(box_pos[1] - clear - 0.06, box_pos[1] + clear + 0.06)
        else:
            x = rng.uniform(*SPAWN_X)
            y = rng.uniform(*SPAWN_Y)
        on_table = SPAWN_X[0] <= x <= SPAWN_X[1] and SPAWN_Y[0] <= y <= SPAWN_Y[1]
        if on_table and (abs(x - box_pos[0]) > clear or abs(y - box_pos[1]) > clear):
            yaw = rng.uniform(-np.pi / 4, np.pi / 4)
            return np.array([x, y]), yaw, ("near_box" if near else "uniform")


def run_episode(ep, cube_xy, cube_yaw, region="uniform", viewer=None):
    global R_DOWN
    log = {
        "episode": ep,
        "spawn_region": region,
        "cube_x": round(float(cube_xy[0]), 4),
        "cube_y": round(float(cube_xy[1]), 4),
        "cube_z": round(float(table_top_z + cube_h), 4),
        "cube_yaw_deg": round(float(np.degrees(cube_yaw)), 2),
        "grasp_yaw_deg": None,
        "success": False,
        "failure_reason": "",
        "failure_detail": "",
        "failed_phase": "",
        "plan_time_s": 0.0,
        "plan_ik_iters": 0,
        "exec_ik_iters": 0,
        "exec_wall_time_s": 0.0,
        "sim_time_s": 0.0,
        "min_table_clearance_mm": None, "min_table_clearance_link": "",
        "min_box_clearance_mm": None, "min_box_clearance_link": "",
        "min_joint_limit_margin_rad": None, "min_joint_limit_joint": "",
        "max_tcp_tracking_err_mm": 0.0,       # peak lag while moving
        "max_settled_tcp_err_mm": 0.0,        # error at the end of each move
        "final_cube_x": None, "final_cube_y": None, "final_cube_z": None,
        "final_cube_yaw_deg": None, "final_cube_tilt_deg": None,
        "waypoints": [],
    }

    # ---------- reset ----------
    mujoco.mj_resetDataKeyframe(model, data, KEY)
    cube_start = np.array([cube_xy[0], cube_xy[1], table_top_z + cube_h])
    cube_quat = np.array([np.cos(cube_yaw / 2), 0, 0, np.sin(cube_yaw / 2)])
    data.qpos[cube_qadr:cube_qadr + 3] = cube_start
    data.qpos[cube_qadr + 3:cube_qadr + 7] = cube_quat
    g_yaw = grasp_yaw_for(cube_yaw)
    R_DOWN = down_rotation(g_yaw)
    log["grasp_yaw_deg"] = round(float(np.degrees(g_yaw)), 2)
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    configuration.update(data.qpos.copy())

    drop = np.array([box_pos[0], box_pos[1]])
    phases = [
        ("pre_grasp",  "move", cube_start + [0, 0, 0.12],                       2.5),
        ("descend",    "move", cube_start.copy(),                               1.5),
        ("close",      "grip", GRIPPER_CLOSED,                                  1.0),
        ("lift",       "move", cube_start + [0, 0, 0.15],                       1.5),
        ("above_box",  "move", np.array([*drop, wall_top + cube_h + 0.05]),     2.5),
        ("lower",      "move", np.array([*drop, wall_top + cube_h + 0.01]),     1.0),
        ("release",    "grip", GRIPPER_OPEN,                                    1.0),
        ("retreat",    "move", np.array([*drop, wall_top + cube_h + 0.12]),     1.5),
    ]

    # ---------- planning: IK for every waypoint ----------
    t0 = time.perf_counter()
    for name, kind, target, _ in phases:
        if kind != "move":
            continue
        ok, iters, pe, oe, q = plan_waypoint(target)
        log["plan_ik_iters"] += iters
        wp = {"phase": name, "tcp": np.round(target, 4).tolist(), "converged": ok,
              "ik_iters": iters, "pos_err_mm": round(pe * 1000, 3),
              "ori_err_rad": round(oe, 5)}
        # static check of the planned pose
        data.qpos[:9] = q[:9]
        mujoco.mj_forward(model, data)
        cl = env_clearance()
        wp["table_clearance_mm"] = round(cl["table"][0] * 1000, 2)
        wp["box_clearance_mm"] = round(cl["box"][0] * 1000, 2)
        wp["joint_limit_margin_rad"] = round(joint_limit_margin(q)[0], 4)
        log["waypoints"].append(wp)

        if not ok:
            log["failure_reason"] = "ik_no_solution"
            log["failure_detail"] = (f"{name}: pos err {pe * 1000:.1f} mm, "
                                     f"ori err {oe:.3f} rad after {iters} iters")
            log["failed_phase"] = name
            break
        if min(cl["table"][0], cl["box"][0]) < 0:
            which = "table" if cl["table"][0] < cl["box"][0] else "box"
            log["failure_reason"] = "planned_pose_collision"
            log["failure_detail"] = (f"{name}: {cl[which][1]} penetrates {which} "
                                     f"by {-cl[which][0] * 1000:.1f} mm")
            log["failed_phase"] = name
            break
    log["plan_time_s"] = round(time.perf_counter() - t0, 4)

    # restore the real state (planning moved data.qpos)
    mujoco.mj_resetDataKeyframe(model, data, KEY)
    data.qpos[cube_qadr:cube_qadr + 3] = cube_start
    data.qpos[cube_qadr + 3:cube_qadr + 7] = cube_quat
    mujoco.mj_forward(model, data)
    configuration.update(data.qpos.copy())

    if log["failure_reason"]:
        if viewer is not None:          # show where the cube was for a moment
            hold_in_viewer(viewer, 1.5)
        _finish(log)
        return log

    # ---------- execution ----------
    q_cmd = data.qpos[:7].copy()
    grip = GRIPPER_OPEN
    tcp_cmd = tcp_pos()
    min_tab, min_box = (MARGIN_RANGE, ""), (MARGIN_RANGE, "")
    min_jl = (np.inf, "")
    max_track = 0.0
    t_exec = time.perf_counter()

    def fail(reason, detail, phase):
        log["failure_reason"], log["failure_detail"], log["failed_phase"] = \
            reason, detail, phase

    for name, kind, target, duration in phases:
        start_tcp = tcp_cmd.copy()
        n = int(duration / CONTROL_DT)
        for k in range(n):
            tic = time.perf_counter()
            if kind == "move":
                a = (k + 1) / n
                a = a * a * (3 - 2 * a)
                tcp_cmd = start_tcp + a * (target - start_tcp)
            else:
                grip = target
            set_tcp_target(tcp_cmd)
            ik_step(EXEC_IK_ITERS)
            log["exec_ik_iters"] += EXEC_IK_ITERS
            q_cmd = configuration.q[:7].copy()

            for _ in range(sim_per_ctrl):
                data.ctrl[:7] = q_cmd
                data.ctrl[7] = grip
                data.qfrc_applied[:7] = data.qfrc_bias[:7]
                mujoco.mj_step(model, data)

            # ----- monitoring -----
            if k % CLEARANCE_EVERY == 0 or k == n - 1:
                cl = env_clearance()
                if cl["table"][0] < min_tab[0]:
                    min_tab = cl["table"]
                if cl["box"][0] < min_box[0]:
                    min_box = cl["box"]
            jl = joint_limit_margin(data.qpos)
            if jl[0] < min_jl[0]:
                min_jl = jl
            max_track = max(max_track, np.linalg.norm(tcp_pos() - tcp_cmd))

            hit = robot_env_contact()
            if hit:
                fail("collision", f"{hit[0]} touched {hit[1]}", name)
                break

            if viewer is not None:
                if not viewer.is_running():
                    raise KeyboardInterrupt
                viewer.sync()
                rest = CONTROL_DT - (time.perf_counter() - tic)
                if rest > 0:
                    time.sleep(rest)

        if log["failure_reason"]:
            break

        # ----- phase-end checks -----
        if kind == "move":
            log["max_settled_tcp_err_mm"] = round(max(
                log["max_settled_tcp_err_mm"],
                np.linalg.norm(tcp_pos() - tcp_cmd) * 1000), 2)
        b = data.xpos[cube_id]
        if name == "descend":
            shift = np.linalg.norm(b[:2] - cube_start[:2])
            turn = abs(np.degrees(wrap(cube_yaw_now() - cube_yaw)))
            if shift > 0.01 or turn > 10:
                fail("cube_disturbed",
                     f"cube pushed {shift * 1000:.0f} mm / rotated {turn:.0f} deg "
                     "while descending", name)
                break
        if name == "lift" and b[2] < cube_start[2] + 0.08:
            fail("grasp_failed", f"cube only raised {(b[2] - cube_start[2]) * 1000:.0f} mm",
                 name)
            break
        if name in ("above_box", "lower") and np.linalg.norm(b - tcp_pos()) > 0.03:
            fail("dropped_in_transport",
                 f"cube {np.linalg.norm(b - tcp_pos()) * 1000:.0f} mm from TCP", name)
            break

    # settle
    if not log["failure_reason"]:
        for _ in range(int(1.0 / model.opt.timestep)):
            data.ctrl[:7] = q_cmd
            data.ctrl[7] = grip
            data.qfrc_applied[:7] = data.qfrc_bias[:7]
            mujoco.mj_step(model, data)
        if cube_in_box():
            log["success"] = True
        else:
            fail("missed_box", f"cube ended at {np.round(data.xpos[cube_id], 3).tolist()}",
                 "release")

    log["exec_wall_time_s"] = round(time.perf_counter() - t_exec, 3)
    log["sim_time_s"] = round(data.time, 3)
    log["min_table_clearance_mm"] = round(min_tab[0] * 1000, 2)
    log["min_table_clearance_link"] = min_tab[1]
    log["min_box_clearance_mm"] = round(min_box[0] * 1000, 2)
    log["min_box_clearance_link"] = min_box[1]
    log["min_joint_limit_margin_rad"] = round(float(min_jl[0]), 4)
    log["min_joint_limit_joint"] = min_jl[1]
    log["max_tcp_tracking_err_mm"] = round(max_track * 1000, 2)
    _finish(log)
    return log


def hold_in_viewer(viewer, seconds):
    """Keep the arm still and refresh the viewer for a while."""
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        if not viewer.is_running():
            raise KeyboardInterrupt
        data.ctrl[:7] = home_q[:7]
        data.ctrl[7] = GRIPPER_OPEN
        data.qfrc_applied[:7] = data.qfrc_bias[:7]
        for _ in range(sim_per_ctrl):
            mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(CONTROL_DT)


def _finish(log):
    b = data.xpos[cube_id]
    log["final_cube_x"], log["final_cube_y"], log["final_cube_z"] = \
        [round(float(v), 4) for v in b]
    log["final_cube_yaw_deg"] = round(float(np.degrees(cube_yaw_now())), 2)
    log["final_cube_tilt_deg"] = round(float(np.degrees(np.arccos(
        np.clip(np.max(np.abs(data.xmat[cube_id].reshape(3, 3)[2])), -1, 1)))), 2)
    # planning-only failures: report margins from the planned waypoints
    if log["min_table_clearance_mm"] is None and log["waypoints"]:
        wps = log["waypoints"]
        log["min_table_clearance_mm"] = min(w["table_clearance_mm"] for w in wps)
        log["min_box_clearance_mm"] = min(w["box_clearance_mm"] for w in wps)
        log["min_joint_limit_margin_rad"] = min(w["joint_limit_margin_rad"] for w in wps)


# ============================================================
# MAIN
# ============================================================

def main():
    rng = np.random.default_rng(args.seed)
    starts = [sample_cube_pose(rng) for _ in range(args.episodes)]
    print(f"MuJoCo {mujoco.__version__}")
    print(f"Robot start pose (rad): {np.round(home_q[:7], 4).tolist()}, gripper open")
    logs = []

    print("=" * 78)
    print(f"PANDA PICK AND PLACE BENCHMARK: {args.episodes} episodes, seed {args.seed}")
    print("=" * 78)
    print(f"{'ep':>3} {'cube x':>7} {'cube y':>7} {'yaw':>5}  {'result':<24} "
          f"{'tab mm':>7} {'box mm':>7} {'plan s':>7} {'planIK':>6}")

    viewer_ctx = None
    if USE_VIEWER:
        viewer_ctx = mujoco.viewer.launch_passive(model, data)
        viewer_ctx.cam.lookat[:] = [0.45, 0.0, 0.45]
        viewer_ctx.cam.distance = 1.8
        viewer_ctx.cam.azimuth = 145
        viewer_ctx.cam.elevation = -30
    try:
        for ep, (xy, cyaw, region) in enumerate(starts):
            log = run_episode(ep, xy, cyaw, region, viewer_ctx)
            logs.append(log)
            if viewer_ctx is not None and log["exec_ik_iters"] > 0:
                viewer_ctx.sync()
                time.sleep(0.8)
            res = "SUCCESS" if log["success"] else log["failure_reason"]
            tab = log["min_table_clearance_mm"]
            box = log["min_box_clearance_mm"]
            print(f"{ep:>3} {xy[0]:7.3f} {xy[1]:7.3f} {np.degrees(cyaw):5.0f}  {res:<24} "
                  f"{tab if tab is not None else float('nan'):7.1f} "
                  f"{box if box is not None else float('nan'):7.1f} "
                  f"{log['plan_time_s']:7.3f} "
                  f"{log['plan_ik_iters']:6d}")
    except KeyboardInterrupt:
        print("\nViewer closed, stopping early.")
    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()

    save(logs)


def save(logs):
    # ---------- JSONL (full detail) ----------
    with open(os.path.join(args.out, "episodes.jsonl"), "w") as f:
        for log in logs:
            f.write(json.dumps(log) + "\n")

    # ---------- CSV (flat) ----------
    cols = [k for k in logs[0] if k != "waypoints"] + ["waypoint_ik_iters"]
    with open(os.path.join(args.out, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for log in logs:
            row = {k: v for k, v in log.items() if k != "waypoints"}
            row["waypoint_ik_iters"] = " ".join(
                f"{wp['phase']}:{wp['ik_iters']}" for wp in log["waypoints"])
            w.writerow(row)

    # ---------- summary ----------
    n = len(logs)
    succ = [l for l in logs if l["success"]]
    reasons = {}
    for l in logs:
        if not l["success"]:
            reasons[l["failure_reason"]] = reasons.get(l["failure_reason"], 0) + 1
    plan_t = np.array([l["plan_time_s"] for l in logs])
    plan_it = np.array([l["plan_ik_iters"] for l in logs])
    executed = [l for l in logs if l["exec_ik_iters"] > 0]

    lines = [
        f"Episodes            : {n}",
        f"Successes           : {len(succ)} ({100 * len(succ) / n:.0f} %)",
        f"Failures            : {n - len(succ)}",
    ]
    for r, k in sorted(reasons.items(), key=lambda x: -x[1]):
        lines.append(f"    {r:<24}: {k}")
    lines += [
        "",
        f"Planning time (s)   : mean {plan_t.mean():.3f}, max {plan_t.max():.3f}",
        f"Planning IK iters   : mean {plan_it.mean():.0f}, max {plan_it.max()}",
    ]
    if executed:
        et = np.array([l["exec_wall_time_s"] for l in executed])
        lines.append(f"Execution wall time : mean {et.mean():.2f} s "
                     f"({len(executed)} executed episodes)")
    if succ:
        lines += [
            "",
            "Successful episodes, worst-case margins:",
            f"    table clearance  : {min(l['min_table_clearance_mm'] for l in succ):.1f} mm",
            f"    box clearance    : {min(l['min_box_clearance_mm'] for l in succ):.1f} mm",
            f"    joint-limit      : {min(l['min_joint_limit_margin_rad'] for l in succ):.3f} rad",
            f"    TCP lag (moving) : {max(l['max_tcp_tracking_err_mm'] for l in succ):.1f} mm (max)",
            f"    TCP err (settled): {max(l['max_settled_tcp_err_mm'] for l in succ):.1f} mm (max)",
        ]
    text = "\n".join(lines)
    print("\n" + "=" * 78 + "\nSUMMARY\n" + "=" * 78 + "\n" + text)
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(text + "\n")

    # ---------- plot ----------
    colours = {"SUCCESS": "tab:green", "ik_no_solution": "tab:red",
               "planned_pose_collision": "tab:purple", "collision": "tab:orange",
               "cube_disturbed": "tab:brown", "grasp_failed": "tab:pink",
               "dropped_in_transport": "tab:olive", "missed_box": "tab:cyan"}
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.add_patch(plt.Rectangle((0.20, -0.25), 0.60, 0.50, fc="#d2b48c", ec="k", alpha=0.5))
    ax.add_patch(plt.Rectangle((box_pos[0] - box_half_outer, box_pos[1] - box_half_outer),
                               2 * box_half_outer, 2 * box_half_outer,
                               fc="royalblue", ec="k", alpha=0.6))
    ax.text(box_pos[0], box_pos[1], "box", ha="center", va="center", color="w")
    ax.plot(0, 0, "ks", ms=10)
    ax.text(0.01, 0.02, "robot base")
    seen = set()
    for l in logs:
        key = "SUCCESS" if l["success"] else l["failure_reason"]
        # draw each cube as a rotated square at its true size and yaw
        a = np.radians(l["cube_yaw_deg"])
        Rz = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        sq = (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * cube_h) @ Rz.T \
            + [l["cube_x"], l["cube_y"]]
        ax.add_patch(plt.Polygon(sq, fc=colours.get(key, "k"), ec="k", zorder=3,
                                 label=None if key in seen else key))
        ax.annotate(str(l["episode"]), (l["cube_x"], l["cube_y"]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
        seen.add(key)
    ax.set_xlim(-0.05, 0.85)
    ax.set_ylim(-0.32, 0.32)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"Cube start poses and outcomes ({len(succ)}/{n} success)")
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "cube_positions.png"), dpi=150)
    plt.close()

    print(f"\nSaved to {os.path.abspath(args.out)}/: episodes.csv, episodes.jsonl, "
          "summary.txt, cube_positions.png")


if __name__ == "__main__":
    main()
