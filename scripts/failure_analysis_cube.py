"""
Franka Panda cube pick and place: FAILURE ANALYSIS + MITIGATIONS.

Runs the same randomized episodes twice (same seed => same cube poses):

  1. baseline   : top-down grasp, gripper yaw closest to home, straight-down
                  pre-grasp, collision margin 0 mm   (= pick_place_batch_cube.py)
  2. mitigated  : tries grasp candidates in order and executes the first one
                  whose plan converges AND keeps >= --margin-mm clearance:
                    a. baseline grasp
                    b. alternate grasp yaw   (the other three 90-deg symmetric yaws)
                    c. tilted approach       (gripper tilted 15/30/45 deg about the
                                              finger-closing axis, hand coming from
                                              the robot side) with a pre-grasp
                                              placed back along the approach axis
                  plus a collision margin on the planned poses.

  3. stress_noise          : mitigated planner, but it only sees a NOISY cube
                              pose (simulated camera error); physics uses the
                              true pose. Exposes execution-time failures.
  4. stress_noise_refined  : same noise, plus a better pre-grasp: the robot
                              stops at the pre-grasp, re-measures the cube from
                              close range (noise / --refine-factor) and corrects
                              the descent before grasping.

Every failure is reproduced and rendered: the arm at the failing configuration,
target TCP (green), achieved TCP (red), target approach axis (green bar),
deepest penetration point (magenta) and the offending link tinted red.

Then a planning-only margin sweep shows how the collision margin trades
safety against the number of plannable episodes.

  python3 failure_analysis_cube.py                 # 25 episodes, seed 0
  python3 failure_analysis_cube.py --episodes 40 --margin-mm 5
  python3 failure_analysis_cube.py --no-snapshots  # faster, no images

Output: ./failure_analysis/
  baseline/episodes.csv, episodes.jsonl        per-episode logs
  mitigated/episodes.csv, episodes.jsonl
  snapshots/<run>_ep<NN>_<reason>.png          rendered failure states
  snapshots/<run>_ep<NN>_<reason>.json         the state (qpos, targets, errors)
  comparison.png                               outcome maps, baseline vs mitigated
  margin_sweep.csv / margin_sweep.png
  analysis.json                                everything the report needs

Headless-safe: snapshots use MuJoCo's offscreen Renderer. On a machine without
a display run with MUJOCO_GL=egl.
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

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=25)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--near-box-frac", type=float, default=0.3)
parser.add_argument("--margin-mm", type=float, default=5.0,
                    help="min planned clearance for the mitigated run")
parser.add_argument("--xml", default=_os.path.join(SCENE_DIR, "pick_place_cube_scene.xml"))
parser.add_argument("--out", default="failure_analysis")
parser.add_argument("--no-snapshots", action="store_true")
parser.add_argument("--pos-noise-mm", type=float, default=15.0,
                    help="std-dev of cube position estimate error (stress runs)")
parser.add_argument("--yaw-noise-deg", type=float, default=20.0,
                    help="std-dev of cube yaw estimate error (stress runs)")
parser.add_argument("--refine-factor", type=float, default=4.0,
                    help="noise reduction of the close-range re-measurement at pre-grasp")
parser.add_argument("--skip-stress", action="store_true")
args = parser.parse_args()

TCP_OFFSET = 0.103
GRIPPER_OPEN, GRIPPER_CLOSED = 255.0, 0.0
CONTROL_DT = 0.01
EXEC_IK_ITERS = 5
PLAN_MAX_ITERS = 500
PLAN_POS_TOL = 0.0015
PLAN_ORI_TOL = 0.010
MARGIN_RANGE = 0.20
CLEARANCE_EVERY = 3
SPAWN_X = (0.24, 0.76)
SPAWN_Y = (-0.21, 0.21)
TILTS_DEG = (15, 30, 45)
PREGRASP_DIST = 0.12

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
cube_h = model.geom("cube_geom").size[0]
cube_reach = cube_h * np.sqrt(2)
box_pos = data.body("box").xpos.copy()
box_half_outer, box_inner_half = 0.07, 0.06
wall_top = box_pos[2] + 0.08
table_top_z = data.body("table").xpos[2] + model.geom("tabletop").size[2]

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
arm_lo, arm_hi = model.jnt_range[:7, 0], model.jnt_range[:7, 1]
home_q = model.key_qpos[KEY].copy()

# ============================================================
# CLEARANCE (version independent, see pick_place_batch.py)
# ============================================================


class BoxClearance:
    def __init__(self, model, robot_geoms, pts_per_geom=300, seed=0):
        rng = np.random.default_rng(seed)
        self.local, self.geoms, owner = [], [], []
        for g in robot_geoms:
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = model.geom_dataid[g]
                V = model.mesh_vert[model.mesh_vertadr[mid]:
                                    model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
                F = model.mesh_face[model.mesh_faceadr[mid]:
                                    model.mesh_faceadr[mid] + model.mesh_facenum[mid]]
                tri = V[F]
                area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                                     tri[:, 2] - tri[:, 0]), axis=1)
                k = rng.choice(len(F), size=pts_per_geom, p=area / area.sum())
                r1, r2 = rng.random(pts_per_geom), rng.random(pts_per_geom)
                s = np.sqrt(r1)
                P = ((1 - s)[:, None] * tri[k, 0] + (s * (1 - r2))[:, None] * tri[k, 1]
                     + (s * r2)[:, None] * tri[k, 2])
                P = np.vstack([V, P])
            elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                g1 = np.linspace(-1, 1, 7)
                P = np.array([[a, b, c] for a in g1 for b in g1 for c in g1
                              if max(abs(a), abs(b), abs(c)) == 1]) * model.geom_size[g]
            else:
                continue
            self.local.append(P)
            self.geoms.append(g)
            owner.append(np.full(len(P), g))
        self.owner = np.concatenate(owner)

    def world_points(self, data):
        return np.concatenate([data.geom_xpos[g] + P @ data.geom_xmat[g].reshape(3, 3).T
                               for g, P in zip(self.geoms, self.local)])

    def query(self, data, boxes, W):
        best, who, pt = np.inf, -1, None
        for e in boxes:
            R = data.geom_xmat[e].reshape(3, 3)
            q = np.abs((W - data.geom_xpos[e]) @ R) - model.geom_size[e]
            sd = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
            i = int(np.argmin(sd))
            if sd[i] < best:
                best, who, pt = sd[i], self.owner[i], W[i].copy()
        return best, who, pt


clearance = BoxClearance(model, robot_geoms)


def env_clearance():
    W = clearance.world_points(data)
    out = {}
    for key, geoms in (("table", table_geoms), ("box", box_geoms)):
        d, g, p = clearance.query(data, geoms, W)
        out[key] = (min(float(d), MARGIN_RANGE), model.body(model.geom_bodyid[g]).name, p)
    return out


def joint_limit_margin(q):
    m = np.minimum(q[:7] - arm_lo, arm_hi - q[:7])
    j = int(np.argmin(m))
    return float(m[j]), f"joint{j + 1}"


def robot_env_contact():
    for c in data.contact[:data.ncon]:
        for r, e in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if r in robot_geom_set and e in env_name:
                if e == floor_geom and r in link0_geoms:
                    continue
                return model.body(model.geom_bodyid[r]).name, env_name[e], c.pos.copy()
    return None


# ============================================================
# ORIENTATIONS AND GRASP CANDIDATES
# ============================================================

def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


R_FLIP = np.diag([1.0, -1.0, -1.0])         # hand z-axis pointing down
mujoco.mj_forward(model, data)
_R0 = data.xmat[hand_id].reshape(3, 3)
HOME_YAW = np.arctan2(_R0[1, 0], _R0[0, 0])


def grasp_rotation(yaw, tilt):
    """Top-down hand at `yaw`, then tilted by `tilt` about its own y-axis (the
    finger-closing axis), so the fingers stay on the same two cube faces."""
    return Rz(yaw) @ R_FLIP @ Ry(tilt)


def grasp_candidates(cube_xy, cube_yaw, strategy):
    """Ordered list of grasps to try: dicts with yaw, tilt, label."""
    yaws = [wrap(cube_yaw + k * np.pi / 2) for k in range(4)]
    yaws.sort(key=lambda y: abs(wrap(y - HOME_YAW)))        # least wrist rotation
    base = [{"yaw": yaws[0], "tilt": 0.0, "label": "baseline"}]
    if strategy == "baseline":
        return base
    cands = base + [{"yaw": y, "tilt": 0.0, "label": "alt_yaw"} for y in yaws[1:]]
    # Tilted approach: prefer yaws whose closing axis is perpendicular to the
    # radial direction, so the tilt happens in the radial plane (toward base).
    radial = np.array([*cube_xy, 0.0]) / np.linalg.norm(cube_xy)
    by_perp = sorted(yaws, key=lambda y: abs(grasp_rotation(y, 0)[:, 1] @ radial))
    for t in TILTS_DEG:
        for y in by_perp[:2]:
            # choose the tilt sign that makes the approach point AWAY from the
            # base (hand sits closer to the robot => shorter reach)
            for sgn in (1, -1):
                R = grasp_rotation(y, sgn * np.radians(t))
                if R[:, 2] @ radial > 0:
                    cands.append({"yaw": y, "tilt": sgn * np.radians(t),
                                  "label": f"tilt_{t}"})
                    break
    return cands


# ============================================================
# MINK
# ============================================================

configuration = Configuration(model)
hand_task = FrameTask("hand", "body", position_cost=1.0,
                      orientation_cost=1.0, lm_damping=1e-3)
posture_task = PostureTask(model, cost=1e-3)
posture_task.set_target(home_q)
tasks = [hand_task, posture_task]


def set_target(tcp, R):
    hand_task.set_target(SE3.from_rotation_and_translation(
        SO3.from_matrix(R), tcp - R[:, 2] * TCP_OFFSET))


def ik_step(n):
    for _ in range(n):
        v = solve_ik(configuration, tasks, CONTROL_DT, "daqp", damping=1e-3)
        configuration.integrate_inplace(v, CONTROL_DT)
        q = configuration.q.copy()
        q[:7] = np.clip(q[:7], arm_lo, arm_hi)
        configuration.update(q)


def plan_waypoint(tcp, R):
    set_target(tcp, R)
    for it in range(1, PLAN_MAX_ITERS + 1):
        ik_step(1)
        err = hand_task.compute_error(configuration)
        pe, oe = np.linalg.norm(err[:3]), np.linalg.norm(err[3:])
        if pe < PLAN_POS_TOL and oe < PLAN_ORI_TOL:
            return True, it, pe, oe, configuration.q.copy()
    return False, PLAN_MAX_ITERS, pe, oe, configuration.q.copy()


def quat_of(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    return q


def slerp_R(R0, R1, a):
    q0, q1 = quat_of(R0), quat_of(R1)
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


def tcp_pos():
    R = data.xmat[hand_id].reshape(3, 3)
    return data.xpos[hand_id] + R[:, 2] * TCP_OFFSET


def cube_corners():
    R = data.xmat[cube_id].reshape(3, 3)
    s = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    return data.xpos[cube_id] + (s * cube_h) @ R.T


def cube_yaw_now():
    R = data.xmat[cube_id].reshape(3, 3)
    return np.arctan2(R[1, 0], R[0, 0])


def cube_tilt_now():
    return float(np.degrees(np.arccos(np.clip(
        np.max(np.abs(data.xmat[cube_id].reshape(3, 3)[2])), -1, 1))))


def cube_in_box():
    C = cube_corners()
    inside = np.all(np.abs(C[:, :2] - box_pos[:2]) <= box_inner_half + 0.002)
    return bool(inside and C[:, 2].max() < wall_top)


# ============================================================
# SNAPSHOTS
# ============================================================

renderer = None
if not args.no_snapshots:
    renderer = mujoco.Renderer(model, 480, 640)
SNAP_DIR = os.path.join(args.out, "snapshots")
os.makedirs(SNAP_DIR, exist_ok=True)


def _add_geom(scene, gtype, size, pos, mat, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom], gtype, np.asarray(size, float),
                        np.asarray(pos, float), np.asarray(mat, float).flatten(),
                        np.asarray(rgba, np.float32))
    scene.ngeom += 1


def _add_bar(scene, p0, p1, radius, rgba):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                        np.eye(3).flatten(), np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                         np.asarray(p0, float), np.asarray(p1, float))
    scene.ngeom += 1


def snapshot(tag, title, markers, highlight_body=None, state=None):
    """Render overview + close-up of the current data state with markers."""
    if renderer is None:
        return None
    mujoco.mj_forward(model, data)
    saved = None
    if highlight_body:
        bid = model.body(highlight_body).id
        gs = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]
        saved = [(g, model.geom_matid[g], model.geom_rgba[g].copy()) for g in gs]
        for g in gs:
            model.geom_matid[g] = -1
            model.geom_rgba[g] = [0.95, 0.15, 0.15, 1]

    # Robot visual geoms/materials, made see-through in the close-up and top
    # views so the target markers and the offending link are visible.
    robot_vis = [g for g in range(model.ngeom)
                 if model.body_rootid[model.geom_bodyid[g]] == root
                 and model.geom_group[g] == 2
                 and (not highlight_body or model.body(model.geom_bodyid[g]).name != highlight_body)]
    robot_mats = sorted({int(model.geom_matid[g]) for g in robot_vis if model.geom_matid[g] >= 0})
    mat_alpha = model.mat_rgba[robot_mats, 3].copy()
    geom_alpha = model.geom_rgba[robot_vis, 3].copy()

    focus = np.asarray(markers.get("target_tcp", tcp_pos()), float)
    # Close-up: side view perpendicular to the reach direction, from the side
    # away from the box, so the box never sits behind the gripper.
    look = focus.copy()
    look[2] = 0.5 * (focus[2] + table_top_z) + 0.03
    radial = np.arctan2(focus[1], focus[0])
    best = None
    for az in (np.degrees(radial) + 90, np.degrees(radial) - 90):
        el = np.radians(-12)
        fwd = np.array([np.cos(el) * np.cos(np.radians(az)),
                        np.cos(el) * np.sin(np.radians(az)), np.sin(el)])
        cam_pos = look - fwd * 0.7
        d_box = np.linalg.norm(cam_pos[:2] - box_pos[:2])
        if best is None or d_box > best[1]:
            best = (az, d_box)
    views = [("overview", [0.42, 0.0, 0.45], 1.55, 145, -28),
             ("close-up (side view)", look, 0.7, best[0], -12),
             ("top (x right, y up)", focus, 0.75, 90, -89)]
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0                     # hide scene sites in snapshots
    imgs = []
    for vi, (_, lookat, dist, az, el) in enumerate(views):
        a = 1.0 if vi == 0 else 0.3
        model.mat_rgba[robot_mats, 3] = mat_alpha * a
        model.geom_rgba[robot_vis, 3] = geom_alpha * a
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        renderer.update_scene(data, cam, opt)
        sc = renderer.scene
        I3 = np.eye(3)
        if "target_tcp" in markers:
            _add_geom(sc, mujoco.mjtGeom.mjGEOM_SPHERE, [0.012, 0, 0],
                      markers["target_tcp"], I3, [0.1, 0.9, 0.2, 1.0])
        if "target_R" in markers and "target_tcp" in markers:
            a = markers["target_R"][:, 2]
            _add_bar(sc, markers["target_tcp"] - a * 0.16, markers["target_tcp"],
                     0.004, [0.1, 0.9, 0.2, 0.6])
            c = markers["target_R"][:, 1]           # finger-closing direction
            _add_bar(sc, markers["target_tcp"] - c * 0.05,
                     markers["target_tcp"] + c * 0.05, 0.003, [0.1, 0.6, 0.95, 0.8])
        _add_geom(sc, mujoco.mjtGeom.mjGEOM_SPHERE, [0.010, 0, 0], tcp_pos(), I3,
                  [0.95, 0.1, 0.1, 0.9])
        if "target_tcp" in markers and np.linalg.norm(tcp_pos() - markers["target_tcp"]) > 0.003:
            _add_bar(sc, tcp_pos(), markers["target_tcp"], 0.002, [1, 0.5, 0, 0.9])
        if markers.get("contact") is not None:
            _add_geom(sc, mujoco.mjtGeom.mjGEOM_SPHERE, [0.014, 0, 0],
                      markers["contact"], I3, [1.0, 0.0, 1.0, 0.9])
        imgs.append(renderer.render().copy())

    model.mat_rgba[robot_mats, 3] = mat_alpha
    model.geom_rgba[robot_vis, 3] = geom_alpha
    if saved:
        for g, mid, rgba in saved:
            model.geom_matid[g] = mid
            model.geom_rgba[g] = rgba

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, img, (name, *_r) in zip(axs, imgs, views):
        ax.imshow(img)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.text(0.5, 0.01, "green sphere/bar = target TCP & approach axis   blue bar = "
             "finger-closing axis   red sphere = achieved TCP   orange = error   "
             "magenta = deepest penetration / contact   red link = offending link",
             ha="center", fontsize=8)
    plt.tight_layout(rect=(0, 0.04, 1, 0.95))
    png = os.path.join(SNAP_DIR, tag + ".png")
    plt.savefig(png, dpi=110)
    plt.close(fig)
    if state is not None:
        with open(os.path.join(SNAP_DIR, tag + ".json"), "w") as f:
            json.dump(state, f, indent=1, default=lambda o: np.round(o, 5).tolist())
    return png


# ============================================================
# EPISODE
# ============================================================

def sample_cube_pose(rng):
    near = rng.random() < args.near_box_frac
    clear = box_half_outer + cube_reach + 0.003
    while True:
        if near:
            x = rng.uniform(box_pos[0] - clear - 0.06, box_pos[0] + clear + 0.06)
            y = rng.uniform(box_pos[1] - clear - 0.06, box_pos[1] + clear + 0.06)
        else:
            x, y = rng.uniform(*SPAWN_X), rng.uniform(*SPAWN_Y)
        on_table = SPAWN_X[0] <= x <= SPAWN_X[1] and SPAWN_Y[0] <= y <= SPAWN_Y[1]
        if on_table and (abs(x - box_pos[0]) > clear or abs(y - box_pos[1]) > clear):
            return np.array([x, y]), rng.uniform(-np.pi / 4, np.pi / 4), \
                ("near_box" if near else "uniform")


def reset_scene(cube_start, cube_quat):
    mujoco.mj_resetDataKeyframe(model, data, KEY)
    data.qpos[cube_qadr:cube_qadr + 3] = cube_start
    data.qpos[cube_qadr + 3:cube_qadr + 7] = cube_quat
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    configuration.update(data.qpos.copy())


def build_phases(cube_start, grasp):
    Rg = grasp_rotation(grasp["yaw"], grasp["tilt"])
    approach = Rg[:, 2]                              # points toward the cube
    drop = np.array([box_pos[0], box_pos[1]])
    # Pre-grasp sits back along the approach axis (straight above when tilt=0)
    pre = cube_start - approach * PREGRASP_DIST
    # The cube is held at the grasp orientation all the way, so it stays level
    # in the world and lands flat in the box.
    return Rg, [
        ("pre_grasp", "move", pre,                                            Rg, 2.5),
        ("descend",   "move", cube_start.copy(),                              Rg, 1.5),
        ("close",     "grip", GRIPPER_CLOSED,                                 Rg, 1.0),
        ("lift",      "move", cube_start + [0, 0, 0.15],                      Rg, 1.5),
        ("above_box", "move", np.array([*drop, wall_top + cube_h + 0.05]),    Rg, 2.5),
        ("lower",     "move", np.array([*drop, wall_top + cube_h + 0.01]),    Rg, 1.0),
        ("release",   "grip", GRIPPER_OPEN,                                   Rg, 1.0),
        ("retreat",   "move", np.array([*drop, wall_top + cube_h + 0.12]),    Rg, 1.5),
    ]


def plan(cube_start, cube_quat, grasp, margin):
    """Plan every move waypoint. Returns (ok, info). info holds the failing
    waypoint, the attempted q and its clearances when ok is False."""
    reset_scene(cube_start, cube_quat)
    Rg, phases = build_phases(cube_start, grasp)
    info = {"waypoints": [], "iters": 0}
    for name, kind, target, R, _ in phases:
        if kind != "move":
            continue
        ok, it, pe, oe, q = plan_waypoint(target, R)
        info["iters"] += it
        data.qpos[:9] = q[:9]
        mujoco.mj_forward(model, data)
        cl = env_clearance()
        wp = {"phase": name, "converged": ok, "ik_iters": it,
              "pos_err_mm": round(pe * 1000, 2), "ori_err_rad": round(oe, 4),
              "table_clearance_mm": round(cl["table"][0] * 1000, 2),
              "box_clearance_mm": round(cl["box"][0] * 1000, 2),
              "joint_limit_margin_rad": round(joint_limit_margin(q)[0], 4)}
        info["waypoints"].append(wp)
        worst = "table" if cl["table"][0] < cl["box"][0] else "box"
        fail = None
        if not ok:
            fail = ("ik_no_solution",
                    f"{name}: pos err {pe * 1000:.1f} mm, ori err {oe:.3f} rad")
        elif cl[worst][0] < margin:
            reason = "planned_pose_collision" if cl[worst][0] < 0 else "below_margin"
            fail = (reason, f"{name}: {cl[worst][1]} {cl[worst][0] * 1000:+.1f} mm "
                            f"from {worst}")
        if fail:
            info.update(reason=fail[0], detail=fail[1], phase=name, q=q.copy(),
                        target=target, R=R, link=cl[worst][1], point=cl[worst][2],
                        pos_err=pe, ori_err=oe)
            return False, info
    return True, info


def execute(cube_start, cube_quat, cube_yaw, grasp, log, est_start=None, refine=None,
            refine_margin=0.0):
    """cube_start/quat/yaw: TRUE pose. est_start: pose the planner believes.
    refine: function () -> (refined_start, refined_yaw) called at pre-grasp."""
    reset_scene(cube_start, cube_quat)
    Rg, phases = build_phases(cube_start if est_start is None else est_start, grasp)
    q_cmd, grip = data.qpos[:7].copy(), GRIPPER_OPEN
    tcp_cmd, R_cmd = tcp_pos(), data.xmat[hand_id].reshape(3, 3).copy()
    min_tab, min_box, min_jl, lag = (MARGIN_RANGE, ""), (MARGIN_RANGE, ""), (np.inf, ""), 0.0
    t0 = time.perf_counter()
    failure = None

    i = -1
    while i + 1 < len(phases):
        i += 1
        name, kind, target, R_t, duration = phases[i]
        p0, R0 = tcp_cmd.copy(), R_cmd.copy()
        n = int(duration / CONTROL_DT)
        for k in range(n):
            a = (k + 1) / n
            a = a * a * (3 - 2 * a)
            if kind == "move":
                tcp_cmd = p0 + a * (target - p0)
            else:
                grip = target
            R_cmd = slerp_R(R0, R_t, a)
            set_target(tcp_cmd, R_cmd)
            ik_step(EXEC_IK_ITERS)
            log["exec_ik_iters"] += EXEC_IK_ITERS
            q_cmd = configuration.q[:7].copy()
            for _ in range(sim_per_ctrl):
                data.ctrl[:7], data.ctrl[7] = q_cmd, grip
                data.qfrc_applied[:7] = data.qfrc_bias[:7]
                mujoco.mj_step(model, data)
            if k % CLEARANCE_EVERY == 0 or k == n - 1:
                cl = env_clearance()
                min_tab = min(min_tab, cl["table"][:2], key=lambda x: x[0])
                min_box = min(min_box, cl["box"][:2], key=lambda x: x[0])
            jl = joint_limit_margin(data.qpos)
            min_jl = min(min_jl, jl, key=lambda x: x[0])
            lag = max(lag, np.linalg.norm(tcp_pos() - tcp_cmd))
            hit = robot_env_contact()
            if hit:
                failure = ("collision", f"{hit[0]} touched {hit[1]}", name,
                           hit[0], hit[2])
                break
        if failure:
            break
        if name == "pre_grasp" and refine is not None:
            r_start, r_yaw = refine()
            # Re-validate the corrected grasp before committing to it: the
            # corrected yaw/position can move the fingers into the box.
            yaws = [wrap(r_yaw + k * np.pi / 2) for k in range(4)]
            same = dict(grasp, yaw=min(yaws, key=lambda y: abs(wrap(y - grasp["yaw"]))))
            cands = [same] + grasp_candidates(r_start[:2], r_yaw, "mitigated")
            saved = (data.qpos.copy(), data.qvel.copy(), data.act.copy(),
                     data.qacc_warmstart.copy(), data.time, configuration.q.copy())
            g2 = None
            for c in cands:
                ok, _ = plan(r_start, cube_quat, c, refine_margin)
                if ok:
                    g2 = c
                    break
            data.qpos[:], data.qvel[:], data.act[:] = saved[0], saved[1], saved[2]
            data.qacc_warmstart[:], data.time = saved[3], saved[4]
            mujoco.mj_forward(model, data)
            configuration.update(saved[5])
            if g2 is None:
                failure = ("no_safe_regrasp", "corrected grasp fails re-validation",
                           name, None, None)
                break
            log["refined_grasp"] = g2["label"]
            new = build_phases(r_start, g2)[1]
            # go to the corrected pre-grasp first, then continue from 'descend'
            phases = (phases[:i + 1] + [("pre_grasp_refined",) + new[0][1:4] + (1.0,)]
                      + new[1:])
            log["refined_correction_mm"] = round(float(np.linalg.norm(
                r_start[:2] - (est_start if est_start is not None else cube_start)[:2]) * 1000), 1)
        b = data.xpos[cube_id]
        if name == "descend":
            shift = np.linalg.norm(b[:2] - cube_start[:2])
            turn = abs(np.degrees(wrap(cube_yaw_now() - cube_yaw)))
            if shift > 0.01 or turn > 10:
                failure = ("cube_disturbed", f"pushed {shift * 1000:.0f} mm / "
                           f"rotated {turn:.0f} deg", name, None, None)
        elif name == "lift" and b[2] < cube_start[2] + 0.08:
            failure = ("grasp_failed", f"cube raised only "
                       f"{(b[2] - cube_start[2]) * 1000:.0f} mm", name, None, None)
        elif name in ("above_box", "lower") and np.linalg.norm(b - tcp_pos()) > 0.035:
            failure = ("dropped_in_transport", f"cube "
                       f"{np.linalg.norm(b - tcp_pos()) * 1000:.0f} mm from TCP",
                       name, None, None)
        if failure:
            break

    if not failure:
        for _ in range(int(1.0 / model.opt.timestep)):
            data.ctrl[:7], data.ctrl[7] = q_cmd, grip
            data.qfrc_applied[:7] = data.qfrc_bias[:7]
            mujoco.mj_step(model, data)
        if not cube_in_box():
            failure = ("missed_box", f"cube at {np.round(data.xpos[cube_id], 3).tolist()}, "
                       f"tilt {cube_tilt_now():.0f} deg", "release", None, None)

    log.update(exec_wall_time_s=round(time.perf_counter() - t0, 3),
               min_table_clearance_mm=round(min_tab[0] * 1000, 2),
               min_table_clearance_link=min_tab[1],
               min_box_clearance_mm=round(min_box[0] * 1000, 2),
               min_box_clearance_link=min_box[1],
               min_joint_limit_margin_rad=round(float(min_jl[0]), 4),
               min_joint_limit_joint=min_jl[1],
               max_tcp_lag_mm=round(lag * 1000, 2))
    return failure, tcp_cmd, R_cmd


def run_episode(run, ep, cube_xy, cube_yaw, region, strategy, margin,
                noise=None, refine=False):
    """noise: (dxy [m], dyaw [rad], dxy_close, dyaw_close) estimate errors."""
    cube_start = np.array([cube_xy[0], cube_xy[1], table_top_z + cube_h])
    cube_quat = np.array([np.cos(cube_yaw / 2), 0, 0, np.sin(cube_yaw / 2)])
    log = {"episode": ep, "spawn_region": region,
           "cube_x": round(float(cube_xy[0]), 4), "cube_y": round(float(cube_xy[1]), 4),
           "cube_z": round(float(cube_start[2]), 4),
           "cube_yaw_deg": round(float(np.degrees(cube_yaw)), 2),
           "strategy": strategy, "margin_mm": margin * 1000,
           "success": False, "failure_reason": "", "failure_detail": "",
           "failed_phase": "", "grasp_used": "", "grasp_yaw_deg": None,
           "grasp_tilt_deg": None, "candidates_tried": 0, "rejected": [],
           "est_err_mm": 0.0, "est_err_yaw_deg": 0.0, "refined_correction_mm": None,
           "refined_grasp": None,
           "plan_time_s": 0.0, "plan_ik_iters": 0, "exec_ik_iters": 0,
           "exec_wall_time_s": 0.0, "snapshot": ""}

    # What the planner believes (true pose + perception error)
    est_xy, est_yaw = cube_xy.copy(), cube_yaw
    if noise is not None:
        est_xy = cube_xy + noise[0]
        est_yaw = cube_yaw + noise[1]
        log["est_err_mm"] = round(float(np.linalg.norm(noise[0]) * 1000), 1)
        log["est_err_yaw_deg"] = round(float(np.degrees(noise[1])), 1)
    est_start = np.array([est_xy[0], est_xy[1], cube_start[2]])

    t0 = time.perf_counter()
    chosen, first_fail = None, None
    for cand in grasp_candidates(est_xy, est_yaw, strategy):
        log["candidates_tried"] += 1
        ok, info = plan(est_start, cube_quat, cand, margin)
        log["plan_ik_iters"] += info["iters"]
        if ok:
            chosen = cand
            break
        log["rejected"].append(f"{cand['label']}@{np.degrees(cand['yaw']):.0f}deg:"
                               f"{info['reason']}({info['phase']})")
        if first_fail is None:
            first_fail = (cand, info)
    log["plan_time_s"] = round(time.perf_counter() - t0, 4)

    if chosen is None:
        cand, info = first_fail                     # report the baseline attempt
        log.update(failure_reason=info["reason"], failure_detail=info["detail"],
                   failed_phase=info["phase"])
        reset_scene(cube_start, cube_quat)
        data.qpos[:9] = info["q"][:9]
        mujoco.mj_forward(model, data)
        tag = f"{run}_ep{ep:02d}_{info['reason']}"
        log["snapshot"] = snapshot(
            tag, f"[{run}] episode {ep}: {info['reason']} at '{info['phase']}' "
                 f"({info['detail']})  -  cube ({cube_xy[0]:.3f}, {cube_xy[1]:.3f}), "
                 f"yaw {np.degrees(cube_yaw):.0f} deg",
            {"target_tcp": info["target"], "target_R": info["R"],
             "contact": info["point"] if info["reason"] != "ik_no_solution" else None},
            highlight_body=info["link"] if info["reason"] != "ik_no_solution" else None,
            state={"episode": ep, "run": run, "reason": info["reason"],
                   "phase": info["phase"], "detail": info["detail"],
                   "cube_pose": [*cube_start, *cube_quat], "arm_qpos": info["q"][:9],
                   "target_tcp": info["target"], "achieved_tcp": tcp_pos(),
                   "pos_err_mm": info["pos_err"] * 1000, "ori_err_rad": info["ori_err"],
                   "grasp_yaw_deg": np.degrees(cand["yaw"]),
                   "grasp_tilt_deg": np.degrees(cand["tilt"]),
                   "joint_limit_margin_rad": joint_limit_margin(info["q"])[0],
                   "waypoints": info["waypoints"]}) or ""
        return log

    log.update(grasp_used=chosen["label"],
               grasp_yaw_deg=round(float(np.degrees(chosen["yaw"])), 2),
               grasp_tilt_deg=round(float(np.degrees(chosen["tilt"])), 2))
    refine_fn = None
    if refine and noise is not None:
        def refine_fn():
            return (np.array([cube_xy[0] + noise[2][0], cube_xy[1] + noise[2][1],
                              cube_start[2]]), cube_yaw + noise[3])
    failure, tcp_cmd, R_cmd = execute(cube_start, cube_quat, cube_yaw, chosen, log,
                                      est_start=est_start, refine=refine_fn,
                                      refine_margin=margin)
    if failure is None:
        log["success"] = True
    else:
        reason, detail, phase, link, point = failure
        log.update(failure_reason=reason, failure_detail=detail, failed_phase=phase)
        tag = f"{run}_ep{ep:02d}_{reason}"
        extra = (f"  -  estimate off by {log['est_err_mm']:.0f} mm / "
                 f"{log['est_err_yaw_deg']:.0f} deg" if noise is not None else "")
        log["snapshot"] = snapshot(
            tag, f"[{run}] episode {ep}: {reason} during '{phase}' ({detail}){extra}",
            {"target_tcp": tcp_cmd, "target_R": R_cmd, "contact": point},
            highlight_body=link if link else None,
            state={"episode": ep, "run": run, "reason": reason, "phase": phase,
                   "detail": detail, "qpos": data.qpos.copy(),
                   "cube_pos": data.xpos[cube_id].copy(), "target_tcp": tcp_cmd,
                   "achieved_tcp": tcp_pos()}) or ""
    b = data.xpos[cube_id]
    log.update(final_cube=[round(float(v), 4) for v in b],
               final_cube_yaw_deg=round(float(np.degrees(cube_yaw_now())), 2),
               final_cube_tilt_deg=round(cube_tilt_now(), 2))
    return log


# ============================================================
# RUNS
# ============================================================

def save_run(run, logs):
    d = os.path.join(args.out, run)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "episodes.jsonl"), "w") as f:
        for l in logs:
            f.write(json.dumps(l, default=float) + "\n")
    keys = []
    for l in logs:
        keys += [k for k in l if k not in keys]
    with open(os.path.join(d, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for l in logs:
            w.writerow({k: (" | ".join(v) if isinstance(v, list) and k == "rejected" else v)
                        for k, v in l.items()})


def do_run(run, starts, strategy, margin, noises=None, refine=False):
    print(f"\n=== {run}: strategy={strategy}, margin={margin * 1000:.0f} mm ===")
    print(f"{'ep':>3} {'x':>6} {'y':>6} {'yaw':>4}  {'result':<24} {'grasp used':<12} "
          f"{'tilt':>4} {'cands':>5} {'plan s':>6}")
    logs = []
    for ep, (xy, cyaw, region) in enumerate(starts):
        l = run_episode(run, ep, xy, cyaw, region, strategy, margin,
                        noise=None if noises is None else noises[ep], refine=refine)
        logs.append(l)
        res = "SUCCESS" if l["success"] else l["failure_reason"]
        tilt = l["grasp_tilt_deg"]
        print(f"{ep:>3} {xy[0]:6.3f} {xy[1]:6.3f} {np.degrees(cyaw):4.0f}  {res:<24} "
              f"{l['grasp_used'] or '-':<12} {tilt if tilt is not None else 0:4.0f} "
              f"{l['candidates_tried']:5d} {l['plan_time_s']:6.2f}")
    save_run(run, logs)
    return logs


def counts(logs):
    c = {}
    for l in logs:
        k = "SUCCESS" if l["success"] else l["failure_reason"]
        c[k] = c.get(k, 0) + 1
    return c


COLOURS = {"SUCCESS": "tab:green", "ik_no_solution": "tab:red",
           "planned_pose_collision": "tab:purple", "below_margin": "tab:gray",
           "collision": "tab:orange", "cube_disturbed": "tab:brown",
           "grasp_failed": "tab:pink", "dropped_in_transport": "tab:olive",
           "missed_box": "tab:cyan", "no_safe_regrasp": "tab:blue"}


def outcome_map(ax, logs, title):
    ax.add_patch(plt.Rectangle((0.20, -0.25), 0.60, 0.50, fc="#d2b48c", ec="k", alpha=0.5))
    ax.add_patch(plt.Rectangle((box_pos[0] - box_half_outer, box_pos[1] - box_half_outer),
                               2 * box_half_outer, 2 * box_half_outer, fc="royalblue",
                               ec="k", alpha=0.6))
    ax.plot(0, 0, "ks", ms=9)
    seen = set()
    for l in logs:
        key = "SUCCESS" if l["success"] else l["failure_reason"]
        a = np.radians(l["cube_yaw_deg"])
        Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        sq = (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * cube_h) @ Rm.T \
            + [l["cube_x"], l["cube_y"]]
        ax.add_patch(plt.Polygon(sq, fc=COLOURS.get(key, "k"), ec="k", zorder=3,
                                 label=None if key in seen else key))
        seen.add(key)
        lab = str(l["episode"])
        if l["success"] and l["grasp_used"] not in ("baseline", ""):
            lab += "\n" + l["grasp_used"]
        ax.annotate(lab, (l["cube_x"], l["cube_y"]), fontsize=6, xytext=(5, 3),
                    textcoords="offset points")
    s = sum(l["success"] for l in logs)
    ax.set_title(f"{title}: {s}/{len(logs)} success", fontsize=10)
    ax.set_xlim(-0.05, 0.85)
    ax.set_ylim(-0.3, 0.3)
    ax.set_aspect("equal")
    ax.legend(loc="lower left", fontsize=7)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")


def margin_sweep(starts, margins_mm):
    """Planning-only: how many episodes still have a valid grasp plan when the
    planner demands a given clearance (mitigated candidate list)."""
    rows = []
    for m in margins_mm:
        n_ok, used = 0, {}
        for xy, cyaw, _ in starts:
            cube_start = np.array([xy[0], xy[1], table_top_z + cube_h])
            cube_quat = np.array([np.cos(cyaw / 2), 0, 0, np.sin(cyaw / 2)])
            for cand in grasp_candidates(xy, cyaw, "mitigated"):
                ok, _ = plan(cube_start, cube_quat, cand, m / 1000)
                if ok:
                    n_ok += 1
                    used[cand["label"]] = used.get(cand["label"], 0) + 1
                    break
        rows.append({"margin_mm": m, "plannable": n_ok, "of": len(starts), **used})
        print(f"  margin {m:5.1f} mm : {n_ok}/{len(starts)} plannable  {used}")
    return rows


def main():
    rng = np.random.default_rng(args.seed)
    starts = [sample_cube_pose(rng) for _ in range(args.episodes)]
    print(f"MuJoCo {mujoco.__version__} | {args.episodes} episodes | seed {args.seed}")

    base = do_run("baseline", starts, "baseline", 0.0)
    mit = do_run("mitigated", starts, "mitigated", args.margin_mm / 1000)

    stress, stress_ref = None, None
    if not args.skip_stress:
        nrng = np.random.default_rng(args.seed + 1000)
        sp, sy = args.pos_noise_mm / 1000, np.radians(args.yaw_noise_deg)
        f = args.refine_factor
        noises = [(nrng.normal(0, sp, 2), nrng.normal(0, sy),
                   nrng.normal(0, sp / f, 2), nrng.normal(0, sy / f))
                  for _ in starts]
        stress = do_run("stress_noise", starts, "mitigated", args.margin_mm / 1000,
                        noises=noises, refine=False)
        stress_ref = do_run("stress_noise_refined", starts, "mitigated",
                            args.margin_mm / 1000, noises=noises, refine=True)

    print("\n=== margin sweep (planning only, mitigated candidates) ===")
    sweep = margin_sweep(starts, [0, 2.5, 5, 10, 15, 20])
    keys = sorted({k for r in sweep for k in r})
    with open(os.path.join(args.out, "margin_sweep.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(sweep)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["margin_mm"] for r in sweep], [r["plannable"] for r in sweep], "o-")
    ax.set_xlabel("required planned clearance (mm)")
    ax.set_ylabel(f"episodes with a valid plan (of {len(starts)})")
    ax.set_title("Collision-margin sweep")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "margin_sweep.png"), dpi=130)
    plt.close()

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    outcome_map(axs[0], base, "Baseline")
    outcome_map(axs[1], mit, f"Mitigated (margin {args.margin_mm:.0f} mm)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "comparison.png"), dpi=130)
    plt.close()
    if stress is not None:
        fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
        outcome_map(axs[0], stress, f"Noisy estimate ({args.pos_noise_mm:.0f} mm, "
                                    f"{args.yaw_noise_deg:.0f} deg)")
        outcome_map(axs[1], stress_ref, "Noisy + pre-grasp re-measurement")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out, "comparison_noise.png"), dpi=130)
        plt.close()

    rescued = [{"episode": b["episode"], "baseline_failure": b["failure_reason"],
                "fixed_by": m["grasp_used"], "grasp_yaw_deg": m["grasp_yaw_deg"],
                "tilt_deg": m["grasp_tilt_deg"]}
               for b, m in zip(base, mit) if not b["success"] and m["success"]]
    still = [{"episode": m["episode"], "reason": m["failure_reason"],
              "detail": m["failure_detail"], "cube": [m["cube_x"], m["cube_y"]],
              "rejected": m["rejected"]} for m in mit if not m["success"]]
    regress = [b["episode"] for b, m in zip(base, mit) if b["success"] and not m["success"]]

    summary = {"episodes": len(starts), "seed": args.seed, "margin_mm": args.margin_mm,
               "baseline": counts(base), "mitigated": counts(mit),
               "rescued": rescued, "still_failing": still, "regressions": regress,
               "margin_sweep": sweep,
               "stress_noise": counts(stress) if stress else None,
               "stress_noise_refined": counts(stress_ref) if stress_ref else None,
               "noise": {"pos_mm": args.pos_noise_mm, "yaw_deg": args.yaw_noise_deg,
                         "refine_factor": args.refine_factor},
               "baseline_success_min_clearance_mm": {
                   "table": min((l["min_table_clearance_mm"] for l in base if l["success"]),
                                default=None),
                   "box": min((l["min_box_clearance_mm"] for l in base if l["success"]),
                              default=None)}}
    with open(os.path.join(args.out, "analysis.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)

    print("\n=== SUMMARY ===")
    print("baseline :", counts(base))
    print("mitigated:", counts(mit))
    print("rescued  :", [(r["episode"], r["baseline_failure"], r["fixed_by"]) for r in rescued])
    print("still failing:", [(s["episode"], s["reason"]) for s in still])
    print("regressions:", regress)
    if stress:
        print("stress_noise        :", counts(stress))
        print("stress_noise_refined:", counts(stress_ref))
    print(f"\nSaved to {os.path.abspath(args.out)}/")


if __name__ == "__main__":
    main()
    if renderer is not None:
        import sys
        sys.stdout.flush()
        os._exit(0)          # skip GL context teardown (noisy on some EGL drivers)
