"""
compare_ik.py  -  Tutorial 4, Stage 2: compare three IK methods on the SAME
seeded cube pick-and-place episodes (Franka Panda, MuJoCo).

  mink : off-the-shelf mink (differential IK solved as a QP), open-loop reference
  dls  : closed-loop Damped Least Squares      qdot = (J^T J + lam I)^-1 J^T xdot
  qp   : closed-loop QP IK with joint-position and joint-velocity constraints

Usage
  python3 compare_ik.py                                  # 25 episodes x 3 methods
  python3 compare_ik.py --methods dls qp --episodes 50 --seed 3
  python3 compare_ik.py --methods qp --episodes 3 --viewer

Each method uses its own IK for BOTH stages:
  planning  : iterative IK (kinematics only) to every waypoint, from the home
              pose -> feasibility, IK iterations to convergence, planning time,
              clearance of the planned poses (>= PLAN_MARGIN)
  execution : the same method tracks the interpolated trajectory in physics
              (position actuators, gravity compensation, contact grasp)

Grasp selection (identical for all methods, the failure-analysis mitigations):
  1. top-down grasp, gripper yaw = cube yaw + k*90 deg closest to home
  2. the other three 90-deg-symmetric yaws            (alternate grasp yaw)
  3. tilted approach 15/30/45 deg about the finger-closing axis, hand on the
     robot side, pre-grasp along the approach axis      (approach direction)

Cube-tray collision avoidance:
  transit height keeps the cube bottom CUBE_TRAY_CLEAR above the tray walls,
  release happens over the cavity centre RELEASE_HEIGHT above the wall top,
  cube-tray clearance is measured every few steps and any cube-tray contact
  before release is a failure ("cube_tray_collision").

Outputs (in --out, default results/):
  <method>/episodes.csv, episodes.jsonl     per-episode logs
  summary.csv, summary.md                   metrics table (Section 10 + 13)
  outcomes.png, metrics.png, outcome_maps.png
"""

import os as _os
SCENE_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "scene")
import argparse
import csv
import json
import os
import time

import numpy as np
import mujoco

# ============================================================
# PARAMETERS
# ============================================================

TCP_OFFSET = 0.103                 # hand origin -> fingertip pad centre (m)
GRIP_OPEN, GRIP_CLOSED = 255.0, 0.0
CONTROL_DT = 0.01                  # control period (s)

PLAN_MAX_ITERS = 500
PLAN_POS_TOL = 0.0015              # 1.5 mm
PLAN_ORI_TOL = 0.010               # rad
PLAN_MARGIN = 0.005                # min planned robot clearance to table / tray (m)
STEP_MAX = 0.2                     # max |dq| per planning iteration (rad)

CUBE_TRAY_CLEAR = 0.05             # cube bottom above tray walls during transit (m)
RELEASE_HEIGHT = 0.01              # cube bottom above wall top at release (m)
CLEARANCE_EVERY = 3                # measure clearances every N control steps
MARGIN_RANGE = 0.20                # clearances above this are reported as 0.20 m

# Franka Panda joint velocity limits (datasheet), used for violation counting
PANDA_QDOT_LIMIT = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
QDOT_MAX = 1.5                     # controller speed cap for DLS / QP (rad/s)
LIMIT_HIT = 0.005                  # joint within 5 mrad of a hard limit = limit hit
LIMIT_MARGIN = 0.02                # controllers keep this far inside hard limits

KP_POS, KP_ORI = 10.0, 8.0         # closed-loop task gains (1/s)
MAX_TASK_SPEED = 0.5               # clamp on the position feedback term (m/s)
K_NULL = 1.0                       # posture (null-space / secondary) gain (1/s)
K_NULL_PLAN = 0.1                  # posture gain inside the planners

# DLS, tutorial form (J^T J + lam I)^-1 J^T, adaptive lam near singularities
DLS_LAMBDA_MIN, DLS_LAMBDA_MAX, W_THRESH = 1e-4, 6.4e-3, 0.02
# QP
W_POS, W_ORI = 1.0, 0.3
QP_LAMBDA, QP_MU = 4e-4, 1e-3
K_LIM = 5.0                        # joint-limit velocity damper gain (1/s)
# mink
MINK_ITERS_PER_STEP = 5
SETTLE_TOL = 0.005                 # convergence gating at the end of each move (m)
SETTLE_TIMEOUT = 2.0               # max extra hold per move (s)

TILTS_DEG = (15, 30, 45)
METHODS = ("mink", "dls", "qp")
METHOD_LABEL = {"mink": "Mink (off-the-shelf)", "dls": "DLS-IK (closed-loop)",
                "qp": "QP-IK (joint-limit constraints)"}


# ============================================================
# SCENE
# ============================================================

def resolve_xml(xml):
    """Find the scene: as given, else in $MUJOCO_MENAGERIE/franka_emika_panda."""
    if os.path.exists(xml):
        return xml
    root = os.environ.get("MUJOCO_MENAGERIE", "")
    cand = os.path.join(root, "franka_emika_panda", os.path.basename(xml))
    if root and os.path.exists(cand):
        return cand
    raise FileNotFoundError(
        f"{xml} not found. Run from mujoco_menagerie/franka_emika_panda or set "
        "MUJOCO_MENAGERIE (see README).")


class Scene:
    def __init__(self, xml=_os.path.join(SCENE_DIR, "pick_place_cube_scene.xml")):
        m = self.model = mujoco.MjModel.from_xml_path(resolve_xml(xml))
        self.data = mujoco.MjData(m)            # physics
        self.pdata = mujoco.MjData(m)           # planning (kinematics only)
        self.key = m.key("pick_place_home").id
        mujoco.mj_resetDataKeyframe(m, self.data, self.key)
        mujoco.mj_forward(m, self.data)
        d = self.data

        self.hand = m.body("hand").id
        self.cube = m.body("cube").id
        self.cube_geom = m.geom("cube_geom").id
        self.cube_qadr = m.jnt_qposadr[m.joint("cube_joint").id]
        self.cube_h = float(m.geom_size[self.cube_geom][0])
        self.cube_reach = self.cube_h * np.sqrt(2)
        self.box_pos = d.body("box").xpos.copy()
        self.box_half_outer, self.box_inner_half = 0.07, 0.06
        self.wall_top = self.box_pos[2] + 0.08
        tg = m.geom("tabletop").id
        tc, ts = d.geom_xpos[tg], m.geom_size[tg]
        self.table_top = float(tc[2] + ts[2])
        self.table_xy = (tc[0] - ts[0], tc[0] + ts[0], tc[1] - ts[1], tc[1] + ts[1])
        self.sim_per_ctrl = max(1, int(round(CONTROL_DT / m.opt.timestep)))

        root = m.body_rootid[m.body("link0").id]
        self.robot_geoms = [g for g in range(m.ngeom)
                            if m.body_rootid[m.geom_bodyid[g]] == root
                            and (m.geom_contype[g] or m.geom_conaffinity[g])]
        self.robot_set = set(self.robot_geoms)
        self.table_geoms = [m.geom(n).id for n in ("tabletop", "table_leg_1", "table_leg_2",
                                                   "table_leg_3", "table_leg_4")]
        self.box_geoms = [m.geom(n).id for n in ("box_floor", "box_wall_px", "box_wall_nx",
                                                 "box_wall_py", "box_wall_ny")]
        self.floor = m.geom("floor").id
        self.env_name = {**{g: "table" for g in self.table_geoms},
                         **{g: "tray" for g in self.box_geoms}, self.floor: "floor"}
        self.link0_geoms = {g for g in self.robot_geoms
                            if m.geom_bodyid[g] == m.body("link0").id}

        self.q_lo_hw = m.jnt_range[:7, 0].copy()
        self.q_hi_hw = m.jnt_range[:7, 1].copy()
        self.q_lo = self.q_lo_hw + LIMIT_MARGIN
        self.q_hi = self.q_hi_hw - LIMIT_MARGIN
        self.home_qpos = m.key_qpos[self.key].copy()
        self.q_home = self.home_qpos[:7].copy()
        R0 = d.xmat[self.hand].reshape(3, 3)
        self.home_yaw = float(np.arctan2(R0[1, 0], R0[0, 0]))

        self._sample_robot_points()
        g = np.linspace(-1, 1, 5)
        self.cube_pts = np.array([[a, b, c] for a in g for b in g for c in g
                                  if max(abs(a), abs(b), abs(c)) == 1]) * self.cube_h
        self._mink = None

    # ---------- clearance (version-independent point-to-box distances) ----------
    def _sample_robot_points(self, n=300, seed=0):
        m, rng = self.model, np.random.default_rng(seed)
        self.rp_local, self.rp_geoms, owner = [], [], []
        for g in self.robot_geoms:
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                F = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
                tri = V[F]
                area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0],
                                                     tri[:, 2] - tri[:, 0]), axis=1)
                k = rng.choice(len(F), size=n, p=area / area.sum())
                r1, r2 = rng.random(n), rng.random(n)
                s = np.sqrt(r1)
                P = ((1 - s)[:, None] * tri[k, 0] + (s * (1 - r2))[:, None] * tri[k, 1]
                     + (s * r2)[:, None] * tri[k, 2])
                P = np.vstack([V, P])
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                gg = np.linspace(-1, 1, 7)
                P = np.array([[a, b, c] for a in gg for b in gg for c in gg
                              if max(abs(a), abs(b), abs(c)) == 1]) * m.geom_size[g]
            else:
                continue
            self.rp_local.append(P)
            self.rp_geoms.append(g)
            owner.append(np.full(len(P), g))
        self.rp_owner = np.concatenate(owner)

    def _pts_to_boxes(self, d, W, boxes):
        best, idx = np.inf, -1
        for e in boxes:
            R = d.geom_xmat[e].reshape(3, 3)
            q = np.abs((W - d.geom_xpos[e]) @ R) - self.model.geom_size[e]
            sd = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
            i = int(np.argmin(sd))
            if sd[i] < best:
                best, idx = float(sd[i]), i
        return best, idx

    def robot_clearance(self, d):
        W = np.concatenate([d.geom_xpos[g] + P @ d.geom_xmat[g].reshape(3, 3).T
                            for g, P in zip(self.rp_geoms, self.rp_local)])
        out = {}
        for key, geoms in (("table", self.table_geoms), ("tray", self.box_geoms)):
            dist, i = self._pts_to_boxes(d, W, geoms)
            link = self.model.body(self.model.geom_bodyid[self.rp_owner[i]]).name
            out[key] = (min(dist, MARGIN_RANGE), link)
        return out

    def cube_tray_clearance(self, d):
        R = d.xmat[self.cube].reshape(3, 3)
        W = d.xpos[self.cube] + self.cube_pts @ R.T
        return min(self._pts_to_boxes(d, W, self.box_geoms)[0], MARGIN_RANGE)

    # ---------- kinematics ----------
    def tcp(self, d):
        R = d.xmat[self.hand].reshape(3, 3)
        return d.xpos[self.hand] + R[:, 2] * TCP_OFFSET, R

    def jac(self, d, p):
        jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, d, jp, jr, p, self.hand)
        return np.vstack([jp[:, :7], jr[:, :7]])

    def mink(self):
        """Lazy import so DLS/QP runs do not need mink installed."""
        if self._mink is None:
            import mink
            conf = mink.Configuration(self.model)
            ft = mink.FrameTask("hand", "body", position_cost=1.0,
                                orientation_cost=1.0, lm_damping=1e-3)
            pt = mink.PostureTask(self.model, cost=1e-3)
            pt.set_target(self.home_qpos)
            self._mink = (mink, conf, ft, pt)
        return self._mink


# ============================================================
# MATH HELPERS
# ============================================================

def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


R_FLIP = np.diag([1.0, -1.0, -1.0])            # hand z-axis pointing down


def mat2quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).flatten())
    return q                                   # MuJoCo convention: (w, x, y, z)


def quat_log_error(R_t, R_c):
    """Tutorial's quaternion error: q_e = q_t (x) q_c^-1, hemisphere flip,
    eps = 2 Log(q_e) = theta * v / sin(theta/2). World-frame rotation vector."""
    qt, qc = mat2quat(R_t), mat2quat(R_c)
    qe = np.zeros(4)
    mujoco.mju_mulQuat(qe, qt, np.array([qc[0], -qc[1], -qc[2], -qc[3]]))
    if qe[0] < 0:
        qe = -qe
    th = 2.0 * np.arccos(np.clip(qe[0], -1.0, 1.0))
    if th < 1e-6:
        return np.zeros(3)
    return th * qe[1:] / np.sin(th / 2.0)


def slerp_R(R0, R1, a):
    q0, q1 = mat2quat(R0), mat2quat(R1)
    if q0 @ q1 < 0:
        q1 = -q1
    dot = np.clip(q0 @ q1, -1, 1)
    if dot > 0.9995:
        q = q0 + a * (q1 - q0)
    else:
        th = np.arccos(dot)
        q = (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)
    q /= np.linalg.norm(q)
    M = np.zeros(9)
    mujoco.mju_quat2Mat(M, q)
    return M.reshape(3, 3)


def smooth(t):
    t = float(np.clip(t, 0, 1))
    return t * t * (3 - 2 * t), 6 * t * (1 - t)


# ============================================================
# IK UPDATES (shared by planning and execution)
# ============================================================

def dls_update(S, J, xdot, q, null_gain):
    """Tutorial eq. (5) + secondary task:
       dq = (J^T J + lam I)^-1 J^T xdot + (I - (J^T J + lam I)^-1 J^T J) dq_null"""
    JtJ = J.T @ J
    w = float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))      # manipulability
    lam = DLS_LAMBDA_MIN
    if w < W_THRESH:
        lam += DLS_LAMBDA_MAX * (1.0 - (w / W_THRESH) ** 2)
    A = JtJ + lam * np.eye(7)
    dq = np.linalg.solve(A, J.T @ xdot)
    N = np.eye(7) - np.linalg.solve(A, JtJ)
    dq += N @ (null_gain * (S.q_home - q))
    return dq, lam, w


def qp_update(S, J, xdot, q, lb, ub, null_gain):
    """Tutorial eq. (22)-(36): min 1/2 x^T H x + c^T x  s.t.  G x <= h,
       G = [I; -I], h = [ub; -lb]."""
    from qpsolvers import solve_qp
    Wt = np.diag([W_POS] * 3 + [W_ORI] * 3)
    H = J.T @ Wt @ J + (QP_LAMBDA + QP_MU) * np.eye(7)
    c = -(J.T @ Wt @ xdot + QP_MU * null_gain * (S.q_home - q))
    ub = np.maximum(ub, lb)
    G = np.vstack([np.eye(7), -np.eye(7)])
    h = np.concatenate([ub, -lb])
    x = solve_qp(H, c, G, h, solver="daqp")
    return (np.zeros(7), False) if x is None else (x, True)


def plan_ik(S, method, p_t, R_t, q0):
    """Iterative IK to a TCP pose (kinematics only).
    Returns ok, iterations, pos_err, ori_err, q (7)."""
    m = S.model
    if method == "mink":
        mink, conf, ft, pt = S.mink()
        qf = S.home_qpos.copy()
        qf[:7] = q0
        conf.update(qf)
        ft.set_target(mink.SE3.from_rotation_and_translation(
            mink.SO3.from_matrix(R_t), p_t - R_t[:, 2] * TCP_OFFSET))
        for it in range(1, PLAN_MAX_ITERS + 1):
            v = mink.solve_ik(conf, [ft, pt], CONTROL_DT, "daqp", damping=1e-3)
            conf.integrate_inplace(v, CONTROL_DT)
            q = conf.q.copy()
            q[:7] = np.clip(q[:7], S.q_lo_hw, S.q_hi_hw)
            conf.update(q)
            p, R = S.tcp(conf.data)
            pe, oe = np.linalg.norm(p_t - p), np.linalg.norm(quat_log_error(R_t, R))
            if pe < PLAN_POS_TOL and oe < PLAN_ORI_TOL:
                return True, it, pe, oe, conf.q[:7].copy()
        return False, PLAN_MAX_ITERS, pe, oe, conf.q[:7].copy()

    d, q = S.pdata, np.array(q0, float)
    d.qpos[:] = S.home_qpos
    for it in range(PLAN_MAX_ITERS + 1):
        d.qpos[:7] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        p, R = S.tcp(d)
        e_p, e_r = p_t - p, quat_log_error(R_t, R)
        pe, oe = np.linalg.norm(e_p), np.linalg.norm(e_r)
        if pe < PLAN_POS_TOL and oe < PLAN_ORI_TOL:
            return True, it, pe, oe, q
        if it == PLAN_MAX_ITERS:
            break
        J = S.jac(d, p)
        e = np.concatenate([e_p, e_r])
        if method == "dls":
            dq = dls_update(S, J, e, q, K_NULL_PLAN)[0]
        else:
            lb = np.maximum(S.q_lo_hw - q, -STEP_MAX)
            ub = np.minimum(S.q_hi_hw - q, STEP_MAX)
            dq = qp_update(S, J, e, q, lb, ub, K_NULL_PLAN)[0]
        s = np.max(np.abs(dq))
        if s > STEP_MAX:
            dq *= STEP_MAX / s
        q = np.clip(q + dq, S.q_lo_hw, S.q_hi_hw)
    return False, PLAN_MAX_ITERS, pe, oe, q


# ============================================================
# TASK DEFINITION
# ============================================================

def grasp_rotation(yaw, tilt):
    return Rz(yaw) @ R_FLIP @ Ry(tilt)


def grasp_candidates(S, cube_xy, cube_yaw, mitigate=True):
    yaws = sorted([wrap(cube_yaw + k * np.pi / 2) for k in range(4)],
                  key=lambda y: abs(wrap(y - S.home_yaw)))
    cands = [{"yaw": yaws[0], "tilt": 0.0, "label": "top_down"}]
    if not mitigate:
        return cands
    cands += [{"yaw": y, "tilt": 0.0, "label": "alt_yaw"} for y in yaws[1:]]
    radial = np.array([cube_xy[0], cube_xy[1], 0.0]) / np.linalg.norm(cube_xy)
    by_perp = sorted(yaws, key=lambda y: abs(grasp_rotation(y, 0)[:, 1] @ radial))
    for t in TILTS_DEG:
        for y in by_perp[:2]:
            for sgn in (1, -1):
                if grasp_rotation(y, sgn * np.radians(t))[:, 2] @ radial > 0:
                    cands.append({"yaw": y, "tilt": sgn * np.radians(t),
                                  "label": f"tilt_{t}"})
                    break
    return cands


def build_phases(S, cube_start, grasp):
    Rg = grasp_rotation(grasp["yaw"], grasp["tilt"])
    approach = Rg[:, 2]
    drop = S.box_pos[:2]
    transit_z = S.wall_top + S.cube_h + CUBE_TRAY_CLEAR        # cube bottom 5 cm over walls
    lift_z = max(cube_start[2] + 0.15, transit_z)
    return [
        ("pre_grasp", "move", cube_start - approach * 0.12,                     2.5),
        ("descend",   "move", cube_start.copy(),                                1.5),
        ("close",     "grip", GRIP_CLOSED,                                      1.0),
        ("lift",      "move", np.array([*cube_start[:2], lift_z]),              1.5),
        ("transit",   "move", np.array([*drop, transit_z]),                     2.5),
        ("lower",     "move", np.array([*drop, S.wall_top + S.cube_h + RELEASE_HEIGHT]), 1.0),
        ("release",   "grip", GRIP_OPEN,                                        1.0),
        ("retreat",   "move", np.array([*drop, transit_z + 0.07]),              1.5),
    ], Rg


def reset_scene(S, cube_start, cube_quat):
    m, d = S.model, S.data
    mujoco.mj_resetDataKeyframe(m, d, S.key)
    d.qpos[S.cube_qadr:S.cube_qadr + 3] = cube_start
    d.qpos[S.cube_qadr + 3:S.cube_qadr + 7] = cube_quat
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)


def plan(S, method, cube_start, grasp):
    phases, Rg = build_phases(S, cube_start, grasp)
    q = S.q_home.copy()
    info = {"iters": 0, "waypoints": []}
    d = S.pdata
    for name, kind, target, _ in phases:
        if kind != "move":
            continue
        ok, it, pe, oe, q = plan_ik(S, method, target, Rg, q)
        info["iters"] += it
        d.qpos[:] = S.home_qpos
        d.qpos[:7] = q
        mujoco.mj_kinematics(S.model, d)
        cl = S.robot_clearance(d)
        wp = {"phase": name, "converged": bool(ok), "ik_iters": int(it),
              "pos_err_mm": round(pe * 1000, 2), "ori_err_rad": round(float(oe), 4),
              "table_clear_mm": round(cl["table"][0] * 1000, 2),
              "tray_clear_mm": round(cl["tray"][0] * 1000, 2)}
        info["waypoints"].append(wp)
        worst = "table" if cl["table"][0] < cl["tray"][0] else "tray"
        if not ok:
            info.update(reason="ik_no_solution", phase=name,
                        detail=f"{name}: pos err {pe * 1000:.1f} mm, ori err {oe:.3f} rad")
            return False, info
        if cl[worst][0] < PLAN_MARGIN:
            info.update(reason="planned_pose_collision", phase=name,
                        detail=f"{name}: {cl[worst][1]} {cl[worst][0] * 1000:+.1f} mm from {worst}")
            return False, info
    return True, info


# ============================================================
# EXECUTION
# ============================================================

def robot_env_contact(S):
    d = S.data
    for c in d.contact[:d.ncon]:
        for r, e in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
            if r in S.robot_set and e in S.env_name:
                if e == S.floor and r in S.link0_geoms:
                    continue
                return S.model.body(S.model.geom_bodyid[r]).name, S.env_name[e]
    return None


def cube_tray_contact(S):
    d = S.data
    for c in d.contact[:d.ncon]:
        if (c.geom1 == S.cube_geom and c.geom2 in S.box_geoms) or \
           (c.geom2 == S.cube_geom and c.geom1 in S.box_geoms):
            return True
    return False


def cube_in_tray(S):
    d = S.data
    R = d.xmat[S.cube].reshape(3, 3)
    s = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)])
    C = d.xpos[S.cube] + (s * S.cube_h) @ R.T
    inside = np.all(np.abs(C[:, :2] - S.box_pos[:2]) <= S.box_inner_half + 0.002)
    return bool(inside and C[:, 2].max() < S.wall_top)


def cube_yaw_tilt(S):
    R = S.data.xmat[S.cube].reshape(3, 3)
    tilt = np.degrees(np.arccos(np.clip(np.max(np.abs(R[2])), -1, 1)))
    return float(np.arctan2(R[1, 0], R[0, 0])), float(tilt)


def execute(S, method, cube_start, cube_quat, cube_yaw, grasp, log, frame_cb=None):
    m, d = S.model, S.data
    reset_scene(S, cube_start, cube_quat)
    phases, Rg = build_phases(S, cube_start, grasp)
    if method == "mink":
        mink, conf, ft, pt = S.mink()
        conf.update(d.qpos.copy())
    st = dict(ik_iters=0, ik_time=0.0, ik_max=0.0, steps=0, lag=0.0, settled=0.0,
              settle_time=0.0, tab=(MARGIN_RANGE, ""), tray=(MARGIN_RANGE, ""),
              cube_tray=MARGIN_RANGE, jmargin=(np.inf, ""), limit_hits=0, vel_viol=0,
              vel_ratio=0.0, qp_pos_active=0, qp_spd_active=0, qp_fail=0,
              lam_max=0.0, manip_min=np.inf)
    ctl = {"q_cmd": d.qpos[:7].copy(), "grip": GRIP_OPEN, "released": False}

    def control_step(name, p_cmd, v_des, R_cmd, measure_clear):
        """One 10 ms control step + physics + monitoring. Returns a failure or None."""
        t0 = time.perf_counter()
        if method == "mink":
            ft.set_target(mink.SE3.from_rotation_and_translation(
                mink.SO3.from_matrix(R_cmd), p_cmd - R_cmd[:, 2] * TCP_OFFSET))
            for _ in range(MINK_ITERS_PER_STEP):
                v = mink.solve_ik(conf, [ft, pt], CONTROL_DT, "daqp", damping=1e-3)
                conf.integrate_inplace(v, CONTROL_DT)
                qq = conf.q.copy()
                qq[:7] = np.clip(qq[:7], S.q_lo_hw, S.q_hi_hw)
                conf.update(qq)
            ctl["q_cmd"] = conf.q[:7].copy()
            st["ik_iters"] += MINK_ITERS_PER_STEP
        else:
            q = d.qpos[:7].copy()                             # MEASURED state
            p, R = S.tcp(d)
            fb = KP_POS * (p_cmd - p)
            nf = np.linalg.norm(fb)
            if nf > MAX_TASK_SPEED:
                fb *= MAX_TASK_SPEED / nf
            xdot = np.concatenate([v_des + fb, KP_ORI * quat_log_error(R_cmd, R)])
            J = S.jac(d, p)
            if method == "dls":
                qdot, lam, w = dls_update(S, J, xdot, q, K_NULL)
                st["lam_max"] = max(st["lam_max"], lam)
                st["manip_min"] = min(st["manip_min"], w)
                mx = np.max(np.abs(qdot))
                if mx > QDOT_MAX:
                    qdot *= QDOT_MAX / mx
            else:
                lb = np.maximum(-QDOT_MAX, K_LIM * (S.q_lo - q))
                ub = np.minimum(QDOT_MAX, K_LIM * (S.q_hi - q))
                qdot, ok = qp_update(S, J, xdot, q, lb, ub, K_NULL)
                st["qp_fail"] += int(not ok)
                pos_lb = K_LIM * (S.q_lo - q) > -QDOT_MAX
                pos_ub = K_LIM * (S.q_hi - q) < QDOT_MAX
                at_lb, at_ub = np.abs(qdot - lb) < 1e-6, np.abs(qdot - ub) < 1e-6
                st["qp_pos_active"] += int(np.any((at_lb & pos_lb) | (at_ub & pos_ub)))
                st["qp_spd_active"] += int(np.any((at_lb & ~pos_lb) | (at_ub & ~pos_ub)))
            qc = ctl["q_cmd"] + qdot * CONTROL_DT
            qc = q + np.clip(qc - q, -0.1, 0.1)                # anti wind-up
            if method == "dls":                                # DLS has no constraints:
                qc = np.clip(qc, S.q_lo, S.q_hi)               # clip as a safety net
            ctl["q_cmd"] = qc
            st["ik_iters"] += 1
        dt_ik = time.perf_counter() - t0
        st["ik_time"] += dt_ik
        st["ik_max"] = max(st["ik_max"], dt_ik)
        st["steps"] += 1

        for _ in range(S.sim_per_ctrl):
            d.ctrl[:7], d.ctrl[7] = ctl["q_cmd"], ctl["grip"]
            d.qfrc_applied[:7] = d.qfrc_bias[:7]               # gravity compensation
            mujoco.mj_step(m, d)
        mujoco.mj_kinematics(m, d)                             # fresh kinematics for
        mujoco.mj_comPos(m, d)                                 # feedback and monitoring

        p, _ = S.tcp(d)
        err = float(np.linalg.norm(p - p_cmd))
        st["lag"] = max(st["lag"], err)
        if measure_clear:
            cl = S.robot_clearance(d)
            st["tab"] = min(st["tab"], cl["table"], key=lambda x: x[0])
            st["tray"] = min(st["tray"], cl["tray"], key=lambda x: x[0])
            if name in ("lift", "transit", "lower"):
                st["cube_tray"] = min(st["cube_tray"], S.cube_tray_clearance(d))
        q = d.qpos[:7]
        marg = np.minimum(q - S.q_lo_hw, S.q_hi_hw - q)
        j = int(np.argmin(marg))
        if marg[j] < st["jmargin"][0]:
            st["jmargin"] = (float(marg[j]), f"joint{j + 1}")
        st["limit_hits"] += int(marg[j] < LIMIT_HIT)
        ratio = float(np.max(np.abs(d.qvel[:7]) / PANDA_QDOT_LIMIT))
        st["vel_ratio"] = max(st["vel_ratio"], ratio)
        st["vel_viol"] += int(ratio > 1.0)
        if frame_cb is not None:
            frame_cb(S, {"phase": name, "p_cmd": p_cmd, "tcp_err": err, "grasp": grasp["label"]})
        hit = robot_env_contact(S)
        if hit:
            return ("collision", f"{hit[0]} touched {hit[1]}", name), err
        if not ctl["released"] and name in ("lift", "transit", "lower") and cube_tray_contact(S):
            return ("cube_tray_collision", "cube touched the tray before release", name), err
        return None, err

    p_cmd, R_cmd = (x.copy() for x in S.tcp(d))
    failure = None
    t_exec = time.perf_counter()
    for name, kind, target, dur in phases:
        p0, R0 = p_cmd.copy(), R_cmd.copy()
        n = int(round(dur / CONTROL_DT))
        for k in range(n):
            s, ds = smooth((k + 1) / n)
            if kind == "move":
                p_cmd = p0 + s * (target - p0)
                v_des = (ds / dur) * (target - p0)
            else:
                ctl["grip"], v_des = target, np.zeros(3)
                ctl["released"] = ctl["released"] or name == "release"
            R_cmd = slerp_R(R0, Rg, s)
            failure, err = control_step(name, p_cmd, v_des, R_cmd,
                                        k % CLEARANCE_EVERY == 0 or k == n - 1)
            if failure:
                break
        if failure:
            break
        if kind == "move":
            # Convergence gating: hold the final target until the TCP is within
            # SETTLE_TOL (or SETTLE_TIMEOUT), so the next phase starts on target.
            extra = 0
            while err > SETTLE_TOL and extra < int(SETTLE_TIMEOUT / CONTROL_DT):
                failure, err = control_step(name, p_cmd, np.zeros(3), R_cmd,
                                            extra % CLEARANCE_EVERY == 0)
                extra += 1
                if failure:
                    break
            st["settle_time"] += extra * CONTROL_DT
            st["settled"] = max(st["settled"], err)
            if failure:
                break
        b = d.xpos[S.cube]
        if name == "descend":
            shift = np.linalg.norm(b[:2] - cube_start[:2])
            turn = abs(np.degrees(wrap(cube_yaw_tilt(S)[0] - cube_yaw))) % 90
            turn = min(turn, 90 - turn)
            if shift > 0.01 or turn > 10:
                failure = ("cube_disturbed", f"pushed {shift * 1000:.0f} mm / rotated "
                           f"{turn:.0f} deg while descending", name)
        elif name == "lift" and b[2] < cube_start[2] + 0.08:
            failure = ("grasp_slip", f"cube raised only {(b[2] - cube_start[2]) * 1000:.0f} mm",
                       name)
        elif name in ("transit", "lower") and np.linalg.norm(b - S.tcp(d)[0]) > 0.035:
            failure = ("dropped_in_transit",
                       f"cube {np.linalg.norm(b - S.tcp(d)[0]) * 1000:.0f} mm from TCP", name)
        if failure:
            break

    if failure is None:
        for i in range(int(1.0 / m.opt.timestep)):            # settle 1 s
            d.ctrl[:7], d.ctrl[7] = ctl["q_cmd"], ctl["grip"]
            d.qfrc_applied[:7] = d.qfrc_bias[:7]
            mujoco.mj_step(m, d)
            if frame_cb is not None and i % S.sim_per_ctrl == 0:
                mujoco.mj_kinematics(m, d)
                frame_cb(S, {"phase": "settle", "p_cmd": p_cmd, "tcp_err": 0.0,
                             "grasp": grasp["label"]})
        mujoco.mj_forward(m, d)
        if not cube_in_tray(S):
            failure = ("missed_tray", f"cube ended at {np.round(d.xpos[S.cube], 3).tolist()}",
                       "release")

    steps = max(st["steps"], 1)
    log.update(
        exec_wall_s=round(time.perf_counter() - t_exec, 3),
        exec_ik_iters=st["ik_iters"],
        exec_ik_ms_mean=round(st["ik_time"] / steps * 1000, 4),
        exec_ik_ms_max=round(st["ik_max"] * 1000, 3),
        settle_time_s=round(st["settle_time"], 2),
        min_table_clear_mm=round(st["tab"][0] * 1000, 2), min_table_clear_link=st["tab"][1],
        min_tray_clear_mm=round(st["tray"][0] * 1000, 2), min_tray_clear_link=st["tray"][1],
        min_cube_tray_clear_mm=round(st["cube_tray"] * 1000, 2),
        min_joint_margin_rad=round(st["jmargin"][0], 4), min_joint_margin_joint=st["jmargin"][1],
        joint_limit_hits=st["limit_hits"], vel_violations=st["vel_viol"],
        max_vel_ratio=round(st["vel_ratio"], 3),
        qp_pos_limit_active_steps=st["qp_pos_active"], qp_speed_limit_active_steps=st["qp_spd_active"],
        qp_failures=st["qp_fail"],
        dls_max_lambda=round(st["lam_max"], 6),
        dls_min_manipulability=None if st["manip_min"] == np.inf else round(st["manip_min"], 4),
        max_tcp_lag_mm=round(st["lag"] * 1000, 2), max_settled_err_mm=round(st["settled"] * 1000, 2))
    return failure


# ============================================================
# EPISODES
# ============================================================

def sample_starts(S, n, seed, near_box_frac=0.3):
    """Cube pose per episode: x, y on the table (4 cm edge band), 30 % in a ring
    around the tray, yaw uniform in [-pi, pi]."""
    rng = np.random.default_rng(seed)
    x0, x1, y0, y1 = S.table_xy
    sx, sy = (x0 + 0.04, x1 - 0.04), (y0 + 0.04, y1 - 0.04)
    clear = S.box_half_outer + S.cube_reach + 0.003
    out = []
    for _ in range(n):
        near = rng.random() < near_box_frac
        while True:
            if near:
                x = rng.uniform(S.box_pos[0] - clear - 0.06, S.box_pos[0] + clear + 0.06)
                y = rng.uniform(S.box_pos[1] - clear - 0.06, S.box_pos[1] + clear + 0.06)
            else:
                x, y = rng.uniform(*sx), rng.uniform(*sy)
            if sx[0] <= x <= sx[1] and sy[0] <= y <= sy[1] and \
                    (abs(x - S.box_pos[0]) > clear or abs(y - S.box_pos[1]) > clear):
                break
        out.append((np.array([x, y]), float(rng.uniform(-np.pi, np.pi)),
                    "near_tray" if near else "uniform"))
    return out


def run_episode(S, method, ep, cube_xy, cube_yaw, region, mitigate=True, frame_cb=None):
    cube_start = np.array([cube_xy[0], cube_xy[1], S.table_top + S.cube_h])
    cube_quat = np.array([np.cos(cube_yaw / 2), 0, 0, np.sin(cube_yaw / 2)])
    log = {"method": method, "episode": ep, "spawn_region": region,
           "cube_x": round(float(cube_xy[0]), 4), "cube_y": round(float(cube_xy[1]), 4),
           "cube_yaw_deg": round(float(np.degrees(cube_yaw)), 2),
           "success": False, "failure_reason": "", "failure_detail": "", "failed_phase": "",
           "grasp_used": "", "grasp_yaw_deg": None, "grasp_tilt_deg": None,
           "candidates_tried": 0, "plan_time_s": 0.0, "plan_ik_iters": 0,
           "waypoint_ik_iters": "", "rejected": ""}
    t0 = time.perf_counter()
    chosen, first, rejected = None, None, []
    for cand in grasp_candidates(S, cube_xy, cube_yaw, mitigate):
        log["candidates_tried"] += 1
        ok, info = plan(S, method, cube_start, cand)
        log["plan_ik_iters"] += info["iters"]
        if ok:
            chosen = (cand, info)
            break
        rejected.append(f"{cand['label']}:{info['reason']}({info['phase']})")
        first = first or (cand, info)
    log["plan_time_s"] = round(time.perf_counter() - t0, 4)
    log["rejected"] = " | ".join(rejected)

    if chosen is None:
        cand, info = first
        wps = info["waypoints"]
        log.update(failure_reason=info["reason"], failure_detail=info["detail"],
                   failed_phase=info["phase"],
                   waypoint_ik_iters=" ".join(f"{w['phase']}:{w['ik_iters']}" for w in wps),
                   min_table_clear_mm=min(w["table_clear_mm"] for w in wps),
                   min_tray_clear_mm=min(w["tray_clear_mm"] for w in wps))
        return log

    cand, info = chosen
    log.update(grasp_used=cand["label"], grasp_yaw_deg=round(float(np.degrees(cand["yaw"])), 2),
               grasp_tilt_deg=round(float(np.degrees(cand["tilt"])), 2),
               waypoint_ik_iters=" ".join(f"{w['phase']}:{w['ik_iters']}"
                                          for w in info["waypoints"]))
    failure = execute(S, method, cube_start, cube_quat, cube_yaw, cand, log, frame_cb)
    if failure is None:
        log["success"] = True
    else:
        log.update(failure_reason=failure[0], failure_detail=failure[1], failed_phase=failure[2])
    b = S.data.xpos[S.cube]
    yaw_f, tilt_f = cube_yaw_tilt(S)
    log.update(final_cube_x=round(float(b[0]), 4), final_cube_y=round(float(b[1]), 4),
               final_cube_z=round(float(b[2]), 4), final_cube_tilt_deg=round(tilt_f, 2))
    return log


# ============================================================
# SUMMARY, PLOTS
# ============================================================

def summarize(logs):
    ex = [l for l in logs if "exec_wall_s" in l]
    ok = [l for l in logs if l["success"]]
    fails = {}
    for l in logs:
        if not l["success"]:
            fails[l["failure_reason"]] = fails.get(l["failure_reason"], 0) + 1
    wp_iters = [int(t.split(":")[1]) for l in logs for t in l["waypoint_ik_iters"].split()
                if t and int(t.split(":")[1]) < PLAN_MAX_ITERS]

    def mn(key, src):
        v = [l[key] for l in src if l.get(key) is not None]
        return min(v) if v else None

    return {
        "episodes": len(logs), "successes": len(ok),
        "success_rate": round(len(ok) / max(len(logs), 1), 3),
        "failures": fails,
        "plan_time_mean_s": round(float(np.mean([l["plan_time_s"] for l in logs])), 4),
        "plan_time_median_s": round(float(np.median([l["plan_time_s"] for l in logs])), 4),
        "plan_iters_mean": round(float(np.mean([l["plan_ik_iters"] for l in logs])), 1),
        "iters_per_converged_waypoint_mean": round(float(np.mean(wp_iters)), 2) if wp_iters else None,
        "exec_ik_ms_per_step_mean": round(float(np.mean([l["exec_ik_ms_mean"] for l in ex])), 3) if ex else None,
        "min_table_clear_mm": mn("min_table_clear_mm", ex),
        "min_tray_clear_mm": mn("min_tray_clear_mm", ex),
        "min_cube_tray_clear_mm": mn("min_cube_tray_clear_mm", ex),
        "min_joint_margin_rad": mn("min_joint_margin_rad", ex),
        "joint_limit_hits": int(sum(l["joint_limit_hits"] for l in ex)),
        "vel_violations": int(sum(l["vel_violations"] for l in ex)),
        "max_vel_ratio": max((l["max_vel_ratio"] for l in ex), default=None),
        "max_tcp_lag_mm": max((l["max_tcp_lag_mm"] for l in ex), default=None),
        "max_settled_err_mm": max((l["max_settled_err_mm"] for l in ex), default=None),
        "settle_time_mean_s": round(float(np.mean([l["settle_time_s"] for l in ex])), 3) if ex else None,
    }


def save(out, method, logs):
    d = os.path.join(out, method)
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
        w.writerows(logs)


ROWS = [("Success rate", "success_rate", "{:.0%}"),
        ("Mean planning time (s)", "plan_time_mean_s", "{:.3f}"),
        ("Planning IK iterations / episode (mean)", "plan_iters_mean", "{:.0f}"),
        ("IK iterations / converged waypoint (mean)", "iters_per_converged_waypoint_mean", "{:.1f}"),
        ("IK compute per control step (ms)", "exec_ik_ms_per_step_mean", "{:.2f}"),
        ("Min robot-table clearance (mm)", "min_table_clear_mm", "{:.1f}"),
        ("Min robot-tray clearance (mm)", "min_tray_clear_mm", "{:.1f}"),
        ("Min cube-tray clearance (mm)", "min_cube_tray_clear_mm", "{:.1f}"),
        ("Min joint-limit margin (rad)", "min_joint_margin_rad", "{:.3f}"),
        ("Joint-limit hits (steps)", "joint_limit_hits", "{:d}"),
        ("Velocity-limit violations (steps)", "vel_violations", "{:d}"),
        ("Peak joint speed / limit", "max_vel_ratio", "{:.2f}"),
        ("Max TCP tracking lag (mm)", "max_tcp_lag_mm", "{:.1f}"),
        ("Max settled TCP error (mm)", "max_settled_err_mm", "{:.1f}"),
        ("Extra settle time / episode (s, mean)", "settle_time_mean_s", "{:.2f}")]


def write_summary(out, summaries):
    methods = list(summaries)
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + methods)
        for label, key, _ in ROWS:
            w.writerow([label] + [summaries[m][key] for m in methods])
        w.writerow(["Failures"] + [json.dumps(summaries[m]["failures"]) for m in methods])
    lines = ["| Metric | " + " | ".join(METHOD_LABEL[m] for m in methods) + " |",
             "| --- |" + " --- |" * len(methods)]
    for label, key, fmt in ROWS:
        cells = []
        for m in methods:
            v = summaries[m][key]
            cells.append("-" if v is None else fmt.format(v))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append("| Failures | " + " | ".join(
        ", ".join(f"{k}: {v}" for k, v in summaries[m]["failures"].items()) or "none"
        for m in methods) + " |")
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return "\n".join(lines)


COLOURS = {"SUCCESS": "#4c9f50", "ik_no_solution": "#d62728",
           "planned_pose_collision": "#9467bd", "collision": "#ff7f0e",
           "cube_tray_collision": "#17becf", "cube_disturbed": "#8c564b",
           "grasp_slip": "#e377c2", "dropped_in_transit": "#bcbd22", "missed_tray": "#1f77b4"}


def plots(out, S, all_logs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    methods = list(all_logs)
    labels = [METHOD_LABEL[m].split(" (")[0] for m in methods]
    mcol = {"mink": "#7f7f7f", "dls": "#1f77b4", "qp": "#d62728"}

    # outcomes
    cats = ["SUCCESS"] + sorted({l["failure_reason"] for m in methods for l in all_logs[m]
                                 if not l["success"]})
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bottom = np.zeros(len(methods))
    for c in cats:
        v = np.array([sum((l["success"] if c == "SUCCESS" else l["failure_reason"] == c)
                          for l in all_logs[m]) for m in methods], float)
        ax.bar(labels, v, bottom=bottom, color=COLOURS.get(c, "k"), label=c, edgecolor="white")
        bottom += v
    for i, m in enumerate(methods):
        s = sum(l["success"] for l in all_logs[m])
        ax.text(i, bottom[i] + 0.3, f"{s}/{len(all_logs[m])}", ha="center", fontweight="bold")
    ax.set_ylabel("episodes")
    ax.set_ylim(0, bottom.max() * 1.15)
    ax.set_title("Episode outcomes (identical seeds)")
    ax.legend(fontsize=8, frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "outcomes.png"), dpi=140)
    plt.close()

    # metrics 2x3
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))

    def box(ax, key, title, ylabel, executed=True, log=False):
        data = [[l[key] for l in all_logs[m] if (not executed or "exec_wall_s" in l)
                 and l.get(key) is not None] for m in methods]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55)
        for patch, m in zip(bp["boxes"], methods):
            patch.set_facecolor(mcol[m])
            patch.set_alpha(0.55)
        for i, dd in enumerate(data):
            ax.scatter(np.full(len(dd), i + 1) + np.random.default_rng(i).uniform(-0.12, 0.12, len(dd)),
                       dd, s=9, color="k", alpha=0.6, zorder=3)
        ax.set_xticks(range(1, len(methods) + 1))
        ax.set_xticklabels(labels)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel)
        if log:
            ax.set_yscale("log")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    box(axs[0, 0], "plan_time_s", "Planning time per episode", "s", executed=False, log=True)
    box(axs[0, 1], "plan_ik_iters", "Planning IK iterations per episode", "iterations",
        executed=False, log=True)
    box(axs[0, 2], "exec_ik_ms_mean", "IK compute per 10 ms control step", "ms")
    box(axs[1, 0], "max_tcp_lag_mm", "Max TCP tracking error while moving", "mm")
    box(axs[1, 1], "max_settled_err_mm", "Max TCP error at end of each move", "mm")
    ax = axs[1, 2]
    keys = [("min_table_clear_mm", "robot-table"), ("min_tray_clear_mm", "robot-tray"),
            ("min_cube_tray_clear_mm", "cube-tray")]
    w = 0.25
    for j, (k, lab) in enumerate(keys):
        vals = [min((l[k] for l in all_logs[m] if "exec_wall_s" in l), default=0) for m in methods]
        ax.bar(np.arange(len(methods)) + (j - 1) * w, vals, w, label=lab)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("mm (worst executed episode)")
    ax.set_title("Minimum clearances", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "metrics.png"), dpi=140)
    plt.close()

    # outcome maps
    fig, axs = plt.subplots(1, len(methods), figsize=(5.2 * len(methods), 4.6))
    axs = np.atleast_1d(axs)
    x0, x1, y0, y1 = S.table_xy
    for ax, m in zip(axs, methods):
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#d2b48c", ec="k", alpha=0.5))
        ax.add_patch(plt.Rectangle((S.box_pos[0] - S.box_half_outer, S.box_pos[1] - S.box_half_outer),
                                   2 * S.box_half_outer, 2 * S.box_half_outer, fc="royalblue",
                                   ec="k", alpha=0.6))
        ax.plot(0, 0, "ks", ms=8)
        seen = set()
        for l in all_logs[m]:
            key = "SUCCESS" if l["success"] else l["failure_reason"]
            a = np.radians(l["cube_yaw_deg"])
            Rm = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            sq = (np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * S.cube_h) @ Rm.T \
                + [l["cube_x"], l["cube_y"]]
            ax.add_patch(plt.Polygon(sq, fc=COLOURS.get(key, "k"), ec="k", zorder=3,
                                     label=None if key in seen else key))
            seen.add(key)
        s = sum(l["success"] for l in all_logs[m])
        ax.set_title(f"{METHOD_LABEL[m]}: {s}/{len(all_logs[m])}", fontsize=10)
        ax.set_xlim(-0.05, 0.85)
        ax.set_ylim(-0.32, 0.32)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.legend(fontsize=7, loc="lower left")
    axs[0].set_ylabel("y (m)")
    plt.tight_layout()
    plt.savefig(os.path.join(out, "outcome_maps.png"), dpi=140)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    ap.add_argument("--episodes", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--xml", default=_os.path.join(SCENE_DIR, "pick_place_cube_scene.xml"))
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-mitigation", action="store_true",
                    help="top-down grasp at the default yaw only (baseline)")
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()

    S = Scene(args.xml)
    os.makedirs(args.out, exist_ok=True)
    starts = sample_starts(S, args.episodes, args.seed)
    print(f"MuJoCo {mujoco.__version__} | {args.episodes} episodes | seed {args.seed} | "
          f"mitigation {'off' if args.no_mitigation else 'on'}")

    viewer, cb = None, None
    if args.viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(S.model, S.data)

        def cb(S_, info):
            viewer.sync()
            time.sleep(CONTROL_DT)

    all_logs, summaries = {}, {}
    for method in args.methods:
        print(f"\n=== {METHOD_LABEL[method]} ===")
        print(f"{'ep':>3} {'x':>6} {'y':>6} {'yaw':>5}  {'result':<22} {'grasp':<9} "
              f"{'plan s':>7} {'planIt':>6} {'ms/step':>7} {'cube-tray':>9}")
        logs = []
        for ep, (xy, yaw, region) in enumerate(starts):
            l = run_episode(S, method, ep, xy, yaw, region, not args.no_mitigation, cb)
            logs.append(l)
            res = "SUCCESS" if l["success"] else l["failure_reason"]
            ct = l.get("min_cube_tray_clear_mm")
            print(f"{ep:>3} {xy[0]:6.3f} {xy[1]:6.3f} {np.degrees(yaw):5.0f}  {res:<22} "
                  f"{l['grasp_used'] or '-':<9} {l['plan_time_s']:7.3f} {l['plan_ik_iters']:6d} "
                  f"{l.get('exec_ik_ms_mean', float('nan')):7.2f} "
                  f"{ct if ct is not None else float('nan'):9.1f}", flush=True)
        save(args.out, method, logs)
        all_logs[method] = logs
        summaries[method] = summarize(logs)

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=1, default=float)
    print("\n" + write_summary(args.out, summaries))
    plots(args.out, S, all_logs)
    print(f"\nSaved to {os.path.abspath(args.out)}/")
    if viewer is not None:
        viewer.close()


if __name__ == "__main__":
    main()
