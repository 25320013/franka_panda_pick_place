import mujoco
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==========================================================
# 1. LOAD THE ORIGINAL FRANKA PANDA SCENE
# ==========================================================

XML_PATH = (
    "/home/iitgn-robotics/Pant/mujoco_menagerie/"
    "franka_emika_panda/scene.xml"
)

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print("=" * 55)
print("FRANKA PANDA REACHABLE WORKSPACE ANALYSIS")
print("=" * 55)


# ==========================================================
# 2. IDENTIFY THE SEVEN ARM JOINTS
# ==========================================================

joint_ids = []

for i in range(model.njnt):
    name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_JOINT, i
    )

    if name and name.startswith("joint") and name[5:].isdigit():
        joint_ids.append(i)

joint_ids.sort(
    key=lambda i: int(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)[5:]
    )
)

if len(joint_ids) != 7:
    raise RuntimeError(
        f"Expected 7 arm joints, found {len(joint_ids)}."
    )

qpos_ids = model.jnt_qposadr[joint_ids]
joint_limits = model.jnt_range[joint_ids]

print("\nArm joints identified:", len(joint_ids))

for i, joint_id in enumerate(joint_ids):
    name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
    )
    low, high = joint_limits[i]
    print(f"{name}: {low:.4f} to {high:.4f}")


# ==========================================================
# 3. IDENTIFY THE HAND BODY
# ==========================================================

hand_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "hand"
)

if hand_id < 0:
    raise RuntimeError("Could not find body named 'hand'.")

print("\nUsing body 'hand' as the TCP approximation.")


# ==========================================================
# 4. SAMPLE VALID JOINT CONFIGURATIONS
# ==========================================================

NUM_SAMPLES = 30000
rng = np.random.default_rng(seed=42)

lower_limits = joint_limits[:, 0]
upper_limits = joint_limits[:, 1]

# Uniform random sampling within joint limits.
joint_samples = rng.uniform(
    low=lower_limits,
    high=upper_limits,
    size=(NUM_SAMPLES, 7)
)

workspace_points = np.empty((NUM_SAMPLES, 3))

for i, q in enumerate(joint_samples):
    # Set the seven arm joint positions.
    data.qpos[qpos_ids] = q

    # Update forward kinematics.
    mujoco.mj_forward(model, data)

    # Record hand position in world coordinates.
    workspace_points[i] = data.xpos[hand_id].copy()

    if (i + 1) % 5000 == 0:
        print(f"Processed {i + 1}/{NUM_SAMPLES} configurations")


# ==========================================================
# 5. REPORT WORKSPACE COORDINATES
# ==========================================================

minimum = workspace_points.min(axis=0)
maximum = workspace_points.max(axis=0)

print("\nWorkspace bounds (metres):")
print(f"X: {minimum[0]:.3f} to {maximum[0]:.3f}")
print(f"Y: {minimum[1]:.3f} to {maximum[1]:.3f}")
print(f"Z: {minimum[2]:.3f} to {maximum[2]:.3f}")


# ==========================================================
# 6. SAVE THE DATA
# ==========================================================

np.save("workspace_reachable.npy", workspace_points)

np.savetxt(
    "workspace_reachable.csv",
    workspace_points,
    delimiter=",",
    header="x,y,z",
    comments=""
)

print("\nSaved:")
print("  workspace_reachable.npy")
print("  workspace_reachable.csv")


# ==========================================================
# 7. VISUALIZE THE REACHABLE WORKSPACE
# ==========================================================

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

scatter = ax.scatter(
    workspace_points[:, 0],
    workspace_points[:, 1],
    workspace_points[:, 2],
    c=workspace_points[:, 2],
    cmap="viridis",
    s=1,
    alpha=0.35
)

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Franka Panda — Reachable Workspace")

fig.colorbar(
    scatter,
    ax=ax,
    label="TCP height Z (m)",
    shrink=0.7
)

ax.set_box_aspect((
    np.ptp(workspace_points[:, 0]),
    np.ptp(workspace_points[:, 1]),
    np.ptp(workspace_points[:, 2])
))

plt.tight_layout()
plt.savefig("workspace_reachable.png", dpi=200)
plt.close()
print("\nSaved workspace_reachable.png")
print("\nReachable workspace analysis complete.")
