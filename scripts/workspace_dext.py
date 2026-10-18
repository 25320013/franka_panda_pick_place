import mujoco
import numpy as np
import matplotlib

# Save plots to files without requiring a graphical display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mink import Configuration, FrameTask, SE3, SO3, solve_ik


# ============================================================
# CONFIGURATION
# ============================================================

XML_PATH = (
    "/home/iitgn-robotics/Pant/mujoco_menagerie/"
    "franka_emika_panda/scene.xml"
)

REACHABLE_FILE = "workspace_reachable.npy"

NUM_CANDIDATES = 500
NUM_YAW_ANGLES = 16

MAX_IK_ITERATIONS = 100
DT = 0.02

POSITION_TOLERANCE = 0.0015       # 15 mm
ORIENTATION_TOLERANCE = 0.010     # approximately 5.7 degrees

RANDOM_SEED = 42


# ============================================================
# 1. LOAD MUJOCO MODEL AND REACHABLE WORKSPACE
# ============================================================

print("=" * 65)
print("FRANKA PANDA DEXTEROUS WORKSPACE ANALYSIS")
print("=" * 65)

model = mujoco.MjModel.from_xml_path(XML_PATH)

# The previous inspection identified 'hand' as body ID 9.
# We use its origin as an approximation of the TCP.
hand_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "hand"
)

if hand_id < 0:
    raise ValueError("Could not find the body named 'hand'.")

reachable_points = np.load(REACHABLE_FILE)

print(f"\nReachable workspace points: {len(reachable_points)}")
print(f"MuJoCo model nq: {model.nq}")

# Randomly select candidate positions from the reachable cloud.
rng = np.random.default_rng(RANDOM_SEED)

num_candidates = min(NUM_CANDIDATES, len(reachable_points))

candidate_indices = rng.choice(
    len(reachable_points),
    size=num_candidates,
    replace=False
)

candidate_points = reachable_points[candidate_indices]


# ============================================================
# 2. INITIALIZE MINK
# ============================================================

configuration = Configuration(model)

# Reference arm configuration used in the earlier FK exercise.
q_seed = np.zeros(model.nq)

q_seed[:7] = [
    0.0,
    -0.5,
    0.0,
    -1.5,
    0.0,
    1.5,
    0.0
]

# Set both finger joints to 0.02 m if this model has 9 qpos values.
if model.nq >= 9:
    q_seed[7] = 0.02
    q_seed[8] = 0.02

configuration.update(q_seed)

# Obtain the reference hand orientation.
reference_pose = configuration.get_transform_frame_to_world(
    "hand",
    "body"
)

reference_rotation = reference_pose.rotation()

print("\nMink initialized.")
print("End-effector frame: body 'hand'")
print("TCP definition: hand body origin")
print(f"Candidate positions: {num_candidates}")
print(f"Yaw orientations per position: {NUM_YAW_ANGLES}")


# ============================================================
# 3. CREATE THE MINK IK TASK
# ============================================================

hand_task = FrameTask(
    frame_name="hand",
    frame_type="body",
    position_cost=1.0,
    orientation_cost=1.0,
    lm_damping=1e-3
)


# ============================================================
# 4. TEST EACH POSITION AT MULTIPLE YAW ANGLES
# ============================================================

yaw_angles = np.linspace(
    0.0,
    2.0 * np.pi,
    NUM_YAW_ANGLES,
    endpoint=False
)

results = []

for i, target_position in enumerate(candidate_points):

    successful_yaws = 0

    for yaw in yaw_angles:

        # Reset to the same starting configuration for every test.
        configuration.update(q_seed)

        # Rotate the reference orientation about WORLD Z.
        world_yaw_rotation = SO3.from_matrix(
            np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw),  np.cos(yaw), 0.0],
                [0.0,          0.0,         1.0]
            ])
        )

        target_rotation = SO3.from_matrix(
            world_yaw_rotation.as_matrix()
            @ reference_rotation.as_matrix()
        )

        # Construct the target pose: position + orientation.
        target_pose = (
            SE3.from_translation(target_position)
            @ SE3.from_rotation(target_rotation)
        )

        hand_task.set_target(target_pose)

        # Differential IK: iteratively reduce pose error.
        for iteration in range(MAX_IK_ITERATIONS):

            velocity = solve_ik(
                configuration,
                [hand_task],
                DT,
                "daqp",
                damping=1e-3
            )

            configuration.integrate_inplace(velocity, DT)

            # Check whether the target pose has been reached.
            current_pose = (
                configuration.get_transform_frame_to_world(
                    "hand",
                    "body"
                )
            )

            position_error = np.linalg.norm(
                current_pose.translation()
                - target_position
            )

            task_error = hand_task.compute_error(configuration)
            orientation_error = np.linalg.norm(task_error[3:])

            if (
                position_error <= POSITION_TOLERANCE
                and orientation_error <= ORIENTATION_TOLERANCE
            ):
                break

        # Count this yaw as successful only if BOTH tolerances pass.
        if (
            position_error <= POSITION_TOLERANCE
            and orientation_error <= ORIENTATION_TOLERANCE
        ):
            successful_yaws += 1

    results.append([
        target_position[0],
        target_position[1],
        target_position[2],
        successful_yaws
    ])

    if (i + 1) % 10 == 0 or i + 1 == num_candidates:
        print(
            f"Tested {i + 1}/{num_candidates} positions"
        )


# ============================================================
# 5. CLASSIFY DEXTEROUS POSITIONS
# ============================================================

results = np.asarray(results)

positions = results[:, :3]
yaw_success_counts = results[:, 3].astype(int)

# Strict sampled definition: all eight yaw targets must succeed.
dexterous_mask = yaw_success_counts == NUM_YAW_ANGLES

dexterous_points = positions[dexterous_mask]

print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)

print(f"Candidate positions tested: {len(positions)}")

print(
    "Positions successful at all eight yaw angles: "
    f"{len(dexterous_points)}"
)

print(
    "Positions successful at four or more yaw angles: "
    f"{np.sum(yaw_success_counts >= 4)}"
)

print("\nYaw success-count distribution:")

for count in range(NUM_YAW_ANGLES + 1):
    number = np.sum(yaw_success_counts == count)
    print(f"  {count}/{NUM_YAW_ANGLES} orientations: {number}")


# ============================================================
# 6. SAVE NUMERICAL RESULTS
# ============================================================

np.save("workspace_dexterous.npy", dexterous_points)

np.savetxt(
    "workspace_dexterous.csv",
    dexterous_points,
    delimiter=",",
    header="x,y,z",
    comments=""
)

np.savetxt(
    "workspace_dexterous_scores.csv",
    results,
    delimiter=",",
    header="x,y,z,successful_yaw_count",
    comments=""
)

print("\nSaved:")
print("  workspace_dexterous.npy")
print("  workspace_dexterous.csv")
print("  workspace_dexterous_scores.csv")


# ============================================================
# 7. VISUALIZE REACHABLE AND DEXTEROUS WORKSPACES
# ============================================================

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(
    reachable_points[:, 0],
    reachable_points[:, 1],
    reachable_points[:, 2],
    s=1,
    alpha=0.08,
    label="Sampled reachable workspace"
)

if len(dexterous_points) > 0:
    ax.scatter(
        dexterous_points[:, 0],
        dexterous_points[:, 1],
        dexterous_points[:, 2],
        s=18,
        label="Sampled yaw-dexterous workspace"
    )

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")

ax.set_title("Franka Panda: Reachable vs Dexterous Workspace")
ax.legend()

plt.tight_layout()
plt.savefig("workspace_dexterous.png", dpi=200)
plt.close()

print("  workspace_dexterous.png")
print("\nAnalysis complete.")