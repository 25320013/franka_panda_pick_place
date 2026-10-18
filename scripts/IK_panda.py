import time
import numpy as np
import mujoco
import mujoco.viewer
import mink

# ---------------------------------------------------------
# 1. Load Franka Emika Panda Model & Data
# ---------------------------------------------------------
XML_PATH = "/home/iitgn-robotics/Pant/mujoco_menagerie/franka_emika_panda/scene.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Find 'hand' body ID
hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")

# Initialize nominal joint configuration (arm start pose)
q_init = np.array([0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398])
data.qpos[:7] = q_init
mujoco.mj_forward(model, data)

# ---------------------------------------------------------
# 2. Setup Mink Configuration & Tasks
# ---------------------------------------------------------
configuration = mink.Configuration(model)

# Primary end-effector task (targeting 'hand' body frame)
ee_task = mink.FrameTask(
    frame_name="hand",
    frame_type="body",
    position_cost=1.0,
    orientation_cost=0.5,
    lm_damping=1e-3,
)

# Secondary posture task (PREVENTS 'TargetNotSet' ERROR)
posture_task = mink.PostureTask(model, cost=1e-2)
posture_task.set_target_from_configuration(configuration)

tasks = [ee_task, posture_task]


# ---------------------------------------------------------
# 3. Define Target Pose (Desired Cartesian Target)
# ---------------------------------------------------------
# Target Position [x, y, z] in meters
target_position = np.array([0.45, 0.20, 0.40])

# Target Orientation: Pointing downward along -Z
target_quaternion = np.array([0.0, 1.0, 0.0, 0.0])  # [w, x, y, z]

# Construct Mink SE3 transformation target
target_transform = mink.SE3.from_rotation_and_translation(
    rotation=mink.SO3(target_quaternion),
    translation=target_position
)
ee_task.set_target(target_transform)

# ---------------------------------------------------------
# 4. Solve Inverse Kinematics Loop
# ---------------------------------------------------------
dt = 0.005
max_steps = 200  # Number of IK iterations to converge
tolerance = 1e-4

print("========================================")
print("SOLVING INVERSE KINEMATICS WITH MINK")
print("========================================")
print(f"Target Position [x, y, z] : {target_position}")

for step in range(max_steps):
    # Compute velocity step using QP solver
    vel = mink.solve_ik(
        configuration=configuration,
        tasks=tasks,
        dt=dt,
        solver="clarabel"
    )
    
    # Integrate velocities into joint positions
    configuration.integrate_inplace(vel, dt)
    
    # Check convergence based on position error
    data.qpos[:] = configuration.q
    mujoco.mj_forward(model, data)
    achieved_pos = data.xpos[hand_id]
    pos_error = np.linalg.norm(achieved_pos - target_position)
    
    if pos_error < tolerance:
        print(f"IK converged in {step + 1} steps!")
        break

# ---------------------------------------------------------
# 5. Output Results
# ---------------------------------------------------------
final_joint_angles = data.qpos[:7]
achieved_position = data.xpos[hand_id]

print("\nComputed Joint Angles (q):")
print(np.round(final_joint_angles, 4))

print("\nAchieved Position [x, y, z]:")
print(achieved_position)

print(f"\nFinal Position Error: {pos_error:.6f} meters")
print("========================================")

# ---------------------------------------------------------
# 6. Launch MuJoCo Viewer to Visualise
# ---------------------------------------------------------
print("\nOpening viewer... Close window to exit.")
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.perf_counter()
        
        viewer.sync()
        
        time_until_next_step = dt - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)