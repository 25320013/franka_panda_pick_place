import mujoco
import numpy as np
from mujoco.viewer import launch

# ---------------------------------------------------------
# Load Franka Emika Panda
# ---------------------------------------------------------

XML_PATH = "/home/iitgn-robotics/Pant/mujoco_menagerie/franka_emika_panda/scene.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# ---------------------------------------------------------
# Find Panda hand body
# ---------------------------------------------------------

hand_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "hand"
)

print("Hand body ID:", hand_id)

# ---------------------------------------------------------
# Set a joint configuration
# ---------------------------------------------------------

q = np.array([
    0.0,      # joint1
    -0.5,     # joint2
    0.0,      # joint3
    -1.5,     # joint4
    0.0,      # joint5
    1.5,      # joint6
    0.0       # joint7
])

# Put joint configuration into MuJoCo
data.qpos[:7] = q

# Keep gripper open
data.qpos[7] = 0.04
data.qpos[8] = 0.04

# ---------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------

mujoco.mj_forward(model, data)

# ---------------------------------------------------------
# Get hand position
# ---------------------------------------------------------

hand_position = data.xpos[hand_id]

# Get hand orientation
hand_rotation = data.xmat[hand_id].reshape(3, 3)

print("\n========================================")
print("FORWARD KINEMATICS")
print("========================================")

print("\nJoint configuration:")
print(q)

print("\nHand position [x, y, z] meters:")
print(hand_position)

print("\nHand orientation:")
print(hand_rotation)

print("\n========================================")

# ---------------------------------------------------------
# Launch viewer
# ---------------------------------------------------------

viewer = launch(model, data)

while viewer and viewer.is_running():
    viewer.sync()
