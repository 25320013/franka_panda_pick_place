import mujoco
import numpy as np
from mujoco.viewer import launch

# Path to your MuJoCo Panda model
URDF_FILE_PATH = "/home/iitgn-robotics/Pant/mujoco_menagerie/franka_emika_panda/scene.xml"

# Load the MuJoCo model
model = mujoco.MjModel.from_xml_path(URDF_FILE_PATH)

# Create a corresponding data object
data = mujoco.MjData(model)

# Perform one simulation step
mujoco.mj_step(model, data)

# Launch the interactive MuJoCo viewer
viewer = launch(model, data)

# Keep the viewer open
while viewer and viewer.is_running():
    viewer.render()
    
    
