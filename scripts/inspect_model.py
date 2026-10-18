import mujoco

# ---------------------------------------------------------
# Load Franka Emika Panda
# ---------------------------------------------------------

XML_PATH = "/home/iitgn-robotics/Pant/mujoco_menagerie/franka_emika_panda/scene.xml"

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

print("\n========================================")
print("FRANKA PANDA MODEL INSPECTION")
print("========================================")

# ---------------------------------------------------------
# 1. JOINTS
# ---------------------------------------------------------

print("\n========== JOINTS ==========")

for i in range(model.njnt):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        i
    )

    print(
        f"Joint ID: {i:2d} | "
        f"Name: {name} | "
        f"Type: {model.jnt_type[i]} | "
        f"qpos address: {model.jnt_qposadr[i]}"
    )

# ---------------------------------------------------------
# 2. JOINT LIMITS
# ---------------------------------------------------------

print("\n========== JOINT LIMITS ==========")

for i in range(model.njnt):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        i
    )

    if model.jnt_limited[i]:

        print(
            f"{name:20s} : "
            f"{model.jnt_range[i][0]: .4f} "
            f"to "
            f"{model.jnt_range[i][1]: .4f}"
        )

# ---------------------------------------------------------
# 3. BODIES
# ---------------------------------------------------------

print("\n========== BODIES ==========")

for i in range(model.nbody):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        i
    )

    print(
        f"Body ID: {i:2d} | "
        f"Name: {name}"
    )

# ---------------------------------------------------------
# 4. SITES
# ---------------------------------------------------------

print("\n========== SITES ==========")

for i in range(model.nsite):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        i
    )

    print(
        f"Site ID: {i:2d} | "
        f"Name: {name}"
    )

# ---------------------------------------------------------
# 5. ACTUATORS
# ---------------------------------------------------------

print("\n========== ACTUATORS ==========")

for i in range(model.nu):

    name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        i
    )

    print(
        f"Actuator ID: {i:2d} | "
        f"Name: {name}"
    )

# ---------------------------------------------------------
# 6. MODEL SIZE
# ---------------------------------------------------------

print("\n========== MODEL SIZE ==========")

print("Number of bodies    :", model.nbody)
print("Number of joints    :", model.njnt)
print("Number of DOF       :", model.nv)
print("Number of qpos      :", model.nq)
print("Number of sites     :", model.nsite)
print("Number of actuators :", model.nu)

print("\n========================================")
print("Inspection complete.")
print("========================================")
