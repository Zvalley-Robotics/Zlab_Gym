"""Z01_std_add_head (量产加头版) humanoid constants.

Model: 29 actuated DOF (7 per arm, 6 per leg, waist_yaw, head_pitch/head_yaw).
Joint names are kept exactly as in the vendor URDF/CSV (no ``_joint`` suffix),
e.g. ``left_knee``, ``right_wrist_roll``, ``waist_yaw``, ``head_pitch``.

The MJCF actually handed to mjlab lives in ``xmls/Z01_std_add_head.xml``; it is
derived from the URDF->MJCF converter output in ``xml/`` (see the header comment
of that file for the exact differences).
"""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg, DelayedActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

Z01_STD_ADD_HEAD_XML: Path = (
    SRC_PATH / "assets" / "robots" / "Z01_std_add_head" / "xmls" / "Z01_std_add_head.xml"
)
assert Z01_STD_ADD_HEAD_XML.exists()

# Meshes are shared with the generated model: xmls/../meshes matches the
# ``meshdir="../meshes/"`` recorded in the MJCF.
Z01_STD_ADD_HEAD_MESHES: Path = Z01_STD_ADD_HEAD_XML.parent.parent / "meshes"
assert Z01_STD_ADD_HEAD_MESHES.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, Z01_STD_ADD_HEAD_MESHES, meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(Z01_STD_ADD_HEAD_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


##
# Actuator config.
##

# Motor classes, taken from csv/Z01_std_add_head_joints.csv.
# ``reflected_inertia`` is the joint-side armature the vendor reports per joint
# (identical to the ``armature`` attribute of the generated MJCF) and is written
# back onto the joint by mjlab; ``velocity_limit`` is informational only, since
# ``BuiltinPositionActuatorCfg`` models a PD setpoint (no velocity curve).
ROTOR_INERTIA_ARM = 0.0027
ROTOR_INERTIA_WRIST = 0.005521
ROTOR_INERTIA_LEG = 0.041
ROTOR_INERTIA_ANKLE_PITCH = 0.0077733
ROTOR_INERTIA_ANKLE_ROLL = 0.0026244

MOTOR_ARM = ElectricActuator(
    reflected_inertia=ROTOR_INERTIA_ARM,
    velocity_limit=25.65,
    effort_limit=25.0,
)
MOTOR_WRIST = ElectricActuator(
    reflected_inertia=ROTOR_INERTIA_WRIST,
    velocity_limit=24.12,
    effort_limit=12.0,
)
MOTOR_LEG = ElectricActuator(
    reflected_inertia=ROTOR_INERTIA_LEG,
    velocity_limit=20.94,
    effort_limit=140.0,
)
MOTOR_ANKLE_PITCH = ElectricActuator(
    reflected_inertia=ROTOR_INERTIA_ANKLE_PITCH,
    velocity_limit=12.82,
    effort_limit=50.0,
)
MOTOR_ANKLE_ROLL = ElectricActuator(
    reflected_inertia=ROTOR_INERTIA_ANKLE_ROLL,
    velocity_limit=25.65,
    effort_limit=25.0,
)

# PD gains are not part of the vendor data. They follow the kp/kd-per-motor-size
# convention used for zoomlion_t4 / tienkung and are meant to be tuned (e.g. the
# 10 Hz critically/over-damped formula in
# zoomlion_t4_tienkung/constants.py: K = J * w_n^2, D = 2 * zeta * J * w_n).
DELAY_MIN_LAG = 0
DELAY_MAX_LAG = 3

Z01_ACTUATOR_ARM = DelayedActuatorCfg(
    base_cfg=BuiltinPositionActuatorCfg(
        target_names_expr=(
            ".*_shoulder_pitch",
            ".*_shoulder_roll",
            ".*_shoulder_yaw",
            ".*_elbow",
            ".*_wrist_yaw",
        ),
        stiffness=20.0,
        damping=1.0,
        effort_limit=MOTOR_ARM.effort_limit,
        armature=MOTOR_ARM.reflected_inertia,
    ),
    delay_target="position",
    delay_min_lag=DELAY_MIN_LAG,
    delay_max_lag=DELAY_MAX_LAG,
)

Z01_ACTUATOR_WRIST_HEAD = DelayedActuatorCfg(
    base_cfg=BuiltinPositionActuatorCfg(
        target_names_expr=(
            ".*_wrist_pitch",
            ".*_wrist_roll",
            "head_pitch",
            "head_yaw",
        ),
        stiffness=10.0,
        damping=1.0,
        effort_limit=MOTOR_WRIST.effort_limit,
        armature=MOTOR_WRIST.reflected_inertia,
    ),
    delay_target="position",
    delay_min_lag=DELAY_MIN_LAG,
    delay_max_lag=DELAY_MAX_LAG,
)

Z01_ACTUATOR_LEG = DelayedActuatorCfg(
    base_cfg=BuiltinPositionActuatorCfg(
        target_names_expr=(
            "waist_yaw",
            ".*_hip_pitch",
            ".*_hip_roll",
            ".*_hip_yaw",
            ".*_knee",
        ),
        stiffness=100.0,
        damping=4.0,
        effort_limit=MOTOR_LEG.effort_limit,
        armature=MOTOR_LEG.reflected_inertia,
    ),
    delay_target="position",
    delay_min_lag=DELAY_MIN_LAG,
    delay_max_lag=DELAY_MAX_LAG,
)

Z01_ACTUATOR_ANKLE_PITCH = DelayedActuatorCfg(
    base_cfg=BuiltinPositionActuatorCfg(
        target_names_expr=(".*_ankle_pitch",),
        stiffness=40.0,
        damping=2.0,
        effort_limit=MOTOR_ANKLE_PITCH.effort_limit,
        armature=MOTOR_ANKLE_PITCH.reflected_inertia,
    ),
    delay_target="position",
    delay_min_lag=DELAY_MIN_LAG,
    delay_max_lag=DELAY_MAX_LAG,
)

Z01_ACTUATOR_ANKLE_ROLL = DelayedActuatorCfg(
    base_cfg=BuiltinPositionActuatorCfg(
        target_names_expr=(".*_ankle_roll",),
        stiffness=20.0,
        damping=1.0,
        effort_limit=MOTOR_ANKLE_ROLL.effort_limit,
        armature=MOTOR_ANKLE_ROLL.reflected_inertia,
    ),
    delay_target="position",
    delay_min_lag=DELAY_MIN_LAG,
    delay_max_lag=DELAY_MAX_LAG,
)

##
# Keyframe config.
##

# Nominal standing pose. Values are inherited from T4_27DOF_HOME_KEYFRAME, whose
# joint axes / sign conventions match this model (all values are inside the Z01
# joint limits). Replace with a measured Z01 pose when available.
# With this pose the lowest foot geom sits at z = 0.0571 below the torso, so the
# base starts ~7 mm above ground contact and settles at reset.
HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0, 0, 0.85),
    joint_pos={
        "left_shoulder_pitch": 0.149569,
        "right_shoulder_pitch": 0.148084,
        "left_shoulder_roll": 0.216868,
        "right_shoulder_roll": -0.22107,
        "left_shoulder_yaw": -0.03033,
        "right_shoulder_yaw": 0.039483,
        "left_elbow": -0.38243,
        "right_elbow": -0.39943,
        "left_wrist_yaw": 0.003242,
        "right_wrist_yaw": -0.00668,
        "left_wrist_pitch": 0.026512,
        "right_wrist_pitch": 0.019646,
        "left_wrist_roll": 0.005913,
        "right_wrist_roll": -0.005913,
        "left_hip_pitch": -0.1112,
        "right_hip_pitch": -0.1162,
        "left_hip_roll": 0.013328,
        "right_hip_roll": -0.00608,
        "left_hip_yaw": 0.014357,
        "right_hip_yaw": -0.011202,
        "left_knee": 0.14,
        "right_knee": 0.14,
        "left_ankle_pitch": -0.10064,
        "right_ankle_pitch": -0.10749,
        "left_ankle_roll": -0.00982,
        "right_ankle_roll": -0.00884,
        "waist_yaw": 0.0,
        "head_pitch": 0.0,
        "head_yaw": 0.0,
    },
    joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
# Geoms named "*_collision" are the collision proxies (boxes / cylinders) of the
# generated model plus the 7 foot capsules per foot; every other geom (visual
# meshes, the head debug sphere) is a non-colliding geom.
FULL_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
    priority={r"^(left|right)_foot[1-7]_collision$": 1},
    friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

##
# Final config.
##

Z01_STD_ADD_HEAD_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        Z01_ACTUATOR_ARM,
        Z01_ACTUATOR_WRIST_HEAD,
        Z01_ACTUATOR_LEG,
        Z01_ACTUATOR_ANKLE_PITCH,
        Z01_ACTUATOR_ANKLE_ROLL,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_z01_std_add_head_robot_cfg() -> EntityCfg:
    """Get a fresh Z01_std_add_head robot configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues when
    the config is shared across multiple places.
    """
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=Z01_STD_ADD_HEAD_ARTICULATION,
    )


Z01_STD_ADD_HEAD_ACTION_SCALE: dict[str, float] = {}
for a in Z01_STD_ADD_HEAD_ARTICULATION.actuators:
    assert isinstance(a, DelayedActuatorCfg)
    e = a.base_cfg.effort_limit
    s = a.base_cfg.stiffness
    names = a.base_cfg.target_names_expr
    assert e is not None
    for n in names:
        Z01_STD_ADD_HEAD_ACTION_SCALE[n] = 0.25  # 0.25 * e / s


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    robot = Entity(get_z01_std_add_head_robot_cfg())

    viewer.launch(robot.spec.compile())
