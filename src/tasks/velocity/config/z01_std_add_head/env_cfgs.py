"""Z01_std_add_head velocity environment configurations.

Modelled on the historical ``zoomlion_t4_27dof_rough_env_cfg`` velocity task
(domain randomization, twist-velocity curriculum, foot-based gait shaping), but
rebuilt on top of mjlab's own velocity base
(``mjlab.tasks.velocity.velocity_env_cfg.make_velocity_env_cfg``) so that this
task only depends on mjlab plus the Z01 robot assets shipped in this repo.
"""
from dataclasses import replace

from mjlab.actuator import BuiltinPositionActuatorCfg, DelayedActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
import src.tasks.velocity.mdp as src_mdp

from src.assets.robots import (
    Z01_STD_ADD_HEAD_ACTION_SCALE,
    get_z01_std_add_head_robot_cfg,
)
from src.assets.robots.Z01_std_add_head import (
    z01_std_add_head_constants as z01c,
)

# Z01_std_add_head is rooted at ``torso_link`` (there is no ``pelvis`` link).
_ROOT_BODY_NAME = "torso_link"
_FOOT_SITE_NAMES = ("left_foot", "right_foot")
_FOOT_GEOM_NAMES = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
)

##
# z01_velocity_flat_base.yaml (run profile) -- deployment-side default joint
# positions and PD gains, applied inside ``z01_std_add_head_flat_env_cfg`` only.
# ``Z01_ACTUATOR_*`` / ``HOME_KEYFRAME`` are module-level singletons shared with
# the tracking task, so they are copied (via ``dataclasses.replace``) instead of
# being mutated in place.
##

#: Default joint positions of the run profile (deployment adapter
#: ``z01_velocity_flat_base.py`` subscribes to the same table).
_YAML_RUN_JOINT_POS: dict[str, float] = {
    # Head.
    "head_yaw": 0.0,
    "head_pitch": 0.0,
    # Arms.
    "left_shoulder_pitch": 0.35,
    "right_shoulder_pitch": 0.35,
    "left_shoulder_roll": 0.18,
    "right_shoulder_roll": -0.18,
    "left_shoulder_yaw": 0.0,
    "right_shoulder_yaw": 0.0,
    "left_elbow": -0.7,
    "right_elbow": -0.7,
    "left_wrist_yaw": 0.0,
    "right_wrist_yaw": 0.0,
    "left_wrist_pitch": 0.0,
    "right_wrist_pitch": 0.0,
    "left_wrist_roll": 0.0,
    "right_wrist_roll": 0.0,
    # Waist.
    "waist_yaw": 0.0,
    # Legs.
    "left_hip_pitch": -0.1,
    "right_hip_pitch": -0.1,
    "left_hip_roll": 0.0,
    "right_hip_roll": 0.0,
    "left_hip_yaw": 0.0,
    "right_hip_yaw": 0.0,
    "left_knee": 0.3,
    "right_knee": 0.3,
    "left_ankle_pitch": -0.2,
    "right_ankle_pitch": -0.2,
    "left_ankle_roll": 0.0,
    "right_ankle_roll": 0.0,
}

#: (target_names_expr, kp, kd, motor) of the run profile. The 6 groups do not
#: overlap and together cover all 29 joints: head 2 + arms 14 + waist/hip_roll/
#: hip_yaw 5 + hip_pitch/knee 4 + ankle_pitch 2 + ankle_roll 2 = 29.
_YAML_PD_GROUPS = (
    (
        ("head_pitch", "head_yaw", ".*_wrist_pitch", ".*_wrist_roll"),
        20.0,
        1.0,
        z01c.MOTOR_WRIST,
    ),
    (
        (
            ".*_shoulder_pitch",
            ".*_shoulder_roll",
            ".*_shoulder_yaw",
            ".*_elbow",
            ".*_wrist_yaw",
        ),
        40.0,
        2.0,
        z01c.MOTOR_ARM,
    ),
    (("waist_yaw", ".*_hip_roll", ".*_hip_yaw"), 50.0, 2.0, z01c.MOTOR_LEG),
    ((".*_hip_pitch", ".*_knee"), 100.0, 4.0, z01c.MOTOR_LEG),
    ((".*_ankle_pitch",), 40.0, 2.0, z01c.MOTOR_ANKLE_PITCH),
    ((".*_ankle_roll",), 40.0, 2.0, z01c.MOTOR_ANKLE_ROLL),
)

#: Root spawn height with the run-profile pose. FK measured: the torso sits
#: 0.8330 above the lowest sole capsule (the foot is level because
#: ankle_pitch == -(hip_pitch + knee)), so 0.84 leaves the sole ~7 mm above the
#: ground -- the same clearance the rough task uses (0.85 for knee=0.5).
_STANDING_BASE_Z = 0.84

#: FK measured ``torso_link`` to foot-site vertical distance of the run-profile
#: pose (foot site is 0.0245 above the sole). Used as the ``base_height`` target.
_STANDING_HEIGHT = 0.823

#: Left/right limb bodies used by the symmetry rewards (1:1 by index).
_ARM_LIMB_BODIES = ("shoulder_roll_link", "elbow_link", "wrist_roll_link")
_LEG_LIMB_BODIES = ("hip_roll_link", "knee_link", "ankle_roll_link")


def z01_std_add_head_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Z01_std_add_head rough terrain velocity configuration."""
    cfg = make_velocity_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 48

    # Z01 的 HOME 姿态继承自 T4 —— 几乎锁膝（knee=0.14）。官方 G1 用的是
    # KNEES_BENT_KEYFRAME（hip_pitch=-0.312, knee=0.669）。锁膝时摆腿只能靠髋，
    # 抬脚高度 = L(1-cos θ) ≈ 2.7 cm，实测 2.8 cm，正是"蹭地碎步"的来源。
    # 该姿态同时决定 pose 奖励目标、reset 姿态和动作默认偏置，所以只改这一处。
    # 只覆盖 velocity 任务，不动 constants.py 里 tracking 共用的 HOME_KEYFRAME。
    robot_cfg = get_z01_std_add_head_robot_cfg()
    robot_cfg.init_state = EntityCfg.InitialStateCfg(
        # 膝弯 0.5 rad 使腿缩短约 1.9 cm，高度从 0.85 下移到 0.83，
        # 保持脚底离地 5~10 mm（落地后我会用 FK 复核）
        pos=(0.0, 0.0, 0.83),
        joint_pos={
            **robot_cfg.init_state.joint_pos,
            "left_hip_pitch": -0.25,  "right_hip_pitch": -0.25,
            "left_knee": 0.50,        "right_knee": 0.50,
            # 脚掌保持贴地：ankle = -(hip + knee)
            "left_ankle_pitch": -0.25, "right_ankle_pitch": -0.25,
        },
        joint_vel={".*": 0.0},
    )
    cfg.scene.entities = {"robot": robot_cfg}

    # Set the raycast sensor frame to the Z01 root body.
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            sensor.frame.name = _ROOT_BODY_NAME
            sensor.include_geom_groups = (0,)  # Terrain only.

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="subtree",
            pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(
            mode="subtree", pattern=_ROOT_BODY_NAME, entity="robot"
        ),
        secondary=ContactMatch(
            mode="subtree", pattern=_ROOT_BODY_NAME, entity="robot"
        ),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        self_collision_cfg,
    )

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = True

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = Z01_STD_ADD_HEAD_ACTION_SCALE

    cfg.viewer.body_name = _ROOT_BODY_NAME
    actor = cfg.observations["actor"]
    actor.terms["base_lin_vel"].noise = Unoise(n_min=-1.0, n_max=1.0)      # was ±0.5
    actor.terms["base_ang_vel"].noise = Unoise(n_min=-0.4, n_max=0.4)      # was ±0.2
    actor.terms["projected_gravity"].noise = Unoise(n_min=-0.1, n_max=0.1) # was ±0.05
    actor.terms["joint_pos"].noise = Unoise(n_min=-0.02, n_max=0.02)       # was ±0.01
    actor.terms["joint_vel"].noise = Unoise(n_min=-3.0, n_max=3.0)         # was ±1.5

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.rel_standing_envs = 0.1
    twist_cmd.rel_heading_envs = 0.25
    twist_cmd.ranges.lin_vel_x = (-1.0, 2.0)
    twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
    twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)
    twist_cmd.viz.z_offset = 1.15

    cfg.observations["critic"].terms["foot_height"].params[
        "asset_cfg"
    ].site_names = _FOOT_SITE_NAMES

    # --- Events -----------------------------------------------------------
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = _FOOT_GEOM_NAMES
    cfg.events["foot_friction"].params["ranges"] = (0.2, 1.4)          # was (0.3, 1.2)
    cfg.events["base_com"].params["asset_cfg"].body_names = (_ROOT_BODY_NAME,)
    cfg.events["base_com"].params["ranges"] = {                        # was (±0.025, ±0.025, ±0.03)
        0: (-0.05, 0.05),
        1: (-0.05, 0.05),
        2: (-0.05, 0.05),
    }
    cfg.events["encoder_bias"].params["bias_range"] = (-0.03, 0.03)    # was (-0.015, 0.015)
    cfg.events["reset_base"].params["velocity_range"] = {
        # "x": (-0.5, 0.5),
        # "y": (-0.5, 0.5),
        # "z": (-0.5, 0.5),
        # "roll": (-0.5, 0.5),
        # "pitch": (-0.5, 0.5),
        # "yaw": (-0.5, 0.5),
        "x": (-1.0, 1.0), "y": (-1.0, 1.0), "z": (-1.0, 1.0),
        "roll": (-1.0, 1.0), "pitch": (-1.0, 1.0), "yaw": (-1.0, 1.0),        
    }
    cfg.events["reset_robot_joints"].params["position_range"] = (-0.7, 0.7)
    cfg.events["reset_robot_joints"].params["velocity_range"] = (-1.0, 1.0)
    # Stronger, more frequent push perturbations.
    cfg.events["push_robot"].interval_range_s = (0.8, 2.5)            # was (1.0, 3.0)
    cfg.events["push_robot"].params["velocity_range"] = {
        "x": (-0.8, 0.8), "y": (-0.8, 0.8), "z": (-0.5, 0.5),
        "roll": (-0.6, 0.6), "pitch": (-0.6, 0.6), "yaw": (-1.0, 1.0),
    }
    cfg.events["base_mass"] = EventTermCfg(
        mode="startup",
        func=dr.body_mass,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=(_ROOT_BODY_NAME,)),
            "operation": "add",
            "ranges": (-8.0, 8.0),
        },
    )
    cfg.events["link_mass"] = EventTermCfg(
        mode="startup",
        func=dr.pseudo_inertia,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=("left_.*_link", "right_.*_link")
            ),
            "alpha_range": (-0.2, 0.2),
        },
    )
    cfg.events["joint_damping"] = EventTermCfg(
        mode="startup",
        func=dr.joint_damping,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "operation": "add",
            "ranges": (0.05, 0.30),
        },
    )
    cfg.events["joint_armature"] = EventTermCfg(
        mode="startup",
        func=dr.joint_armature,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "operation": "scale",
            "ranges": (0.8, 1.2),
        },
    )
    cfg.events["joint_friction"] = EventTermCfg(
        mode="startup",
        func=dr.joint_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "operation": "add",
            "ranges": (0.05, 0.30),
        },
    )
    cfg.events["joint_offset"] = EventTermCfg(
        mode="startup",
        func=dr.joint_default_pos,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "operation": "add",
            "ranges": (-0.02, 0.02),
        },
    )
    cfg.events["pd_gains"] = EventTermCfg(
        mode="startup",
        func=dr.pd_gains,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "operation": "scale",
            "kp_range": (0.7, 1.3),
            "kd_range": (0.7, 1.3),
        },
    )

    # --- Rewards ----------------------------------------------------------
    # Rationale for std values:
    # - Knees/hip_pitch get the loosest std to allow natural leg bending during stride.
    # - Hip roll/yaw stay tighter to prevent excessive lateral sway and keep gait stable.
    # - Ankle roll is very tight for balance; ankle pitch looser for foot clearance.
    # - Waist yaw stays tight to keep the torso upright and stable.
    # - Shoulders/elbows get moderate freedom for natural arm swing during walking.
    # - Wrists are loose (0.3) since they don't affect balance much.
    # - Head stays tight so the added head joints do not drift.
    # Running values are ~1.5-2x walking values to accommodate larger motion range.
    cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
    cfg.rewards["pose"].params["std_walking"] = {
        # Lower body.
        r".*hip_pitch.*": 0.3,
        r".*hip_roll.*": 0.15,
        r".*hip_yaw.*": 0.15,
        r".*knee.*": 0.35,
        r".*ankle_pitch.*": 0.25,
        r".*ankle_roll.*": 0.1,
        # Waist / head.
        r".*waist_yaw.*": 0.2,
        r".*head.*": 0.1,
        # Arms.
        r".*shoulder_pitch.*": 0.15,
        r".*shoulder_roll.*": 0.15,
        r".*shoulder_yaw.*": 0.1,
        r".*elbow.*": 0.15,
        r".*wrist.*": 0.3,
    }
    cfg.rewards["pose"].params["std_running"] = {
        # Lower body.
        r".*hip_pitch.*": 0.5,
        r".*hip_roll.*": 0.2,
        r".*hip_yaw.*": 0.2,
        r".*knee.*": 0.6,
        r".*ankle_pitch.*": 0.35,
        r".*ankle_roll.*": 0.15,
        # Waist / head.
        r".*waist_yaw.*": 0.3,
        r".*head.*": 0.15,
        # Arms.
        r".*shoulder_pitch.*": 0.5,
        r".*shoulder_roll.*": 0.2,
        r".*shoulder_yaw.*": 0.15,
        r".*elbow.*": 0.35,
        r".*wrist.*": 0.3,
    }

    cfg.rewards["upright"].params["asset_cfg"].body_names = (_ROOT_BODY_NAME,)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (_ROOT_BODY_NAME,)

    for reward_name in ("foot_clearance", "foot_swing_height", "foot_slip"):
        cfg.rewards[reward_name].params["asset_cfg"].site_names = _FOOT_SITE_NAMES

    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["angular_momentum"].weight = -0.02
    cfg.rewards["action_rate_l2"].weight = -0.01
    cfg.rewards["air_time"].weight = 0.0
    cfg.rewards["action_acc_l2"] = RewardTermCfg(
        func=envs_mdp.action_acc_l2, weight=-0.01
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
    )


    cfg.rewards["track_linear_velocity"].weight = 3.5
    cfg.rewards["track_angular_velocity"].weight = 2.5
    # --- Curriculum -------------------------------------------------------
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
        {"step": 0, "lin_vel_x": (-1.0, 1.0), "ang_vel_z": (-0.5, 0.5)},
        {"step": 10000 * 24, "lin_vel_x": (-1.5, 1.5), "lin_vel_y": (-1.5, 1.5), "ang_vel_z": (-1.0, 1.0)},
        {"step": 30000 * 24, "lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
    ]

    # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )

        if cfg.scene.terrain is not None:
            if cfg.scene.terrain.terrain_generator is not None:
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg


def _yaml_delayed_actuator(
    expr: tuple[str, ...],
    kp: float,
    kd: float,
    motor: "z01c.ElectricActuator",
) -> DelayedActuatorCfg:
    """Build a frozen ``DelayedActuatorCfg`` with the yaml run-profile kp/kd.

    ``effort_limit`` / ``armature`` are reused from the shared motor constants so
    only the PD gains differ from ``z01_std_add_head_constants.py``.
    """
    return DelayedActuatorCfg(
        base_cfg=BuiltinPositionActuatorCfg(
            target_names_expr=tuple(expr),
            stiffness=kp,
            damping=kd,
            effort_limit=motor.effort_limit,
            armature=motor.reflected_inertia,
        ),
        delay_target="position",
        delay_min_lag=z01c.DELAY_MIN_LAG,
        delay_max_lag=z01c.DELAY_MAX_LAG,
    )


def _apply_yaml_run_profile(robot_cfg: EntityCfg) -> None:
    """Overwrite default joint positions and PD gains with the yaml run profile.

    Only the flat velocity task calls this: the copies are installed on
    ``robot_cfg`` (which is per-task) while ``HOME_KEYFRAME`` and the
    ``Z01_ACTUATOR_*`` singletons stay untouched, so the tracking task keeps its
    original default pose and gains. The action scale is not touched either: it
    is still 0.25 per joint (``Z01_STD_ADD_HEAD_ACTION_SCALE``), matching the
    ``_ACTION_SCALE`` of the deployment adapter.
    """
    prev = robot_cfg.init_state
    robot_cfg.init_state = EntityCfg.InitialStateCfg(
        pos=(0.0, 0.0, _STANDING_BASE_Z),
        rot=prev.rot,
        lin_vel=prev.lin_vel,
        ang_vel=prev.ang_vel,
        joint_pos={**(prev.joint_pos or {}), **_YAML_RUN_JOINT_POS},
        joint_vel={".*": 0.0},
    )

    articulation = robot_cfg.articulation
    assert isinstance(articulation, EntityArticulationInfoCfg)
    robot_cfg.articulation = replace(
        articulation,
        actuators=tuple(_yaml_delayed_actuator(*group) for group in _YAML_PD_GROUPS),
    )


def _retune_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
    """Reward rework of the flat task.

    1. Trim the redundant foot-shaping terms (only velocity tracking, standing
       height and stability/gait terms are kept);
    2. raise the stability weights and add the standing-height plus the
       left/right limb symmetry rewards.
    """
    # --- (1) Drop redundant foot shaping ---------------------------------
    for name in ("foot_clearance", "foot_swing_height", "soft_landing"):
        cfg.rewards.pop(name, None)

    # --- Velocity tracking ------------------------------------------------
    cfg.rewards["track_linear_velocity"].weight = 3.5
    cfg.rewards["track_linear_velocity"].params["std"] = 0.3   # 默认 sqrt(0.25)=0.5

    cfg.rewards["track_angular_velocity"].weight = 2.5

    # --- (2) Standing height (new) ---------------------------------------
    cfg.rewards["base_height"] = RewardTermCfg(
        func=src_mdp.base_height,
        weight=0.5,
        params={
            "sensor_name": "feet_ground_contact",
            "std": 0.05,
            "target_height": _STANDING_HEIGHT,
            # Active for standing and walking alike (no command gating).
            "command_name": None,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=(_ROOT_BODY_NAME,),
                site_names=_FOOT_SITE_NAMES,
            ),
        },
    )

    # --- (2) Posture / stability: raise the weights -----------------------
    cfg.rewards["upright"].weight = 2.0
    cfg.rewards["body_ang_vel"].weight = -0.2
    cfg.rewards["angular_momentum"].weight = -0.05
    cfg.rewards["action_rate_l2"].weight = -0.02
    cfg.rewards["action_acc_l2"].weight = -0.02

    # --- Gait stability ---------------------------------------------------
    # NOTE: this task builds on mjlab's upstream velocity base, so the air-time
    # term is keyed ``air_time``; the ``foot_gait`` key of this repo's own
    # ``src/tasks/velocity`` package does not exist upstream (KeyError).
    # cfg.rewards["air_time"].weight = 0.4
    # cfg.rewards["air_time"].params["threshold_min"] = 0.35
    # cfg.rewards["air_time"].params["threshold_max"] = 0.7
    # --- Gait stability: periodic gait-clock reward (replaces air_time) ----
    # ``air_time`` only shapes the swing duration; ``foot_gait`` instead aligns
    # each foot's contact with a global gait phase for a cleaner alternation.
    # Keep the upstream term but switch it off.
    cfg.rewards["air_time"].weight = 0.0
    cfg.rewards["foot_gait"] = RewardTermCfg(
        func=src_mdp.feet_gait,
        weight=0.8,
        params={
            "period": 0.6,          # gait cycle (s)
            "offset": [0.0, 0.5],   # two legs, half a cycle apart
            "threshold": 0.56,      # fraction of the cycle spent in stance
            "command_threshold": 0.1,
            "command_name": "twist",
            "sensor_name": "feet_ground_contact",
        },
    )

    cfg.rewards["foot_slip"].weight = -0.15
    cfg.rewards["feet_lateral_distance"] = RewardTermCfg(
        func=src_mdp.feet_lateral_distance,
        weight=-0.5,
        params={
            "min_distance": 0.16,
            "command_name": "twist",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=(_ROOT_BODY_NAME,),
                site_names=_FOOT_SITE_NAMES,
            ),
        },
    )

    # --- (2) Left/right limb symmetry (new) --------------------------------
    # Arms: the vertical distance to the torso xy-plane must match on both sides.
    cfg.rewards["arm_symmetry"] = RewardTermCfg(
        func=src_mdp.limb_symmetry,
        weight=0.5,
        params={
            "std": 0.05,
            "root_body_name": _ROOT_BODY_NAME,
            "left_body_names": tuple(f"left_{n}" for n in _ARM_LIMB_BODIES),
            "right_body_names": tuple(f"right_{n}" for n in _ARM_LIMB_BODIES),
            "mode": "z",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    # Legs: mirror symmetry about the torso sagittal plane. Using "z" here would
    # penalize the leg lift the gait needs; switch to "z" only if a strictly
    # height-matched stance is wanted.
    cfg.rewards["leg_symmetry"] = RewardTermCfg(
        func=src_mdp.limb_symmetry,
        weight=0.3,
        params={
            "std": 0.05,
            "root_body_name": _ROOT_BODY_NAME,
            "left_body_names": tuple(f"left_{n}" for n in _LEG_LIMB_BODIES),
            "right_body_names": tuple(f"right_{n}" for n in _LEG_LIMB_BODIES),
            "mode": "y",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


def z01_std_add_head_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Z01_std_add_head flat terrain velocity configuration.

    Differences from the rough version:
      1. reward trimming (velocity tracking / standing height / stability only);
      2. higher stability weights plus new left/right limb symmetry rewards;
      3. default joint positions and PD gains aligned with the run profile of
         ``z01_velocity_flat_base.yaml``.
    """
    cfg = z01_std_add_head_rough_env_cfg(play=play)

    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None

    # Switch to flat terrain.
    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # Remove raycast sensor and height scan (no terrain to scan).
    cfg.scene.sensors = tuple(
        s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
    )
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    # --- (3) Default joint positions + PD gains (yaml run profile) ---------
    robot_cfg = cfg.scene.entities["robot"]
    assert isinstance(robot_cfg, EntityCfg)
    _apply_yaml_run_profile(robot_cfg)

    # --- (1)(2) Reward trimming / stability weights / new height+symmetry --
    _retune_rewards(cfg)

    # Disable terrain curriculum (not present in play mode since rough clears all).
    cfg.curriculum.pop("terrain_levels", None)

    if play:
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
        twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

    return cfg
