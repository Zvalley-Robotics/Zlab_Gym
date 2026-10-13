"""Z01_std_add_head flat tracking environment configuration."""

from src.assets.robots import (
    Z01_STD_ADD_HEAD_ACTION_SCALE,
    get_z01_std_add_head_robot_cfg,
)
from mjlab.envs import mdp as envs_mdp
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
import src.tasks.tracking.mdp as src_mdp
from src.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg


def z01_std_add_head_flat_tracking_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the Z01_std_add_head flat-terrain motion-tracking environment.

    Single base task for secondary development. The robot uses the standard
    ``DelayedActuatorCfg(base_cfg=BuiltinPositionActuatorCfg(...))`` actuator
    combination defined in
    ``src/assets/robots/Z01_std_add_head/z01_std_add_head_constants.py``;
    the action scale is derived automatically from ``effort_limit``/``stiffness``
    (``Z01_STD_ADD_HEAD_ACTION_SCALE``), no manual tuning required.
    """
    cfg = make_tracking_env_cfg()

    cfg.scene.entities = {"robot": get_z01_std_add_head_robot_cfg()}

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(
            mode="subtree", pattern="torso_link", entity="robot"),
        secondary=ContactMatch(
            mode="subtree", pattern="torso_link", entity="robot"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (self_collision_cfg,)

    # ------------------------------------------------------------------ #
    # Head/neck is ignored by the policy: the clips carry no neck data, so
    # the two neck joints are excluded from both the action space and the
    # joint observations. They stay in the model and are held at their zero
    # reference by the passive spring in "head_joint_lock" below (no action
    # channel is left that could command them).
    # ------------------------------------------------------------------ #
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    # "(?!head).*" keeps every joint but the two neck ones -- the same regex the
    # reset randomization below already uses.
    joint_pos_action.actuator_names = ("(?!head).*",)
    # BaseAction resolves the scale dict with resolve_matching_names_values(),
    # which raises on keys that match no actuator -- so the two head entries have
    # to be dropped together with the actuators themselves.
    joint_pos_action.scale = {
        k: v
        for k, v in Z01_STD_ADD_HEAD_ACTION_SCALE.items()
        if k not in ("head_pitch", "head_yaw")
    }
    # Drop the two neck slots from the joint observations as well (actor and
    # critic), otherwise the policy would still receive a 29-dim joint state.
    for group_name in ("actor", "critic"):
        for term_name in ("joint_pos", "joint_vel"):
            cfg.observations[group_name].terms[term_name].params["asset_cfg"] = (
                SceneEntityCfg("robot", joint_names=("(?!head).*",))
            )

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.anchor_body_name = "torso_link"
    # 训练期也关掉 RSI 随机化（原来只在 play 里关）。reset 瞬间给躯干灌
    # ±0.5 m/s / ±0.52 rad/s 随机速度，对"原地比心"是必死项。
    # 照抄参考仓库对 demo_act 类动作的处理。
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (-0.03, 0.03)
    motion_cmd.sampling_mode = "start"    
    motion_cmd.body_names = (
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_roll_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_roll_link",
        "waist_yaw_link",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
    )

    cfg.events["foot_friction"].params[
        "asset_cfg"
    ].geom_names = r"^(left|right)_foot[1-7]_collision$"
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
    cfg.events["reset_robot_joints"] = EventTermCfg(
        func=envs_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.5, 0.5),
            "velocity_range": (-0.5, 0.5),
            # Head/neck are locked at their zero reference, so they are excluded
            # from the reset randomization (everything not starting with "head").
            "asset_cfg": SceneEntityCfg("robot", joint_names=("(?!head).*",)),
        },
    )
    cfg.events["base_mass"] = EventTermCfg(
        mode="startup",
        func=dr.body_mass,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
            "operation": "add",
            "ranges": (-5.0, 5.0),
        },
    )
    cfg.events["link_mass"] = EventTermCfg(
        mode="startup",
        func=dr.pseudo_inertia,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=("left_.*_link", "right_.*_link")),
            "alpha_range": (-0.1, 0.1),
        },
    )
    cfg.events["joint_damping"] = EventTermCfg(
        mode="startup",
        func=dr.joint_damping,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "operation": "add",
            "ranges": (0.05, 0.15),
        },
    )
    # Head/neck are not animated by the motion clips (their reference is a
    # constant 0) and are excluded from the action space above, so no action
    # channel ever writes a target for them: this passive spring
    # (-stiffness * (q - qpos0), qpos0 = 0) plus the damping below is the *only*
    # thing holding them at zero. Both are applied *after* "joint_damping" above,
    # which uses operation="add"; "abs" here therefore wins.
    cfg.events["head_joint_lock"] = EventTermCfg(
        mode="startup",
        func=dr.jnt_stiffness,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=("head_pitch", "head_yaw")),
            "operation": "abs",
            "ranges": (100.0, 100.0),
        },
    )
    cfg.events["head_joint_damping"] = EventTermCfg(
        mode="startup",
        func=dr.joint_damping,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=("head_pitch", "head_yaw")),
            "operation": "abs",
            "ranges": (10.0, 10.0),
        },
    )
    cfg.events["joint_armature"] = EventTermCfg(
        mode="startup",
        func=dr.joint_armature,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "operation": "scale",
            "ranges": (0.9, 1.1),
        },
    )
    cfg.events["joint_friction"] = EventTermCfg(
        mode="startup",
        func=dr.joint_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "operation": "add",
            "ranges": (0.05, 0.15),
        },
    )
    cfg.events["joint_offset"] = EventTermCfg(
        mode="startup",
        func=dr.joint_default_pos,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "operation": "add",
            "ranges": (-0.01, 0.01),
        },
    )
    cfg.events["pd_gains"] = EventTermCfg(
        mode="startup",
        func=dr.pd_gains,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            "operation": "scale",
            "kp_range": (0.8, 1.2),
            "kd_range": (0.8, 1.2),
        },
    )

    cfg.rewards["neg_power"] = RewardTermCfg(
        func=src_mdp.neg_power,
        weight=0,
        params={
            "threshold": 150.0,
            "norm": 500.0,
            "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_knee",)),
        },
    )
    cfg.rewards["action_acc_l2"] = RewardTermCfg(
        func=envs_mdp.action_acc_l2, weight=-1e-1)
    cfg.rewards["motion_body_pos"].params["body_axis_weights"] = {
        "left_shoulder_roll_link":  (2.0, 2.0, 2.0),
        "left_elbow_link":          (2.0, 2.0, 2.0),
        "left_wrist_roll_link":     (2.0, 2.0, 2.0),
        "right_shoulder_roll_link": (2.0, 2.0, 2.0),
        "right_elbow_link":         (2.0, 2.0, 2.0),
        "right_wrist_roll_link":    (2.0, 2.0, 2.0),
    }

    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "left_wrist_roll_link",
        "right_wrist_roll_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    )

    # 动作 0 = 默认姿态（手垂下/直腿），参考是"手抬到胸前"。策略输出 ~0 时
    # PD 在 ~0.1 s 内把手脚拉到默认姿态，手腕相对躯干偏 0.3~0.5 m →
    # 0.25 m 阈值下第 8 步必死，且死亡率恒定（所以 2122 轮都没有 timeout）。
    # 0.5 m 是 BeyondMimic 一系的常用值，足够放过冷启动瞬态。
    cfg.terminations["ee_body_pos"].params["threshold"] = 0.5
    cfg.terminations["anchor_pos"].params["threshold"] = 0.5
    cfg.viewer.body_name = "torso_link"

    # Apply play mode overrides.
    if play:
        # Effectively infinite episode length.
        cfg.episode_length_s = int(1e9)

        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)

        # Disable RSI randomization.
        motion_cmd.pose_range = {}
        motion_cmd.velocity_range = {}

        motion_cmd.sampling_mode = "start"

    return cfg
