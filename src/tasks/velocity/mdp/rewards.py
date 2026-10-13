from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def feet_lateral_distance(
    env: ManagerBasedRlEnv,
    min_distance: float,
    command_name: str | None = None,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize feet getting too close in the pelvis-yaw lateral direction."""
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
    base_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :].squeeze(1)
    base_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)

    foot_pos_rel_w = foot_pos_w - base_pos_w.unsqueeze(1)
    base_yaw_quat = yaw_quat(base_quat_w).unsqueeze(1).expand(-1, foot_pos_rel_w.shape[1], -1)
    foot_pos_yaw = quat_apply_inverse(base_yaw_quat, foot_pos_rel_w)
    lateral_distance = torch.abs(foot_pos_yaw[:, 0, 1] - foot_pos_yaw[:, 1, 1])

    cost = torch.square(torch.clamp(
        (min_distance - lateral_distance) / min_distance, min=0.0))
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            active = (linear_norm + angular_norm > command_threshold).float()
            cost *= active

    env.extras["log"]["Metrics/feet_lateral_distance_mean"] = torch.mean(lateral_distance)
    return cost


def base_height(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    std: float,
    target_height: float,
    command_name: str | None = None,
    command_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward the standing height of the base.

    The height is measured as the vertical distance between the root body and the
    foot sites that are currently in contact, i.e. ``root_z - foot_site_z``. Being
    a root-to-foot distance, the term is independent of the terrain height.

    ``sensor_name`` must refer to a ``ContactSensor`` created with
    ``track_air_time=True`` (``feet_ground_contact`` in the velocity task).
    """
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    in_contact = contact_sensor.data.current_contact_time > 0
    base_z = asset.data.body_link_pos_w[:, asset_cfg.body_ids, 2].squeeze(1)  # [B]
    foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
    ref_height = base_z.unsqueeze(-1) - foot_z
    height_error = torch.abs(ref_height - target_height) * in_contact.float()
    num_contacts = in_contact.float().sum(dim=-1).clamp(min=1.0)
    mean_height_error = height_error.sum(dim=-1) / num_contacts
    reward = torch.exp(-mean_height_error / std)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            total_command = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
            reward = reward * (total_command > command_threshold).float()

    env.extras["log"]["Metrics/base_height_mean"] = torch.mean(
        (ref_height * in_contact.float()).sum(dim=-1) / num_contacts
    )
    return reward


def limb_symmetry(
    env: ManagerBasedRlEnv,
    std: float,
    left_body_names: tuple[str, ...],
    right_body_names: tuple[str, ...],
    root_body_name: str = "torso_link",
    mode: str = "z",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward left/right limb symmetry, evaluated in the torso body frame.

    The left and right body positions are first expressed in the root (torso)
    frame, then compared:

    * ``mode="z"``: |z_left - z_right|, i.e. the vertical distance to the torso
      xy-plane must match on both sides (used for the arms, which hang at the
      same height when the posture is symmetric);
    * ``mode="y"``: |y_left + y_right|, i.e. mirror symmetry about the torso
      sagittal plane (used for the legs, whose vertical positions necessarily
      alternate while walking -- penalizing their z difference would fight the
      gait);
    * ``mode="both"``: sum of the two.

    ``left_body_names`` / ``right_body_names`` must be pairwise corresponding.
    """
    asset: Entity = env.scene[asset_cfg.name]
    left_ids, left_names = asset.find_bodies(left_body_names, preserve_order=True)
    right_ids, right_names = asset.find_bodies(right_body_names, preserve_order=True)
    root_ids, _ = asset.find_bodies((root_body_name,), preserve_order=True)
    if len(left_ids) != len(right_ids):
        raise ValueError(
            f"limb_symmetry left/right mismatch: {left_names} vs {right_names}"
        )

    root_pos_w = asset.data.body_link_pos_w[:, root_ids, :].squeeze(1)  # [B, 3]
    root_quat_w = asset.data.body_link_quat_w[:, root_ids, :].squeeze(1)  # [B, 4]
    body_pos_w = asset.data.body_link_pos_w[:, left_ids + right_ids, :]  # [B, 2K, 3]

    rel_w = body_pos_w - root_pos_w.unsqueeze(1)
    rel_b = quat_apply_inverse(
        root_quat_w.unsqueeze(1).expand(-1, rel_w.shape[1], -1), rel_w
    )

    n = len(left_ids)
    left_b, right_b = rel_b[:, :n, :], rel_b[:, n:, :]
    error = torch.zeros_like(left_b[:, 0, 0])
    if mode in ("z", "both"):
        error = error + torch.sum(
            torch.square(left_b[..., 2] - right_b[..., 2]), dim=-1
        )
    if mode in ("y", "both"):
        error = error + torch.sum(
            torch.square(left_b[..., 1] + right_b[..., 1]), dim=-1
        )

    env.extras["log"]["Metrics/limb_symmetry_err_mean"] = torch.mean(torch.sqrt(error))
    return torch.exp(-error / (std * std))

def feet_gait(
    env: ManagerBasedRlEnv,
    period: float,
    offset: list[float],
    threshold: float,
    command_threshold: float,
    command_name: str,
    sensor_name: str,
) -> torch.Tensor:
    """Reward feet gait."""
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) /
                    period).unsqueeze(1)
    offsets = torch.as_tensor(
        offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward
