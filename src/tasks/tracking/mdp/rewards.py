from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.utils.lab_api.math import quat_error_magnitude

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_body_indexes(
  command: MotionCommand, body_names: tuple[str, ...] | None
) -> list[int]:
  return [
    i
    for i, name in enumerate(command.cfg.body_names)
    if (body_names is None) or (name in body_names)
  ]


def motion_global_anchor_position_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.sum(
    torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1
  )
  return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(
  env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
  return torch.exp(-error / std**2)


def _build_body_axis_weight(
    axis_weights,
    body_axis_weights,
    tracked_names: list[str],
    device,
    dtype,
) -> torch.Tensor:
    """构建形状为 (M, 3) 的逐轴权重，M = 被追踪 body 的数量。

    设计目标：在不破坏现有行为的前提下，支持"全局逐轴加权 + 局部 body 覆盖"。

    - axis_weights 为 None            -> 全局退化为 (1, 1, 1)，即所有 body 的 x/y/z 等权（与原实现逐位一致）
    - axis_weights 为 (wx, wy, wz)    -> 全局应用于所有被追踪 body
    - body_axis_weights 为 dict       -> 仅对 dict 中列出的 body 逐 body 覆盖其 (wx, wy, wz)

    返回结果会按 body 顺序排布，便于直接对 (N, M, 3) 的位置误差做广播乘法。
    """
    M = len(tracked_names)
    # 全局基准权重：None 时全 1，否则取传入的三元权重
    if axis_weights is None:
        base = torch.ones(3, device=device, dtype=dtype)
    else:
        base = torch.as_tensor(axis_weights, device=device, dtype=dtype)
    # 由 (3,) 扩展为 (M, 3)，每个 body 一份基准权重
    w = base.unsqueeze(0).expand(M, 3).clone()
    # 局部覆盖：仅改写 dict 中出现的 body 行
    if body_axis_weights:
        for i, name in enumerate(tracked_names):
            if name in body_axis_weights:
                w[i] = torch.as_tensor(body_axis_weights[name], device=device, dtype=dtype)
    return w


def motion_relative_body_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
  axis_weights=None,
  body_axis_weights=None,
) -> torch.Tensor:
  """运动相对 body 位置误差的指数奖励（支持逐轴加权）。

  参数:
    axis_weights      全局三轴权重 (wx, wy, wz)，默认 None 表示等权 (1,1,1)
    body_axis_weights 局部覆盖字典 {body_name: (wx, wy, wz)}，默认 None 表示不覆盖

  不传 axis_weights / body_axis_weights 时，行为与原始实现逐位一致（全 1 等权）。
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  tracked_names = [command.cfg.body_names[i] for i in body_indexes]
  # 位置误差张量，形状 (N, M, 3)
  diff = (
    command.body_pos_relative_w[:, body_indexes]
    - command.robot_body_pos_w[:, body_indexes]
  )
  # 构建 (M, 3) 逐轴权重并对每个 body 的 x/y/z 分别加权
  aw = _build_body_axis_weight(
    axis_weights, body_axis_weights, tracked_names, diff.device, diff.dtype
  )
  diff = diff * aw.unsqueeze(0)
  error = torch.sum(torch.square(diff), dim=-1)
  return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = (
    quat_error_magnitude(
      command.body_quat_relative_w[:, body_indexes],
      command.robot_body_quat_w[:, body_indexes],
    )
    ** 2
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_lin_vel_w[:, body_indexes]
      - command.robot_body_lin_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  body_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  body_indexes = _get_body_indexes(command, body_names)
  error = torch.sum(
    torch.square(
      command.body_ang_vel_w[:, body_indexes]
      - command.robot_body_ang_vel_w[:, body_indexes]
    ),
    dim=-1,
  )
  return torch.exp(-error.mean(-1) / std**2)


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


class neg_power:
  """Penalize negative electrical power consumption of actuators.

  负机械功率惩罚（再生制动）：tau*qd < -threshold 时按平方惩罚，
  挂载于 env_cfgs 的 ``cfg.rewards["neg_power"]``（weight=0 默认关闭）。
  """

  def __init__(
    self,
    cfg: RewardTermCfg,
    env: ManagerBasedRlEnv,
  ):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]

    joint_ids, _ = asset.find_joints(
      cfg.params["asset_cfg"].joint_names,
    )
    actuator_ids, _ = asset.find_actuators(
      cfg.params["asset_cfg"].joint_names,
    )
    self._joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
    self._actuator_ids = torch.tensor(actuator_ids, device=env.device, dtype=torch.long)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    threshold: float,
    norm: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    tau = asset.data.actuator_force[:, self._actuator_ids]
    qd = asset.data.joint_vel[:, self._joint_ids]
    mech = tau * qd
    mech_pos = torch.clamp(-mech - threshold, min=0.0) / norm
    return torch.sum(torch.square(mech_pos), dim=1)
