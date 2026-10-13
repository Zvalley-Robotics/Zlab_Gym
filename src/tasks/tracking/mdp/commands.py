"""Motion command that keeps the Z01 neck out of the tracking task.

The bundled clips carry no neck data: ``joint_pos`` / ``joint_vel`` are either
full-length (29 columns, ``head_pitch`` / ``head_yaw`` constant 0) or only carry
the 27 body joints. ``mjlab.tasks.tracking.mdp.MotionCommand`` reads those arrays
*positionally* and assumes ``motion.joint_pos.shape[1] == robot.num_joints``, so a
27-column clip raises ``RuntimeError: tensor a (27) vs b (29)`` inside
``_resample_command`` (RSI / episode resample) and ``_update_metrics``.

This subclass keeps the base behaviour for full-length clips and otherwise
scatters the 27 reference columns into the 27 non-head joint slots, filling the
two head slots with 0 (their locked reference). Either way the ``command``
observation is returned *without* the neck slots, so the policy never sees or
controls the neck.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.tasks.tracking.mdp import MotionCommand, MotionCommandCfg

__all__ = ["HeadAgnosticMotionCommand", "HeadAgnosticMotionCommandCfg"]

HEAD_JOINT_NAMES = ("head_pitch", "head_yaw")


@dataclass(kw_only=True)
class HeadAgnosticMotionCommandCfg(MotionCommandCfg):
  """``MotionCommandCfg`` whose reference covers only the non-head joints."""

  joint_names: tuple[str, ...] = ("(?!head).*",)
  """Robot joints the motion file provides data for (natural joint order)."""

  def build(self, env) -> "HeadAgnosticMotionCommand":
    return HeadAgnosticMotionCommand(self, env)


class HeadAgnosticMotionCommand(MotionCommand):
  cfg: HeadAgnosticMotionCommandCfg

  def __init__(self, cfg: HeadAgnosticMotionCommandCfg, env) -> None:
    super().__init__(cfg, env)
    joint_ids, joint_names = self.robot.find_joints(cfg.joint_names)
    self._tracked_joint_ids = torch.tensor(
      joint_ids, dtype=torch.long, device=self.device
    )
    self._tracked_joint_names = tuple(joint_names)

    n_motion = int(self.motion.joint_pos.shape[1])
    n_robot = int(self.robot.num_joints)
    if n_motion == n_robot:
      self._headless = False  # 29-column clip, the head columns are already 0.
    elif n_motion == len(joint_ids):
      self._headless = True  # 27-column clip, the columns need scattering.
    else:
      raise ValueError(
        f"motion joint columns ({n_motion}) must be either the robot joint "
        f"count ({n_robot}) or the tracked subset ({len(joint_ids)})"
      )

  def _expand(self, ref: torch.Tensor) -> torch.Tensor:
    """(N, len(tracked)) -> (N, num_joints); head slots filled with 0."""
    full = torch.zeros(
      ref.shape[0], self.robot.num_joints, dtype=ref.dtype, device=ref.device
    )
    full[:, self._tracked_joint_ids] = ref
    return full

  # ``_resample_command`` (RSI), ``_update_metrics`` and the ghost debug
  # visualisation all consume the full-length reference.
  @property
  def joint_pos(self) -> torch.Tensor:
    ref = self.motion.joint_pos[self.time_steps]
    return self._expand(ref) if self._headless else ref

  @property
  def joint_vel(self) -> torch.Tensor:
    ref = self.motion.joint_vel[self.time_steps]
    return self._expand(ref) if self._headless else ref

  @property
  def command(self) -> torch.Tensor:
    """Reference (q, qd) handed to the policy, without the neck slots."""
    if not self._headless:
      return super().command
    return torch.cat(
      [self.motion.joint_pos[self.time_steps], self.motion.joint_vel[self.time_steps]],
      dim=1,
    )
