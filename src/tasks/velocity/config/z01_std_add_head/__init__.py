from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
    z01_std_add_head_flat_env_cfg,
    z01_std_add_head_rough_env_cfg,
)
from .rl_cfg import z01_std_add_head_velocity_ppo_runner_cfg

register_mjlab_task(
    task_id="Z01-StdAddHead-Velocity-Rough",
    env_cfg=z01_std_add_head_rough_env_cfg(),
    play_env_cfg=z01_std_add_head_rough_env_cfg(play=True),
    rl_cfg=z01_std_add_head_velocity_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Z01-StdAddHead-Velocity-Flat",
    env_cfg=z01_std_add_head_flat_env_cfg(),
    play_env_cfg=z01_std_add_head_flat_env_cfg(play=True),
    rl_cfg=z01_std_add_head_velocity_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
