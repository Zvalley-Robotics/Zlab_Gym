from mjlab.tasks.registry import register_mjlab_task
from src.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import z01_std_add_head_flat_tracking_env_cfg
from .rl_cfg import z01_std_add_head_tracking_ppo_runner_cfg

register_mjlab_task(
    task_id="Z01-StdAddHead-Tracking",
    env_cfg=z01_std_add_head_flat_tracking_env_cfg(),
    play_env_cfg=z01_std_add_head_flat_tracking_env_cfg(play=True),
    rl_cfg=z01_std_add_head_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)
