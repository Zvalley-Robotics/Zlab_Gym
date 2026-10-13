# 本仓库 velocity 任务的本地 mdp 扩展（mjlab 上游未提供 feet_lateral_distance）。
# 用于把自定义奖励/观测项挂到 make_velocity_env_cfg 生成的配置上。
from mjlab.envs.mdp import * 
from .rewards import *