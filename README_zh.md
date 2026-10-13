# Zlab_Gym — 针对Z01机器人的动作模仿 & 基础行走训练框架，让使用者可以通过改框架训练自己所需的运动模型

基于 [mjlab](https://github.com/mujocolab/mjlab)（`mjlab==1.2.0`）+ MuJoCo 的精简二次开发框架。
统一注册**一个机器人**（`Z01_std_add_head`，29 个自由度，策略实际控制 27 个）、
一个动作模仿任务与两个速度任务：

| 任务 id | 任务组 | 说明 |
|---|---|---|
| `Z01-StdAddHead-Tracking` | `tracking` | 全身动作模仿（平地） |
| `Z01-StdAddHead-Velocity-Rough` | `velocity` | 盲走速度跟踪，崎岖地形 + 地形课程 |
| `Z01-StdAddHead-Velocity-Flat` | `velocity` | 盲走速度跟踪，平地 |

---

## 1. 安装

```bash
conda create -n zlab_tracking python=3.11 -y
conda activate zlab_tracking

# 先按你的 CUDA 版本安装 torch，例如：
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

pip install -e .        # 自动安装 mjlab==1.2.0
```

`scripts/train.py` / `scripts/play.py` 会把仓库 `src/` 提前插入 `sys.path`，
以保证**内置补丁版 `rsl_rl`**（`src/rsl_rl/`）优先于 site-packages 中的旧版。
请勿删除这两个脚本顶部的 `sys.path` 逻辑。

## 2. 查看可用任务

```bash
python scripts/list_envs.py                  # 列出所有已注册任务
python scripts/list_envs.py --keyword Z01    # 按关键字过滤
```

## 3. 快速开始

### 3.1 动作模仿任务 —— `--motion-file` 必填

```bash
# 训练
python scripts/train.py Z01-StdAddHead-Tracking \
  --motion-file=src/assets/motions/z01_std_add_head/demo_act/z01_std_base_heart.npz \
  --env.scene.num-envs=256 \
  --agent.experiment-name=z01_std_add_head_tracking
# 日志 -> logs/rsl_rl/z01_std_add_head_tracking/<时间戳>/（每 500 次迭代存 model_<iter>.pt）

# 回放
python scripts/play.py Z01-StdAddHead-Tracking \
  --checkpoint-file=logs/rsl_rl/z01_std_add_head_tracking/<RUN_DIR>/model_0.pt \
  --motion-file=src/assets/motions/z01_std_add_head/demo_act/z01_std_base_heart.npz \
  --num-envs 1 --device cpu --viewer native
```

### 3.2 速度任务 —— 无需动作文件

```bash
# 训练（崎岖地形，开启地形等级课程）
python scripts/train.py Z01-StdAddHead-Velocity-Rough \
  --env.scene.num-envs=4096 \
  --agent.experiment-name=z01_std_add_head_velocity

# 训练（平地）
python scripts/train.py Z01-StdAddHead-Velocity-Flat \
  --env.scene.num-envs=4096 \
  --agent.experiment-name=z01_std_add_head_velocity

# 回放
python scripts/play.py Z01-StdAddHead-Velocity-Rough \
  --checkpoint-file=logs/rsl_rl/z01_std_add_head_velocity/<RUN_DIR>/model_0.pt \
  --num-envs 1 --device cpu --viewer native
```

## 4. 常用参数速查

| 参数 | 适用 | 说明 |
|---|---|---|
| `--motion-file` | train / play（tracking） | 动作 `.npz`，动作模仿任务**必填** |
| `--checkpoint-file` | play | 本地 `.pt` 权重，**必填**（未接入 WandB 下载） |
| `--agent.experiment-name` | train | 日志子目录名 |
| `--agent.logger` | train | 默认 `tensorboard`，可改 `wandb` |
| `--agent.max-iterations` | train | 覆盖训练迭代数 |
| `--env.scene.num-envs` | train | 并行环境数 |
| `--num-envs` | play | 并行环境数 |
| `--device` | play | `cpu` / `cuda:0`，默认优先 `cuda:0` |
| `--gpu-ids` | train | `[0]`、`all` 或显式 id（多卡走 torchrunx） |
| `--video` / `--video-length` | train / play | 录制视频到 `<log_dir>/videos/` |
| `--viewer` | play | `auto` / `native` / `viser`；`auto` 在无 `DISPLAY` 时回落 `viser` |
| `--agent zero\|random\|trained` | play | 空策略调试用 |
| `--no-terminations` | play | 关闭所有终止项（便于完整观察动作） |
| `--log-root` / `--log-dir` / 环境变量 `ZLAB_LOG_ROOT` | train | 日志根目录（默认 `logs`） |
| `--enable-nan-guard` | train | 开启 NaN 守卫 |

## 5. 内置资源

```
src/assets/motions/z01_std_add_head/demo_act/
├── z01_std_base_heart.npz        # demo 动作（比心），供 --motion-file 使用
├── z01_std_base_heart.onnx       # tracking 预训练策略（动作已内嵌）
└── z01_velocity_flat_base.onnx   # flat 速度预训练策略
```

两个 ONNX 是对应训练跑次的导出产物，供下游部署参考。本仓库**未**提供直接播放
ONNX 的脚本，回放请用 `scripts/play.py` 加载 `.pt` 权重。

## 6. ONNX 导出

动作模仿任务的 runner `MotionTrackingOnPolicyRunner` 在每次 `save()` 时，
会在 checkpoint 同目录导出两个模型：

- `<ckpt_stem>_policy.onnx` —— 策略网络（`actor.as_onnx()`）。
- `<ckpt_stem>_motion.onnx` —— `_OnnxMotionModel`：策略 + 参考动作缓冲内嵌为
  ONNX 常量，部署时无需外部数据文件即可回放同一动作片段。

两者均通过 `attach_metadata_to_onnx` 写入 mjlab 元数据。速度任务使用 mjlab 的
`VelocityOnPolicyRunner`（仅导出策略）。

## 7. 目录结构

```
scripts/                                    # train.py / play.py / list_envs.py
tools/                                      # 离线动作格式转换工具
src/
├── assets/
│   ├── robots/Z01_std_add_head/            # 常量 + xmls/（实际加载） + xml/（生成器输出）
│   │                                       #   + urdf/ csv/ meshes/ urdf_to_xml.yaml
│   └── motions/z01_std_add_head/demo_act/  # demo 动作 + ONNX 产物
├── rsl_rl/                                 # 内置补丁版 PPO（请勿改动）
└── tasks/
    ├── __init__.py                         # 注册 "tracking" + "velocity" 任务组
    ├── tracking/
    │   ├── tracking_env_cfg.py             # make_tracking_env_cfg()：公共环境骨架
    │   ├── config/z01_std_add_head/        # env/rl 配置 + 任务注册
    │   ├── rl/runner.py                    # MotionTrackingOnPolicyRunner（含 ONNX 导出）
    │   └── mdp/                            # commands / observations / rewards / terminations
    └── velocity/
        ├── config/z01_std_add_head/        # env/rl 配置 + 任务注册
        └── mdp/rewards.py                  # 本仓库自定义奖励项
```

`tools/`：

- `convert_t4_motion_to_z01.py` —— T4（27 自由度）→ Z01（29 自由度）动作转换，主力工具：
  `python tools/convert_t4_motion_to_z01.py <motion.npz>`（加 `--dry-run` 预览，
  目录参数可批量处理）。
- `convert_pkl_to_50hz_npz.py` —— pkl → 50 Hz npz 动作。
- `convert_npz_to_csv.py` / `csv_to_npz.py` —— **遗留** G1 专用脚本，路径写死，
  Z01 流程不使用。

## 8. 二次开发切入点

1. **新动作**：向 `src/assets/motions/...` 放入同结构 `.npz`，用 `--motion-file` 传入。
2. **新奖励 / 终止项**：在 `src/tasks/<group>/mdp/` 实现，并在该组的 `env_cfgs.py` 挂载。
3. **新机器人**：仿照 `Z01_std_add_head` 复制常量与 `xmls/`、`meshes/`，提供
   `get_<robot>_robot_cfg()` 与 `<ROBOT>_ACTION_SCALE`。
4. **新任务 id**：复制一个 `register_mjlab_task` 块，并把模块路径加入
   `src/tasks/__init__.py` 的 `_TASK_GROUPS`。

## 9. 注意事项

- `src/rsl_rl` 是**补丁版** rsl-rl（熵退火 / std 下限）。不要删除，也不要改用
  site-packages 中的版本。
- 策略只控制 **27 个关节**：`head_pitch` / `head_yaw` 不参与动作与 `joint_pos` /
  `joint_vel` 观测，在 `env_cfgs.py` 中由被动刚度（`jnt_stiffness=100`）+ 阻尼
  （`dof_damping=10`）锁在零位，并由 `HeadAgnosticMotionCommand` 从 command 观测中剔除；
  但 `pose` 奖励的 std 字典仍需覆盖全部 29 个关节。
- `Z01_STD_ADD_HEAD_ACTION_SCALE` 对每个关节**硬编码为 0.25**（并非由
  `effort_limit / stiffness` 自动推导）。确需覆盖时修改
  `z01_std_add_head_constants.py` 中的该常量。
- Z01 的关节名**不带 `_joint` 后缀**（如 `left_knee`、`waist_yaw`、`head_pitch`），
  `target_names_expr` / `joint_names` 正则需按此约定编写。
- 速度任务 **Flat** 与 Rough 不同：删掉了 `foot_clearance` / `foot_swing_height` /
  `soft_landing`，`air_time` 权重置 0，新增 `base_height`(0.5)、`foot_gait`(0.8)、
  `feet_lateral_distance`(-0.5)、`arm_symmetry`(0.5)、`leg_symmetry`(0.3)；
  并额外套用 yaml run profile（`z01_velocity_flat_base`）——固定的默认关节角
  与一整套 PD 增益（`_YAML_PD_GROUPS`，站高 0.823 m）。
- Z01 躯干下方的 body 是 `waist_yaw_link`（没有 `pelvis`）；速度任务统一用根 body
  `torso_link` 作为 `terrain_scan`、`upright`、`body_ang_vel` 与 `self_collision` 的参考 body。
- Z01 MJCF 带 `left_foot` / `right_foot` 两个足底中心 site，供基于足部的奖励项使用。
- `neg_power`（负机械功率 / 再生制动惩罚）挂载于 `cfg.rewards["neg_power"]`，
  默认 `weight=0` 关闭。
- 回放必须提供本地 `--checkpoint-file`；本基础框架未接入 WandB 权重下载。
