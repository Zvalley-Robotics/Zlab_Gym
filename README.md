# Zlab_Gym — A Motion Imitation & Basic Locomotion Training Framework for the Z01 Robot, Enabling Users to Train Custom Motion Models by Adapting the Framework

A minimal secondary-development framework built on
[mjlab](https://github.com/mujocolab/mjlab) (`mjlab==1.2.0`) + MuJoCo.
It registers **one robot** (`Z01_std_add_head`, 29 DOF, 27 policy-controlled)
with one motion-tracking task and two velocity tasks:

| Task id | Group | Description |
|---|---|---|
| `Z01-StdAddHead-Tracking` | `tracking` | whole-body motion imitation (flat terrain) |
| `Z01-StdAddHead-Velocity-Rough` | `velocity` | blind twist-velocity locomotion, rough terrain + terrain curriculum |
| `Z01-StdAddHead-Velocity-Flat` | `velocity` | blind twist-velocity locomotion, flat plane |

---

## 1. Installation

```bash
conda create -n zlab_tracking python=3.11 -y
conda activate zlab_tracking

# Install torch first, matching your CUDA, e.g.:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

pip install -e .        # installs mjlab==1.2.0 automatically
```

`src/` is prepended to `sys.path` by `scripts/train.py` / `scripts/play.py`
so the **vendored, patched `rsl_rl`** under `src/rsl_rl` takes precedence over
any site-packages copy. Do not remove those `sys.path` lines.

## 2. List available tasks

```bash
python scripts/list_envs.py            # all registered tasks
python scripts/list_envs.py --keyword Z01   # filter
```

## 3. Quick start

### 3.1 Tracking (motion imitation) — `--motion-file` is mandatory

```bash
# Train
python scripts/train.py Z01-StdAddHead-Tracking \
  --motion-file=src/assets/motions/z01_std_add_head/demo_act/z01_std_base_heart.npz \
  --env.scene.num-envs=256 \
  --agent.experiment-name=z01_std_add_head_tracking
# logs -> logs/rsl_rl/z01_std_add_head_tracking/<timestamp>/  (model_<iter>.pt every 500 iters)

# Play
python scripts/play.py Z01-StdAddHead-Tracking \
  --checkpoint-file=logs/rsl_rl/z01_std_add_head_tracking/<RUN_DIR>/model_0.pt \
  --motion-file=src/assets/motions/z01_std_add_head/demo_act/z01_std_base_heart.npz \
  --num-envs 1 --device cpu --viewer native
```

### 3.2 Velocity tasks — no motion file needed

```bash
# Train (rough terrain, terrain-level curriculum)
python scripts/train.py Z01-StdAddHead-Velocity-Rough \
  --env.scene.num-envs=4096 \
  --agent.experiment-name=z01_std_add_head_velocity

# Train (flat plane)
python scripts/train.py Z01-StdAddHead-Velocity-Flat \
  --env.scene.num-envs=4096 \
  --agent.experiment-name=z01_std_add_head_velocity

# Play
python scripts/play.py Z01-StdAddHead-Velocity-Rough \
  --checkpoint-file=logs/rsl_rl/z01_std_add_head_velocity/<RUN_DIR>/model_0.pt \
  --num-envs 1 --device cpu --viewer native
```

## 4. Common flags

| Flag | Applies to | Description |
|---|---|---|
| `--motion-file` | train / play (tracking) | motion `.npz`, **required** for tracking tasks |
| `--checkpoint-file` | play | local `.pt` weight, **required** (no WandB download) |
| `--agent.experiment-name` | train | log sub-directory name |
| `--agent.logger` | train | default `tensorboard`; `wandb` also available |
| `--agent.max-iterations` | train | override training length |
| `--env.scene.num-envs` | train | number of parallel envs |
| `--num-envs` | play | number of parallel envs |
| `--device` | play | `cpu` / `cuda:0`, defaults to `cuda:0` if available |
| `--gpu-ids` | train | `[0]`, `all`, or explicit ids (multi-GPU via torchrunx) |
| `--video` / `--video-length` | train / play | record video into `<log_dir>/videos/` |
| `--viewer` | play | `auto` / `native` / `viser`; `auto` falls back to `viser` without `DISPLAY` |
| `--agent zero\|random\|trained` | play | dummy policy for debugging |
| `--no-terminations` | play | disable all terminations (watch the full motion) |
| `--log-root` / `--log-dir` / env `ZLAB_LOG_ROOT` | train | log root (default `logs`) |
| `--enable-nan-guard` | train | enable NaN guard |

## 5. Bundled assets

```
src/assets/motions/z01_std_add_head/demo_act/
├── z01_std_base_heart.npz        # demo motion ("heart" gesture), input for --motion-file
├── z01_std_base_heart.onnx       # pretrained tracking policy (motion baked in)
└── z01_velocity_flat_base.onnx   # pretrained flat-velocity policy
```

The two ONNX files are exported artifacts of the corresponding trained runs,
provided as references for downstream deployment. This repo does **not** ship a
script that plays ONNX directly — run the `.pt` checkpoints with `scripts/play.py`.

## 6. ONNX export

Every `save()` of the tracking runner `MotionTrackingOnPolicyRunner` writes two
models next to the checkpoint:

- `<ckpt_stem>_policy.onnx` — the policy network (`actor.as_onnx()`).
- `<ckpt_stem>_motion.onnx` — `_OnnxMotionModel`: policy + reference motion
  buffers baked in as ONNX constants, so the deployed policy replays the clip
  without external data files.

Both carry mjlab metadata (`attach_metadata_to_onnx`). Velocity tasks use
mjlab's `VelocityOnPolicyRunner` (policy-only export).

## 7. Repo layout

```
scripts/                                    # train.py / play.py / list_envs.py
tools/                                      # offline motion conversion utilities
src/
├── assets/
│   ├── robots/Z01_std_add_head/            # constants + xmls/(loaded) + xml/(generator output)
│   │                                       #   + urdf/ csv/ meshes/ urdf_to_xml.yaml
│   └── motions/z01_std_add_head/demo_act/  # demo motion + ONNX artifacts
├── rsl_rl/                                 # vendored patched PPO (keep as-is!)
└── tasks/
    ├── __init__.py                         # registers "tracking" + "velocity" groups
    ├── tracking/
    │   ├── tracking_env_cfg.py             # make_tracking_env_cfg(): shared env skeleton
    │   ├── config/z01_std_add_head/        # env/rl cfg + task registration
    │   ├── rl/runner.py                    # MotionTrackingOnPolicyRunner (ONNX export)
    │   └── mdp/                            # commands / observations / rewards / terminations
    └── velocity/
        ├── config/z01_std_add_head/        # env/rl cfg + task registration
        └── mdp/rewards.py                  # repo-specific reward terms
```

`tools/`:

- `convert_t4_motion_to_z01.py` — T4 (27-DOF) → Z01 (29-DOF) motion conversion, the
  main tool: `python tools/convert_t4_motion_to_z01.py <motion.npz>`
  (`--dry-run` to preview, a directory argument to batch).
- `convert_pkl_to_50hz_npz.py` — pkl → 50 Hz npz motion.
- `convert_npz_to_csv.py` / `csv_to_npz.py` — **legacy** G1-specific scripts with
  hard-coded paths, not used by the Z01 workflow.

## 8. Extension points

1. **New motion**: drop a `.npz` with the same schema under `src/assets/motions/...`
   and pass it via `--motion-file`.
2. **New reward / termination**: implement in `src/tasks/<group>/mdp/` and mount it
   in the group's `env_cfgs.py`.
3. **New robot**: clone `Z01_std_add_head` constants + `xmls/` + `meshes/`, exposing
   `get_<robot>_robot_cfg()` and `<ROBOT>_ACTION_SCALE`.
4. **New task id**: duplicate a `register_mjlab_task` block and add the module path
   to `_TASK_GROUPS` in `src/tasks/__init__.py`.

## 9. Notes & caveats

- `src/rsl_rl` is a **patched** copy of rsl-rl (entropy annealing / std clamp).
  Do not delete it, and do not replace it with the site-packages version.
- The policy controls **27 joints**: `head_pitch` / `head_yaw` are excluded from
  actions and from the `joint_pos` / `joint_vel` observations, and are locked at
  zero by passive stiffness (`jnt_stiffness=100`) + damping (`dof_damping=10`)
  in `env_cfgs.py`; `HeadAgnosticMotionCommand` also strips them from the command
  observation. The `pose` std dicts, however, still cover all 29 joints.
- `Z01_STD_ADD_HEAD_ACTION_SCALE` is **hard-coded to 0.25** for every joint
  (not derived from `effort_limit / stiffness`). Override it in
  `z01_std_add_head_constants.py` if needed.
- Z01 joint names carry **no `_joint` suffix** (`left_knee`, `waist_yaw`,
  `head_pitch`, …), so `target_names_expr` / `joint_names` regexes must follow that.
- The velocity **Flat** variant differs from Rough: it drops
  `foot_clearance` / `foot_swing_height` / `soft_landing`, sets `air_time` weight to 0,
  and adds `base_height` (0.5), `foot_gait` (0.8), `feet_lateral_distance` (-0.5),
  `arm_symmetry` (0.5), `leg_symmetry` (0.3). It also applies a yaml run profile
  (`z01_velocity_flat_base`): a fixed default joint pose plus a full set of PD gains
  (`_YAML_PD_GROUPS`, standing height 0.823 m).
- The Z01 body directly below the torso is `waist_yaw_link` (no `pelvis`); velocity
  tasks use the root body `torso_link` for `terrain_scan`, `upright`, `body_ang_vel`
  and the `self_collision` sensor.
- The Z01 MJCF ships two sole-centre sites, `left_foot` / `right_foot`, required by
  the foot-based reward terms.
- `neg_power` (negative mechanical power / regenerative braking) is mounted at
  `cfg.rewards["neg_power"]` with `weight=0` — off by default.
- Play requires a local `--checkpoint-file`; WandB checkpoint download is not wired
  in this base.
