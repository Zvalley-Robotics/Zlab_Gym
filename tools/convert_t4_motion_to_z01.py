#!/usr/bin/env python3
"""Convert a T4-27DOF retargeted motion clip into the Z01_std_add_head layout.

Two mismatches are fixed:

1. **Joints (27 -> 29).**  Z01_std_add_head adds a 2-DOF neck that T4 does not
   have.  Z01 joint order is the T4 order with the two head joints appended::

     0-6   left  arm   shoulder_pitch/roll/yaw, elbow, wrist_yaw/pitch/roll
     7-13  right arm   same
     14    waist_yaw
     15-20 left  leg   hip_pitch/roll/yaw, knee, ankle_pitch/roll
     21-26 right leg   same
     27    head_pitch  <- padded with 0 (no source data)
     28    head_yaw    <- padded with 0 (no source data)

   The first 27 columns are copied verbatim: T4 and Z01 share the same joint
   axes and limits for every one of them, so no sign flips or reordering are
   needed.  ``joint_vel`` gets the same treatment.  The padded head columns are
   constant zero for both position and velocity, i.e. the neck reference is
   "hold still".

2. **Bodies (30 -> 30, order differs).**  Both robots expose 30 named bodies,
   so the wrong order fails silently: mjlab slices the motion body array with
   the *robot's* body indices::

     Z01 index : 0-14         15            16-27         28, 29
     Z01 name  : torso+arms   waist_yaw_lnk legs          head_pitch/yaw_link
     T4 index  : 0-14         17 (pelvis)   18-29         (absent)

   T4 welds ``waist_pitch_link``/``waist_roll_link``/``pelvis`` at the same
   offset where Z01 puts ``waist_yaw_link`` (`0 0 -0.043` from ``torso_link``),
   so ``pelvis`` is the exact stand-in for ``waist_yaw_link``.  The two head
   bodies have no source data; since their joints are held at zero they are
   rigidly attached to ``torso_link``, so their world pose is reconstructed
   from the torso pose plus the fixed Z01 head offset.

Only the ``body_*_w`` arrays are treated as body-indexed; all other keys
(``fps``, ``joint_names``, ...) are copied through unchanged.

Usage::

    # rewrite in place, keeping a .bak backup
    python tools/convert_t4_motion_to_z01.py path/to/z01_heart.npz

    # write elsewhere
    python tools/convert_t4_motion_to_z01.py in.npz -o out.npz

    # preview without touching anything
    python tools/convert_t4_motion_to_z01.py in.npz --dry-run

    # whole directory
    python tools/convert_t4_motion_to_z01.py src/assets/motions/.../demo_act
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Layouts
# --------------------------------------------------------------------------- #

Z01_JOINT_NAMES = (
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_yaw", "left_wrist_pitch", "left_wrist_roll",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_yaw", "right_wrist_pitch", "right_wrist_roll",
    "waist_yaw",
    "left_hip_pitch", "left_hip_roll", "left_hip_yaw",
    "left_knee", "left_ankle_pitch", "left_ankle_roll",
    "right_hip_pitch", "right_hip_roll", "right_hip_yaw",
    "right_knee", "right_ankle_pitch", "right_ankle_roll",
    "head_pitch", "head_yaw",
)
N_Z01_JOINTS = len(Z01_JOINT_NAMES)  # 29
N_T4_JOINTS = 27

N_Z01_BODIES = 30

# Z01 body index -> source T4 body index.  28/29 are the head bodies, which are
# rebuilt from the torso pose (marker -1), see ``_head_bodies_from_torso``.
BODY_SRC_FROM_T4: tuple[int, ...] = (
    *range(15),   # 0-14  : torso + both arms, identical order
    17,           # 15    : Z01 waist_yaw_link  <- T4 pelvis
    *range(18, 30),  # 16-27: both legs, shifted by +2 in T4
    -1,           # 28    : Z01 head_pitch_link (no T4 source)
    -1,           # 29    : Z01 head_yaw_link   (no T4 source)
)

Z01_BODY_NAMES = (
    "torso_link",
    *(f"{s}_{n}_link" for s in ("left", "right")
      for n in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                "wrist_yaw", "wrist_pitch", "wrist_roll")),
    "waist_yaw_link",
    *(f"{s}_{n}_link" for s in ("left", "right")
      for n in ("hip_pitch", "hip_roll", "hip_yaw", "knee",
                "ankle_pitch", "ankle_roll")),
    "head_pitch_link",
    "head_yaw_link",
)

BODY_KEYS = ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")

DEFAULT_XML = (
    Path(__file__).resolve().parents[1]
    / "src/assets/robots/Z01_std_add_head/xmls/Z01_std_add_head.xml"
)

HEAD_BODY_NAMES = ("head_pitch_link", "head_yaw_link")


# --------------------------------------------------------------------------- #
# Quaternion helpers (MuJoCo convention: w, x, y, z)
# --------------------------------------------------------------------------- #

def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz -> (..., 3, 3)."""
    q = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    w, x, y, z = (q[..., i] for i in range(4))
    out = np.empty(q.shape[:-1] + (3, 3), dtype=q.dtype)
    out[..., 0, 0] = 1 - 2 * (y * y + z * z)
    out[..., 0, 1] = 2 * (x * y - z * w)
    out[..., 0, 2] = 2 * (x * z + y * w)
    out[..., 1, 0] = 2 * (x * y + z * w)
    out[..., 1, 1] = 1 - 2 * (x * x + z * z)
    out[..., 1, 2] = 2 * (y * z - x * w)
    out[..., 2, 0] = 2 * (x * z - y * w)
    out[..., 2, 1] = 2 * (y * z + x * w)
    out[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return out


# --------------------------------------------------------------------------- #
# Conversion
# --------------------------------------------------------------------------- #

def _head_offsets(xml_path: Path) -> np.ndarray:
    """Offset of each Z01 head body in the ``torso_link`` frame, joints at 0.

    ``(2, 3)``; the head bodies are rigidly attached to the torso because their
    joints are held at zero, so a pure translation is enough.
    """
    import mujoco  # local import: only needed for the head reconstruction

    model = mujoco.MjSpec.from_file(str(xml_path)).compile()
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qpos[3] = 1.0  # free joint quat = identity (w, x, y, z)
    mujoco.mj_forward(model, data)

    torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    torso_pos = data.xpos[torso].copy()
    torso_quat = data.xquat[torso].copy()
    if not np.allclose(torso_pos, 0.0) or not np.allclose(torso_quat, [1, 0, 0, 0]):
        raise RuntimeError(
            f"unexpected nominal torso pose {torso_pos} {torso_quat}; "
            "cannot derive head offsets"
        )
    return np.stack([
        data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()
        for name in HEAD_BODY_NAMES
    ])


def convert_joints(arr: np.ndarray, key: str) -> np.ndarray:
    """(T, 27) -> (T, 29); the two head columns stay zero."""
    if arr.ndim != 2:
        raise ValueError(f"{key}: expected (T, J), got {arr.shape}")
    if arr.shape[1] == N_Z01_JOINTS:
        return arr
    if arr.shape[1] != N_T4_JOINTS:
        raise ValueError(
            f"{key}: expected {N_T4_JOINTS} or {N_Z01_JOINTS} columns, got {arr.shape[1]}"
        )
    out = np.zeros((arr.shape[0], N_Z01_JOINTS), dtype=arr.dtype)
    out[:, :N_T4_JOINTS] = arr
    return out


def convert_bodies(
    blob: dict[str, np.ndarray], head_offsets: np.ndarray
) -> dict[str, np.ndarray]:
    """Reorder the 30 body entries into Z01 order and add the head bodies."""
    src = np.asarray(BODY_SRC_FROM_T4, dtype=np.int64)
    dst_idx = np.arange(N_Z01_BODIES, dtype=np.int64)
    keep = src >= 0
    head_idx = dst_idx[~keep]
    if head_idx.size != len(HEAD_BODY_NAMES) or len(src) != N_Z01_BODIES:
        raise AssertionError("BODY_SRC_FROM_T4 is malformed")

    out: dict[str, np.ndarray] = {}
    torso_pos = blob["body_pos_w"][:, 0]                      # (T, 3)
    torso_quat = blob["body_quat_w"][:, 0]                    # (T, 4)
    torso_lin_vel = blob["body_lin_vel_w"][:, 0]              # (T, 3)
    torso_ang_vel = blob["body_ang_vel_w"][:, 0]              # (T, 3)
    rot = quat_to_mat(torso_quat)                             # (T, 3, 3)

    for key in BODY_KEYS:
        if key not in blob:
            continue
        arr = blob[key].astype(np.float64, copy=False)
        if arr.shape[1] != N_Z01_BODIES:
            raise ValueError(
                f"{key}: expected {N_Z01_BODIES} bodies, got {arr.shape[1]}"
            )
        new = np.empty_like(arr)
        new[:, dst_idx[keep]] = arr[:, src[keep]]
        out[key] = new

    for head_slot, offset in zip(head_idx, head_offsets):
        offset = offset.astype(np.float64)
        rotated = rot @ offset                                   # (T, 3)
        if "body_pos_w" in out:
            out["body_pos_w"][:, head_slot] = torso_pos + rotated
        if "body_quat_w" in out:
            out["body_quat_w"][:, head_slot] = torso_quat
        if "body_lin_vel_w" in out:
            # v_head = v_torso + omega_torso x r_head
            out["body_lin_vel_w"][:, head_slot] = (
                torso_lin_vel + np.cross(torso_ang_vel, rotated)
            )
        if "body_ang_vel_w" in out:
            out["body_ang_vel_w"][:, head_slot] = torso_ang_vel

    for key in out:
        out[key] = out[key].astype(blob[key].dtype, copy=False)
    return out


def convert_file(path: Path, out_path: Path, xml: Path, dry_run: bool) -> bool:
    with np.load(path, allow_pickle=True) as data:
        blob = {k: data[k] for k in data.files}

    if "joint_pos" not in blob or "body_pos_w" not in blob:
        print(f"  skip (not a motion clip): {path}")
        return False

    n_joints = blob["joint_pos"].shape[1]
    n_bodies = blob["body_pos_w"].shape[1]
    if n_joints == N_Z01_JOINTS and n_bodies == N_Z01_BODIES:
        print(f"  skip (already Z01 29-DOF): {path}")
        return False

    head_offsets = _head_offsets(xml)
    before = (blob["joint_pos"].shape, blob["body_pos_w"].shape)

    for key in ("joint_pos", "joint_vel"):
        if key in blob:
            blob[key] = convert_joints(np.asarray(blob[key]), key)
    blob.update(convert_bodies(blob, head_offsets))

    blob["joint_names"] = np.array(Z01_JOINT_NAMES, dtype="<U32")
    blob["body_names"] = np.array(Z01_BODY_NAMES, dtype="<U64")

    after = (blob["joint_pos"].shape, blob["body_pos_w"].shape)
    print(f"  {path}")
    print(f"    joint_pos {before[0]} -> {after[0]}   (head_pitch/head_yaw = 0)")
    print(f"    body_pos_w {before[1]} -> {after[1]} (reordered to Z01)")

    if dry_run:
        print("    [dry-run] not written")
        return True

    if out_path == path and not (path.parent / (path.name + ".bak")).exists():
        shutil.copy2(path, path.parent / (path.name + ".bak"))
        print(f"    backup: {path.name}.bak")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **blob)
    print(f"    written: {out_path}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help=".npz file or a directory of them")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="output file (default: rewrite the input in place)")
    ap.add_argument("--xml", type=Path, default=DEFAULT_XML,
                    help="Z01 MJCF, used to derive the head body offsets")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.input.is_dir():
        files = sorted(args.input.rglob("*.npz"))
        if not files:
            raise SystemExit(f"no .npz under {args.input}")
    else:
        files = [args.input]

    if args.output is not None and len(files) > 1:
        raise SystemExit("-o cannot be combined with a directory input")

    print(f"Z01 layout: {N_Z01_JOINTS} joints / {N_Z01_BODIES} bodies")
    changed = 0
    for path in files:
        if convert_file(path, args.output or path, args.xml, args.dry_run):
            changed += 1
    print(f"done: {changed}/{len(files)} file(s) converted")


if __name__ == "__main__":
    main()
