"""
Convert pkl motion data files to 50Hz npz format.

Source pkl format (dict):
  - fps: float
  - root_pos: (N, 3)
  - root_rot: (N, 4)  quaternion (w, x, y, z)
  - dof_pos: (N, 27)
  - loop_mode: int
  - key_body_pos: (N, num_key_bodies, 3)

Target npz format (same keys, resampled to 50Hz):
  - fps: float (50.0)
  - root_pos: (M, 3)
  - root_rot: (M, 4)
  - dof_pos: (M, 27)
  - loop_mode: int
  - key_body_pos: (M, num_key_bodies, 3)
"""

import os
import sys
import pickle
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp, Rotation as R

TARGET_FPS = 50.0


def resample_array(data: np.ndarray, src_fps: float, tgt_fps: float) -> np.ndarray:
    """Linearly resample a 1D/2D/3D array along the time axis (axis=0)."""
    N = data.shape[0]
    if N < 2:
        return data
    src_t = np.arange(N) / src_fps
    tgt_N = max(2, int(np.round((N - 1) / src_fps * tgt_fps + 1)))
    tgt_t = np.arange(tgt_N) / tgt_fps
    tgt_t = np.clip(tgt_t, 0, src_t[-1])

    original_shape = data.shape
    flat_data = data.reshape(N, -1)
    interpolator = interp1d(src_t, flat_data, axis=0, kind='linear',
                            fill_value='extrapolate')
    resampled_flat = interpolator(tgt_t)
    resampled = resampled_flat.reshape(tgt_N, *original_shape[1:])
    return resampled.astype(data.dtype)


def resample_quaternion(quat: np.ndarray, src_fps: float, tgt_fps: float) -> np.ndarray:
    """Resample quaternion array using slerp."""
    N = quat.shape[0]
    if N < 2:
        return quat
    src_t = np.arange(N) / src_fps
    tgt_N = max(2, int(np.round((N - 1) / src_fps * tgt_fps + 1)))
    tgt_t = np.arange(tgt_N) / tgt_fps
    tgt_t = np.clip(tgt_t, 0, src_t[-1])

    rotations = R.from_quat(quat[:, [1, 2, 3, 0]])  # scipy uses (x,y,z,w)
    slerp = Slerp(src_t, rotations)
    resampled_rot = slerp(tgt_t)
    resampled_quat = resampled_rot.as_quat()  # (x,y,z,w)
    return np.concatenate([resampled_quat[:, 3:4], resampled_quat[:, :3]], axis=1).astype(quat.dtype)


def convert_pkl_to_npz(pkl_path: str, npz_path: str):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    src_fps = float(data['fps'])
    print(f"  Source fps: {src_fps:.2f}, frames: {data['root_pos'].shape[0]}")

    if abs(src_fps - TARGET_FPS) < 0.5:
        print(
            f"  Already at ~{TARGET_FPS} Hz, skipping resample (copying as-is).")
        resampled = {k: v for k, v in data.items()}
    else:
        resampled = {}
        resampled['fps'] = TARGET_FPS
        resampled['loop_mode'] = data['loop_mode']

        resampled['root_pos'] = resample_array(
            data['root_pos'], src_fps, TARGET_FPS)
        resampled['root_rot'] = resample_quaternion(
            data['root_rot'], src_fps, TARGET_FPS)
        resampled['dof_pos'] = resample_array(
            data['dof_pos'], src_fps, TARGET_FPS)
        resampled['key_body_pos'] = resample_array(
            data['key_body_pos'], src_fps, TARGET_FPS)

    tgt_frames = resampled['root_pos'].shape[0]
    print(f"  Target fps: {TARGET_FPS}, frames: {tgt_frames}")

    np.savez(npz_path, **resampled)
    print(f"  Saved to: {npz_path}")


def main():
    amp_dir = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(os.path.dirname(__file__), '..', 'src',
                     'assets', 'motions', 'z01_std_add_head', 'amp')

    amp_dir = os.path.abspath(amp_dir)
    if not os.path.isdir(amp_dir):
        print(f"Error: directory not found: {amp_dir}")
        sys.exit(1)

    pkl_files = sorted([f for f in os.listdir(amp_dir) if f.endswith('.pkl')])
    if not pkl_files:
        print(f"No .pkl files found in {amp_dir}")
        sys.exit(0)

    print(f"Found {len(pkl_files)} .pkl files in {amp_dir}")
    print(f"Target: {TARGET_FPS} Hz npz format\n")

    for pkl_file in pkl_files:
        pkl_path = os.path.join(amp_dir, pkl_file)
        npz_file = pkl_file.replace('.pkl', '.npz')
        npz_path = os.path.join(amp_dir, npz_file)
        print(f"[{pkl_file}]")
        convert_pkl_to_npz(pkl_path, npz_path)
        print()

    print("Done!")


if __name__ == '__main__':
    main()
