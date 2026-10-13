"""Convert 29-joint T4 motion .npz files to CSV format."""

import numpy as np
import csv
import os
import sys

# Joint names from t4_std.xml (29 joints)
JOINT_NAMES = [
    "J_arm_l_01", "J_arm_l_02", "J_arm_l_03", "J_arm_l_04",
    "J_arm_l_05", "J_arm_l_06", "J_arm_l_07",
    "J_arm_r_01", "J_arm_r_02", "J_arm_r_03", "J_arm_r_04",
    "J_arm_r_05", "J_arm_r_06", "J_arm_r_07",
    "J_waist_pitch", "J_waist_roll", "J_waist_yaw",
    "J_hip_l_pitch", "J_hip_l_roll", "J_hip_l_yaw", "J_knee_l_pitch",
    "J_ankle_l_pitch", "J_ankle_l_roll",
    "J_hip_r_pitch", "J_hip_r_roll", "J_hip_r_yaw", "J_knee_r_pitch",
    "J_ankle_r_pitch", "J_ankle_r_roll",
]

# Body names from t4_std.xml (30 bodies)
BODY_NAMES = [
    "Trunk",
    "AL1", "AL2", "AL3", "AL4", "AL5", "AL6", "AL7",
    "AR1", "AR2", "AR3", "AR4", "AR5", "AR6", "AR7",
    "Waist_pitch", "Waist_roll", "Waist_yaw",
    "Hip_Pitch_Left", "Hip_Roll_Left", "Hip_Yaw_Left", "Shank_Left",
    "Ankle_Cross_Left", "left_foot_link",
    "Hip_Pitch_Right", "Hip_Roll_Right", "Hip_Yaw_Right", "Shank_Right",
    "Ankle_Cross_Right", "right_foot_link",
]


def build_header():
    header = ["time_step", "fps"]
    for jn in JOINT_NAMES:
        header.append(f"joint_pos_{jn}")
    for jn in JOINT_NAMES:
        header.append(f"joint_vel_{jn}")
    for bn in BODY_NAMES:
        for axis in ["x", "y", "z"]:
            header.append(f"body_pos_w_{bn}_{axis}")
    for bn in BODY_NAMES:
        for comp in ["w", "x", "y", "z"]:
            header.append(f"body_quat_w_{bn}_{comp}")
    for bn in BODY_NAMES:
        for axis in ["x", "y", "z"]:
            header.append(f"body_lin_vel_w_{bn}_{axis}")
    for bn in BODY_NAMES:
        for axis in ["x", "y", "z"]:
            header.append(f"body_ang_vel_w_{bn}_{axis}")
    return header


def npz_to_csv(npz_path, csv_path):
    data = np.load(npz_path)
    T = data["joint_pos"].shape[0]
    fps = float(data["fps"][0])

    header = build_header()

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for t in range(T):
            row = [t, fps]
            row.extend(data["joint_pos"][t].tolist())
            row.extend(data["joint_vel"][t].tolist())
            row.extend(data["body_pos_w"][t].flatten().tolist())
            row.extend(data["body_quat_w"][t].flatten().tolist())
            row.extend(data["body_lin_vel_w"][t].flatten().tolist())
            row.extend(data["body_ang_vel_w"][t].flatten().tolist())
            writer.writerow(row)

    size_mb = os.path.getsize(csv_path) / 1024 / 1024
    print(
        f"Saved: {csv_path}  ({T} rows, {len(header)} cols, {size_mb:.1f} MB)")


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base)
    outdir = os.path.join(project_root, "src", "assets", "motions", "z01_std_add_head")

    files = [
        "high_jump_R_001__A500_M_sonic",
        "jump_forward03_poses_smplx_t4",
    ]

    for name in files:
        npz_path = os.path.join(outdir, f"{name}.npz")
        csv_path = os.path.join(outdir, f"{name}.csv")
        npz_to_csv(npz_path, csv_path)


if __name__ == "__main__":
    main()
