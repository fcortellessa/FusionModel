#sample_id,pred_translation,pred_rotation,pred_quaternion,gt_pose


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
import re

# Load the CSV file
df = pd.read_csv("results/results.csv")
# Convert string representations of arrays back to numpy arrays
def parse_array_string(s):
    try:
        # Handles Python-list style strings with commas, e.g. "[[1, 2, 3]]"
        parsed = ast.literal_eval(str(s))
        return np.asarray(parsed, dtype=float).reshape(-1)
    except Exception:
        # Handles whitespace-separated values, e.g. "[[ 0 -0.002 0.19 ]]"
        values = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", str(s))
        if values:
            return np.asarray([float(v) for v in values], dtype=float)
        return np.array([])
df["pred_translation"] = df["pred_translation"].apply(parse_array_string)
df["pred_rotation"] = df["pred_rotation"].apply(parse_array_string)
df["pred_quaternion"] = df["pred_quaternion"].apply(parse_array_string)
df["gt_pose"] = df["gt_pose"].apply(parse_array_string)


def normalize_quaternion_xyzw(q, eps=1e-12):
    q = np.asarray(q, dtype=float).reshape(4)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < eps:
        raise ValueError(f"Invalid quaternion norm: {n}")
    return q / n


def quaternion_geodesic_deg(q_pred, q_gt):
    q_pred = normalize_quaternion_xyzw(q_pred)
    q_gt = normalize_quaternion_xyzw(q_gt)
    dot = float(np.clip(np.abs(np.dot(q_pred, q_gt)), 0.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


def quat_xyzw_to_euler_xyz_deg(q):
    x, y, z, w = normalize_quaternion_xyzw(q)

    # Roll (X axis)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))

    # Pitch (Y axis)
    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))

    # Yaw (Z axis)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))

    return np.array([roll, pitch, yaw], dtype=float)


def wrapped_abs_angle_diff_deg(a_deg, b_deg):
    diff = np.abs(a_deg - b_deg)
    return np.minimum(diff, 360.0 - diff)


def is_valid_quaternion_array(arr):
    return arr.size == 4 and np.all(np.isfinite(arr))


# Prefer explicit quaternion column if present; otherwise fallback.
quat_col = "pred_quaternion"
if df[quat_col].apply(lambda x: x.size == 4).sum() == 0:
    quat_col = "pred_rotation"

invalid_quat_rows = [
    int(i)
    for i, arr in enumerate(df[quat_col])
    if not is_valid_quaternion_array(arr)
]
if invalid_quat_rows:
    preview = invalid_quat_rows[:10]
    raise ValueError(
        f"Column '{quat_col}' is not valid quaternion data (expected 4 finite values per row). "
        f"Invalid rows (first 10): {preview}"
    )

abs_translation_error = []
abs_translation_z_error = []
abs_translation_y_error = []
abs_translation_x_error = []
abs_rotation_error = []
abs_rotation_x_error = []
abs_rotation_z_error = []
abs_rotation_y_error = []
skipped_rows = 0
for idx, row in df.iterrows():
    pred_translation = row["pred_translation"]
    pred_quat = row[quat_col]
    gt_pose = row["gt_pose"]

    if pred_translation.size < 3 or gt_pose.size < 7:
        skipped_rows += 1
        continue

    gt_translation = gt_pose[:3]
    gt_quat = gt_pose[3:7]
    pred_translation = pred_translation[:3]

    abs_translation_error.append(np.linalg.norm(pred_translation - gt_translation))
    abs_translation_x_error.append(abs(pred_translation[0] - gt_translation[0]))
    abs_translation_y_error.append(abs(pred_translation[1] - gt_translation[1]))
    abs_translation_z_error.append(abs(pred_translation[2] - gt_translation[2]))

    # Rotation error as quaternion geodesic angle on SO(3), in degrees.
    abs_rotation_error.append(quaternion_geodesic_deg(pred_quat, gt_quat))

    pred_euler = quat_xyzw_to_euler_xyz_deg(pred_quat)
    gt_euler = quat_xyzw_to_euler_xyz_deg(gt_quat)
    axis_diffs = wrapped_abs_angle_diff_deg(pred_euler, gt_euler)
    abs_rotation_x_error.append(axis_diffs[0])
    abs_rotation_y_error.append(axis_diffs[1])
    abs_rotation_z_error.append(axis_diffs[2])

if not abs_translation_error:
    raise ValueError("No valid rows found in results.csv after parsing. Check column formats.")

# Plot translation error in mm
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(abs_translation_error, bins=20, color='blue', alpha=0.7)
plt.title("Absolute Translation Error (mm)")
plt.xlabel("Error (mm)")
plt.ylabel("Frequency")
plt.subplot(1, 2, 2)
plt.hist(abs_rotation_error, bins=20, color='orange', alpha=0.7)
plt.title("Absolute Rotation Error (degrees)")
plt.xlabel("Error (degrees)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Plot rotation error in degrees
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(abs_rotation_error, bins=20, color='orange', alpha=0.7)
plt.title("Absolute Rotation Error (degrees)")
plt.xlabel("Error (degrees)")
plt.ylabel("Frequency")
plt.subplot(1, 2, 2)
plt.hist(abs_rotation_x_error, bins=20, color='red', alpha=0.7, label="X-axis")
plt.hist(abs_rotation_y_error, bins=20, color='green', alpha=0.7, label="Y-axis")
plt.hist(abs_rotation_z_error, bins=20, color='blue', alpha=0.7, label="Z-axis")
plt.title("Absolute Rotation Error by Axis (degrees)")
plt.xlabel("Error (degrees)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()

# print summary statistics
print(f"Mean Translation Error: {np.mean(abs_translation_error)*1000:.2f} mm")
print(f"Mean Rotation Error: {np.mean(abs_rotation_error):.4f} degrees")
print(f"Mean Translation Error X: {np.mean(abs_translation_x_error)*1000:.2f} mm")
print(f"Mean Translation Error Y: {np.mean(abs_translation_y_error)*1000:.2f} mm")
print(f"Mean Translation Error Z: {np.mean(abs_translation_z_error)*1000:.2f} mm")
print(f"Mean Rotation Error X: {np.mean(abs_rotation_x_error):.4f} degrees")
print(f"Mean Rotation Error Y: {np.mean(abs_rotation_y_error):.4f} degrees")
print(f"Mean Rotation Error Z: {np.mean(abs_rotation_z_error):.4f} degrees")
print(f"Skipped Rows: {skipped_rows}")
print(f"Rotation Source Column: {quat_col}")



