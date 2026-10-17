import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Change this to the path where your JSON files are stored
DATA_DIR = r"D:\MA\USB_TCP_combined"


def analyze_rotations(data_dir):
    json_files = glob.glob(os.path.join(data_dir, "*.json"))

    if not json_files:
        print(f"No JSON files found in {data_dir}. Please check the path.")
        return

    data = []

    for file_path in json_files:
        with open(file_path, 'r') as f:
            try:
                label_data = json.load(f)
                orient = label_data['orientation']

                # Scipy expects quaternions in [x, y, z, w] format
                quat = [orient['x'], orient['y'], orient['z'], orient['w']]

                # Convert to Euler angles in degrees (using 'xyz' intrinsic sequence)
                # You can change the sequence ('xyz', 'zyx', etc.) based on your coordinate system
                euler = R.from_quat(quat).as_euler('xyz', degrees=True)

                data.append({
                    'filename': os.path.basename(file_path),
                    'rot_x': euler[0],
                    'rot_y': euler[1],
                    'rot_z': euler[2]
                })
            except Exception as e:
                print(f"Error parsing {file_path}: {e}")

    df = pd.DataFrame(data)

    # Calculate Statistics
    means = df[['rot_x', 'rot_y', 'rot_z']].mean()
    stds = df[['rot_x', 'rot_y', 'rot_z']].std()

    print("--- Dataset Rotation Statistics (Degrees) ---")
    for axis in ['x', 'y', 'z']:
        print(f"Rotation {axis.upper()} -> Mean: {means[f'rot_{axis}']:6.2f}°, Std Dev: {stds[f'rot_{axis}']:6.2f}°")
    print("---------------------------------------------")

    # Plot Distribution
    plt.figure(figsize=(10, 4))
    plt.hist(df['rot_x'], bins=30, alpha=0.7, color='crimson', edgecolor='black')
    plt.axvline(means['rot_x'], color='blue', linestyle='dashed', linewidth=2, label=f"Mean ({means['rot_x']:.2f}°)")
    plt.title("Distribution of X Rotation (Pitch/Roll depending on setup)")
    plt.xlabel("Degrees")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # Run the analysis
    analyze_rotations(DATA_DIR)