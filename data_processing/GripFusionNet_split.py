import os
import json
import shutil
import random

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from utils.GripFusionNet_utils import load_config, get_triplets

# -------------------------------------------------
# Utility: read list of corrupted/excluded samples
# -------------------------------------------------
def get_excluded_timestamps(exclude_file):
    """
    Read a text file containing corrupted image paths and return their timestamps.
    Only entries ending in '_digit.png' are considered.
    """
    excluded = set()
    if exclude_file and os.path.exists(exclude_file):
        with open(exclude_file, 'r') as f:
            for line in f:
                path = line.strip()
                if path:
                    filename = os.path.basename(path)
                    if filename.endswith('_digit.png'):
                        # remove suffix to extract timestamp
                        timestamp = filename.replace('_digit.png', '')
                        excluded.add(timestamp)
    return excluded



# -------------------------------------------------
# Utility: copy one triplet into a target folder
# -------------------------------------------------
def copy_triplet(triplet, src_folder, dest_folder):
    """
    Copy all three files of a triplet (RGB, digit, JSON) from the source folder
    into the destination folder.
    """
    timestamp, rgb_file, digit_file, json_file = triplet
    os.makedirs(dest_folder, exist_ok=True)
    for fname in [rgb_file, digit_file, json_file]:
        shutil.copy(
            os.path.join(src_folder, fname),
            os.path.join(dest_folder, fname)
        )


# -------------------------------------------------
# Main splitting function
# -------------------------------------------------
def split_dataset(input_folder, output_root, train_ratio=0.8, val_ratio=0.15, test_ratio=0.05):
    """
    Split dataset triplets into train/val/test sets and copy them into
    separate subfolders.

    Args:
        input_folder: path containing all triplet files
        output_root: root output folder for train/val/test sets
        exclude_file: path to file listing corrupted images
        train_ratio, val_ratio, test_ratio: split proportions
    """
    # Load corrupted/excluded timestamps
    # Collect valid triplets
    triplets = get_triplets(input_folder)
    print(f"Found {len(triplets)} valid triplets.")

    # Shuffle for random splitting
    random.shuffle(triplets)
    total = len(triplets)

    # Compute split indices
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    # Partition into splits
    train_triplets = triplets[:train_end]
    val_triplets = triplets[train_end:val_end]
    test_triplets = triplets[val_end:]

    print(f"Splitting into: {len(train_triplets)} train, {len(val_triplets)} val, {len(test_triplets)} test")

    # Recreate split folders to avoid stale files from previous runs.
    for split_name in ["train", "val", "test"]:
        split_dir = os.path.join(output_root, split_name)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)

    # Copy files into respective subfolders
    for triplet in train_triplets:
        copy_triplet(triplet, input_folder, os.path.join(output_root, "train"))
    for triplet in val_triplets:
        copy_triplet(triplet, input_folder, os.path.join(output_root, "val"))
    for triplet in test_triplets:
        copy_triplet(triplet, input_folder, os.path.join(output_root, "test"))


# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    # config_path = r"GripFusionNet\Config\config.yaml"
    # config = load_config(config_path)
    # input_folder = config["data_path"]
    # output_root = os.path.join(config["data_path"],"Data_split/")

    output_root = os.path.join("D:/MA/GripFusionNet/GripFusionNet/dummy_data","Data_split/")
    split_dataset("D:/MA/GripFusionNet/GripFusionNet/dummy_data", output_root)
