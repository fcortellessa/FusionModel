
import os
import sys
import shutil
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics.data.augment import LetterBox
import cv2
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from data_processing.GripFusionNet_split import split_dataset
from data_processing.GripFusionNet_get_norm_params import (
    compute_mean_std_digit_images,
    compute_mean_std_rgb_images_no_crop
)


# ----------------------------
# SETTINGS
# ----------------------------
DEBUG = True          # Set True to disable multiprocessing (recommended first run)
NUM_WORKERS = 4        # safer than cpu_count()


# ----------------------------
# WORKER FUNCTION
# ----------------------------

def _create_image_transforms(rgb_mean, rgb_std, digit_mean, digit_std):
    """Create image transformation pipelines."""
    rgb_transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=tuple(rgb_mean), std=tuple(rgb_std))
    ])

    digit_transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=tuple(digit_mean), std=tuple(digit_std))
    ])
    
    return rgb_transform, digit_transform


def _process_image(image_path, transform, use_letterbox=True):
    """Load and process a single image."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"cv2.imread failed for {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if use_letterbox:
        letterbox = LetterBox(new_shape=(640, 640), auto=False,
                              scale_fill=False, scaleup=False,
                              center=True, stride=32)
        img = letterbox(image=img)
    
    img = Image.fromarray(img).convert("RGB")
    tensor = transform(img)
    return tensor


def _process_pose(pose_path, transf_matrix):
    """Load and transform pose from JSON file."""
    with open(pose_path, 'r') as f:
        data = json.load(f)

    try:
        pos = np.array([data["position"][k] for k in ["x", "y", "z"]])
        quat = np.array([data["orientation"][k] for k in ["x", "y", "z", "w"]])
    except KeyError as e:
        raise KeyError(f"Malformed JSON in {pose_path}: missing {e}")

    rot = R.from_quat(quat).as_matrix()
    homo_tcp = np.eye(4)
    homo_tcp[:3, :3] = rot
    homo_tcp[:3, 3] = pos

    hub_pose = np.linalg.inv(homo_tcp) @ np.array(transf_matrix)

    pose_vec = torch.tensor(
        np.concatenate([
            hub_pose[:3, 3],
            R.from_matrix(hub_pose[:3, :3]).as_quat()
        ]),
        dtype=torch.float32
    )
    return pose_vec


def process_sample(ts, folder_path, output_dir,
                   transf_matrix, rgb_mean, rgb_std, digit_mean, digit_std):
    """Process a single sample: load, transform, and save."""
    try:
        print(f"Processing {ts}", flush=True)

        # ---- Paths ----
        rgb_path = os.path.join(folder_path, f"{ts}.png")
        digit_path = os.path.join(folder_path, f"{ts}_digit.png")
        pose_path = os.path.join(folder_path, f"{ts}.json")

        # ---- File checks ----
        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"Missing RGB: {rgb_path}")
        if not os.path.exists(digit_path):
            raise FileNotFoundError(f"Missing Digit: {digit_path}")
        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"Missing Pose: {pose_path}")

        # ---- Create transforms ----
        rgb_transform, digit_transform = _create_image_transforms(
            rgb_mean, rgb_std, digit_mean, digit_std
        )

        # ---- Process images ----
        rgb_tensor = _process_image(rgb_path, rgb_transform, use_letterbox=True)
        digit_tensor = _process_image(digit_path, digit_transform, use_letterbox=True)

        # ---- Process pose ----
        pose_vec = _process_pose(pose_path, transf_matrix)

        # ---- Save ----
        torch.save({
            "rgb": rgb_tensor,
            "digit": digit_tensor,
            "pose": pose_vec
        }, os.path.join(output_dir, f"{ts}.pt"))

        return ts

    except Exception as e:
        print(f"\n❌ ERROR processing {ts}: {e}", flush=True)
        raise


# ----------------------------
# PREPROCESSING FUNCTION
# ----------------------------

def _collect_valid_timestamps(folder_path):
    """Collect all valid sample timestamps from folder."""
    all_files = os.listdir(folder_path)

    rgb_ts = {f.replace(".png", "") for f in all_files
              if f.endswith(".png") and "_digit" not in f}
    digit_ts = {f.replace("_digit.png", "") for f in all_files
                if f.endswith("_digit.png")}
    pose_ts = {f.replace(".json", "") for f in all_files
               if f.endswith(".json")}

    valid_timestamps = sorted(rgb_ts & digit_ts & pose_ts)
    return valid_timestamps


def _process_samples_serial(valid_timestamps, folder_path, output_dir,
                           transf_matrix, rgb_mean, rgb_std, digit_mean, digit_std):
    """Process samples sequentially (no multiprocessing)."""
    for ts in tqdm(valid_timestamps, desc="DEBUG preprocessing"):
        process_sample(
            ts, folder_path, output_dir,
            transf_matrix, rgb_mean, rgb_std, digit_mean, digit_std
        )


def _process_samples_parallel(valid_timestamps, folder_path, output_dir,
                             transf_matrix, rgb_mean, rgb_std, digit_mean, digit_std):
    """Process samples in parallel using multiprocessing."""
    with Pool(NUM_WORKERS) as pool:
        process_fn = partial(
            process_sample,
            folder_path=folder_path,
            output_dir=output_dir,
            transf_matrix=transf_matrix,
            rgb_mean=rgb_mean,
            rgb_std=rgb_std,
            digit_mean=digit_mean,
            digit_std=digit_std
        )

        list(tqdm(
            pool.imap_unordered(process_fn, valid_timestamps, chunksize=1),
            total=len(valid_timestamps),
            desc=f"Preprocessing {os.path.basename(folder_path)}"
        ))


def preprocess_and_save(
    transf_matrix,
    folder_path,
    output_dir,
    rgb_mean,
    rgb_std,
    digit_mean,
    digit_std,
):
    """Preprocess and save all samples in a folder."""
    os.makedirs(output_dir, exist_ok=True)

    # ---- Collect valid timestamps ----
    valid_timestamps = _collect_valid_timestamps(folder_path)
    print(f"Found {len(valid_timestamps)} samples", flush=True)

    # ---- Process samples ----
    if DEBUG:
        _process_samples_serial(valid_timestamps, folder_path, output_dir,
                               transf_matrix, rgb_mean, rgb_std, digit_mean, digit_std)
    else:
        _process_samples_parallel(valid_timestamps, folder_path, output_dir,
                                 transf_matrix, rgb_mean, rgb_std, digit_mean, digit_std)


def _save_normalization_stats(output_path, digit_mean, digit_std, rgb_mean, rgb_std):
    stats_path = os.path.join(output_path, "normalization_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "digit_mean": digit_mean,
                "digit_std": digit_std,
                "rgb_mean": rgb_mean,
                "rgb_std": rgb_std,
            },
            f,
            indent=2,
        )


# ----------------------------
def preprocess_data(
    root_dir,
    use_photometric_augmentation=False,
    aug_seed=42,
    aug_photo_count=1,
    aug_overwrite=False,
    aug_output_folder="train_augmentation_default",
    use_preaugmented_train_dir=False,
):
    """Main preprocessing pipeline: split, augment (if needed), and preprocess all splits."""
    output_path = os.path.join(root_dir, "Data_split/")
    
    # ---- Stage 1: Dataset splitting ----
    print("Splitting dataset into train/val/test")
    split_dataset(root_dir, output_path)

    train_split_dir = os.path.join(output_path, "train")

    if use_preaugmented_train_dir or use_photometric_augmentation:
        print(
            "Online augmentation is handled in the training dataloader; "
            "preprocessing will only build the standard train/val/test caches."
        )

    # ---- Stage 3: Preprocess all splits ----
    transf_matrix = [
        [1.0, 0.0, 0.0, 0.456577],
        [0.0, 1.0, 0.0, -0.163301],
        [0.0, 0.0, 1.0, -0.024417],
        [0.0, 0.0, 0.0, 1.0]
    ]
    
    split_input_dirs = {
        "train": train_split_dir,
        "val": os.path.join(output_path, "val"),
        "test": os.path.join(output_path, "test"),
    }

    print("Computing normalization stats...", flush=True)
    digit_mean, digit_std = compute_mean_std_digit_images(train_split_dir)
    rgb_mean, rgb_std = compute_mean_std_rgb_images_no_crop(train_split_dir)

    print("Preprocessing all splits")
    for split in ["train", "val", "test"]:
        split_output_dir = os.path.join(output_path, split, f"{split}_preprocessing")
        if os.path.exists(split_output_dir):
            shutil.rmtree(split_output_dir)
        preprocess_and_save(
            transf_matrix,
            folder_path=split_input_dirs[split],
            output_dir=split_output_dir,
            rgb_mean=rgb_mean,
            rgb_std=rgb_std,
            digit_mean=digit_mean,
            digit_std=digit_std,
        )

    _save_normalization_stats(output_path, digit_mean, digit_std, rgb_mean, rgb_std)
        