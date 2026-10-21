
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
from multiprocessing import Pool
from functools import partial

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


LETTERBOX_PAD_VALUE = 114  # ultralytics LetterBox fill value (grey)


def load_letterboxed_uint8(image_path, use_letterbox=True):
    """Read an image as RGB uint8 (H, W, 3), letterboxed to 640x640 (no upscaling).

    Shared by the legacy .pt cache (via _process_image) and the compact uint8 cache
    (data_processing/compact_cache.py) so both start from byte-identical pixels.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"cv2.imread failed for {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if use_letterbox:
        letterbox = LetterBox(new_shape=(640, 640), auto=False,
                              scale_fill=False, scaleup=False,
                              center=True, stride=32)
        img = letterbox(image=img)
    return img


def _process_image(image_path, transform, use_letterbox=True):
    """Load and process a single image."""
    img = load_letterboxed_uint8(image_path, use_letterbox=use_letterbox)
    img = Image.fromarray(img).convert("RGB")
    tensor = transform(img)
    return tensor


def _hub_frame_pose(pose_path, transf_matrix):
    """Load a raw pose JSON and transform it into the hub frame.

    Returns a plain numpy array [x, y, z, qx, qy, qz, qw] in raw (un-normalized)
    units. Shared by per-sample caching and train-split translation stats so
    both read the exact same transform.
    """
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

    return np.concatenate([
        hub_pose[:3, 3],
        R.from_matrix(hub_pose[:3, :3]).as_quat()
    ])


def _process_pose(pose_path, transf_matrix, trans_norm_stats=None):
    """Load and transform pose from JSON file, normalizing y/z if stats are given.

    trans_norm_stats: optional dict with mean_y/std_y/mean_z/std_z (computed from
    the train split only; see compute_translation_norm_stats). When None, y/z are
    left in raw meters (matches historical no-op behavior).
    """
    pose_vec = _hub_frame_pose(pose_path, transf_matrix)

    if trans_norm_stats is not None:
        pose_vec[1] = (pose_vec[1] - trans_norm_stats["mean_y"]) / trans_norm_stats["std_y"]
        pose_vec[2] = (pose_vec[2] - trans_norm_stats["mean_z"]) / trans_norm_stats["std_z"]

    return torch.tensor(pose_vec, dtype=torch.float32)


def compute_translation_norm_stats(folder_path, transf_matrix):
    """Compute mean/std of hub-frame y and z translation over one split's raw poses.

    Intended to be called on the train split only; val/test reuse these stats
    (see preprocess_data) so normalization never leaks split-specific statistics.
    """
    valid_timestamps = _collect_valid_timestamps(folder_path)
    ys, zs = [], []
    for ts in valid_timestamps:
        pose_path = os.path.join(folder_path, f"{ts}.json")
        hub_pose = _hub_frame_pose(pose_path, transf_matrix)
        ys.append(hub_pose[1])
        zs.append(hub_pose[2])

    ys = np.array(ys, dtype=np.float64)
    zs = np.array(zs, dtype=np.float64)

    return {
        "mean_y": float(ys.mean()),
        "std_y": float(ys.std()),
        "mean_z": float(zs.mean()),
        "std_z": float(zs.std()),
    }


def process_sample(ts, folder_path, output_dir,
                   transf_matrix, rgb_mean, rgb_std, digit_mean, digit_std,
                   trans_norm_stats=None):
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
        pose_vec = _process_pose(pose_path, transf_matrix, trans_norm_stats=trans_norm_stats)

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
                           transf_matrix, rgb_mean, rgb_std, digit_mean, digit_std,
                           trans_norm_stats=None):
    """Process samples sequentially (no multiprocessing)."""
    for ts in tqdm(valid_timestamps, desc="DEBUG preprocessing"):
        process_sample(
            ts, folder_path, output_dir,
            transf_matrix, rgb_mean, rgb_std, digit_mean, digit_std,
            trans_norm_stats=trans_norm_stats
        )


def _process_samples_parallel(valid_timestamps, folder_path, output_dir,
                             transf_matrix, rgb_mean, rgb_std, digit_mean, digit_std,
                             trans_norm_stats=None):
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
            digit_std=digit_std,
            trans_norm_stats=trans_norm_stats
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
    trans_norm_stats=None,
):
    """Preprocess and save all samples in a folder."""
    os.makedirs(output_dir, exist_ok=True)

    # ---- Collect valid timestamps ----
    valid_timestamps = _collect_valid_timestamps(folder_path)
    print(f"Found {len(valid_timestamps)} samples", flush=True)

    # ---- Process samples ----
    if DEBUG:
        _process_samples_serial(valid_timestamps, folder_path, output_dir,
                               transf_matrix, rgb_mean, rgb_std, digit_mean, digit_std,
                               trans_norm_stats=trans_norm_stats)
    else:
        _process_samples_parallel(valid_timestamps, folder_path, output_dir,
                                 transf_matrix, rgb_mean, rgb_std, digit_mean, digit_std,
                                 trans_norm_stats=trans_norm_stats)


def _save_normalization_stats(output_path, digit_mean, digit_std, rgb_mean, rgb_std, trans_norm_stats):
    stats_path = os.path.join(output_path, "normalization_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "digit_mean": digit_mean,
                "digit_std": digit_std,
                "rgb_mean": rgb_mean,
                "rgb_std": rgb_std,
                "trans_mean_y": trans_norm_stats["mean_y"],
                "trans_std_y": trans_norm_stats["std_y"],
                "trans_mean_z": trans_norm_stats["mean_z"],
                "trans_std_z": trans_norm_stats["std_z"],
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
    split_seed=None,
    force_split=False,
    group_aware_split=True,
):
    """Main preprocessing pipeline: split, augment (if needed), and preprocess all splits.

    split_seed/force_split/group_aware_split are forwarded to split_dataset():
    the split is only reshuffled when its manifest doesn't already match
    (seed/ratios/source data/method), or when force_split is True. See
    GripFusionNet_split.py::split_dataset -- group_aware_split=False reproduces
    the legacy per-frame shuffle (comparison only; leaks grasp events).
    """
    output_path = os.path.join(root_dir, "Data_split/")

    # ---- Stage 1: Dataset splitting ----
    print("Splitting dataset into train/val/test")
    split_dataset(root_dir, output_path, seed=split_seed, force=force_split, group_aware=group_aware_split)

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
    # Translation (y/z) stats are computed from the train split only, then reused
    # to normalize val/test as well, so normalization never leaks split-specific
    # statistics (same convention as the RGB/digit image stats above).
    trans_norm_stats = compute_translation_norm_stats(train_split_dir, transf_matrix)

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
            trans_norm_stats=trans_norm_stats,
        )

    _save_normalization_stats(output_path, digit_mean, digit_std, rgb_mean, rgb_std, trans_norm_stats)

    return trans_norm_stats


def ensure_test_split_current(cfg, data_root, log=print):
    """Make sure <data_root>/Data_split/test/test_preprocessing matches this run's
    configured split seed/ratios/source data before it's read for evaluation,
    rebuilding it (via preprocess_data) if it doesn't or is missing.

    Shared by scripts/test.py and scripts/test_multiple_runs.py so evaluation
    always reads the split that corresponds to the run's own training-time seed
    (data.split_seed, falling back to training.seed), not whatever happens to
    already be on disk at data_root.
    """
    from data_processing.GripFusionNet_split import split_manifest_matches
    from utils.utils import resolve_split_seed, resolve_group_aware_split

    split_root = os.path.join(data_root, "Data_split")
    split_seed = resolve_split_seed(cfg)
    group_aware_split = resolve_group_aware_split(cfg)
    test_cache_dir = os.path.join(split_root, "test", "test_preprocessing")

    split_matches = split_manifest_matches(
        data_root, split_root, seed=split_seed, group_aware=group_aware_split
    )
    cache_exists = os.path.exists(test_cache_dir)

    if split_matches and cache_exists:
        return split_seed

    reason = []
    if not split_matches:
        reason.append("split manifest mismatch")
    if not cache_exists:
        reason.append("test preprocessing cache missing")
    log(f"Preparing split at {data_root} (seed={split_seed}, reason={', '.join(reason)})")
    preprocess_data(
        root_dir=data_root, split_seed=split_seed, force_split=False, group_aware_split=group_aware_split
    )
    return split_seed
