# --- dataloader_preprocessed.py ---
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from data_processing.GripFusionNet_get_norm_params import (
    compute_mean_std_digit_images,
    compute_mean_std_rgb_images_no_crop,
)
from data_processing.augment_data import (
    _apply_rgb_augmentation,
    _apply_tactile_augmentation,
    _rng_for,
)
from data_processing.preprocess_data import _hub_frame_pose

# Hub-frame calibration matrix; must match data_processing/preprocess_data.py's
# transf_matrix so online-augmented samples land in the same frame as the cached
# (non-augmented) preprocessing path.
_HUB_TRANSF_MATRIX = [
    [1.0, 0.0, 0.0, 0.456577],
    [0.0, 1.0, 0.0, -0.163301],
    [0.0, 0.0, 1.0, -0.024417],
    [0.0, 0.0, 0.0, 1.0]
]


def _pose_json_to_tensor(json_path, normalize_translation=True, trans_norm_stats=None):
    pose_vec = _hub_frame_pose(json_path, _HUB_TRANSF_MATRIX)

    if normalize_translation:
        stats = trans_norm_stats or {}
        mean_y = stats.get("trans_mean_y", 0.0)
        std_y = stats.get("trans_std_y", 1.0)
        mean_z = stats.get("trans_mean_z", 0.0)
        std_z = stats.get("trans_std_z", 1.0)
        pose_vec[1] = (pose_vec[1] - mean_y) / std_y
        pose_vec[2] = (pose_vec[2] - mean_z) / std_z

    return torch.tensor(pose_vec, dtype=torch.float32)


def _build_raw_samples(folder_path):
    folder = Path(folder_path)
    samples = []
    for json_path in sorted(folder.glob("*.json")):
        timestamp = json_path.stem
        rgb_path = folder / f"{timestamp}.png"
        digit_path = folder / f"{timestamp}_digit.png"
        if rgb_path.exists() and digit_path.exists():
            samples.append(
                {
                    "timestamp": timestamp,
                    "rgb_path": rgb_path,
                    "digit_path": digit_path,
                    "json_path": json_path,
                }
            )
    return samples


_REQUIRED_NORM_STATS_KEYS = (
    "digit_mean", "digit_std", "rgb_mean", "rgb_std",
    "trans_mean_y", "trans_std_y", "trans_mean_z", "trans_std_z",
)


def _load_or_compute_norm_stats(stats_path, train_raw_dir):
    from data_processing.preprocess_data import compute_translation_norm_stats

    stats_file = Path(stats_path)
    if stats_file.exists():
        with open(stats_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        # A stats file written by an older code version (before translation
        # normalization was added) will be missing the trans_* keys -- recompute
        # rather than silently handing back a partial dict that later KeyErrors
        # deep in training.
        if all(key in cached for key in _REQUIRED_NORM_STATS_KEYS):
            return cached

    digit_mean, digit_std = compute_mean_std_digit_images(train_raw_dir)
    rgb_mean, rgb_std = compute_mean_std_rgb_images_no_crop(train_raw_dir)
    trans_stats = compute_translation_norm_stats(train_raw_dir, _HUB_TRANSF_MATRIX)
    stats = {
        "digit_mean": digit_mean,
        "digit_std": digit_std,
        "rgb_mean": rgb_mean,
        "rgb_std": rgb_std,
        "trans_mean_y": trans_stats["mean_y"],
        "trans_std_y": trans_stats["std_y"],
        "trans_mean_z": trans_stats["mean_z"],
        "trans_std_z": trans_stats["std_z"],
    }
    stats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


class PreprocessedTactileDataset(Dataset):
    def __init__(self, folder_path):
        self.samples = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.pt')]
        if not self.samples:
            raise ValueError(f"No .pt files found in {folder_path}")
        print(f"[INFO] Loaded {len(self.samples)} preprocessed samples from {folder_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = torch.load(self.samples[idx])
        return data["rgb"], data["digit"], data["pose"]


class RawOnlineAugmentedTactileDataset(Dataset):
    def __init__(self, folder_path, image_size=640, augmentation_cfg=None, norm_stats=None, normalize_translation=True):
        self.folder_path = folder_path
        self.samples = _build_raw_samples(folder_path)
        if not self.samples:
            raise ValueError(f"No raw samples found in {folder_path}")

        self.image_size = int(image_size)
        self.augmentation_cfg = augmentation_cfg or {}
        self.normalize_translation = normalize_translation
        self.enable_augmentation = bool(self.augmentation_cfg.get("enabled", False))
        self.photo_count = max(0, int(self.augmentation_cfg.get("photo_count", 1)))
        self.seed = int(self.augmentation_cfg.get("seed", 42))
        self.variants_per_sample = 1 + self.photo_count if self.enable_augmentation else 1

        norm_stats = norm_stats or {}
        rgb_mean = norm_stats.get("rgb_mean", [0.485, 0.456, 0.406])
        rgb_std = norm_stats.get("rgb_std", [0.229, 0.224, 0.225])
        digit_mean = norm_stats.get("digit_mean", [0.5, 0.5, 0.5])
        digit_std = norm_stats.get("digit_std", [0.5, 0.5, 0.5])
        self.trans_norm_stats = norm_stats

        self.rgb_transform = T.Compose(
            [
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=rgb_mean, std=rgb_std),
            ]
        )
        self.digit_transform = T.Compose(
            [
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=digit_mean, std=digit_std),
            ]
        )

        if self.enable_augmentation:
            print(
                f"[INFO] Loaded {len(self.samples)} raw training samples from {folder_path} "
                f"with {self.variants_per_sample} online variants per sample"
            )
        else:
            print(f"[INFO] Loaded {len(self.samples)} raw training samples from {folder_path}")

    def __len__(self):
        return len(self.samples) * self.variants_per_sample

    def __getitem__(self, idx):
        sample_idx, variant_idx = divmod(idx, self.variants_per_sample)
        sample = self.samples[sample_idx]

        rgb_img = Image.open(sample["rgb_path"]).convert("RGB")
        digit_img = Image.open(sample["digit_path"]).convert("RGB")

        if self.enable_augmentation and variant_idx > 0:
            local_rng = _rng_for(self.seed, sample["timestamp"], "online", variant_idx - 1)
            rgb_img, _ = _apply_rgb_augmentation(rgb_img, local_rng)
            digit_img, _ = _apply_tactile_augmentation(digit_img, local_rng, aug_config=self.augmentation_cfg)

        rgb_tensor = self.rgb_transform(rgb_img)
        digit_tensor = self.digit_transform(digit_img)
        pose_tensor = _pose_json_to_tensor(
            sample["json_path"],
            normalize_translation=self.normalize_translation,
            trans_norm_stats=self.trans_norm_stats,
        )

        return rgb_tensor, digit_tensor, pose_tensor

def tactile_collate_fn(batch):
    rgb_images, digit_images, poses = zip(*batch)
    return {
        "rgb_image": torch.stack(rgb_images),
        "digit_image": torch.stack(digit_images),
        "pose": torch.stack(poses),
    }

def get_dataloaders(
    root_dir,
    batch_size=16,
    shuffle=True,
    num_workers=2,
    train_folder="train",
    use_online_augmentation=False,
    augmentation_cfg=None,
    image_size=640,
    normalize_translation=True,
):
    train_raw_dir = os.path.join(root_dir, train_folder)
    train_preprocessed_dir = os.path.join(root_dir, train_folder, "train_preprocessing")
    val_dir = os.path.join(root_dir, "val", "val_preprocessing")
    test_dir = os.path.join(root_dir, "test", "test_preprocessing")

    # normalization_stats.json is written by preprocess_data() (which always runs
    # before this, see DataPipelineManager.get_loaders); loaded here regardless of
    # augmentation mode so the caller can snapshot the translation stats actually
    # used for this split into the run's config.
    stats = _load_or_compute_norm_stats(os.path.join(root_dir, "normalization_stats.json"), train_raw_dir)

    if use_online_augmentation:
        train_set = RawOnlineAugmentedTactileDataset(
            train_raw_dir,
            image_size=image_size,
            augmentation_cfg=augmentation_cfg,
            norm_stats=stats,
            normalize_translation=normalize_translation,
        )
    else:
        train_set = PreprocessedTactileDataset(train_preprocessed_dir)
    val_set = PreprocessedTactileDataset(val_dir)
    test_set = PreprocessedTactileDataset(test_dir)

    loader_args = dict(
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        collate_fn=tactile_collate_fn
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, **loader_args)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **loader_args)

    trans_norm_stats = {
        "mean_y": stats.get("trans_mean_y", 0.0),
        "std_y": stats.get("trans_std_y", 1.0),
        "mean_z": stats.get("trans_mean_z", 0.0),
        "std_z": stats.get("trans_std_z", 1.0),
    }

    return train_loader, val_loader, test_loader, trans_norm_stats

def get_test_dataloader(
    root_dir,
    batch_size=16,
    shuffle=True,
    num_workers=2
):
    test_dir = os.path.join(root_dir, "test", "test_preprocessing")
    test_set = PreprocessedTactileDataset(test_dir)

    loader_args = dict(
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=num_workers > 0,
        collate_fn=tactile_collate_fn
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **loader_args)

    return test_loader
