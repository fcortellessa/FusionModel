# --- dataloader_preprocessed.py ---
import json
import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from utils.normalization import MEAN_Y, STD_Y, MEAN_Z, STD_Z
from data_processing.GripFusionNet_get_norm_params import (
    compute_mean_std_digit_images,
    compute_mean_std_rgb_images_no_crop,
)
from data_processing.augment_data import (
    _apply_rgb_augmentation,
    _apply_tactile_augmentation,
    _rng_for,
)


def _pose_json_to_tensor(json_path, normalize_translation=True):
    with open(json_path, "r", encoding="utf-8") as f:
        pose_data = json.load(f)

    y = float(pose_data["position"]["y"])
    z = float(pose_data["position"]["z"])
    x = float(pose_data["position"]["x"])

    if normalize_translation:
        y = (y - MEAN_Y) / STD_Y
        z = (z - MEAN_Z) / STD_Z

    pos = [x, y, z]
    quat = [float(pose_data["orientation"][k]) for k in ("x", "y", "z", "w")]
    return torch.tensor(pos + quat, dtype=torch.float32)


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


def _load_or_compute_norm_stats(stats_path, train_raw_dir):
    stats_file = Path(stats_path)
    if stats_file.exists():
        with open(stats_file, "r", encoding="utf-8") as f:
            return json.load(f)

    digit_mean, digit_std = compute_mean_std_digit_images(train_raw_dir)
    rgb_mean, rgb_std = compute_mean_std_rgb_images_no_crop(train_raw_dir)
    stats = {
        "digit_mean": digit_mean,
        "digit_std": digit_std,
        "rgb_mean": rgb_mean,
        "rgb_std": rgb_std,
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
        pose_tensor = _pose_json_to_tensor(sample["json_path"], normalize_translation=self.normalize_translation)

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

    if use_online_augmentation:
        stats = _load_or_compute_norm_stats(os.path.join(root_dir, "normalization_stats.json"), train_raw_dir)
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

    return train_loader, val_loader, test_loader

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
