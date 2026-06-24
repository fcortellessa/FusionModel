import argparse
import hashlib
import json
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import albumentations as A
import numpy as np
from PIL import Image
import yaml
from torch.utils.data import Dataset


BRIGHTNESS_ONLY_TACTILE_AUG_CONFIG = {
    "enabled": True,
    "output_folder": "train_augmentation_brightness_only",
    "enabled_noise": False,
    "enabled_brightness": True,
    "enabled_contrast": False,
    "enabled_blur": False,
    "enabled_channel_scaling": False,
    "seed": 42,
    "photo_count": 1,
    "overwrite": False,
}


@dataclass(frozen=True)
class Triplet:
    timestamp: str
    rgb_path: Path
    digit_path: Path
    json_path: Path


def list_triplets(input_dir: Path) -> List[Triplet]:
    triplets: List[Triplet] = []
    for json_path in sorted(input_dir.glob("*.json")):
        ts = json_path.stem
        rgb_path = input_dir / f"{ts}.png"
        digit_path = input_dir / f"{ts}_digit.png"
        if rgb_path.exists() and digit_path.exists():
            triplets.append(Triplet(ts, rgb_path, digit_path, json_path))
    return triplets


class OnlineAugmentedDataset(Dataset):
    """
    An online data loading pipeline that applies deterministic albumentation 
    transforms per sample using a cryptographic hashing scheme.
    """
    def __init__(
        self,
        input_dir: str | Path,
        seed: int,
        tactile_aug_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.input_dir = Path(input_dir)
        self.seed = seed
        self.tactile_aug_config = tactile_aug_config if tactile_aug_config is not None else {}
        
        # Discover and inventory available raw data triplets
        self.triplets = list_triplets(self.input_dir)
        if not self.triplets:
            raise ValueError(f"No valid triplets found in {self.input_dir}")
            
        # Build the static Albumentations compositions up front
        self.rgb_transform = self._build_rgb_transform()
        self.tactile_transform = self._build_tactile_transform()

    def _build_rgb_transform(self) -> A.Compose:
        """RGB-specific photometric augmentation pipeline."""
        return A.Compose(
            [
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.MotionBlur(blur_limit=7, p=1.0),
                ], p=0.2),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=0, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            ],
            p=1.0,
        )

    def _build_tactile_transform(self) -> A.Compose:
        """Tactile-specific photometric augmentation based on configurations."""
        enabled = {
            k: self.tactile_aug_config.get(k, True) 
            for k in ["enabled_noise", "enabled_brightness", "enabled_contrast", "enabled_blur", "enabled_channel_scaling"]
        }

        transforms = []
        if enabled["enabled_noise"]:
            transforms.append(A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)) 

        if enabled["enabled_brightness"] or enabled["enabled_contrast"]:
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=(-0.2, 0.2) if enabled["enabled_brightness"] else 0.0,
                contrast_limit=(-0.2, 0.2) if enabled["enabled_contrast"] else 0.0,
                p=1.0
            ))

        if enabled["enabled_blur"]:
            transforms.append(A.MotionBlur(blur_limit=7, p=1.0))

        if enabled["enabled_channel_scaling"]:
            transforms.append(A.RGBShift(
                r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0
            ))

        return A.Compose(transforms, p=1.0) if transforms else A.Compose([A.NoOp()], p=1.0)

    def _get_local_rng(self, timestamp: str) -> np.random.Generator:
        """
        Creates a dedicated random number generator for a specific sample.
        Ensures perfect reproducibility across data loaders.
        """
        token = f"{self.seed}|{timestamp}|online_augm|0"
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        local_seed = int.from_bytes(digest, byteorder="little", signed=False) % (2**32 - 1)
        return np.random.default_rng(local_seed)

    def _apply_deterministic(
        self, 
        img: Image.Image, 
        transform: A.Compose, 
        rng: np.random.Generator
    ) -> Image.Image:
        """Applies albumentation to images deterministically using state hijacking."""
        seed = int(rng.integers(0, 2**32 - 1))
        
        # Save pre-existing dynamic global states
        np_state = np.random.get_state()
        py_state = random.getstate()
        
        # Seed local context execution
        np.random.seed(seed)
        random.seed(seed)
        try:
            result = transform(image=np.asarray(img))
        finally:
            # Revert global state tracking safely
            np.random.set_state(np_state)
            random.setstate(py_state)
            
        out_arr = result["image"]
        return Image.fromarray(np.clip(out_arr, 0, 255).astype(np.uint8))

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, Image.Image, Dict, Triplet]:
        """
        Retrieves files, applies deterministic online processing pipeline,
        and returns the augmented structures.
        """
        triplet = self.triplets[idx]
        
        # 1. Load Images
        rgb_img = Image.open(triplet.rgb_path).convert("RGB")
        digit_img = Image.open(triplet.digit_path).convert("RGB")
        
        # 2. Parse associated labels/meta
        with open(triplet.json_path, "r", encoding="utf-8") as f:
            label_data = json.load(f)

        # 3. Secure deterministic local random number generation object
        local_rng = self._get_local_rng(triplet.timestamp)

        # 4. Transform images online
        rgb_augmented = self._apply_deterministic(rgb_img, self.rgb_transform, local_rng)
        digit_augmented = self._apply_deterministic(digit_img, self.tactile_transform, local_rng)

        return rgb_augmented, digit_augmented, label_data, triplet


def build_brightness_only_training_config(seed: int, template_config: Dict) -> Dict:
    """Return a training config adjusted for online brightness-only augmentation."""
    config = deepcopy(template_config)
    data_cfg = config.setdefault("data", {})
    data_cfg["use_preaugmented_train_dir"] = False
    data_cfg["root"] = ""

    augmentation_cfg = deepcopy(BRIGHTNESS_ONLY_TACTILE_AUG_CONFIG)
    augmentation_cfg["seed"] = int(seed)
    data_cfg["augmentation"] = augmentation_cfg

    training_cfg = config.setdefault("training", {})
    training_cfg["seed"] = int(seed)
    return config


def write_seeded_online_configs(
    template_config_path: Path,
    output_dir: Path,
    seed_start: int,
    seed_end: int,
) -> List[Path]:
    """Write one brightness-only online config per seed in the inclusive range."""
    if seed_end < seed_start:
        raise ValueError("seed_end must be greater than or equal to seed_start")

    with template_config_path.open("r", encoding="utf-8") as handle:
        template_config = yaml.safe_load(handle)

    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: List[Path] = []

    for seed in range(seed_start, seed_end + 1):
        seeded_config = build_brightness_only_training_config(seed, template_config)
        output_path = output_dir / f"tactile_ablation_brightness_only_seed{seed}.yml"
        with output_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(seeded_config, handle, sort_keys=False)
        written_paths.append(output_path)

    return written_paths


def _parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate online brightness-only ablation configs")
    parser.add_argument(
        "--template-config",
        type=Path,
        default=Path("configs/tactile_ablation_brightness_only.yml"),
        help="Base training config to copy and adjust",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("configs/generated/online_brightness_only"),
        help="Directory where seed-specific configs will be written",
    )
    parser.add_argument("--seed-start", type=int, default=40, help="Inclusive start seed")
    parser.add_argument("--seed-end", type=int, default=46, help="Inclusive end seed")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli()
    written = write_seeded_online_configs(
        template_config_path=args.template_config,
        output_dir=args.output_dir,
        seed_start=args.seed_start,
        seed_end=args.seed_end,
    )
    print("Wrote the following online brightness-only configs:")
    for path in written:
        print(path)