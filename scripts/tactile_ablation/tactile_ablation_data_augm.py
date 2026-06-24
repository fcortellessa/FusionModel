#!/usr/bin/env python3
# tactile_ablation_data_augm.py - Generate tactile ablation augmentation datasets expected by training scripts.
"""
Generate tactile ablation augmentation datasets expected by training scripts.

Creates the following folders under Data/Data_split by default:
    - train_augmentation_no_aug
    - train_augmentation_noise_only
    - train_augmentation_brightness_only
    - train_augmentation_contrast_only
    - train_augmentation_blur_only
    - train_augmentation_channel_scaling_only
    - train_augmentation_all
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_processing.augment_data import augment_offline


ABLATION_VARIANTS = {
    "no_aug": {
        "enabled_noise": False,
        "enabled_brightness": False,
        "enabled_contrast": False,
        "enabled_blur": False,
        "enabled_channel_scaling": False,
    },
    "noise_only": {
        "enabled_noise": True,
        "enabled_brightness": False,
        "enabled_contrast": False,
        "enabled_blur": False,
        "enabled_channel_scaling": False,
    },
    "brightness_only": {
        "enabled_noise": False,
        "enabled_brightness": True,
        "enabled_contrast": False,
        "enabled_blur": False,
        "enabled_channel_scaling": False,
    },
    "contrast_only": {
        "enabled_noise": False,
        "enabled_brightness": False,
        "enabled_contrast": True,
        "enabled_blur": False,
        "enabled_channel_scaling": False,
    },
    "blur_only": {
        "enabled_noise": False,
        "enabled_brightness": False,
        "enabled_contrast": False,
        "enabled_blur": True,
        "enabled_channel_scaling": False,
    },
    "channel_scaling_only": {
        "enabled_noise": False,
        "enabled_brightness": False,
        "enabled_contrast": False,
        "enabled_blur": False,
        "enabled_channel_scaling": True,
    },
    "all": {
        "enabled_noise": True,
        "enabled_brightness": True,
        "enabled_contrast": True,
        "enabled_blur": True,
        "enabled_channel_scaling": True,
    },
}


def _resolve_seed_values(seed: int, seed_start: int | None, seed_end: int | None) -> list[int]:
    if seed_start is None and seed_end is None:
        return [seed]
    if seed_start is None or seed_end is None:
        raise ValueError("Provide both --seed-start and --seed-end, or neither.")
    if seed_end < seed_start:
        raise ValueError("--seed-end must be greater than or equal to --seed-start.")
    return list(range(seed_start, seed_end + 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate tactile ablation datasets with expected folder naming"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("Data/Data_split/train"),
        help="Input train split folder",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("Data/Data_split"),
        help="Base output folder where train_augmentation_* folders are created",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic augmentation",
    )
    parser.add_argument(
        "--augmentation-count",
        type=int,
        default=1,
        help="Number of augmented copies per source sample",
    )
    parser.add_argument(
        "--no-copy-originals",
        action="store_true",
        help="Do not copy originals into each output variant folder",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files if output folders already contain previous runs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing files",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on first problematic sample",
    )
    parser.add_argument(
        "--variant",
        choices=list(ABLATION_VARIANTS.keys()),
        default="brightness_only",
        help="Ablation variant to generate",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=None,
        help="Inclusive start of the seed sweep",
    )
    parser.add_argument(
        "--seed-end",
        type=int,
        default=None,
        help="Inclusive end of the seed sweep",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seed_values = _resolve_seed_values(args.seed, args.seed_start, args.seed_end)

    if not args.input.exists():
        raise FileNotFoundError(f"Input folder does not exist: {args.input}")

    print("=" * 80)
    print("Generating tactile ablation datasets")
    print(f"Input: {args.input}")
    print(f"Output base: {args.output_base}")
    print(f"Variant: {args.variant}")
    print(f"Seeds: {', '.join(str(value) for value in seed_values)}")
    print("=" * 80)

    tactile_aug_config = ABLATION_VARIANTS[args.variant]

    for current_seed in seed_values:
        output_suffix = args.variant if len(seed_values) == 1 else f"{args.variant}_seed{current_seed}"
        out_dir = args.output_base / f"train_augmentation_{output_suffix}"
        print(f"\n[Variant: {args.variant}] seed={current_seed} -> {out_dir}")

        augment_offline(
            input_dir=args.input,
            output_dir=out_dir,
            seed=current_seed,
            augmentation_count=args.augmentation_count,
            copy_originals=not args.no_copy_originals,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            strict=args.strict,
            tactile_aug_config=tactile_aug_config,
        )

    print("\nDone. Tactile ablation datasets are ready.")


if __name__ == "__main__":
    main()
