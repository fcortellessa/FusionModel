#!/usr/bin/env python3
"""
Generate augmented data for tactile augmentation ablation study.
Creates variants with each parameter individually enabled/disabled.
Run once to generate all augmented datasets, then train with different configs.
"""

import os
import sys
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.insert(0, repo_root)

from data_processing.augment_data import augment_offline

# Define ablation variants
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


def generate_ablation_variants(
    input_dir: str = "Data/Data_split/train",
    output_base: str = "Data/Data_split",
    seed: int = 42,
    photo_count: int = 1,
    copy_originals: bool = True,
    overwrite: bool = True,
):
    """Generate augmented data for each ablation variant."""
    
    input_path = Path(input_dir)
    base_path = Path(output_base)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    print(f"\n{'='*80}")
    print(f"Tactile Augmentation Ablation Study")
    print(f"{'='*80}")
    print(f"Input: {input_path}")
    print(f"Seed: {seed}")
    print(f"Photo count per sample: {photo_count}")
    print(f"Copy originals: {copy_originals}")
    print(f"Variants to generate: {len(ABLATION_VARIANTS)}")
    print(f"{'='*80}\n")
    
    for variant_name, variant_config in ABLATION_VARIANTS.items():
        output_dir = base_path / f"train_augmentation_{variant_name}"
        
        print(f"\nGenerating variant: {variant_name}")
        print(f"  Config: {variant_config}")
        print(f"  Output: {output_dir}")
        
        augment_offline(
            input_dir=input_path,
            output_dir=output_dir,
            seed=seed,
            augmentation_count=photo_count,
            copy_originals=copy_originals,
            overwrite=overwrite,
            dry_run=False,
            strict=False,
            tactile_aug_config=variant_config,
        )
        
        print(f"  ✓ Completed: {output_dir}")
    
    print(f"\n{'='*80}")
    print("All variants generated successfully!")
    print(f"{'='*80}\n")
    
    # Print summary instructions
    print("Next steps:")
    print("1. Create config files for each variant in configs/")
    print("2. For each ablation, modify train.py to use the corresponding augmented folder")
    print("3. Or update preprocess_data() to support variant selection\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate tactile ablation augmentation variants")
    parser.add_argument("--input", type=str, default="Data/Data_split/train", help="Input folder")
    parser.add_argument("--output", type=str, default="Data/Data_split", help="Output base folder")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--photo-count", type=int, default=1, help="Augmentation variants per sample")
    parser.add_argument("--no-copy-originals", action="store_true", help="Don't include originals")
    parser.add_argument("--no-overwrite", action="store_true", help="Don't overwrite existing")
    
    args = parser.parse_args()
    
    generate_ablation_variants(
        input_dir=args.input,
        output_base=args.output,
        seed=args.seed,
        photo_count=args.photo_count,
        copy_originals=not args.no_copy_originals,
        overwrite=not args.no_overwrite,
    )
