import os
import json
import yaml
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import argparse


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_transforms(image_size, augment=False):
    base_transforms = [
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ]

    if augment:
        augment_transforms = [
            T.RandomHorizontalFlip(),
            T.RandomCrop(image_size, padding=4),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        ]
        return T.Compose(augment_transforms + base_transforms)

    return T.Compose(base_transforms)


def preprocess_sample(sample, transforms_dict, output_dir, cache_poses):
    timestamp = sample["json"].stem
    processed_data = {}

    # Process images
    for stream, path in sample.items():
        if stream == "json":
            continue
        img = Image.open(path).convert("RGB")
        processed_data[stream] = transforms_dict[stream](img)

    # Cache pose
    if cache_poses:
        with open(sample["json"], "r") as f:
            pose_data = json.load(f)
        processed_data["pose"] = pose_data

    # Save preprocessed sample
    sample_path = output_dir / f"{timestamp}.pt"
    torch.save(processed_data, sample_path)


def build_index(root, streams):
    samples = []
    for json_file in root.glob("*.json"):
        timestamp = json_file.stem
        sample = {"json": json_file}
        valid = True

        for stream in streams:
            if stream == "mono_cam":
                path = root / f"{timestamp}.png"
            elif stream == "tactile":
                path = root / f"{timestamp}_digit.png"
            elif stream == "depth_cam":
                path = root / f"{timestamp}_depth.png"
            else:
                path = root / f"{timestamp}_{stream}.png"

            if not path.exists():
                valid = False
                break
            sample[stream] = path

        if valid:
            samples.append(sample)

    return samples


def main(config_path, output_base_dir, augment=False, cache_poses=True):
    cfg = load_config(config_path)
    root = Path(cfg["data_path"])
    streams = cfg["streams"]
    image_size = cfg["image_size"]

    splits = ["train", "val", "test"]

    # Get transforms for each stream
    transforms_dict = {stream: get_transforms(image_size, augment) for stream in streams}

    for split in splits:
        split_root = root / split
        if not split_root.exists():
            print(f"Split folder {split_root} does not exist, skipping.")
            continue

        output_dir = Path(output_base_dir) / f"{split}_preprocessed"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build sample index for this split
        samples = build_index(split_root, streams)
        print(f"Found {len(samples)} valid samples in {split}")

        # Preprocess each sample
        for i, sample in enumerate(samples):
            preprocess_sample(sample, transforms_dict, output_dir, cache_poses)
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(samples)} samples in {split}")

        print(f"Preprocessing complete for {split}. Saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess dataset for GripFusionNet")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--output", type=str, required=True, help="Base output directory for preprocessed data (will create train_preprocessed, val_preprocessed, test_preprocessed)")
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation")
    parser.add_argument("--no-cache", action="store_true", help="Disable JSON pose caching")

    args = parser.parse_args()
    main(args.config, args.output, args.augment, not args.no_cache)