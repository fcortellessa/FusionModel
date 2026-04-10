import os
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


class PoseDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.root = Path(cfg["root"])
        self.image_size = cfg.get("image_size", 640)
        self.streams = tuple(cfg.get("streams", ("mono_cam", "tactile")))

        # Preprocessing transforms
        self.rgb_tf = T.Compose([
            T.Resize((self.image_size, self.image_size)),   # Resize to match square YOLO input size
            T.ToTensor(),   # Convert to [0, 1]
        ])

        # Build index
        self.samples = self._build_index(self.streams)

    def _build_index(self, streams=("mono_cam", "tactile")):
        """
        Scans the root directory for samples and builds an index.
        Each sample consists of a JSON file (pose) and one image per stream.
        JSON pose files are expected to be named {timestamp}.json."""
        samples = []

        for json_file in self.root.glob("*.json"):
            timestamp = json_file.stem
            sample = {"json": json_file}

            for stream in streams:
                if stream == "mono_cam":
                    path = self.root / f"{timestamp}.png"
                elif stream == "tactile":
                    path = self.root / f"{timestamp}_digit.png"
                elif stream == "depth_cam":
                    path = self.root / f"{timestamp}_depth.png"
                else:
                    path = self.root / f"{timestamp}_{stream}.png"

                sample[stream] = path

            if any(not sample[stream].exists() for stream in streams):
                continue

            samples.append(sample)

        print(f"Loaded {len(samples)} samples")
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_pose_label(self, json_path):
        with open(json_path, "r") as f:
            data = json.load(f)

        y = data["position"]["y"]
        z = data["position"]["z"]

        q = data["orientation"]
        quat = torch.tensor([q["x"], q["y"], q["z"], q["w"]], dtype=torch.float32)

        return torch.tensor([y, z], dtype=torch.float32), quat

    def __getitem__(self, idx):
        """
         Each sample consists of:
         - A JSON file containing the pose label (translation in y, z and rotation as quaternion)
         - An RGB image from the monocular camera (named {timestamp}.png)
         - An RGB image from the tactile sensor (named {timestamp}_digit.png)
         - (Optional) A depth image from the depth camera (named {timestamp}_depth.png)
         The __getitem__ method loads the images, applies transformations, and returns a dictionary containing the processed images and the pose label."""
        sample = self.samples[idx]

        # Load images
        rgb = Image.open(sample["rgb"]).convert("RGB")
        tactile = Image.open(sample["tactile"]).convert("RGB")

        rgb = self.rgb_tf(rgb)
        tactile = self.tactile_tf(tactile)

        # Load pose
        (y, z), quat = self._load_pose(sample["json"])

        return {
            "mono_cam": rgb,
            "tactile": tactile,
            "pose": torch.cat([y.unsqueeze(0), z.unsqueeze(0), quat])
        }
    

"""
Inside training loop:
for batch in loader:
    inputs = {
        "mono_cam": batch["mono_cam"].to(device),
        "tactile": batch["tactile"].to(device),
    }

    pose = batch["pose"].to(device)

    y, z, q = model(inputs)

    
ToDo:)
Normalise images (important for YOLO)
Add augmentation
Cache JSON (speed boost)

T.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
"""