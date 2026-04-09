# --- preprocess_and_save.py ---
import os
import json
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from ultralytics.data.augment import LetterBox
import cv2
from scipy.spatial.transform import Rotation as R
import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)


from utils.GripFusionNet_utils import load_config
from data_processing.GripFusionNet_get_norm_params import compute_mean_std_digit_images, compute_mean_std_rgb_images, compute_mean_std_rgb_images_no_crop

def process_sample(ts, folder_path, output_dir, letterbox, rgb_transform, digit_transform, homo_matrix_hub_base):
    """Process a single timestamped sample and save it as a .pt tensor file."""
    try:
        # File paths
        rgb_path = os.path.join(folder_path, f"{ts}.png")
        digit_path = os.path.join(folder_path, f"{ts}_digit.png")
        pose_path = os.path.join(folder_path, f"{ts}.json")

        # ----- RGB preprocessing -----
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = letterbox(image=rgb_img)
        rgb_img = Image.fromarray(rgb_img).convert("RGB")
        rgb_tensor = rgb_transform(rgb_img)

        # ----- Digit preprocessing -----
        digit_img = cv2.imread(digit_path)
        digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2RGB)
        digit_img = letterbox(image=digit_img)
        digit_img = Image.fromarray(digit_img).convert("RGB")
        digit_tensor = digit_transform(digit_img)

        # ----- Pose preprocessing -----
        with open(pose_path, 'r') as f:
            data = json.load(f)
        pos = np.array([data["position"][k] for k in ["x", "y", "z"]])
        quat = np.array([data["orientation"][k] for k in ["x", "y", "z", "w"]])
        rot = R.from_quat(quat).as_matrix()

        homo_tcp = np.eye(4)
        homo_tcp[:3, :3] = rot
        homo_tcp[:3, 3] = pos

        hub_pose = np.linalg.inv(homo_tcp) @ homo_matrix_hub_base
        pose_vec = torch.tensor(
            np.concatenate([hub_pose[:3, 3], R.from_matrix(hub_pose[:3, :3]).as_quat()]),
            dtype=torch.float32
        )

        # Save all tensors to disk
        torch.save({
            "rgb": rgb_tensor,
            "digit": digit_tensor,
            "pose": pose_vec
        }, os.path.join(output_dir, f"{ts}.pt"))

    except Exception as e:
        print(f"⚠️ Error processing {ts}: {e}")


def preprocess_and_save(data_path, transf_matrix, folder_path, output_dir):
    """
    Preprocess all valid triplets (RGB, Digit, Pose) in a dataset folder
    and save them as .pt files for fast training.
    """
    os.makedirs(output_dir, exist_ok=True)

    # ROI cropping and letterbox resizing (not used, since data is expected in the correct ROI shape)
    #roi = (1800, 1600, 2250 - 1800, 3000 - 1600)
    letterbox = LetterBox(new_shape=(640, 640), auto=False, scale_fill=False, scaleup=False, center=True, stride=32)

    # Compute stats separately for Digit & RGB
    digit_mean, digit_std = compute_mean_std_digit_images(os.path.join(data_path,"Data_split/train/"))
    rgb_mean, rgb_std = compute_mean_std_rgb_images_no_crop(os.path.join(data_path,"Data_split/train/"))

    rgb_transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=rgb_mean, std=rgb_std)
    ])
    digit_transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=digit_mean, std=digit_std)
    ])

    # Base hub transformation
    homo_matrix_hub_base = np.array(transf_matrix)

    # Collect valid samples = intersection of rgb, digit, and pose files
    all_files = os.listdir(folder_path)
    rgb_timestamps = {f.replace(".png", "") for f in all_files if f.endswith(".png") and "_digit" not in f}
    digit_timestamps = {f.replace("_digit.png", "") for f in all_files if f.endswith("_digit.png")}
    pose_timestamps = {f.replace(".json", "") for f in all_files if f.endswith(".json")}
    valid_timestamps = sorted(rgb_timestamps & digit_timestamps & pose_timestamps)
    valid_timestamps = [ts for ts in valid_timestamps]

    # Parallel preprocessing across CPU cores
    with Pool(cpu_count()) as pool:
        process = partial(
            process_sample,
            folder_path=folder_path,
            output_dir=output_dir,
            letterbox=letterbox,
            rgb_transform=rgb_transform,
            digit_transform=digit_transform,
            homo_matrix_hub_base=homo_matrix_hub_base
        )
        list(tqdm(
            pool.imap_unordered(process, valid_timestamps),
            total=len(valid_timestamps),
            desc=f"Preprocessing {os.path.basename(folder_path)}"
        ))


# ----------------------------
# Script Entry Point
# ----------------------------
if __name__ == "__main__":
    # config_path = r"GripFusionNet\Config\config.yaml"
    # config = load_config(config_path)

    # root = os.path.join(config["data_path"],"Data_split/")
    # for split in ["train_augmented", "val", "test"]:
        # preprocess_and_save(config["data_path"], config["homo_matrix_hub_base"]
        #     folder_path=os.path.join(root, split),
        #     output_dir=os.path.join(root, f"{split}_preprocessed")
        # )

    transf_matrix = [
    [1.0, 0.0, 0.0, 0.456577],
    [0.0, 1.0, 0.0, -0.163301],
    [0.0, 0.0, 1.0, -0.024417],
    [0.0, 0.0, 0.0, 1.0]
    ]
    data_path = "D:/MA/GripFusionNet/GripFusionNet/dummy_data"
    root = os.path.join(data_path,"Data_split/")

    for split in ["train", "val", "test"]:
        preprocess_and_save(data_path, transf_matrix,
            folder_path=os.path.join(root, split),
            output_dir=os.path.join(root, f"{split}_preprocessed")
        )