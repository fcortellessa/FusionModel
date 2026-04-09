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
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# ----------------------------
# SETTINGS
# ----------------------------
DEBUG = True          # 🔥 Set True to disable multiprocessing (recommended first run)
NUM_WORKERS = 4        # safer than cpu_count()

# ----------------------------
# PATH SETUP
# ----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from data_processing.GripFusionNet_get_norm_params import (
    compute_mean_std_digit_images,
    compute_mean_std_rgb_images_no_crop
)

# ----------------------------
# WORKER FUNCTION
# ----------------------------
def process_sample(ts, folder_path, output_dir,
                   transf_matrix, rgb_mean, rgb_std, digit_mean, digit_std):

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

        # ---- Transforms ----
        letterbox = LetterBox(new_shape=(640, 640), auto=False,
                              scale_fill=False, scaleup=False,
                              center=True, stride=32)

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

        # ---- RGB ----
        rgb_img = cv2.imread(rgb_path)
        if rgb_img is None:
            raise ValueError(f"cv2.imread failed for {rgb_path}")

        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = letterbox(image=rgb_img)
        rgb_img = Image.fromarray(rgb_img).convert("RGB")
        rgb_tensor = rgb_transform(rgb_img)

        # ---- Digit ----
        digit_img = cv2.imread(digit_path)
        if digit_img is None:
            raise ValueError(f"cv2.imread failed for {digit_path}")

        digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2RGB)
        digit_img = letterbox(image=digit_img)
        digit_img = Image.fromarray(digit_img).convert("RGB")
        digit_tensor = digit_transform(digit_img)

        # ---- Pose ----
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

        # ---- Save ----
        torch.save({
            "rgb": rgb_tensor,
            "digit": digit_tensor,
            "pose": pose_vec
        }, os.path.join(output_dir, f"{ts}.pt"))

        return ts  # helps debugging

    except Exception as e:
        print(f"\n❌ ERROR processing {ts}: {e}", flush=True)
        raise  # 🔥 DO NOT REMOVE


# ----------------------------
# MAIN PREPROCESS
# ----------------------------
def preprocess_and_save(data_path, transf_matrix, folder_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    print("Computing normalization stats...", flush=True)

    digit_mean, digit_std = compute_mean_std_digit_images(
        os.path.join(data_path, "Data_split/train/")
    )
    rgb_mean, rgb_std = compute_mean_std_rgb_images_no_crop(
        os.path.join(data_path, "Data_split/train/")
    )

    # ---- Collect valid timestamps ----
    all_files = os.listdir(folder_path)

    rgb_ts = {f.replace(".png", "") for f in all_files
              if f.endswith(".png") and "_digit" not in f}
    digit_ts = {f.replace("_digit.png", "") for f in all_files
                if f.endswith("_digit.png")}
    pose_ts = {f.replace(".json", "") for f in all_files
               if f.endswith(".json")}

    valid_timestamps = sorted(rgb_ts & digit_ts & pose_ts)

    print(f"Found {len(valid_timestamps)} samples", flush=True)

    # ----------------------------
    # DEBUG MODE (NO MULTIPROCESSING)
    # ----------------------------
    if DEBUG:
        for ts in tqdm(valid_timestamps, desc="DEBUG preprocessing"):
            process_sample(
                ts, folder_path, output_dir,
                transf_matrix, rgb_mean, rgb_std, digit_mean, digit_std
            )
        return

    # ----------------------------
    # MULTIPROCESSING
    # ----------------------------
    with Pool(NUM_WORKERS) as pool:
        process = partial(
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
            pool.imap_unordered(process, valid_timestamps, chunksize=1),
            total=len(valid_timestamps),
            desc=f"Preprocessing {os.path.basename(folder_path)}"
        ))


# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == "__main__":

    transf_matrix = [
        [1.0, 0.0, 0.0, 0.456577],
        [0.0, 1.0, 0.0, -0.163301],
        [0.0, 0.0, 1.0, -0.024417],
        [0.0, 0.0, 0.0, 1.0]
    ]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "../Data")

    for split in ["train", "val", "test"]:
        preprocess_and_save(
            data_path,
            transf_matrix,
            folder_path=os.path.join(data_path, "Data_split", split),
            output_dir=os.path.join(data_path, "Data_split", f"{split}_preprocessed")
        )