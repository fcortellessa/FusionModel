
import os
import sys
import shutil
from pathlib import Path

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from data_processing.GripFusionNet_split import split_dataset
from data_processing.GripFusionNet_preprocess_data import preprocess_and_save
from data_processing.augment_data import augment_offline


def preprocess_data(
    root_dir,
    use_photometric_augmentation=False,
    aug_seed=42,
    aug_photo_count=1,
    aug_overwrite=False,
):
    output_path = os.path.join(root_dir,"Data_split/")
    split_dataset(root_dir, output_path)

    train_split_dir = os.path.join(output_path, "train")
    train_source_dir = train_split_dir

    if use_photometric_augmentation:
        train_augmented_dir = os.path.join(output_path, "train_augmented")
        if os.path.exists(train_augmented_dir):
            shutil.rmtree(train_augmented_dir)
        augment_offline(
            input_dir=Path(train_split_dir),
            output_dir=Path(train_augmented_dir),
            seed=int(aug_seed),
            geo_count=0,
            photo_count=int(aug_photo_count),
            rotation_deg=0.0,
            copy_originals=True,
            overwrite=bool(aug_overwrite),
            dry_run=False,
            strict=False,
        )
        train_source_dir = train_augmented_dir

    transf_matrix = [
    [1.0, 0.0, 0.0, 0.456577],
    [0.0, 1.0, 0.0, -0.163301],
    [0.0, 0.0, 1.0, -0.024417],
    [0.0, 0.0, 0.0, 1.0]
    ]
    split_input_dirs = {
        "train": train_source_dir,
        "val": os.path.join(output_path, "val"),
        "test": os.path.join(output_path, "test"),
    }

    for split in ["train", "val", "test"]:
        split_output_dir = os.path.join(output_path, f"{split}_preprocessed")
        if os.path.exists(split_output_dir):
            shutil.rmtree(split_output_dir)
        preprocess_and_save(root_dir, transf_matrix,
            folder_path=split_input_dirs[split],
            output_dir=split_output_dir
        )
        