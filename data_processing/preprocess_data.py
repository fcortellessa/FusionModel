
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from data_processing.GripFusionNet_split import split_dataset
from data_processing.GripFusionNet_preprocess_data import preprocess_and_save


def preprocess_data(root_dir):
    output_path = os.path.join(root_dir,"Data_split/")
    split_dataset(root_dir, output_path)

    transf_matrix = [
    [1.0, 0.0, 0.0, 0.456577],
    [0.0, 1.0, 0.0, -0.163301],
    [0.0, 0.0, 1.0, -0.024417],
    [0.0, 0.0, 0.0, 1.0]
    ]
    for split in ["train", "val", "test"]:
        preprocess_and_save(root_dir, transf_matrix,
            folder_path=os.path.join(output_path, split),
            output_dir=os.path.join(output_path, f"{split}_preprocessed")
        )
        