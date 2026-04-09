# --- dataloader_preprocessed.py ---
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

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
    num_workers=2
):
    train_set = PreprocessedTactileDataset(os.path.join(root_dir, 'train_preprocessed'))
    val_set = PreprocessedTactileDataset(os.path.join(root_dir, 'val_preprocessed'))
    test_set = PreprocessedTactileDataset(os.path.join(root_dir, 'test_preprocessed'))

    loader_args = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=tactile_collate_fn
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, **loader_args)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **loader_args)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, **loader_args)

    return train_loader, val_loader, test_loader
