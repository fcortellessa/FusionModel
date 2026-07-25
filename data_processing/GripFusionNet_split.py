import os
import json
import shutil
import random
import hashlib
from datetime import datetime

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)

from utils.GripFusionNet_utils import load_config, get_triplets

DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.05

MANIFEST_FILENAME = "split_manifest.json"

# -------------------------------------------------
# Utility: read list of corrupted/excluded samples
# -------------------------------------------------
def get_excluded_timestamps(exclude_file):
    """
    Read a text file containing corrupted image paths and return their timestamps.
    Only entries ending in '_digit.png' are considered.
    """
    excluded = set()
    if exclude_file and os.path.exists(exclude_file):
        with open(exclude_file, 'r') as f:
            for line in f:
                path = line.strip()
                if path:
                    filename = os.path.basename(path)
                    if filename.endswith('_digit.png'):
                        # remove suffix to extract timestamp
                        timestamp = filename.replace('_digit.png', '')
                        excluded.add(timestamp)
    return excluded



# -------------------------------------------------
# Utility: copy one triplet into a target folder
# -------------------------------------------------
def copy_triplet(triplet, src_folder, dest_folder):
    """
    Copy all three files of a triplet (RGB, digit, JSON) from the source folder
    into the destination folder.
    """
    timestamp, rgb_file, digit_file, json_file = triplet
    os.makedirs(dest_folder, exist_ok=True)
    for fname in [rgb_file, digit_file, json_file]:
        shutil.copy(
            os.path.join(src_folder, fname),
            os.path.join(dest_folder, fname)
        )


# -------------------------------------------------
# Split manifest: makes a given split reproducible and self-describing,
# independent of any other RNG usage (e.g. training.seed) in the process.
# -------------------------------------------------
def _triplets_hash(triplets):
    """Order-independent fingerprint of the source triplet set (detects raw-data drift)."""
    timestamps = sorted(t[0] for t in triplets)
    return hashlib.sha1("|".join(timestamps).encode("utf-8")).hexdigest()


def _manifest_path(output_root):
    return os.path.join(output_root, MANIFEST_FILENAME)


def _load_manifest(output_root):
    path = _manifest_path(output_root)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_manifest(output_root, seed, train_ratio, val_ratio, test_ratio, triplets, split_assignment):
    manifest = {
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "num_triplets": len(triplets),
        "source_timestamps_hash": _triplets_hash(triplets),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "splits": {
            split_name: [t[0] for t in split_triplets]
            for split_name, split_triplets in split_assignment.items()
        },
    }
    os.makedirs(output_root, exist_ok=True)
    with open(_manifest_path(output_root), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def split_manifest_matches(
    input_folder,
    output_root,
    train_ratio=DEFAULT_TRAIN_RATIO,
    val_ratio=DEFAULT_VAL_RATIO,
    test_ratio=DEFAULT_TEST_RATIO,
    seed=None,
):
    """
    Cheaply check whether an existing split already matches the requested
    seed/ratios/source data, without touching any files. Used both internally
    by split_dataset() and by callers (e.g. DataPipelineManager) that want to
    know up front whether a resplit is needed.
    """
    manifest = _load_manifest(output_root)
    if manifest is None:
        return False

    if manifest.get("seed") != seed:
        return False
    if (
        manifest.get("train_ratio") != train_ratio
        or manifest.get("val_ratio") != val_ratio
        or manifest.get("test_ratio") != test_ratio
    ):
        return False

    triplets = get_triplets(input_folder)
    return manifest.get("source_timestamps_hash") == _triplets_hash(triplets)


# -------------------------------------------------
# Main splitting function
# -------------------------------------------------
def split_dataset(
    input_folder,
    output_root,
    train_ratio=DEFAULT_TRAIN_RATIO,
    val_ratio=DEFAULT_VAL_RATIO,
    test_ratio=DEFAULT_TEST_RATIO,
    seed=None,
    force=False,
):
    """
    Split dataset triplets into train/val/test sets and copy them into
    separate subfolders.

    The split is deterministic for a given (seed, ratios, source data) and is
    recorded in a manifest (`split_manifest.json`) under `output_root`. If an
    existing split already matches the requested seed/ratios/source data and
    `force` is False, the existing split is reused as-is (no reshuffle, no
    copying) so unrelated changes (e.g. training.seed) don't perturb it.

    Args:
        input_folder: path containing all triplet files
        output_root: root output folder for train/val/test sets
        train_ratio, val_ratio, test_ratio: split proportions
        seed: seed for the split shuffle only, independent of any training RNG
        force: if True, always reshuffle/recopy even if the manifest matches
    """
    triplets = get_triplets(input_folder)
    print(f"Found {len(triplets)} valid triplets.")

    if not force and split_manifest_matches(input_folder, output_root, train_ratio, val_ratio, test_ratio, seed):
        print(f"Reusing existing split at {output_root} (seed={seed}, ratios/source unchanged).")
        return

    # Canonical order before shuffling: os.listdir() order (which get_triplets
    # relies on) isn't guaranteed stable across machines/filesystems, so sorting
    # first is required for `seed` to reproduce the same split everywhere.
    triplets = sorted(triplets, key=lambda t: t[0])

    # Local RNG instance: shuffling no longer touches global random state.
    rng = random.Random(seed)
    rng.shuffle(triplets)
    total = len(triplets)

    # Compute split indices
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)

    # Partition into splits
    train_triplets = triplets[:train_end]
    val_triplets = triplets[train_end:val_end]
    test_triplets = triplets[val_end:]

    print(f"Splitting into: {len(train_triplets)} train, {len(val_triplets)} val, {len(test_triplets)} test")

    # Recreate split folders to avoid stale files from previous runs.
    for split_name in ["train", "val", "test"]:
        split_dir = os.path.join(output_root, split_name)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)

    # Copy files into respective subfolders
    for triplet in train_triplets:
        copy_triplet(triplet, input_folder, os.path.join(output_root, "train"))
    for triplet in val_triplets:
        copy_triplet(triplet, input_folder, os.path.join(output_root, "val"))
    for triplet in test_triplets:
        copy_triplet(triplet, input_folder, os.path.join(output_root, "test"))

    _write_manifest(
        output_root,
        seed,
        train_ratio,
        val_ratio,
        test_ratio,
        triplets,
        {"train": train_triplets, "val": val_triplets, "test": test_triplets},
    )


# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    # config_path = r"GripFusionNet\Config\config.yaml"
    # config = load_config(config_path)
    # input_folder = config["data_path"]
    # output_root = os.path.join(config["data_path"],"Data_split/")

    output_root = os.path.join("D:/MA/GripFusionNet/GripFusionNet/dummy_data","Data_split/")
    split_dataset("D:/MA/GripFusionNet/GripFusionNet/dummy_data", output_root)
