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

from utils.GripFusionNet_utils import get_triplets

DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.05

MANIFEST_FILENAME = "split_manifest.json"

# Recorded in the manifest so a manifest written by one method is never
# silently reused for the other, and so old (pre-flag) manifests -- which
# only ever mean the group-aware method -- still invalidate correctly.
SPLIT_METHOD_GROUP_AWARE = "pose_group_v1"
SPLIT_METHOD_PER_FRAME = "per_frame_v1"  # legacy pre-fix behavior; comparison only

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
# Grasp-event grouping: several consecutive frames can share one recorded
# TCP pose (the pose sensor is read at a lower rate than the camera, and/or
# one grasp is deliberately photographed against several backgrounds). All
# frames sharing an identical (position, orientation) reading describe the
# same physical grasp and must stay together in one split -- otherwise the
# model can see one background of a grasp in train and another in test and
# "solve" pose estimation by recognizing the grasp instead of the pose.
# -------------------------------------------------
def _pose_key(input_folder, timestamp):
    """Exact (position, orientation) tuple for a triplet's JSON -- the grouping key."""
    with open(os.path.join(input_folder, f"{timestamp}.json"), "r", encoding="utf-8") as f:
        d = json.load(f)
    p, o = d["position"], d["orientation"]
    return (p["x"], p["y"], p["z"], o["x"], o["y"], o["z"], o["w"])


def group_triplets_by_pose(triplets, input_folder):
    """
    Partition triplets into groups that share an identical recorded pose
    (i.e. the same grasp event). Frames with a unique pose form singleton
    groups. Groups are returned in a canonical order (sorted by their
    earliest timestamp, members sorted by timestamp) so that shuffling them
    with a seeded RNG is reproducible across machines/filesystems.
    """
    by_pose = {}
    for triplet in triplets:
        key = _pose_key(input_folder, triplet[0])
        by_pose.setdefault(key, []).append(triplet)

    groups = list(by_pose.values())
    for group in groups:
        group.sort(key=lambda t: t[0])
    groups.sort(key=lambda group: group[0][0])
    return groups


def _assign_groups_to_ratios(groups, train_ratio, val_ratio, test_ratio, rng):
    """
    Shuffle groups (as atomic units) and greedily assign each one to whichever
    split is currently furthest below its target *frame* count. Group sizes
    are uneven (1 grasp = 1-10+ frames), so hitting the exact ratios isn't
    possible, but this greedy fill keeps the realized ratios close to
    train_ratio/val_ratio/test_ratio without ever splitting a group.
    """
    groups = list(groups)
    rng.shuffle(groups)

    total = sum(len(g) for g in groups)
    targets = {
        "train": train_ratio * total,
        "val": val_ratio * total,
        "test": test_ratio * total,
    }
    counts = {"train": 0, "val": 0, "test": 0}
    buckets = {"train": [], "val": [], "test": []}

    for group in groups:
        # Assign to the split with the largest remaining shortfall vs. its target.
        split_name = max(targets, key=lambda name: targets[name] - counts[name])
        buckets[split_name].append(group)
        counts[split_name] += len(group)

    train_triplets = [t for g in buckets["train"] for t in g]
    val_triplets = [t for g in buckets["val"] for t in g]
    test_triplets = [t for g in buckets["test"] for t in g]
    return train_triplets, val_triplets, test_triplets


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


def _write_manifest(
    output_root, seed, train_ratio, val_ratio, test_ratio, triplets, split_assignment,
    group_aware, num_groups, num_leaky_groups,
):
    manifest = {
        "seed": seed,
        "split_method": SPLIT_METHOD_GROUP_AWARE if group_aware else SPLIT_METHOD_PER_FRAME,
        "group_aware": group_aware,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "num_triplets": len(triplets),
        "num_grasp_groups": num_groups,
        "num_leaky_groups": num_leaky_groups,  # grasp-event groups spanning >1 split; 0 iff group_aware
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
    group_aware=True,
):
    """
    Cheaply check whether an existing split already matches the requested
    seed/ratios/source data/method, without touching any files. Used both
    internally by split_dataset() and by callers (e.g. DataPipelineManager)
    that want to know up front whether a resplit is needed.
    """
    manifest = _load_manifest(output_root)
    if manifest is None:
        return False

    if manifest.get("seed") != seed:
        return False
    expected_method = SPLIT_METHOD_GROUP_AWARE if group_aware else SPLIT_METHOD_PER_FRAME
    if manifest.get("split_method") != expected_method:
        # Either the method flag changed, or this manifest predates the flag
        # (pre-fix manifests only ever mean the leaky per-frame method).
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
    group_aware=True,
):
    """
    Split dataset triplets into train/val/test sets and copy them into
    separate subfolders.

    The split is deterministic for a given (seed, ratios, source data, method)
    and is recorded in a manifest (`split_manifest.json`) under `output_root`.
    If an existing split already matches the request and `force` is False, the
    existing split is reused as-is (no reshuffle, no copying) so unrelated
    changes (e.g. training.seed) don't perturb it.

    Args:
        input_folder: path containing all triplet files
        output_root: root output folder for train/val/test sets
        train_ratio, val_ratio, test_ratio: split proportions
        seed: seed for the split shuffle only, independent of any training RNG
        force: if True, always reshuffle/recopy even if the manifest matches
        group_aware: if True (default, recommended), frames sharing an
            identical recorded pose -- the same grasp event, photographed
            against several backgrounds -- are kept together in one split, so
            no grasp event can end up split across train/val/test. If False,
            reproduces the legacy per-frame shuffle (pre-fix behavior), which
            *can* leak grasp events across splits. This exists to run the two
            methods side by side (e.g. into different output_root paths) and
            compare resulting metrics -- do not use False for results you
            intend to report. Either way, a `Grasp-event leakage audit` is
            printed and written into the manifest (`num_leaky_groups`), so you
            can quantify the difference directly without retraining anything.
    """
    triplets = get_triplets(input_folder)
    print(f"Found {len(triplets)} valid triplets.")

    if not force and split_manifest_matches(
        input_folder, output_root, train_ratio, val_ratio, test_ratio, seed, group_aware=group_aware
    ):
        print(f"Reusing existing split at {output_root} (seed={seed}, group_aware={group_aware}, ratios/source unchanged).")
        return

    # Grasp-event groups are computed either way: needed to build the split
    # when group_aware=True, and needed to audit the resulting leakage either way.
    groups = group_triplets_by_pose(triplets, input_folder)
    multi_frame_groups = [g for g in groups if len(g) > 1]
    print(
        f"Found {len(groups)} grasp-event groups "
        f"({len(multi_frame_groups)} multi-frame, "
        f"{sum(len(g) for g in multi_frame_groups)} frames in them)."
    )

    # Local RNG instance: shuffling no longer touches global random state.
    rng = random.Random(seed)

    if group_aware:
        train_triplets, val_triplets, test_triplets = _assign_groups_to_ratios(
            groups, train_ratio, val_ratio, test_ratio, rng
        )
    else:
        print("group_aware=False: reproducing the legacy per-frame shuffle "
              "(comparison only -- this can leak grasp events across splits).")
        triplets_sorted = sorted(triplets, key=lambda t: t[0])
        rng.shuffle(triplets_sorted)
        total = len(triplets_sorted)
        train_end = int(train_ratio * total)
        val_end = train_end + int(val_ratio * total)
        train_triplets = triplets_sorted[:train_end]
        val_triplets = triplets_sorted[train_end:val_end]
        test_triplets = triplets_sorted[val_end:]

    print(f"Splitting into: {len(train_triplets)} train, {len(val_triplets)} val, {len(test_triplets)} test")

    # Audit: how many grasp-event groups actually end up spanning more than
    # one partition in the split we just built, regardless of which method
    # produced it. Should be exactly 0 when group_aware=True.
    split_of = {}
    for split_name, split_triplets in (("train", train_triplets), ("val", val_triplets), ("test", test_triplets)):
        for t in split_triplets:
            split_of[t[0]] = split_name
    num_leaky_groups = sum(1 for g in groups if len({split_of[t[0]] for t in g}) > 1)
    print(f"Grasp-event leakage audit: {num_leaky_groups} / {len(groups)} groups span more than one split.")

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
        group_aware=group_aware,
        num_groups=len(groups),
        num_leaky_groups=num_leaky_groups,
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
