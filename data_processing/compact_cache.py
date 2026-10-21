"""Compact, split-independent data pipeline for cluster / array-job training (`data.pipeline: compact`).

Why: the legacy pipeline keeps ONE split-specific cache at <data_root>/Data_split (~60 GB of float32
640x640 tensors per split) and rebuilds it in place whenever a run asks for a different split -- unsafe
when several jobs run at once, and every epoch re-reads ~48 GB from disk.

This module replaces it with two read-only artefacts that are built once and then shared by every job:

  <data_root>/splits/split_<seed>.json     frozen split: train/val/test timestamp lists + that split's
                                           normalization stats. Written by freeze_splits(); training never
                                           re-derives a split, so splits stay fixed even if Data/ changes.
  <data_root>/compact_cache/               every sample's letterboxed image *content* as uint8 (the grey
                                           letterbox padding is not stored), plus the raw hub-frame pose.
                                           ~0.63 MB/sample (~4 GB total) instead of ~9.8 MB/sample per split.

CompactCacheDataset rebuilds exactly the tensors the legacy .pt cache holds (same letterbox code, same
ToTensor/Normalize, same float64 pose normalization), so a given split + stats yields identical inputs.
Only the sample *order* differs: here it is the sorted timestamp list (reproducible on any machine),
whereas the legacy dataset used os.listdir order (filesystem-dependent).
"""

import hashlib
import json
import os
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from data_processing.GripFusionNet_split import (
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    SPLIT_METHOD_GROUP_AWARE,
    SPLIT_METHOD_PER_FRAME,
    _assign_groups_to_ratios,
    _triplets_hash,
    group_triplets_by_pose,
)
from data_processing.preprocess_data import LETTERBOX_PAD_VALUE, _hub_frame_pose, load_letterboxed_uint8
from utils.GripFusionNet_utils import get_triplets

PIPELINE_NAME = "compact"
SPLIT_FORMAT = "fusionmodel-frozen-split-v1"
CACHE_FORMAT = "fusionmodel-compact-cache-v1"
CANVAS = 640
STREAMS = ("rgb", "digit")

# Hub-frame calibration matrix; must match preprocess_data.preprocess_data's transf_matrix.
HUB_TRANSF_MATRIX = [
    [1.0, 0.0, 0.0, 0.456577],
    [0.0, 1.0, 0.0, -0.163301],
    [0.0, 0.0, 1.0, -0.024417],
    [0.0, 0.0, 0.0, 1.0],
]


# ------------------------------------------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------------------------------------------
def is_compact_pipeline(cfg):
    return str(cfg.get("data", {}).get("pipeline", "legacy")).lower() == PIPELINE_NAME


def _under(data_root, configured, default):
    path = configured or default
    return path if os.path.isabs(path) else os.path.join(data_root, path)


def resolve_cache_dir(cfg, data_root):
    return _under(data_root, cfg.get("data", {}).get("compact_cache_dir"), "compact_cache")


def split_filename(split_seed, group_aware=True):
    return f"split_{split_seed}.json" if group_aware else f"split_{split_seed}_perframe.json"


def resolve_split_file(cfg, data_root):
    from utils.utils import resolve_group_aware_split, resolve_split_seed

    explicit = cfg.get("data", {}).get("split_file")
    if explicit:
        return explicit if os.path.isabs(explicit) else os.path.join(data_root, explicit)
    split_dir = _under(data_root, cfg.get("data", {}).get("split_dir"), "splits")
    return os.path.join(split_dir, split_filename(resolve_split_seed(cfg), resolve_group_aware_split(cfg)))


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_frozen_split(cfg, data_root):
    """Load and validate the frozen split for this config. Never builds anything: fails fast instead."""
    from utils.utils import resolve_group_aware_split, resolve_split_seed

    path = resolve_split_file(cfg, data_root)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Frozen split file not found: {path}\n"
            "The compact pipeline never creates splits during training. Build it once with\n"
            "  python scripts/cluster/prepare_data.py freeze-splits --seeds <seed ...>"
        )
    with open(path, "r", encoding="utf-8") as f:
        split = json.load(f)
    if split.get("format") != SPLIT_FORMAT:
        raise ValueError(f"{path}: unexpected format {split.get('format')!r}")
    seed, group_aware = resolve_split_seed(cfg), resolve_group_aware_split(cfg)
    if split["split_seed"] != seed or bool(split["group_aware"]) != bool(group_aware):
        raise ValueError(
            f"{path} holds split_seed={split['split_seed']}, group_aware={split['group_aware']} but the config "
            f"asks for split_seed={seed}, group_aware={group_aware}."
        )
    split["_path"] = path
    split["_sha256"] = file_sha256(path)
    return split


def translation_stats_from_split(split):
    s = split["stats"]
    return {"mean_y": s["trans_mean_y"], "std_y": s["trans_std_y"], "mean_z": s["trans_mean_z"], "std_z": s["trans_std_z"]}


# ------------------------------------------------------------------------------------------------
# Freezing splits
# ------------------------------------------------------------------------------------------------
def assign_split(data_root, split_seed, group_aware=True,
                 ratios=(DEFAULT_TRAIN_RATIO, DEFAULT_VAL_RATIO, DEFAULT_TEST_RATIO)):
    """Exactly the assignment GripFusionNet_split.split_dataset performs, without copying any files."""
    import random

    triplets = get_triplets(data_root)
    rng = random.Random(split_seed)
    if group_aware:
        groups = group_triplets_by_pose(triplets, data_root)
        train, val, test = _assign_groups_to_ratios(groups, *ratios, rng)
    else:
        ordered = sorted(triplets, key=lambda t: t[0])
        rng.shuffle(ordered)
        n = len(ordered)
        a = int(ratios[0] * n)
        b = a + int(ratios[1] * n)
        train, val, test = ordered[:a], ordered[a:b], ordered[b:]
    return triplets, {"train": [t[0] for t in train], "val": [t[0] for t in val], "test": [t[0] for t in test]}


def _accumulate_image_stats(paths, transform):
    """Same arithmetic as GripFusionNet_get_norm_params.compute_mean_std_* (float32 running sums),
    but over an explicit, sorted file list instead of a split folder's glob order."""
    sum_ = torch.zeros(3)
    sum_sq = torch.zeros(3)
    total_pixels = 0
    for path in paths:
        tensor = transform(Image.open(path).convert("RGB"))
        sum_ += tensor.sum(dim=(1, 2))
        sum_sq += (tensor ** 2).sum(dim=(1, 2))
        total_pixels += tensor.shape[1] * tensor.shape[2]
    mean = sum_ / total_pixels
    std = (sum_sq / total_pixels - mean ** 2).sqrt()
    return mean.tolist(), std.tolist()


def compute_split_stats(data_root, train_timestamps):
    train_timestamps = sorted(train_timestamps)
    digit_mean, digit_std = _accumulate_image_stats(
        [os.path.join(data_root, f"{t}_digit.png") for t in train_timestamps],
        transforms.Compose([transforms.Resize((320, 240)), transforms.ToTensor()]),
    )
    rgb_mean, rgb_std = _accumulate_image_stats(
        [os.path.join(data_root, f"{t}.png") for t in train_timestamps], transforms.ToTensor()
    )
    hub = np.array([_hub_frame_pose(os.path.join(data_root, f"{t}.json"), HUB_TRANSF_MATRIX) for t in train_timestamps],
                   dtype=np.float64)
    return {
        "rgb_mean": rgb_mean, "rgb_std": rgb_std, "digit_mean": digit_mean, "digit_std": digit_std,
        "trans_mean_y": float(hub[:, 1].mean()), "trans_std_y": float(hub[:, 1].std()),
        "trans_mean_z": float(hub[:, 2].mean()), "trans_std_z": float(hub[:, 2].std()),
    }


def write_frozen_split(data_root, out_dir, split_seed, group_aware=True, stats_override=None,
                       stats_source=None, overwrite=False):
    """Freeze one split (timestamps + stats) to <out_dir>/split_<seed>.json. Refuses to overwrite."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, split_filename(split_seed, group_aware))
    if os.path.exists(path) and not overwrite:
        raise FileExistsError(f"{path} already exists; frozen splits are never overwritten implicitly.")
    triplets, lists = assign_split(data_root, split_seed, group_aware)
    stats = compute_split_stats(data_root, lists["train"])
    source = {k: "recomputed from Data/ (sorted train file list)" for k in stats}
    for k, v in (stats_override or {}).items():
        stats[k] = v
        source[k] = (stats_source or {}).get(k, "override")
    payload = {
        "format": SPLIT_FORMAT,
        "split_seed": split_seed,
        "group_aware": bool(group_aware),
        "split_method": SPLIT_METHOD_GROUP_AWARE if group_aware else SPLIT_METHOD_PER_FRAME,
        "ratios": [DEFAULT_TRAIN_RATIO, DEFAULT_VAL_RATIO, DEFAULT_TEST_RATIO],
        "num_triplets": len(triplets),
        "source_timestamps_hash": _triplets_hash(triplets),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "stats": stats,
        "stats_source": source,
        "train": sorted(lists["train"]),
        "val": sorted(lists["val"]),
        "test": sorted(lists["test"]),
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)  # atomic: readers never see a half-written split
    return path, payload


# ------------------------------------------------------------------------------------------------
# Building the compact cache
# ------------------------------------------------------------------------------------------------
def _nonpad_box(img):
    """(top, left, height, width) of pixels that differ from the letterbox pad value."""
    mask = (img != LETTERBOX_PAD_VALUE).any(axis=2)
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    if len(rows) == 0:
        return (0, 0, 0, 0)
    return (int(rows[0]), int(cols[0]), int(rows[-1] - rows[0] + 1), int(cols[-1] - cols[0] + 1))


def _sample_boxes(args):
    data_root, ts = args
    out = []
    for name, path in (("rgb", f"{ts}.png"), ("digit", f"{ts}_digit.png")):
        img = load_letterboxed_uint8(os.path.join(data_root, path))
        if img.shape != (CANVAS, CANVAS, 3):
            raise ValueError(f"{path}: letterboxed shape {img.shape}, expected {(CANVAS, CANVAS, 3)}")
        out.append(_nonpad_box(img))
    return ts, out


def _sample_crops(args):
    data_root, ts, boxes = args
    crops = []
    for (path, (t, l, h, w)) in zip((f"{ts}.png", f"{ts}_digit.png"), boxes):
        img = load_letterboxed_uint8(os.path.join(data_root, path))
        # Everything outside the stored box must be pad, otherwise reconstruction would not be exact.
        outside = img.copy()
        outside[t:t + h, l:l + w] = LETTERBOX_PAD_VALUE
        if (outside != LETTERBOX_PAD_VALUE).any():
            raise ValueError(f"{path}: content outside the union box -- cache would not be exact")
        crops.append(np.ascontiguousarray(img[t:t + h, l:l + w]))
    pose = _hub_frame_pose(os.path.join(data_root, f"{ts}.json"), HUB_TRANSF_MATRIX)
    return ts, crops, np.asarray(pose, dtype=np.float64)


def build_compact_cache(data_root, out_dir, workers=8, overwrite=False):
    """Write <out_dir>/{rgb.npy, digit.npy, pose_hub.npy, meta.json} for every valid triplet in data_root."""
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path) and not overwrite:
        raise FileExistsError(f"{out_dir} already holds a compact cache (pass overwrite=True to rebuild).")
    os.makedirs(out_dir, exist_ok=True)
    triplets = get_triplets(data_root)
    timestamps = sorted(t[0] for t in triplets)

    with Pool(workers) as pool:
        boxes = dict(pool.imap_unordered(_sample_boxes, [(data_root, t) for t in timestamps], chunksize=16))
    union = []
    for i in range(len(STREAMS)):
        b = [boxes[t][i] for t in timestamps if boxes[t][i][2] > 0]
        top, left = min(x[0] for x in b), min(x[1] for x in b)
        bottom, right = max(x[0] + x[2] for x in b), max(x[1] + x[3] for x in b)
        union.append((top, left, bottom - top, right - left))

    n = len(timestamps)
    index = {t: i for i, t in enumerate(timestamps)}
    arrays = {
        name: np.lib.format.open_memmap(os.path.join(out_dir, f"{name}.npy.tmp"), mode="w+", dtype=np.uint8,
                                        shape=(n, box[2], box[3], 3))
        for name, box in zip(STREAMS, union)
    }
    poses = np.zeros((n, 7), dtype=np.float64)
    with Pool(workers) as pool:
        for ts, crops, pose in pool.imap_unordered(_sample_crops, [(data_root, t, union) for t in timestamps], chunksize=8):
            i = index[ts]
            for name, crop in zip(STREAMS, crops):
                arrays[name][i] = crop
            poses[i] = pose
    for name, arr in arrays.items():
        arr.flush()
        del arr
        os.replace(os.path.join(out_dir, f"{name}.npy.tmp"), os.path.join(out_dir, f"{name}.npy"))
    np.save(os.path.join(out_dir, "pose_hub.npy"), poses)
    meta = {
        "format": CACHE_FORMAT,
        "canvas": CANVAS,
        "pad_value": LETTERBOX_PAD_VALUE,
        "boxes": {name: list(box) for name, box in zip(STREAMS, union)},  # top, left, height, width
        "num_samples": n,
        "source_timestamps_hash": _triplets_hash(triplets),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "timestamps": timestamps,
    }
    with open(meta_path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(meta, f)
    os.replace(meta_path + ".tmp", meta_path)  # meta.json last: its presence marks a complete cache
    return meta


# ------------------------------------------------------------------------------------------------
# Dataset / loaders
# ------------------------------------------------------------------------------------------------
class CompactCacheDataset(Dataset):
    """Returns (rgb, digit, pose) exactly like PreprocessedTactileDataset, built from the compact cache.

    Arrays are memory-mapped lazily per worker process, so the OS page cache shares one copy of the
    ~4 GB cache between all workers (and all jobs on a node)."""

    def __init__(self, cache_dir, timestamps, stats):
        with open(os.path.join(cache_dir, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("format") != CACHE_FORMAT:
            raise ValueError(f"{cache_dir}: unexpected cache format {meta.get('format')!r}")
        row = {t: i for i, t in enumerate(meta["timestamps"])}
        missing = [t for t in timestamps if t not in row]
        if missing:
            raise KeyError(f"{len(missing)} split timestamps are not in the compact cache, e.g. {missing[:3]}")
        self.cache_dir = cache_dir
        self.rows = [row[t] for t in timestamps]
        self.timestamps = list(timestamps)
        self.boxes = {k: tuple(v) for k, v in meta["boxes"].items()}
        self.canvas = int(meta["canvas"])
        self.pad = int(meta["pad_value"])
        self.trans = {k: stats[k] for k in ("trans_mean_y", "trans_std_y", "trans_mean_z", "trans_std_z")}
        # Same ops as preprocess_data._create_image_transforms minus Resize((640, 640)), a no-op on the 640x640 canvas.
        self.transforms = {
            "rgb": transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=tuple(stats["rgb_mean"]), std=tuple(stats["rgb_std"]))]),
            "digit": transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=tuple(stats["digit_mean"]), std=tuple(stats["digit_std"]))]),
        }
        self._arrays = None

    def _open(self):
        if self._arrays is None:
            self._arrays = {name: np.load(os.path.join(self.cache_dir, f"{name}.npy"), mmap_mode="r") for name in STREAMS}
            self._arrays["pose"] = np.load(os.path.join(self.cache_dir, "pose_hub.npy"), mmap_mode="r")
        return self._arrays

    def __getstate__(self):  # never pickle open memmaps into worker processes
        state = self.__dict__.copy()
        state["_arrays"] = None
        return state

    def __len__(self):
        return len(self.rows)

    def _image(self, name, i):
        t, l, h, w = self.boxes[name]
        canvas = np.full((self.canvas, self.canvas, 3), self.pad, dtype=np.uint8)
        canvas[t:t + h, l:l + w] = self._open()[name][i]
        return self.transforms[name](Image.fromarray(canvas).convert("RGB"))

    def __getitem__(self, idx):
        i = self.rows[idx]
        pose = np.array(self._open()["pose"][i], dtype=np.float64)
        # Same float64 arithmetic as preprocess_data._process_pose; the legacy cache always stores
        # normalized y/z, independent of data.normalize_translation.
        pose[1] = (pose[1] - self.trans["trans_mean_y"]) / self.trans["trans_std_y"]
        pose[2] = (pose[2] - self.trans["trans_mean_z"]) / self.trans["trans_std_z"]
        return self._image("rgb", i), self._image("digit", i), torch.tensor(pose, dtype=torch.float32)


def get_compact_dataloaders(cfg, data_root, batch_size, num_workers=2):
    """Train/val/test loaders with the same loader settings as dataloader_preprocessed.get_dataloaders."""
    from data_processing.dataloader_preprocessed import tactile_collate_fn

    if cfg.get("data", {}).get("augmentation", {}).get("enabled", False):
        raise NotImplementedError("data.pipeline: compact does not support online augmentation yet; use the legacy pipeline.")
    split = load_frozen_split(cfg, data_root)
    cache_dir = resolve_cache_dir(cfg, data_root)
    if not os.path.exists(os.path.join(cache_dir, "meta.json")):
        raise FileNotFoundError(
            f"Compact cache not found at {cache_dir}. Build it once with\n"
            "  python scripts/cluster/prepare_data.py build-cache"
        )
    sets = {name: CompactCacheDataset(cache_dir, split[name], split["stats"]) for name in ("train", "val", "test")}
    loader_args = dict(num_workers=num_workers, pin_memory=False, persistent_workers=num_workers > 0,
                       collate_fn=tactile_collate_fn)
    train_loader = DataLoader(sets["train"], batch_size=batch_size, shuffle=True, **loader_args)
    val_loader = DataLoader(sets["val"], batch_size=batch_size, shuffle=False, **loader_args)
    test_loader = DataLoader(sets["test"], batch_size=1, shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader, translation_stats_from_split(split), split


def get_compact_test_dataloader(cfg, data_root, num_workers=2):
    """Test loader for scripts/test.py; refuses to evaluate if the split file changed since training."""
    from data_processing.dataloader_preprocessed import tactile_collate_fn

    split = load_frozen_split(cfg, data_root)
    trained_sha = cfg.get("data", {}).get("split_file_sha256")
    if trained_sha and trained_sha != split["_sha256"]:
        raise ValueError(f"{split['_path']} changed since this run was trained (sha256 mismatch).")
    dataset = CompactCacheDataset(resolve_cache_dir(cfg, data_root), split["test"], split["stats"])
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=False,
                      persistent_workers=num_workers > 0, collate_fn=tactile_collate_fn)
