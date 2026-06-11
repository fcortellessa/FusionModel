import argparse
import hashlib
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import numpy as np
from PIL import Image
from tqdm import tqdm


@dataclass(frozen=True)
class Triplet:
	timestamp: str
	rgb_path: Path
	digit_path: Path
	json_path: Path


def list_triplets(input_dir: Path) -> List[Triplet]:
	triplets: List[Triplet] = []
	for json_path in sorted(input_dir.glob("*.json")):
		ts = json_path.stem
		rgb_path = input_dir / f"{ts}.png"
		digit_path = input_dir / f"{ts}_digit.png"
		if rgb_path.exists() and digit_path.exists():
			triplets.append(Triplet(ts, rgb_path, digit_path, json_path))
	return triplets


def _rng_for(seed: int, timestamp: str, mode: str, variant_idx: int) -> np.random.Generator:
	# Set a fixed random number generator: a deterministic per-sample seed so ordering changes do not affect outputs.
	token = f"{seed}|{timestamp}|{mode}|{variant_idx}"
	digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
	local_seed = int.from_bytes(digest, byteorder="little", signed=False) % (2**32 - 1)
	return np.random.default_rng(local_seed)


def _apply_albumentations_deterministic(
	img: Image.Image,
	transform: A.Compose,
	rng: np.random.Generator,
) -> np.ndarray:
	"""Run an albumentations transform deterministically with local RNG objects."""
	seed = int(rng.integers(0, 2**32 - 1))
	np_state = np.random.get_state()
	py_state = random.getstate()
	np.random.seed(seed)
	random.seed(seed)
	try:
		result = transform(image=np.asarray(img))
	finally:
		np.random.set_state(np_state)
		random.setstate(py_state)
	return result["image"]


def _apply_rgb_augmentation(
	rgb_img: Image.Image,
	rng: np.random.Generator,
) -> Tuple[Image.Image, Dict]:
	"""RGB-specific photometric augmentation."""
	transform = A.Compose(
		[
			A.OneOf([A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
					A.MotionBlur(blur_limit=7, p=1.0),], 
					p=0.2,),
			A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=0, p=0.5),
			A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
		],
		p=1.0,
	)
	out_arr = _apply_albumentations_deterministic(rgb_img, transform, rng)
	out = Image.fromarray(np.clip(out_arr, 0, 255).astype(np.uint8))
	meta = {"type": "photometric_rgb", "albumentations": True}
	return out, meta


def _apply_tactile_augmentation(
    digit_img: Image.Image,
    rng: np.random.Generator,
    aug_config: Dict = None,
) -> Tuple[Image.Image, Dict]:
    """Tactile-specific photometric augmentation."""
    if aug_config is None:
        aug_config = {}

    # Extract configs with default fallback to True
    enabled = {k: aug_config.get(k, True) for k in [
        "enabled_noise", "enabled_brightness", "enabled_contrast", 
        "enabled_blur", "enabled_channel_scaling"
    ]}

    transforms = []

    if enabled["enabled_noise"]:
        transforms.append(A.GaussNoise(var_limit=(10.0, 50.0), p=1.0)) 

    if enabled["enabled_brightness"] or enabled["enabled_contrast"]:
        transforms.append(A.RandomBrightnessContrast(
			brightness_limit=(-0.2, 0.2) if enabled["enabled_brightness"] else 0.0,
			contrast_limit=(-0.2, 0.2) if enabled["enabled_contrast"] else 0.0,
            p=1.0
        ))

    if enabled["enabled_blur"]:
        transforms.append(A.MotionBlur(blur_limit=7, p=1.0))

    if enabled["enabled_channel_scaling"]:
        transforms.append(A.RGBShift(
            r_shift_limit=20, 
            g_shift_limit=20, 
            b_shift_limit=20, 
            p=1.0
        ))

    # Fallback to NoOp if everything is disabled
    transform = A.Compose(transforms, p=1.0) if transforms else A.Compose([A.NoOp()], p=1.0)
    
    # Run pipeline deterministically
    out_arr = _apply_albumentations_deterministic(digit_img, transform, rng)
    
    out = Image.fromarray(out_arr)
    meta = {
        "type": "photometric_tactile",
        "albumentations": True,
        "config": aug_config,
    }
    return out, meta


def _copy_original(triplet: Triplet, output_dir: Path, overwrite: bool) -> str:
	out_ts = triplet.timestamp
	out_rgb = output_dir / f"{out_ts}.png"
	out_digit = output_dir / f"{out_ts}_digit.png"
	out_json = output_dir / f"{out_ts}.json"

	if overwrite or not out_rgb.exists():
		shutil.copy2(triplet.rgb_path, out_rgb)
	if overwrite or not out_digit.exists():
		shutil.copy2(triplet.digit_path, out_digit)
	if overwrite or not out_json.exists():
		shutil.copy2(triplet.json_path, out_json)

	return out_ts


def _save_augmented(
	out_ts: str,
	output_dir: Path,
	rgb_img: Image.Image,
	digit_img: Image.Image,
	source_json_path: Path,
	overwrite: bool,
) -> None:
	out_rgb = output_dir / f"{out_ts}.png"
	out_digit = output_dir / f"{out_ts}_digit.png"
	out_json = output_dir / f"{out_ts}.json"

	if overwrite or not out_rgb.exists():
		rgb_img.save(out_rgb)
	if overwrite or not out_digit.exists():
		digit_img.save(out_digit)
	if overwrite or not out_json.exists():
		shutil.copy2(source_json_path, out_json)


def augment_offline(
	input_dir: Path,
	output_dir: Path,
	seed: int,
	augmentation_count: int,
	copy_originals: bool,
	overwrite: bool,
	dry_run: bool,
	strict: bool,
	tactile_aug_config: Dict = None,
) -> None:
	triplets = list_triplets(input_dir)
	if not triplets:
		raise ValueError(f"No valid triplets found in {input_dir}")


	output_dir.mkdir(parents=True, exist_ok=True)

	failed: List[Dict] = []
	generated: Dict[str, List[Dict]] = {}
	total_new_per_sample = augmentation_count + (1 if copy_originals else 0)

	print(f"Found {len(triplets)} source samples in {input_dir}")
	print(f"Will generate {total_new_per_sample} samples per source in {output_dir}")

	for triplet in tqdm(triplets, desc="Offline augmentation"):
		try:
			rgb_img = Image.open(triplet.rgb_path).convert("RGB")
			digit_img = Image.open(triplet.digit_path).convert("RGB")

			generated[triplet.timestamp] = []

			if copy_originals:
				if not dry_run:
					original_ts = _copy_original(triplet, output_dir, overwrite=overwrite)
				else:
					original_ts = triplet.timestamp
				generated[triplet.timestamp].append(
					{"timestamp": original_ts, "kind": "original", "params": {}}
				)

			for idx in range(augmentation_count):
				local_rng = _rng_for(seed, triplet.timestamp, "augm", idx)
				rgb_aug, rgb_meta = _apply_rgb_augmentation(rgb_img, local_rng)
				digit_aug, tactile_meta = _apply_tactile_augmentation(digit_img, local_rng, aug_config=tactile_aug_config)
				out_ts = f"{triplet.timestamp}__augm{idx}"
				if not dry_run:
					_save_augmented(
						out_ts,
						output_dir,
						rgb_aug,
						digit_aug,
						triplet.json_path,
						overwrite=overwrite,
					)
				generated[triplet.timestamp].append(
					{"timestamp": out_ts, "kind": "photometric", "params": {"rgb": rgb_meta, "tactile": tactile_meta}}
				)

		except Exception as exc:
			failed_item = {
				"timestamp": triplet.timestamp,
				"error": str(exc),
			}
			failed.append(failed_item)
			if strict:
				raise

	manifest = {
		"seed": seed,
		"input_dir": str(input_dir),
		"output_dir": str(output_dir),
		"copy_originals": copy_originals,
		"augmentation_count": augmentation_count,
		"photometric": {
			"rgb_augmentation": {
				"gaussian_noise_p": 0.002,
				"hue_range_degrees": [-5.0, 5.0],
				"brightness_range": [0.7, 1.3],
				"contrast_range": [0.7, 1.3],
				"blur_p": 0.2,
				"gamma": 0.1,
			},
			"tactile_augmentation": {
				"gaussian_noise_p": 0.002,
				"brightness_range": [0.9, 1.1],
				"contrast_range": [0.9, 1.1],
				"blur_p": 0.2,
				"channel_scaling_range": [0.95, 1.05],
			},
		},
		"num_source_samples": len(triplets),
		"num_failed_samples": len(failed),
		"generated": generated,
	}

	if not dry_run:
		manifest_path = output_dir / "augmentation_manifest.json"
		with manifest_path.open("w", encoding="utf-8") as fh:
			json.dump(manifest, fh, indent=2)

		failed_path = output_dir / "failed_samples.json"
		with failed_path.open("w", encoding="utf-8") as fh:
			json.dump(failed, fh, indent=2)

	print(f"Done. Failed samples: {len(failed)}")
	if dry_run:
		print("Dry run enabled: no files were written.")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Offline data augmentation for train split benchmarking"
	)
	parser.add_argument(
		"--input-dir",
		type=str,
		default="Data/Data_split/train",
		help="Input split folder with triplets (*.png, *_digit.png, *.json)",
	)
	parser.add_argument(
		"--output-dir",
		type=str,
		default="Data/Data_split/train_augmented",
		help="Output folder for augmented samples",
	)
	parser.add_argument(
		"--augmentation-count",
		type=int,
		default=1,
		help="Photometric variants per source sample",
	)
	parser.add_argument(
		"--no-copy-originals",
		action="store_true",
		help="Do not copy original samples into output folder",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Overwrite existing output files. Useful if re-running augmentation.",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Preview run without writing files",
	)
	parser.add_argument(
		"--strict",
		action="store_true",
		help="Fail fast on first sample error",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	augment_offline(
		input_dir=Path(args.input_dir),
		output_dir=Path(args.output_dir),
		seed=args.seed,
		augmentation_count=args.augmentation_count,
		copy_originals=not args.no_copy_originals,
		overwrite=args.overwrite,
		dry_run=args.dry_run,
		strict=args.strict,
	)
