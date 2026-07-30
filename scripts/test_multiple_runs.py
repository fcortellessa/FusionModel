"""
This script runs test.py for multiple run folders.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ==============================================================================
# CONFIGURATION: Paste your folders here
# ==============================================================================

# Option 1: Paste an explicit list of run folders
RUN_FOLDERS_TO_BE_INFERENCED = [
    # "runs/run1",
    # "runs/run2",
]

# Subdirectories within the ablation/run folder to completely skip
IGNORE_FOLDERS = ["results"]


def _resolve_run_config_path(folder_path):
    manifest_candidates = [
        os.path.join(folder_path, "manifest.json"),
        os.path.join(folder_path, "run_manifest.json"),
    ]

    for manifest_path in manifest_candidates:
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            config_path = manifest.get("config_snapshot_path") or manifest.get("config_path")
            if config_path and os.path.exists(config_path):
                return config_path

    for fallback_name in ["config.yml", "config.resolved.yml"]:
        fallback_path = os.path.join(folder_path, fallback_name)
        if os.path.exists(fallback_path):
            return fallback_path

    raise FileNotFoundError(f"Could not resolve a config file for run folder: {folder_path}")


def _ensure_matching_split(cfg, folder_path):
    from utils.utils import resolve_data_root
    from data_processing.preprocess_data import ensure_test_split_current

    data_root = resolve_data_root(cfg, REPO_ROOT)
    split_seed = ensure_test_split_current(
        cfg, data_root, log=lambda msg: print(f"🔄 {msg} ({folder_path})")
    )
    print(f"✅ Split ready (seed={split_seed}) for {folder_path}")


# ==============================================================================
# EXECUTION LOGIC
# ==============================================================================

def run_inference_on_folder(folder_path, plot_formats):
    print(f"\n" + "="*80)
    print(f"🚀 Processing: {folder_path}")
    print("="*80 + "\n")

    from utils.utils import load_config

    cfg_path = _resolve_run_config_path(folder_path)
    cfg = load_config(cfg_path)
    _ensure_matching_split(cfg, folder_path)
    
    # Explicitly specify results subfolder inside the run folder
    results_dir = os.path.join(folder_path, "results")
    
    test_script = REPO_ROOT / "scripts" / "test.py"
    cmd = [
        sys.executable, str(test_script),
        "--run_folder", folder_path,
        "--results_dir", results_dir,
        "--plot_format", *plot_formats
    ]
    
    try:
        subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    except subprocess.CalledProcessError as e:
        print(f"❌ Error occurred while processing: {folder_path}")
        print(f"Details: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Batch inference runner")
    parser.add_argument(
        "--plot_format", 
        nargs="+", 
        choices=["png", "svg", "html", "all"], 
        default=["all"],
        help="Plot formats to generate (default: all)"
    )
    parser.add_argument(
        "--run_dir", 
        type=str, 
        default=None, 
        help="Directory containing sub-run folders, or a single run directory."
    )
    args = parser.parse_args()

    folders_to_process = []

    # 1. Add explicitly listed folders from top configuration
    if RUN_FOLDERS_TO_BE_INFERENCED:
        folders_to_process.extend(RUN_FOLDERS_TO_BE_INFERENCED)

    # 2. Add folders specified by --run_dir argument
    if args.run_dir:
        run_path = Path(args.run_dir)
        if run_path.is_dir():
            # Find subdirectories inside run_dir that aren't ignored
            subdirs = [
                str(d) for d in run_path.iterdir() 
                if d.is_dir() and d.name not in IGNORE_FOLDERS
            ]
            
            if subdirs:
                subdirs.sort()
                folders_to_process.extend(subdirs)
            else:
                # If no child run folders exist, assume run_dir itself is the run directory
                folders_to_process.append(str(run_path))
        else:
            print(f"⚠️ Warning: Provided --run_dir does not exist: {args.run_dir}")

    # Remove duplicates while preserving order
    folders_to_process = list(dict.fromkeys(folders_to_process))

    if not folders_to_process:
        print("⚠️ No folders found to process. Please check your CLI arguments or script config.")
        return

    print(f"📋 Total runs found to process: {len(folders_to_process)}")

    # Loop through and run inference sequentially
    for folder in folders_to_process:
        absolute_folder = os.path.abspath(folder)
        if os.path.exists(absolute_folder):
            run_inference_on_folder(absolute_folder, args.plot_format)
        else:
            print(f"⚠️ Warning: Folder path does not exist, skipping: {folder}")

    print("\n✅ All runs processed successfully.")


if __name__ == "__main__":
    main()