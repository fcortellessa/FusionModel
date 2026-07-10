"""
This script runs test.py for multiple run folders.
"""

import os
import subprocess

# ==============================================================================
# CONFIGURATION: Paste your folders here
# ==============================================================================

# Option 1: Paste an explicit list of run folders
RUN_FOLDERS_TO_BE_INFERENCED = [
    # "runs/run1",
    # "runs/run2",
]

# Option 2: Provide a parent directory (it will scan all direct subdirectories)
ABLATION_FOLDER = "runs/data_augmentation/"

# Subdirectories within the ablation folder to completely skip
IGNORE_FOLDERS = ["results"]

# ==============================================================================
# EXECUTION LOGIC
# ==============================================================================

def run_inference_on_folder(folder_path):
    print(f"\n" + "="*80)
    print(f"🚀 Processing: {folder_path}")
    print("="*80 + "\n")
    
    # Constructs the exact command line arguments for your test.py
    cmd = ["python", "test.py", "--run_folder", folder_path]
    
    try:
        # Runs the script and streams output directly to your current console
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error occurred while processing: {folder_path}")
        print(f"Details: {e}\n")

def main():
    folders_to_process = []

    # 1. Add explicitly listed folders
    if RUN_FOLDERS_TO_BE_INFERENCED:
        folders_to_process.extend(RUN_FOLDERS_TO_BE_INFERENCED)

    # 2. Automatically find and add subfolders from the ablation directory
    if ABLATION_FOLDER and os.path.isdir(ABLATION_FOLDER):
        subdirs = []
        for d in os.listdir(ABLATION_FOLDER):
            # Check if it's a directory and NOT in our ignore list
            if os.path.isdir(os.path.join(ABLATION_FOLDER, d)) and d not in IGNORE_FOLDERS:
                subdirs.append(os.path.join(ABLATION_FOLDER, d))
        subdirs.sort()  # Sorts them alphabetically/numerically
        folders_to_process.extend(subdirs)

    # Remove duplicates while preserving order
    folders_to_process = list(dict.fromkeys(folders_to_process))

    if not folders_to_process:
        print("⚠️ No folders found to process. Please check your configuration at the top of the script.")
        return

    print(f"📋 Total runs found to process: {len(folders_to_process)}")

    # Loop through and run inference sequentially
    for folder in folders_to_process:
        if os.path.exists(folder):
            run_inference_on_folder(folder)
        else:
            print(f"⚠️ Warning: Folder path does not exist, skipping: {folder}")

    print("\n✅ All runs processed successfully.")

if __name__ == "__main__":
    main()
