"""Strip optimizer_state_dict from existing checkpoints to reduce file size by ~70%.

Usage:
    python scripts/strip_optimizer_from_checkpoints.py --run-dir runs/<run_name>  # All .pth in folder
    python scripts/strip_optimizer_from_checkpoints.py --run-dir runs/<run_name>/checkpoint_best.pth  # Single file
    python scripts/strip_optimizer_from_checkpoints.py --glob "runs/*/checkpoint_best.pth"
    python scripts/strip_optimizer_from_checkpoints.py --all-runs  # Process all runs/ recursively
"""

import argparse
import glob
import os
import sys

import torch


def get_checkpoint_size_mb(path):
    """Return file size in MB."""
    return os.path.getsize(path) / (1024 ** 2)


def strip_checkpoint(checkpoint_path, dry_run=False, verbose=True):
    """
    Load checkpoint, remove optimizer_state_dict, re-save to same path.

    Args:
        checkpoint_path: Path to checkpoint file
        dry_run: If True, don't actually write; just report what would happen
        verbose: Print progress

    Returns:
        Tuple of (success: bool, old_size_mb: float, new_size_mb: float, had_optimizer: bool)
    """
    checkpoint_path = os.path.abspath(checkpoint_path)

    if not os.path.exists(checkpoint_path):
        if verbose:
            print(f"[NOT FOUND] {checkpoint_path}")
        return False, 0.0, 0.0, False

    old_size_mb = get_checkpoint_size_mb(checkpoint_path)

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        if verbose:
            print(f"[LOAD ERROR] {checkpoint_path}")
            print(f"   Error: {e}")
        return False, old_size_mb, 0.0, False

    had_optimizer = 'optimizer_state_dict' in checkpoint

    if not had_optimizer:
        if verbose:
            print(f"[SKIP] No optimizer state: {checkpoint_path} ({old_size_mb:.1f} MB)")
        return True, old_size_mb, old_size_mb, False

    # Remove optimizer state
    del checkpoint['optimizer_state_dict']

    if dry_run:
        # Estimate new size (in-memory size is approximate)
        temp_file = checkpoint_path + ".tmp_size_estimate"
        try:
            torch.save(checkpoint, temp_file)
            new_size_mb = get_checkpoint_size_mb(temp_file)
            os.remove(temp_file)
        except Exception:
            new_size_mb = old_size_mb * 0.3  # rough estimate
    else:
        # Re-save to same path
        temp_file = checkpoint_path + ".tmp"
        try:
            torch.save(checkpoint, temp_file)
            os.replace(temp_file, checkpoint_path)
            new_size_mb = get_checkpoint_size_mb(checkpoint_path)
        except Exception as e:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            if verbose:
                print(f"[SAVE ERROR] {checkpoint_path}")
                print(f"   Error: {e}")
            return False, old_size_mb, 0.0, True

    savings_mb = old_size_mb - new_size_mb
    savings_pct = 100.0 * savings_mb / old_size_mb if old_size_mb > 0 else 0.0

    if verbose:
        status = "[DRY RUN]" if dry_run else "[STRIPPED]"
        print(
            f"{status} {checkpoint_path}\n"
            f"         {old_size_mb:.1f} MB -> {new_size_mb:.1f} MB "
            f"(saved {savings_mb:.1f} MB, {savings_pct:.1f}%)"
        )

    return True, old_size_mb, new_size_mb, had_optimizer


def main():
    parser = argparse.ArgumentParser(
        description="Strip optimizer_state_dict from checkpoints to reduce file size",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single run folder (strips all .pth files in it)
  python scripts/strip_optimizer_from_checkpoints.py --run-dir runs/20260731_214607_bs16_gated_concat

  # Single checkpoint file
  python scripts/strip_optimizer_from_checkpoints.py --run-dir runs/20260731_214607_bs16_gated_concat/checkpoint_best.pth

  # Glob pattern
  python scripts/strip_optimizer_from_checkpoints.py --glob "runs/*/checkpoint_best.pth"

  # All runs recursively
  python scripts/strip_optimizer_from_checkpoints.py --all-runs

  # Dry run (preview changes without writing)
  python scripts/strip_optimizer_from_checkpoints.py --all-runs --dry-run
        """
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        help="Run directory or checkpoint file path. If directory, finds all *.pth files in it."
    )
    parser.add_argument(
        "--glob",
        type=str,
        help="Glob pattern for checkpoint files (e.g. 'runs/*/checkpoint_best.pth')"
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Process all checkpoint_best.pth files under runs/ recursively"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing to disk"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file output; only show summary"
    )

    args = parser.parse_args()

    # Determine which checkpoints to process
    checkpoint_paths = []

    if args.run_dir:
        # Support both single checkpoint file and run directory (find all .pth files)
        run_dir = os.path.abspath(args.run_dir)
        if os.path.isfile(run_dir):
            # User passed a file path directly
            checkpoint_paths = [run_dir]
        elif os.path.isdir(run_dir):
            # User passed a directory; find all .pth checkpoint files in it
            checkpoint_paths = glob.glob(os.path.join(run_dir, "*.pth"))
            if not checkpoint_paths:
                # Fallback: look for checkpoint_best.pth specifically
                fallback = os.path.join(run_dir, "checkpoint_best.pth")
                if os.path.exists(fallback):
                    checkpoint_paths = [fallback]
        else:
            checkpoint_paths = []
    elif args.glob:
        checkpoint_paths = glob.glob(args.glob)
    elif args.all_runs:
        checkpoint_paths = glob.glob("runs/**/checkpoint_best.pth", recursive=True)
    else:
        parser.print_help()
        sys.exit(1)

    if not checkpoint_paths:
        print("[ERROR] No checkpoints found matching criteria")
        sys.exit(1)

    print(f"[INFO] Found {len(checkpoint_paths)} checkpoint(s) to process")
    if args.dry_run:
        print("       [DRY RUN MODE - no files will be modified]")
    print()

    # Process each checkpoint
    results = {
        'processed': 0,
        'stripped': 0,
        'skipped': 0,
        'failed': 0,
        'total_old_size_mb': 0.0,
        'total_new_size_mb': 0.0,
    }

    for ckpt_path in sorted(checkpoint_paths):
        success, old_mb, new_mb, had_opt = strip_checkpoint(
            ckpt_path,
            dry_run=args.dry_run,
            verbose=not args.quiet
        )

        results['processed'] += 1
        if had_opt:
            results['stripped'] += 1
        else:
            results['skipped'] += 1
        if not success:
            results['failed'] += 1

        results['total_old_size_mb'] += old_mb
        results['total_new_size_mb'] += new_mb

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Processed:    {results['processed']}")
    print(f"Stripped:     {results['stripped']}")
    print(f"Skipped:      {results['skipped']} (no optimizer state)")
    print(f"Failed:       {results['failed']}")

    total_savings_mb = results['total_old_size_mb'] - results['total_new_size_mb']
    total_savings_pct = (
        100.0 * total_savings_mb / results['total_old_size_mb']
        if results['total_old_size_mb'] > 0 else 0.0
    )

    print()
    print(f"Total size:   {results['total_old_size_mb']:.1f} MB -> {results['total_new_size_mb']:.1f} MB")
    print(f"Saved:        {total_savings_mb:.1f} MB ({total_savings_pct:.1f}%)")

    if args.dry_run:
        print()
        print("[DRY RUN] No changes were written. Re-run without --dry-run to apply.")

    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
