#!/usr/bin/env python3
"""
Analyze tactile augmentation ablation study results.
Compares metrics across all variants and seeds.
"""

import pandas as pd
import glob
import os
from pathlib import Path
import argparse

VARIANTS = [
    "no_aug",
    "noise_only",
    "brightness_only",
    "contrast_only",
    "blur_only",
    "channel_scaling_only",
    "all",
]


def analyze_ablation_results(runs_dir: str = "runs", output_file: str = "ablation_results.csv"):
    """Analyze all ablation study runs and compare metrics."""
    
    results = []
    
    # Collect metrics from all runs
    for metrics_file in glob.glob(os.path.join(runs_dir, "*/metrics.csv")):
        run_dir = os.path.dirname(metrics_file)
        run_name = os.path.basename(run_dir)
        
        try:
            df = pd.read_csv(metrics_file)
            final_row = df.iloc[-1] if len(df) > 0 else None
            
            if final_row is not None:
                # 1. Fallback variant identification from run_name
                variant = "unknown"
                for v in VARIANTS:
                    if v in run_name:
                        variant = v
                        break
                
                # 2. Extract explicit augmentation setting from results.txt if it exists
                results_txt_path = Path(run_dir) / "results.txt"
                if results_txt_path.exists():
                    with open(results_txt_path, "r") as f:
                        for line in f:
                            # FIXED: Match your actual format 'augmentation=all'
                            if line.startswith("augmentation="):
                                variant = line.strip().split("=")[1].strip()
                                break
                
                # FIXED: Mapped 'train_loss' and set a fallback for missing val_loss
                results.append({
                    "run_name": run_name,
                    "variant": variant,
                    "final_epoch": int(final_row.get("epoch", len(df))),
                    "final_val_loss": final_row.get("val_loss", None), # Keeps None safely if missing
                    "final_val_abs_trans": final_row.get("val_abs_trans", None),
                    "final_val_abs_trans_y": final_row.get("val_abs_trans_y", None),
                    "final_val_abs_trans_z": final_row.get("val_abs_trans_z", None),
                    "final_val_abs_rot_x": final_row.get("val_abs_rot_x", None),
                    "final_train_loss": final_row.get("train_loss", None), # FIXED: was 'loss'
                })
                
        except Exception as e:
            print(f"Error reading {metrics_file}: {e}")
    
    if not results:
        print("No valid results found! Check your directory paths or file structures.")
        return
    
    results_df = pd.DataFrame(results)
    
    # Save raw results
    results_df.to_csv(output_file, index=False)
    print(f"\nSaved results to {output_file}")
    
    # Aggregate by variant
    print("\n" + "="*100)
    print("TACTILE AUGMENTATION ABLATION STUDY RESULTS")
    print("="*100)
    
    summary = []
    
    # Determine metrics to show based on what actually exists in data
    metrics_to_show = ["final_val_abs_trans", "final_val_abs_trans_y", "final_val_abs_trans_z", "final_val_abs_rot_x", "final_train_loss"]
    if results_df["final_val_loss"].notna().any():
        metrics_to_show.insert(0, "final_val_loss")

    for variant in VARIANTS:
        group = results_df[results_df["variant"] == variant]
        
        if len(group) == 0:
            print(f"\n{variant}: NO RUNS FOUND")
            continue
        
        print(f"\n{variant} (n={len(group)} runs)")
        print("-" * 100)
        
        for metric in metrics_to_show:
            mean_val = group[metric].mean()
            std_val = group[metric].std()
            # Handle cases where std is NaN (e.g., only 1 run) cleanly
            std_str = f"{std_val:8.4f}" if not pd.isna(std_val) else "  N/A  "
            print(f"  {metric:25s}: {mean_val:8.4f} ± {std_str}")
            
            summary.append({
                "variant": variant,
                "metric": metric,
                "mean": mean_val,
                "std": std_val if not pd.isna(std_val) else 0.0,
            })
    
    # Comparison table
    print("\n" + "="*100)
    print("COMPARISON: Best Performance by Metric")
    print("="*100)
    
    if not summary:
        return
        
    summary_df = pd.DataFrame(summary)
    
    # Rank validation metrics
    rank_metrics = [m for m in ["final_val_abs_trans", "final_val_abs_trans_y", "final_val_abs_trans_z", "final_val_abs_rot_x"] if m in metrics_to_show]
    
    for metric in rank_metrics:
        metric_group = summary_df[summary_df["metric"] == metric]
        if metric_group.empty:
            continue
            
        best_idx = metric_group["mean"].idxmin()
        best_variant = metric_group.loc[best_idx, "variant"]
        best_mean = metric_group.loc[best_idx, "mean"]
        best_std = metric_group.loc[best_idx, "std"]
        
        print(f"\n{metric}:")
        print(f"  Best: {best_variant:30s} = {best_mean:8.4f} ± {best_std:8.4f}")
        
        # Show ranking
        ranking = metric_group.sort_values("mean")
        for idx, (_, row) in enumerate(ranking.iterrows(), 1):
            if idx <= 7:
                print(f"  {idx}. {row['variant']:28s} = {row['mean']:8.4f} ± {row['std']:8.4f}")
    
    print("\n" + "="*100)
    print("INTERPRETATION GUIDE:")
    print("="*100)
    print("""
Metrics to optimize (lower is better):
  - final_val_abs_trans: Mean absolute translation error (meters)
  - final_val_abs_trans_y: Mean absolute translation error along Y-axis (meters)
  - final_val_abs_trans_z: Mean absolute translation error along Z-axis (meters)
  - final_val_abs_rot_x: Mean absolute rotation error around X-axis (radians)

Interpretation:
  - If ALL variants similar: The augmentation parameters don't significantly help
  - If ONE parameter dominates: That's the most important augmentation
  - If COMBINATION better than singles: Parameters work synergistically
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze tactile ablation study results")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Directory containing run folders")
    parser.add_argument("--output", type=str, default="ablation_results2.csv", help="Output CSV file")
    
    args = parser.parse_args()
    analyze_ablation_results(runs_dir=args.runs_dir, output_file=args.output)