#!/usr/bin/env python3
"""
Fully Dynamic Ablation Study Analyzer.
Automatically detects *any* configuration changes across runs and groups results.
"""

import pandas as pd
import glob
import os
from pathlib import Path
import argparse
import yaml

def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """Flattens a nested dictionary, concatenating keys with a separator."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def analyze_any_ablation(runs_dir: str = "runs", output_file: str = "ablation_results.csv"):
    """Dynamically detects configuration variances and analyzes run metrics."""
    
    metrics_files = glob.glob(os.path.join(runs_dir, "*/metrics.csv"))
    if not metrics_files:
        print(f"No runs with metrics.csv found in: {runs_dir}")
        return

    print(f"Processing {len(metrics_files)} runs...")

    all_runs_data = []
    all_configs = {}  # run_name -> flattened_config_dict

    # 1. First Pass: Gather Metrics and Flattened Configurations
    for metrics_file in metrics_files:
        run_dir = os.path.dirname(metrics_file)
        run_name = os.path.basename(run_dir)
        config_path = Path(run_dir) / "config.yml"
        
        # Load Config if it exists
        flat_config = {}
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    raw_config = yaml.safe_load(f) or {}
                    flat_config = flatten_dict(raw_config)
            except Exception as e:
                print(f"Warning: Failed to parse config for {run_name}: {e}")
        
        all_configs[run_name] = flat_config

        # Load Metrics
        try:
            df = pd.read_csv(metrics_file)
            if df.empty:
                continue
            final_row = df.iloc[-1]
            
            run_data = {
                "run_name": run_name,
                "final_epoch": int(final_row.get("epoch", len(df))),
            }
            # Gather all logged metrics
            for col in df.columns:
                if col not in ["epoch", "step"]:
                    run_data[f"final_{col}"] = final_row[col]
                    
            all_runs_data.append(run_data)
        except Exception as e:
            print(f"Error reading metrics for {run_name}: {e}")

    if not all_runs_data:
        print("No valid metrics extracted.")
        return

    # 2. Identify Ablated Hyperparameters (Keys with more than 1 unique value)
    config_df = pd.DataFrame.from_dict(all_configs, orient='index')
    
    # Drop columns that are completely missing everywhere, or ignore irrelevant metadata seeds
    ignore_keys = ["data.augmentation.seed", "training.uncertainty_weighting.init_log_var_rot", "training.uncertainty_weighting.init_log_var_trans"]
    config_df = config_df.drop(columns=[c for c in ignore_keys if c in config_df.columns], errors='ignore')
    
    # Find which config keys actually vary across the folders
    unique_counts = config_df.nunique()
    ablated_keys = unique_counts[unique_counts > 1].index.tolist()

    print("\n" + "-"*60)
    if ablated_keys:
        print(f"Detected Ablated Variables: {', '.join(ablated_keys)}")
    else:
        print("Detected No Varied Parameters! (All config.yml files look identical or are missing). Using run names as fallback.")
    print("-"*60)

    # 3. Create Variant Names Based on Changing Hyperparameters
    variant_mapping = {}
    for run_name, config in all_configs.items():
        if ablated_keys:
            # Construct a string label like "model.fusion=spatial_gated_2d | training.lr_schedule.type=one_cycle"
            label_parts = [f"{key}={config.get(key, 'N/A')}" for key in ablated_keys]
            variant_mapping[run_name] = " | ".join(label_parts)
        else:
            variant_mapping[run_name] = run_name # Fallback to folder name if configs are identical

    # Merge Variant Mapping and configurations into the final results dataframe
    results_df = pd.DataFrame(all_runs_data)
    results_df["variant"] = results_df["run_name"].map(variant_mapping)
    
    # Optional: Join the config columns to the final CSV output for deep inspection
    results_df = results_df.merge(config_df, left_on="run_name", right_index=True, how="left")
    results_df.to_csv(output_file, index=False)
    print(f"Saved complete dynamic results matrix to: {output_file}")

    # 4. Aggregate & Display Results
    discovered_variants = results_df["variant"].unique()
    ignore_cols = ["run_name", "variant", "final_epoch"] + list(config_df.columns)
    metrics_to_show = [col for col in results_df.columns if col not in ignore_cols and pd.api.types.is_numeric_dtype(results_df[col])]
    
    print("\n" + "="*100)
    print("DYNAMIC CONFIGURATION ABLATION STUDY RESULTS")
    print("="*100)
    
    summary = []
    for variant in discovered_variants:
        group = results_df[results_df["variant"] == variant]
        print(f"\nVariant Set: {variant}\n(n={len(group)} runs)")
        print("-" * 100)
        
        for metric in metrics_to_show:
            mean_val = group[metric].mean()
            std_val = group[metric].std()
            std_str = f"{std_val:8.4f}" if not pd.isna(std_val) else "  N/A  "
            
            if not pd.isna(mean_val):
                print(f"  {metric:25s}: {mean_val:8.4f} ± {std_str}")
                summary.append({"variant": variant, "metric": metric, "mean": mean_val, "std": std_val if not pd.isna(std_val) else 0.0})

    # 5. Leaderboard Ranking
    print("\n" + "="*100)
    print("RANKINGS (Lower Mean is Better)")
    print("="*100)
    
    if not summary:
        return
    summary_df = pd.DataFrame(summary)
    
    for metric in metrics_to_show:
        metric_group = summary_df[summary_df["metric"] == metric]
        if metric_group.empty:
            continue
        ranking = metric_group.sort_values("mean")
        print(f"\n🏆 Top Performers for {metric}:")
        for idx, (_, row) in enumerate(ranking.iterrows(), 1):
            if idx <= 5:
                print(f"  {idx}. {row['variant']} -> Mean: {row['mean']:8.4f} (± {row['std']:8.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Config Dynamic Ablation Analyzer")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Directory containing run folders")
    parser.add_argument("--output", type=str, default="dynamic_study_results.csv", help="Output CSV file")
    
    args = parser.parse_args()
    analyze_any_ablation(runs_dir=args.runs_dir, output_file=args.output)