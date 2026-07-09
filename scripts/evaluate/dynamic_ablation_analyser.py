#!/usr/bin/env python3
"""
Fully Dynamic Ablation Study Analyzer with Plotting.
Automatically detects configuration changes across runs, aggregates results,
and plots training curves for specified metrics and losses over epochs.
"""

import pandas as pd
import glob
import os
from pathlib import Path
import argparse
import yaml
import matplotlib.pyplot as plt

def flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """Flattens a nested dictionary, converting lists to tuples so they are hashable."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            tuple_val = tuple(flatten_dict(x) if isinstance(x, dict) else x for x in v)
            items.append((new_key, tuple_val))
        else:
            items.append((new_key, v))
    return dict(items)


def plot_ablation_curves(history_df: pd.DataFrame, metrics_to_plot: list, output_dir: str = None, default_parent: str = "runs"):
    """Plots training/validation curves over epochs, aggregating seeds per variant."""
    if history_df.empty:
        return
        
    # Implement user-specified output directory routing logic for plots
    if output_dir is None:
        # If no output directory is specified, default to the user specified runs directory/plots
        output_dir = Path(default_parent) / "results" / "plots"
    
    os.makedirs(output_dir, exist_ok=True)
    
    variants = history_df['variant'].unique()
    cmap = plt.get_cmap('tab10')
    variant_colors = {variant: cmap(i % 10) for i, variant in enumerate(variants)}
    
    for metric in metrics_to_plot:
        if metric not in history_df.columns:
            continue
            
        fig, ax = plt.subplots(figsize=(11, 6))
        has_data = False
        
        for variant in variants:
            variant_df = history_df[history_df['variant'] == variant]
            clean_df = variant_df.dropna(subset=[metric])
            if clean_df.empty:
                continue
                
            has_data = True
            grouped = clean_df.groupby('epoch')[metric].agg(['mean', 'std']).reset_index()
            color = variant_colors[variant]
            display_label = variant if len(variant) <= 50 else variant[:47] + "..."
            
            ax.plot(grouped['epoch'], grouped['mean'], label=display_label, color=color, linewidth=2)
            
            if 'std' in grouped.columns and grouped['std'].notna().any() and (grouped['std'] > 0).any():
                ax.fill_between(
                    grouped['epoch'], 
                    grouped['mean'] - grouped['std'], 
                    grouped['mean'] + grouped['std'], 
                    color=color, 
                    alpha=0.15
                )
        
        if not has_data:
            plt.close(fig)
            continue
            
        ax.set_title(f'{metric} performance over Epochs', fontsize=13, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'{metric}_curve.png')
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Generated learning curve plot: {plot_path}")


def analyze_any_ablation(runs_dir: str = "runs", output_file: str = None, plots_dir: str = None):
    """Dynamically detects configuration variances, analyzes run metrics, and generates plots."""
    
    metrics_files = glob.glob(os.path.join(runs_dir, "*/metrics.csv"))
    if not metrics_files:
        print(f"No runs with metrics.csv found in: {runs_dir}")
        return

    print(f"Processing {len(metrics_files)} runs...")

    all_runs_data = []
    all_histories_dfs = []
    all_configs = {}

    for metrics_file in metrics_files:
        run_dir = os.path.dirname(metrics_file)
        run_name = os.path.basename(run_dir)
        config_path = Path(run_dir) / "config.yml"
        
        flat_config = {}
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    raw_config = yaml.safe_load(f) or {}
                    flat_config = flatten_dict(raw_config)
            except Exception as e:
                print(f"Warning: Failed to parse config for {run_name}: {e}")
        
        all_configs[run_name] = flat_config

        try:
            df = pd.read_csv(metrics_file)
            if df.empty:
                continue
            
            all_histories_dfs.append((run_name, df))
            final_row = df.iloc[-1]
            run_data = {
                "run_name": run_name,
                "final_epoch": int(final_row.get("epoch", len(df))),
            }
            for col in df.columns:
                if col not in ["epoch", "step", "timestamp", "checkpoint_path", "model_name", "streams", "fusion", "head_type"]:
                    run_data[f"final_{col}"] = final_row[col]
                    
            all_runs_data.append(run_data)
        except Exception as e:
            print(f"Error reading metrics for {run_name}: {e}")

    if not all_runs_data:
        print("No valid metrics extracted.")
        return

    config_df = pd.DataFrame.from_dict(all_configs, orient='index')
    ignore_keys = ["data.augmentation.seed", "training.uncertainty_weighting.init_log_var_rot", "training.uncertainty_weighting.init_log_var_trans", "seed"]
    config_df = config_df.drop(columns=[c for c in ignore_keys if c in config_df.columns], errors='ignore')
    
    unique_counts = config_df.nunique()
    ablated_keys = unique_counts[unique_counts > 1].index.tolist()

    print("\n" + "-"*60)
    if ablated_keys:
        print(f"Detected Ablated Variables: {', '.join(ablated_keys)}")
    else:
        print("Detected No Varied Parameters! (All config.yml files look identical). Using run names as fallback.")
    print("-"*60)

    variant_mapping = {}
    for run_name, config in all_configs.items():
        if ablated_keys:
            label_parts = [f"{key}={config.get(key, 'N/A')}" for key in ablated_keys]
            variant_mapping[run_name] = " | ".join(label_parts)
        else:
            variant_mapping[run_name] = run_name

    results_df = pd.DataFrame(all_runs_data)
    results_df["variant"] = results_df["run_name"].map(variant_mapping)
    results_df = results_df.merge(config_df, left_on="run_name", right_index=True, how="left")
    
    # UPDATED: If no explicit output file path is given, save in active runs directory under 'results'
    if output_file is None:
        results_dir = Path(runs_dir) / "results"
        os.makedirs(results_dir, exist_ok=True)
        output_file = str(results_dir / "dynamic_study_results.csv")
        
    results_df.to_csv(output_file, index=False)
    print(f"Saved complete dynamic results summary matrix to: {output_file}")

    processed_histories = []
    for run_name, df in all_histories_dfs:
        df = df.copy()
        df["run_name"] = run_name
        df["variant"] = variant_mapping.get(run_name, run_name)
        processed_histories.append(df)
        
    history_df = pd.concat(processed_histories, ignore_index=True)

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

    print("\n" + "="*100)
    print("RANKINGS (Lower Mean is Better)")
    print("="*100)
    
    if summary:
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

    print("\n" + "="*100)
    print("GENERATING ACCLAIMED PERFORMANCE CURVES")
    print("="*100)
    
    loss_metrics = [col for col in history_df.columns if 'loss' in col.lower() and col not in ['best_loss', 'checkpoint_path']]
    requested_metrics = ['train_abs_y', 'train_abs_z', 'train_abs_rot_x']
    metrics_to_plot = list(set(requested_metrics + loss_metrics))
    
    # Call plotting with the dynamic routing configuration
    plot_ablation_curves(history_df, metrics_to_plot, output_dir=plots_dir, default_parent=runs_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Config Dynamic Ablation Analyzer with Plotting")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Directory containing run folders")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file (Defaults to inside --runs-dir/results)")
    parser.add_argument("--plots-dir", type=str, default=None, help="Directory to save generated plots (Defaults to inside --runs-dir/plots)")
    
    args = parser.parse_args()
    analyze_any_ablation(runs_dir=args.runs_dir, output_file=args.output, plots_dir=args.plots_dir)