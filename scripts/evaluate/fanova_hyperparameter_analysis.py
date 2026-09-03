#!/usr/bin/env python3
"""
Functional ANOVA (fANOVA) Hyperparameter Relevance Analysis
============================================================

Analyzes the importance of hyperparameters on model performance using functional
ANOVA decomposition. This script quantifies how much each hyperparameter (or
combination of hyperparameters) explains the variance in model performance.

Key concepts:
- Identifies which hyperparameters have the strongest effect on the performance metric
- Computes partial dependence functions showing how performance changes with each param
- Decomposes total variance into main effects and interaction effects
- Supports both numerical and categorical hyperparameters
- Handles missing/incomplete hyperparameter sweeps gracefully

Usage:
    python fanova_hyperparameter_analysis.py --path runs/ --metric val_loss --output results/fanova/
    python fanova_hyperparameter_analysis.py --path runs/ --metric test_rotation_error --output results/fanova/ --n-points 50

Author: Claude Code
Date: 2026-09-03
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import interpolate, stats
from scipy.spatial.distance import pdist, squareform

# Try to import optional dependencies
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("plotly not available; interactive HTML plots will be skipped")

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.utils import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class HyperparameterInfo:
    """Metadata about a hyperparameter."""
    name: str
    param_type: str  # 'numerical', 'categorical'
    values: np.ndarray
    is_log_scale: bool = False

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return isinstance(other, HyperparameterInfo) and self.name == other.name


class FunctionalANOVA:
    """
    Functional ANOVA decomposition for hyperparameter sensitivity analysis.

    Decomposes the performance function as:
        f(x1, x2, ..., xd) = f0 + Σ fi(xi) + Σ fij(xi, xj) + ... + residual

    Where:
    - f0: global mean performance
    - fi(xi): main effect of hyperparameter i
    - fij(xi, xj): interaction effect between parameters i and j
    - The variance explained by each term quantifies its importance
    """

    def __init__(self, data: pd.DataFrame, metric_name: str,
                 param_names: List[str], n_points: int = 50):
        """
        Initialize fANOVA analyzer.

        Args:
            data: DataFrame with columns [param_names..., metric_name]
            metric_name: name of the performance metric column
            param_names: list of hyperparameter column names
            n_points: number of points for partial dependence interpolation
        """
        self.data = data.copy()
        self.metric_name = metric_name
        self.param_names = param_names
        self.n_points = n_points

        # Filter to complete cases
        required_cols = param_names + [metric_name]
        self.data = self.data[required_cols].dropna()

        if len(self.data) < 3:
            raise ValueError(f"Need at least 3 complete samples, got {len(self.data)}")

        # Categorize hyperparameters
        self.param_info = self._categorize_params()

        # Normalize numerical params to [0, 1] for fair comparison
        self._normalize_numerical_params()

        # Store performance metric
        self.y = self.data[metric_name].values
        self.y_mean = np.mean(self.y)
        self.y_std = np.std(self.y)
        self.total_variance = np.var(self.y)

        logger.info(f"Initialized fANOVA with {len(self.data)} samples")
        logger.info(f"Analyzing {len(self.param_names)} hyperparameters")
        logger.info(f"Metric: {metric_name} (mean={self.y_mean:.4f}, std={self.y_std:.4f})")

    def _categorize_params(self) -> Dict[str, HyperparameterInfo]:
        """Categorize hyperparameters as numerical or categorical."""
        param_info = {}

        for param in self.param_names:
            col = self.data[param]
            unique_vals = col.unique()

            # Check if parameter is numerical
            try:
                numeric_vals = pd.to_numeric(col)
                is_numerical = True
            except (ValueError, TypeError):
                is_numerical = False

            # Decide: numerical if >5 unique values and numeric type, else categorical
            if is_numerical and len(unique_vals) > 5:
                param_info[param] = HyperparameterInfo(
                    name=param,
                    param_type='numerical',
                    values=np.sort(np.unique(pd.to_numeric(col)))
                )
                logger.info(f"  {param}: NUMERICAL ({len(unique_vals)} values)")
            else:
                param_info[param] = HyperparameterInfo(
                    name=param,
                    param_type='categorical',
                    values=np.array(sorted(set(col.astype(str))))
                )
                logger.info(f"  {param}: CATEGORICAL ({len(unique_vals)} values)")

        return param_info

    def _normalize_numerical_params(self):
        """Normalize numerical parameters to [0, 1]."""
        for param, info in self.param_info.items():
            if info.param_type == 'numerical':
                col_vals = pd.to_numeric(self.data[param])
                min_val = col_vals.min()
                max_val = col_vals.max()

                if max_val > min_val:
                    self.data[param] = (col_vals - min_val) / (max_val - min_val)
                    # Update param info with normalized values
                    info.values = np.sort(np.unique(self.data[param]))

    def compute_main_effects(self) -> Dict[str, Dict[str, Any]]:
        """
        Compute main effects for each hyperparameter.

        Main effect: how performance changes when varying one parameter
        while averaging over all others (marginalizing out other params).

        Returns:
            Dictionary mapping param_name -> {
                'x_vals': parameter values (for plotting),
                'y_vals': mean performance at each value,
                'y_err': std error of performance,
                'importance': variance explained by this parameter,
                'sparsity': fraction of unexplained variance after main effect
            }
        """
        main_effects = {}

        for param in self.param_names:
            logger.info(f"Computing main effect for {param}...")

            if self.param_info[param].param_type == 'categorical':
                # For categorical: compute mean performance per category
                grouped = self.data.groupby(param)[self.metric_name].agg(['mean', 'std', 'count'])
                x_vals = grouped.index.values
                y_vals = grouped['mean'].values
                y_err = grouped['std'].values / np.sqrt(grouped['count'].values)

            else:  # numerical
                # For numerical: fit spline to partial dependence
                x_vals, y_vals, y_err = self._fit_partial_dependence(param)

            # Compute importance: variance in partial dependence
            importance_variance = np.var(y_vals)
            importance_fraction = importance_variance / (self.total_variance + 1e-10)

            main_effects[param] = {
                'x_vals': x_vals,
                'y_vals': y_vals,
                'y_err': y_err,
                'importance': importance_variance,
                'importance_fraction': importance_fraction,
            }

            logger.info(f"  Importance: {importance_fraction:.4f} ({100*importance_fraction:.1f}% of total variance)")

        return main_effects

    def _fit_partial_dependence(self, param: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit partial dependence for a numerical hyperparameter.

        Partial dependence: average prediction when varying param, marginalizing others.
        Uses binning + spline interpolation for smooth curves.
        """
        # Bin the parameter into groups
        n_bins = min(10, len(np.unique(self.data[param])))
        bins = pd.cut(self.data[param], bins=n_bins, duplicates='drop')

        # Compute mean performance per bin
        grouped = self.data.groupby(bins)[self.metric_name].agg(['mean', 'std', 'count'])

        # Use bin centers as x-values
        bin_centers = np.array([interval.mid for interval in grouped.index])
        y_vals = grouped['mean'].values
        y_err = grouped['std'].values / np.sqrt(grouped['count'].values)

        # Fit cubic spline if we have enough points
        if len(bin_centers) >= 4:
            try:
                spl = interpolate.UnivariateSpline(
                    bin_centers, y_vals, k=min(3, len(bin_centers)-1),
                    s=np.sum(y_err**2)  # Smoothing based on error bars
                )
                x_fine = np.linspace(bin_centers.min(), bin_centers.max(), self.n_points)
                y_fine = spl(x_fine)
                y_err_fine = np.interp(x_fine, bin_centers, y_err)
                return x_fine, y_fine, y_err_fine
            except Exception as e:
                logger.warning(f"Spline fitting failed for {param}: {e}")

        # Fallback: return binned values
        return bin_centers, y_vals, y_err

    def compute_interaction_effects(self, top_k: int = 5) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """
        Compute pairwise interaction effects for the top-k most important parameters.

        Interaction effect: how the effect of one parameter depends on another.
        Quantified as unexplained variance after removing both main effects.

        Args:
            top_k: analyze interactions only for top-k hyperparameters by main effect

        Returns:
            Dictionary mapping (param1, param2) -> {
                'importance': interaction variance,
                'importance_fraction': interaction as fraction of total variance
            }
        """
        # Get main effects to identify top parameters
        main_effects = self.compute_main_effects()
        top_params = sorted(
            main_effects.items(),
            key=lambda x: x[1]['importance'],
            reverse=True
        )[:top_k]
        top_param_names = [p[0] for p in top_params]

        interaction_effects = {}

        for i, param1 in enumerate(top_param_names):
            for param2 in top_param_names[i+1:]:
                logger.info(f"Computing interaction effect between {param1} and {param2}...")

                # Group by both parameters
                grouped = self.data.groupby([param1, param2])[self.metric_name].agg(['mean', 'count'])

                if len(grouped) < 4:
                    logger.warning(f"  Insufficient data points for {param1} x {param2} interaction")
                    continue

                # ANOVA: compute residual after removing main effects
                residuals = self._compute_residuals_after_main_effects(param1, param2)
                interaction_variance = np.var(residuals)
                interaction_fraction = interaction_variance / (self.total_variance + 1e-10)

                interaction_effects[(param1, param2)] = {
                    'importance': interaction_variance,
                    'importance_fraction': interaction_fraction,
                }

                logger.info(f"  Interaction importance: {interaction_fraction:.4f} "
                          f"({100*interaction_fraction:.1f}% of total variance)")

        return interaction_effects

    def _compute_residuals_after_main_effects(self, param1: str, param2: str) -> np.ndarray:
        """Compute residuals after removing both main effects (for interaction analysis)."""
        residuals = np.zeros_like(self.y)

        # Subtract main effects
        for i, row in self.data.iterrows():
            pred = self.y_mean

            # Add param1 main effect
            mask1 = self.data[param1] == row[param1]
            pred += np.mean(self.y[mask1]) - self.y_mean

            # Add param2 main effect
            mask2 = self.data[param2] == row[param2]
            pred += np.mean(self.y[mask2]) - self.y_mean

            residuals[i] = self.y[i] - pred

        return residuals

    def compute_importance_ranking(self) -> pd.DataFrame:
        """
        Rank hyperparameters by importance.

        Returns:
            DataFrame with columns [hyperparameter, importance, importance_fraction,
                                    pct_of_total_variance]
        """
        main_effects = self.compute_main_effects()

        ranking = []
        for param, effects in main_effects.items():
            ranking.append({
                'Hyperparameter': param,
                'Importance (Variance)': effects['importance'],
                'Importance (Fraction)': effects['importance_fraction'],
                'Pct of Total Variance': 100 * effects['importance_fraction'],
                'Param Type': self.param_info[param].param_type,
            })

        df = pd.DataFrame(ranking).sort_values('Importance (Fraction)', ascending=False)
        return df

    def summary(self) -> str:
        """Generate a text summary of the fANOVA analysis."""
        ranking = self.compute_importance_ranking()

        summary_text = f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                  Functional ANOVA (fANOVA) Summary                        ║
╚═══════════════════════════════════════════════════════════════════════════╝

Metric Analyzed: {self.metric_name}
Number of Experiments: {len(self.data)}
Mean Performance: {self.y_mean:.6f}
Std Deviation: {self.y_std:.6f}
Total Variance: {self.total_variance:.6f}

═══════════════════════════════════════════════════════════════════════════
HYPERPARAMETER IMPORTANCE RANKING
═══════════════════════════════════════════════════════════════════════════

"""
        for idx, row in ranking.iterrows():
            bar_length = int(40 * row['Importance (Fraction)'])
            bar = '█' * bar_length + '░' * (40 - bar_length)
            summary_text += f"{idx+1:2d}. {row['Hyperparameter']:<30s} [{bar}] {row['Pct of Total Variance']:>6.2f}%\n"

        summary_text += """
═══════════════════════════════════════════════════════════════════════════
INTERPRETATION
═══════════════════════════════════════════════════════════════════════════

• Hyperparameter importance is measured as the variance in the model's
  performance metric when that parameter varies, averaged over all other
  parameter configurations.

• A high importance value (e.g., 20%) indicates that this hyperparameter
  has a strong effect on model performance and should be carefully tuned.

• Parameters with low importance (<5%) have minimal effect and can often
  be set to sensible defaults without extensive tuning.

• Interaction effects (not shown here) reveal when two parameters jointly
  affect performance in ways not captured by their individual effects.

═══════════════════════════════════════════════════════════════════════════
"""
        return summary_text


class FANOVAVisualizer:
    """Generates plots and visualizations for fANOVA results."""

    def __init__(self, fanova: FunctionalANOVA, output_dir: Path):
        self.fanova = fanova
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)

    def plot_importance_ranking(self) -> Path:
        """Plot hyperparameter importance as a horizontal bar chart."""
        ranking = self.fanova.compute_importance_ranking()

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = plt.cm.RdYlGn_r(ranking['Importance (Fraction)'] / ranking['Importance (Fraction)'].max())
        bars = ax.barh(ranking['Hyperparameter'], ranking['Pct of Total Variance'], color=colors)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, ranking['Pct of Total Variance'])):
            ax.text(val + 0.5, i, f'{val:.1f}%', va='center', fontsize=10)

        ax.set_xlabel('Importance (% of Total Variance)', fontsize=12)
        ax.set_ylabel('Hyperparameter', fontsize=12)
        ax.set_title(f'Hyperparameter Importance for {self.fanova.metric_name}', fontsize=14, fontweight='bold')
        ax.set_xlim(0, max(ranking['Pct of Total Variance']) * 1.15)

        plt.tight_layout()
        output_path = self.output_dir / 'importance_ranking.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved importance ranking plot to {output_path}")
        return output_path

    def plot_main_effects(self) -> List[Path]:
        """Plot partial dependence functions for each hyperparameter."""
        main_effects = self.fanova.compute_main_effects()

        n_params = len(main_effects)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_params == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for ax, (param, effects) in zip(axes, main_effects.items()):
            x_vals = effects['x_vals']
            y_vals = effects['y_vals']
            y_err = effects['y_err']

            # Plot with confidence interval
            ax.plot(x_vals, y_vals, 'o-', linewidth=2, markersize=6, label=param, color='#2E86AB')
            ax.fill_between(x_vals, y_vals - y_err, y_vals + y_err, alpha=0.3, color='#2E86AB')

            # Add horizontal line at global mean
            ax.axhline(self.fanova.y_mean, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Global mean')

            importance_pct = 100 * effects['importance_fraction']
            ax.set_title(f'{param}\n(Importance: {importance_pct:.1f}%)', fontsize=11, fontweight='bold')
            ax.set_ylabel(self.fanova.metric_name, fontsize=10)

            if self.fanova.param_info[param].param_type == 'categorical':
                ax.set_xlabel('Category', fontsize=10)
                ax.set_xticks(range(len(x_vals)))
                ax.set_xticklabels(x_vals, rotation=45, ha='right')
            else:
                ax.set_xlabel(f'{param} (normalized)', fontsize=10)

            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

        # Hide unused subplots
        for i in range(len(main_effects), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        output_path = self.output_dir / 'main_effects.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved main effects plot to {output_path}")
        return [output_path]

    def plot_interaction_heatmap(self, top_k: int = 5) -> Optional[Path]:
        """Plot pairwise interaction effects as a heatmap."""
        main_effects = self.fanova.compute_importance_ranking()
        top_params = main_effects.head(top_k)['Hyperparameter'].tolist()

        if len(top_params) < 2:
            logger.warning("Need at least 2 parameters for interaction heatmap")
            return None

        # Compute interaction matrix
        n = len(top_params)
        interaction_matrix = np.zeros((n, n))

        for i, param1 in enumerate(top_params):
            for j, param2 in enumerate(top_params):
                if i < j:
                    try:
                        residuals = self.fanova._compute_residuals_after_main_effects(param1, param2)
                        interaction_var = np.var(residuals)
                        interaction_frac = interaction_var / (self.fanova.total_variance + 1e-10)
                        interaction_matrix[i, j] = interaction_frac
                        interaction_matrix[j, i] = interaction_frac
                    except Exception as e:
                        logger.warning(f"Failed to compute interaction {param1}×{param2}: {e}")

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(8, 7))

        sns.heatmap(
            interaction_matrix,
            xticklabels=top_params,
            yticklabels=top_params,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Interaction Importance (Fraction of Variance)'},
            ax=ax,
            vmin=0,
        )

        ax.set_title(f'Pairwise Interaction Effects (Top {top_k} Parameters)',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = self.output_dir / 'interaction_heatmap.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved interaction heatmap to {output_path}")
        return output_path

    def generate_all_plots(self) -> Dict[str, Path]:
        """Generate all visualizations."""
        plots = {}

        logger.info("Generating importance ranking plot...")
        plots['importance_ranking'] = self.plot_importance_ranking()

        logger.info("Generating main effects plots...")
        plots['main_effects'] = self.plot_main_effects()[0]

        logger.info("Generating interaction heatmap...")
        interaction_plot = self.plot_interaction_heatmap(top_k=5)
        if interaction_plot:
            plots['interaction_heatmap'] = interaction_plot

        return plots


def extract_hyperparams_from_runs(run_dirs: List[Path], metric_name: str) -> pd.DataFrame:
    """
    Extract hyperparameters and performance metrics from run directories.

    Expected directory structure:
        run_dir/
          ├── config.yml (hyperparameters)
          └── metrics.csv (performance metrics)

    Args:
        run_dirs: list of run directory paths
        metric_name: which column from metrics.csv to analyze

    Returns:
        DataFrame with columns [hyperparams..., metric_name]
    """
    data = []

    for run_dir in run_dirs:
        run_dir = Path(run_dir)

        # Read config
        config_path = run_dir / 'config.yml'
        if not config_path.exists():
            logger.warning(f"No config.yml in {run_dir}, skipping")
            continue

        try:
            with open(config_path) as f:
                config = load_config(config_path)
        except Exception as e:
            logger.warning(f"Failed to load config from {run_dir}: {e}")
            continue

        # Read metrics
        metrics_path = run_dir / 'metrics.csv'
        if not metrics_path.exists():
            logger.warning(f"No metrics.csv in {run_dir}, skipping")
            continue

        try:
            metrics_df = pd.read_csv(metrics_path)

            # Get best validation metric
            if f'val_{metric_name}' in metrics_df.columns:
                best_metric = metrics_df[f'val_{metric_name}'].min()
            elif metric_name in metrics_df.columns:
                best_metric = metrics_df[metric_name].min()
            else:
                logger.warning(f"Metric '{metric_name}' not found in {metrics_path}")
                continue

        except Exception as e:
            logger.warning(f"Failed to read metrics from {run_dir}: {e}")
            continue

        # Flatten config to dot-notation
        flattened_config = _flatten_config(config)

        # Combine into one row
        row = {**flattened_config, metric_name: best_metric}
        data.append(row)

    if not data:
        raise ValueError(f"No valid runs found. Checked {len(run_dirs)} directories.")

    return pd.DataFrame(data)


def _flatten_config(config: dict, prefix: str = '') -> dict:
    """Flatten nested config dictionary to dot-notation."""
    flattened = {}

    for key, value in config.items():
        if prefix:
            full_key = f'{prefix}.{key}'
        else:
            full_key = key

        if isinstance(value, dict):
            flattened.update(_flatten_config(value, full_key))
        else:
            flattened[full_key] = value

    return flattened


def main():
    parser = argparse.ArgumentParser(
        description='Functional ANOVA (fANOVA) Hyperparameter Relevance Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze validation loss across all runs
  python fanova_hyperparameter_analysis.py --path runs/ --metric loss

  # Analyze test rotation error with custom output directory
  python fanova_hyperparameter_analysis.py --path runs/ --metric test_rotation_error \\
    --output results/fanova/ --n-points 50

  # Analyze specific runs only
  python fanova_hyperparameter_analysis.py --path runs/run_001 runs/run_002 runs/run_003 \\
    --metric loss --verbose
        """
    )

    parser.add_argument(
        '--path', nargs='+', required=True, type=Path,
        help='Path(s) to run directories or a parent directory containing runs'
    )
    parser.add_argument(
        '--metric', required=True,
        help='Performance metric to analyze (e.g., "loss", "rotation_error")'
    )
    parser.add_argument(
        '--output', type=Path, default='results/fanova/',
        help='Output directory for plots and reports (default: results/fanova/)'
    )
    parser.add_argument(
        '--n-points', type=int, default=50,
        help='Number of points for partial dependence interpolation (default: 50)'
    )
    parser.add_argument(
        '--top-k', type=int, default=5,
        help='Top-k hyperparameters to analyze for interactions (default: 5)'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Collect run directories
    run_dirs = []
    for path in args.path:
        if path.is_file():
            logger.warning(f"{path} is a file, treating parent as run directory")
            run_dirs.append(path.parent)
        elif path.is_dir():
            # Check if this is a run directory (has config.yml) or parent directory
            if (path / 'config.yml').exists():
                run_dirs.append(path)
            else:
                # Assume it's a parent directory; find all subdirs with config.yml
                subdirs = [d for d in path.iterdir() if (d / 'config.yml').exists()]
                if subdirs:
                    run_dirs.extend(subdirs)
                    logger.info(f"Found {len(subdirs)} run directories in {path}")
                else:
                    logger.warning(f"No run directories found in {path}")

    if not run_dirs:
        logger.error("No run directories found. Check --path argument.")
        sys.exit(1)

    logger.info(f"Analyzing {len(run_dirs)} runs")

    # Extract data from runs
    logger.info(f"Extracting hyperparameters and metrics (looking for '{args.metric}')...")
    try:
        df = extract_hyperparams_from_runs(run_dirs, args.metric)
    except Exception as e:
        logger.error(f"Failed to extract data: {e}")
        sys.exit(1)

    logger.info(f"Extracted {len(df)} complete configurations with {len(df.columns)-1} hyperparameters")

    # Filter columns: keep only those with some variance
    param_cols = [col for col in df.columns if col != args.metric]
    varying_params = [col for col in param_cols if df[col].nunique() > 1]

    if not varying_params:
        logger.error("No hyperparameters with variance found. Check data.")
        sys.exit(1)

    logger.info(f"Using {len(varying_params)} hyperparameters with variance")
    logger.info(f"  Parameters: {', '.join(varying_params)}")

    # Run fANOVA
    logger.info("Running Functional ANOVA...")
    fanova = FunctionalANOVA(
        data=df[varying_params + [args.metric]],
        metric_name=args.metric,
        param_names=varying_params,
        n_points=args.n_points
    )

    # Generate summary
    summary = fanova.summary()
    print(summary)

    # Generate visualizations
    logger.info("Generating visualizations...")
    visualizer = FANOVAVisualizer(fanova, args.output)
    plots = visualizer.generate_all_plots()

    # Save results
    results_dir = Path(args.output)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save summary
    summary_path = results_dir / 'fanova_summary.txt'
    with open(summary_path, 'w') as f:
        f.write(summary)
    logger.info(f"Saved summary to {summary_path}")

    # Save ranking as CSV
    ranking = fanova.compute_importance_ranking()
    ranking_path = results_dir / 'hyperparameter_importance.csv'
    ranking.to_csv(ranking_path, index=False)
    logger.info(f"Saved importance ranking to {ranking_path}")

    # Save detailed results as JSON
    results_json = {
        'metric': args.metric,
        'n_experiments': len(df),
        'n_hyperparameters': len(varying_params),
        'mean_performance': float(fanova.y_mean),
        'std_performance': float(fanova.y_std),
        'total_variance': float(fanova.total_variance),
        'hyperparameter_importance': {
            param: {
                'variance': float(effects['importance']),
                'fraction_of_total': float(effects['importance_fraction']),
                'type': fanova.param_info[param].param_type,
            }
            for param, effects in fanova.compute_main_effects().items()
        }
    }

    results_json_path = results_dir / 'fanova_results.json'
    with open(results_json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Saved detailed results to {results_json_path}")

    logger.info(f"\n✓ fANOVA analysis complete!")
    logger.info(f"  Results saved to: {results_dir}")
    logger.info(f"  Plots: {list(plots.keys())}")


if __name__ == '__main__':
    main()
