#!/usr/bin/env python3
"""
Scatter plots of TEST-set errors (not validation) across fusion strategies,
built for comparing the 7 fusion candidates in the "which fusion strategy
wins" research question:

    1. Concatenation                                          (fusion=concat)
    2. Gating                                                 (fusion=gated / gated_Softmax)
    3. Softmax gated concatenation (scalar)                   (fusion=gated_concat)
    4. Residual sigmoid gated concatenation (scalar)          (fusion=residual_gated_concat)
    5. Channelwise local sigmoid gated concatenation          (fusion=channelwise_gated_concat_local)
    6. Channelwise hybrid residual sigmoid gated concatenation(fusion=channelwise_residual_gated_concat_hybrid)
    7. Dual Fusion                                            (fusion=gated_concat, dual_fusion=true)

Any run whose (fusion, dual_fusion) doesn't match one of the 7 above is still
included, grouped under its own "<fusion>[+dual]" label, so the script isn't
limited to exactly this candidate set.

Runs are discovered by scanning each `--run_dir` (a single run folder, or a
container directory holding many run folders) for `results/results_*.csv`
files written by `scripts/test.py`. Each run's config (`config.yml` /
`config.resolved.yml`) supplies `model.fusion` / `model.dual_fusion` /
`training.seed`. When a run has multiple test csvs (re-ran `test.py` more
than once), only the most recent (by the timestamp in the filename) is used.

Test-set metrics used (from the per-sample results csv, NOT metrics.csv):
    - translation error: sqrt(val_abs_y^2 + val_abs_z^2), meters -> plotted in mm.
      Deliberately NOT val_abs_trans (which is the L2 norm over x/y/z): the model
      only ever predicts y/z (x is fixed to 0, per CLAUDE.md), but gt_trans_x in
      the cached pose is not exactly 0 (~1-2mm residual from the hub-frame
      calibration transform), so val_abs_trans silently folds that never-predicted,
      never-zero x "error" into the translation number and inflates it.
    - rotation error: val_abs_rot_z (quaternion geodesic distance, degrees) by
      default, or val_abs_rot_x (roll only) via --rot_metric

Two plots are produced (matplotlib .png + interactive plotly .html each):
    scatter_test_by_run.{png,html}      - one point per run (mean test error
                                           across all its test samples),
                                           colored by candidate, labeled by seed
    scatter_test_per_sample.{png,html}  - every individual test sample as a
                                           point, colored by candidate (shows
                                           error-distribution shape/spread,
                                           not just the mean)

Usage:
    # single run folder
    python scripts/evaluate/scatter_test_results_by_fusion.py --run_dir runs/q1_ConcatVsGated_gradclip/20260825_144103_bs16_gated

    # multiple run folders / container dirs combined into one plot
    python scripts/evaluate/scatter_test_results_by_fusion.py \\
        --run_dir runs/q1_ConcatVsGated_gradclip runs/q2_channelwise_local_hybrid runs/Q4_dualFusion \\
        --output_dir runs/scatter_test_7candidates

    # use the full geodesic rotation error instead of roll-only
    python scripts/evaluate/scatter_test_results_by_fusion.py --run_dir runs/q1_ConcatVsGated --rot_metric val_abs_rot_x
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# (fusion, dual_fusion) -> display label for the 7 target candidates.
# "gated_Softmax" is included alongside "gated" since both instantiate the
# same GatedSoftmaxFusion class (see fusion/registry.py aliasing).
CANDIDATE_MAP = {
    ("concat", False): "1. Concatenation",
    ("gated", False): "2. Gating",
    ("gated_Softmax", False): "2. Gating",
    ("gated_concat", False): "3. Softmax gated concatenation (scalar)",
    ("residual_gated_concat", False): "4. Residual sigmoid gated concatenation (scalar)",
    ("channelwise_gated_concat_local", False): "5. Channelwise local sigmoid gated concatenation",
    ("channelwise_residual_gated_concat_hybrid", False): "6. Channelwise hybrid residual sigmoid gated concatenation",
    ("gated_concat", True): "7. Dual Fusion",
}

CANDIDATE_ORDER = [
    "1. Concatenation",
    "2. Gating",
    "3. Softmax gated concatenation (scalar)",
    "4. Residual sigmoid gated concatenation (scalar)",
    "5. Channelwise local sigmoid gated concatenation",
    "6. Channelwise hybrid residual sigmoid gated concatenation",
    "7. Dual Fusion",
]

# Short row labels for the summary table (matches the raw config.fusion values,
# with a "dual_" prefix for the dual-fusion candidate).
CANDIDATE_SHORT_NAME = {
    "1. Concatenation": "concat",
    "2. Gating": "gated",
    "3. Softmax gated concatenation (scalar)": "gated_concat",
    "4. Residual sigmoid gated concatenation (scalar)": "residual_gated_concat",
    # Deliberately matches the exact model.fusion registry string (fusion/gated_concat.py),
    # not a shortened form -- the repo also has plain, non-suffixed
    # "channelwise_gated_concat"/"channelwise_residual_gated_concat" fusion types
    # (different strategies, not just a naming shorthand for these), so a shortened
    # label here would silently collide with those in the printed table.
    "5. Channelwise local sigmoid gated concatenation": "channelwise_gated_concat_local",
    "6. Channelwise hybrid residual sigmoid gated concatenation": "channelwise_residual_gated_concat_hybrid",
    "7. Dual Fusion": "dual_fusion (gated_concat)",
}

CANDIDATE_COLORS = {
    "1. Concatenation": "#4C72B0",
    "2. Gating": "#DD8452",
    "3. Softmax gated concatenation (scalar)": "#55A868",
    "4. Residual sigmoid gated concatenation (scalar)": "#C44E52",
    "5. Channelwise local sigmoid gated concatenation": "#8172B2",
    "6. Channelwise hybrid residual sigmoid gated concatenation": "#937860",
    "7. Dual Fusion": "#CCB974",
}
FALLBACK_COLORS = ["#64B5CD", "#8C8C8C", "#DA8BC3", "#CCCCCC"]

# Which per-sample results the plots are built from. Default = test split (results/results_*.csv written
# by scripts/test.py). "val" reads <run>/results_val/results_val*.csv (per-sample validation errors of
# checkpoint_best.pth) and relabels titles / axes / file names accordingly. Set via set_split().
SPLIT_LABEL = "Test"
SPLIT_KEY = "test"
RESULTS_SUBDIR = "results"
RESULTS_GLOB = "results_*.csv"


def set_split(split: str):
    global SPLIT_LABEL, SPLIT_KEY, RESULTS_SUBDIR, RESULTS_GLOB
    if split == "val":
        SPLIT_LABEL, SPLIT_KEY, RESULTS_SUBDIR, RESULTS_GLOB = "Validation", "val", "results_val", "results_val*.csv"
    elif split == "test":
        SPLIT_LABEL, SPLIT_KEY, RESULTS_SUBDIR, RESULTS_GLOB = "Test", "test", "results", "results_*.csv"
    else:
        raise ValueError(f"unknown split: {split}")


def has_task_adapter(model_cfg: dict) -> bool:
    """True if either the legacy or current task-specific feature-adapter
    mechanism is switched on (see CLAUDE.md's "Task-specific mechanisms").
    These are independent, off-by-default research axes -- a run with one
    enabled isn't a clean instance of its `model.fusion` strategy alone, so
    it must not be silently pooled into that strategy's baseline candidate."""
    if model_cfg.get("task_adapter"):
        return True
    return bool((model_cfg.get("task_feature_adapter", {}) or {}).get("enabled"))


def classify_candidate(fusion: str, dual_fusion: bool, model_cfg: dict = None) -> str:
    label = CANDIDATE_MAP.get((fusion, bool(dual_fusion)))
    if label is None:
        label = f"{fusion}{'+dual' if dual_fusion else ''}"
    if model_cfg is not None and has_task_adapter(model_cfg):
        # Deliberately falls outside CANDIDATE_ORDER so it's excluded from the
        # clean 7-candidate summary table by default, while still visible in
        # the scatter plots / per-run CSV under its own label.
        label = f"{label} [+task_adapter]"
    return label


def find_run_config(run_folder: Path):
    for name in ("config.yml", "config.resolved.yml"):
        candidate = run_folder / name
        if candidate.exists():
            return candidate
    return None


def latest_results_csv(results_dir: Path):
    """Picks the most recent results_*.csv by the timestamp embedded in the filename."""
    candidates = sorted(results_dir.glob(RESULTS_GLOB))
    return candidates[-1] if candidates else None


def discover_run_folders(run_dirs):
    """Each run_dir is either a run folder itself (has config + results/results_*.csv)
    or a container directory scanned recursively for such run folders."""
    found = {}
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        if not run_dir.is_absolute():
            run_dir = (REPO_ROOT / run_dir) if (REPO_ROOT / run_dir).exists() else run_dir.resolve()

        results_dir = run_dir / RESULTS_SUBDIR
        if results_dir.is_dir() and latest_results_csv(results_dir) is not None:
            found[str(run_dir)] = run_dir
            continue

        for csv_path in run_dir.rglob(f"{RESULTS_SUBDIR}/{RESULTS_GLOB}"):
            candidate = csv_path.parent.parent
            found[str(candidate)] = candidate

    return sorted(found.values())


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = []
    for k, v in (d or {}).items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, tuple(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def collect_runs(run_folders):
    rows = []
    per_sample_frames = []
    excluded = []
    configs = {}
    seen_run_names = {}  # run_name (basename) -> first run_folder it was seen at

    for run_folder in run_folders:
        run_name = run_folder.name

        cfg_path = find_run_config(run_folder)
        if cfg_path is None:
            excluded.append({"run_folder": str(run_folder), "reason": "no config.yml/config.resolved.yml found"})
            continue
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            excluded.append({"run_folder": str(run_folder), "reason": f"failed to parse config: {e}"})
            continue

        model_cfg = cfg.get("model", {}) or {}
        fusion = model_cfg.get("fusion")
        dual_fusion = bool(model_cfg.get("dual_fusion", False))
        seed = (cfg.get("training", {}) or {}).get("seed")
        candidate = classify_candidate(fusion, dual_fusion, model_cfg)
        flat_cfg = flatten_dict(cfg)

        # Multiple --run_dir container folders can hold a copy of the same physical
        # run (e.g. a baseline gated_concat run reused as the comparison point in
        # two different ablation groups). train.py timestamps run folders to the
        # second, so an identical run_name (basename) with an identical config is
        # treated as a duplicate and only counted once, to avoid silently doubling
        # its weight in the per-candidate mean/std.
        if run_name in seen_run_names:
            prev_folder, prev_flat_cfg = seen_run_names[run_name]
            if prev_flat_cfg == flat_cfg:
                excluded.append({"run_folder": str(run_folder), "fusion": fusion,
                                  "reason": f"duplicate of {prev_folder} (identical run_name + config)"})
                continue
        else:
            seen_run_names[run_name] = (str(run_folder), flat_cfg)

        results_dir = run_folder / RESULTS_SUBDIR
        csv_path = latest_results_csv(results_dir)
        if csv_path is None:
            excluded.append({"run_folder": str(run_folder), "fusion": fusion, "reason": "no results_*.csv in results/"})
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            excluded.append({"run_folder": str(run_folder), "fusion": fusion, "reason": f"failed to read {csv_path.name}: {e}"})
            continue

        required_cols = {"val_abs_y", "val_abs_z", "val_abs_rot_x", "val_abs_rot_z"}
        missing = required_cols - set(df.columns)
        if missing or df.empty:
            excluded.append({"run_folder": str(run_folder), "fusion": fusion, "reason": f"results csv missing columns {missing} or empty"})
            continue

        df = df.copy()
        df["run_name"] = run_name
        df["run_folder"] = str(run_folder)
        df["fusion"] = fusion
        df["dual_fusion"] = dual_fusion
        df["candidate"] = candidate
        df["seed"] = seed
        df["trans_l2_mm"] = np.sqrt(df["val_abs_y"] ** 2 + df["val_abs_z"] ** 2) * 1000.0
        per_sample_frames.append(df)
        configs[str(run_folder)] = flat_cfg

        rows.append({
            "run_name": run_name,
            "run_folder": str(run_folder),
            "fusion": fusion,
            "dual_fusion": dual_fusion,
            "candidate": candidate,
            "seed": seed,
            "n_samples": len(df),
            "test_csv": str(csv_path),
            "trans_l2_mm": df["trans_l2_mm"].mean(),
            "rot_x_deg": df["val_abs_rot_x"].mean(),
            "rot_z_geodesic_deg": df["val_abs_rot_z"].mean(),
            "y_mm": (df["val_abs_y"] * 1000.0).mean(),
            "z_mm": (df["val_abs_z"] * 1000.0).mean(),
        })

    per_run_df = pd.DataFrame(rows)
    per_sample_df = pd.concat(per_sample_frames, ignore_index=True) if per_sample_frames else pd.DataFrame()
    excluded_df = pd.DataFrame(excluded)

    if not per_run_df.empty:
        per_run_df["candidate"] = pd.Categorical(
            per_run_df["candidate"],
            categories=CANDIDATE_ORDER + sorted(set(per_run_df["candidate"]) - set(CANDIDATE_ORDER)),
            ordered=True,
        )
        per_run_df = per_run_df.sort_values(["candidate", "seed"]).reset_index(drop=True)

    return per_run_df, per_sample_df, excluded_df, configs


def compute_config_drift_notes(df: pd.DataFrame, configs: dict) -> list:
    """For each candidate group with >=2 included runs, flags config keys that
    differ across its runs (besides seed) -- averaging test errors across runs
    is only a fair "same candidate, different seed" comparison if nothing else
    varies between them."""
    notes = []
    if df.empty:
        return notes

    ignore_prefixes = ("training.seed",)
    for candidate, group in df.groupby("candidate", observed=True):
        run_folders = group["run_folder"].tolist()
        if len(run_folders) < 2:
            continue
        flat_configs = [configs[rf] for rf in run_folders]
        all_keys = set().union(*(fc.keys() for fc in flat_configs))
        diffs = []
        for key in sorted(all_keys):
            if key.startswith(ignore_prefixes):
                continue
            values = [fc.get(key, "<missing>") for fc in flat_configs]
            if len(set(values)) > 1:
                diffs.append((key, values))
        if diffs:
            notes.append({"candidate": str(candidate), "run_folders": run_folders, "diffs": diffs})
    return notes


def get_color(candidate: str, seen: dict):
    if candidate in CANDIDATE_COLORS:
        return CANDIDATE_COLORS[candidate]
    if candidate not in seen:
        seen[candidate] = FALLBACK_COLORS[len(seen) % len(FALLBACK_COLORS)]
    return seen[candidate]


def ordered_candidates(candidates):
    present = list(dict.fromkeys(candidates))
    ordered = [c for c in CANDIDATE_ORDER if c in present]
    ordered += sorted(c for c in present if c not in CANDIDATE_ORDER)
    return ordered


def build_summary_table(df: pd.DataFrame, target_candidates_only: bool = True) -> pd.DataFrame:
    """Mean +/- std (across runs/seeds, of each run's own mean-over-test-samples)
    per candidate, for the 4 metrics: rot_x (roll, deg), trans L2 -- y/z only, mm --
    (deg), y (mm), z (mm).

    By default restricted to the 7 named target candidates (matching
    CANDIDATE_MAP) -- other fusion variants discovered under --run_dir (e.g.
    the unsuffixed channelwise_gated_concat, or the _wide/_floor variants) are
    excluded from this table since they aren't part of that comparison, even
    though they still appear in test_comparison_by_run.csv. Pass
    target_candidates_only=False to include every discovered candidate.
    """
    if df.empty:
        return pd.DataFrame()

    rows = []
    candidates = ordered_candidates(df["candidate"].astype(str).tolist())
    if target_candidates_only:
        candidates = [c for c in candidates if c in CANDIDATE_ORDER]
    for cand in candidates:
        cdf = df[df["candidate"].astype(str) == cand]
        n = len(cdf)
        run_folders = sorted(cdf["run_folder"].tolist())
        # Container dir = the run folder's parent (e.g. "q1_ConcatVsGated_gradclip"),
        # i.e. which --run_dir group each contributing run actually came from.
        container_dirs = sorted(set(Path(rf).parent.name for rf in run_folders))
        rows.append({
            "candidate": cand,
            "short_name": CANDIDATE_SHORT_NAME.get(cand, cand),
            "n_runs": n,
            "seeds": sorted(cdf["seed"].dropna().astype(int).unique().tolist()),
            "run_folders": run_folders,
            "container_dirs": container_dirs,
            "rot_x_deg_mean": cdf["rot_x_deg"].mean(), "rot_x_deg_std": cdf["rot_x_deg"].std(),
            "trans_mm_mean": cdf["trans_l2_mm"].mean(), "trans_mm_std": cdf["trans_l2_mm"].std(),
            "y_mm_mean": cdf["y_mm"].mean(), "y_mm_std": cdf["y_mm"].std(),
            "z_mm_mean": cdf["z_mm"].mean(), "z_mm_std": cdf["z_mm"].std(),
        })
    return pd.DataFrame(rows)


def flag_comparability(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Adds a 'flags' column (list of strings) noting, per candidate row, why it
    might NOT be a fair apples-to-apples comparison against the others: fewer
    seeds than the majority, or sourced from a container dir no other candidate
    used (so any dir-level config difference wouldn't be caught by the
    same-candidate config-drift check, which only compares runs already grouped
    together)."""
    if summary_df.empty:
        return summary_df

    summary_df = summary_df.copy()
    max_n = summary_df["n_runs"].max()
    all_container_dirs = {}
    for _, row in summary_df.iterrows():
        for d in row["container_dirs"]:
            all_container_dirs.setdefault(d, []).append(row["short_name"])

    flags_col = []
    for _, row in summary_df.iterrows():
        flags = []
        if row["n_runs"] < max_n:
            flags.append(f"only {row['n_runs']} seed(s) vs {max_n} for the others")
        exclusive_dirs = [d for d in row["container_dirs"] if len(all_container_dirs[d]) == 1]
        if exclusive_dirs and len(summary_df) > 1:
            flags.append(f"only sourced from {exclusive_dirs} (no other candidate shares that run_dir)")
        flags_col.append(flags)
    summary_df["flags"] = flags_col
    return summary_df


def write_summary_table(summary_df: pd.DataFrame, output_dir: Path, run_dir_args):
    if summary_df.empty:
        return

    def fmt(mean, std, n):
        if pd.isna(mean):
            return "-"
        if n < 2 or pd.isna(std):
            return f"{mean:.3f}"
        return f"{mean:.3f} ± {std:.3f}"

    lines = [
        f"Run dirs scanned (`--run_dir`): {run_dir_args}\n",
        "| fusion | n_seeds | run folders used | test rot_x (deg) | test trans (mm) | test y (mm) | test z (mm) | flags |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in summary_df.iterrows():
        folder_names = ", ".join(Path(rf).name for rf in row["run_folders"])
        flags_txt = "; ".join(row.get("flags", [])) or "-"
        lines.append(
            f"| {row['short_name']} | {row['n_runs']} | {folder_names} | "
            f"{fmt(row['rot_x_deg_mean'], row['rot_x_deg_std'], row['n_runs'])} | "
            f"{fmt(row['trans_mm_mean'], row['trans_mm_std'], row['n_runs'])} | "
            f"{fmt(row['y_mm_mean'], row['y_mm_std'], row['n_runs'])} | "
            f"{fmt(row['z_mm_mean'], row['z_mm_std'], row['n_runs'])} | {flags_txt} |"
        )
    table_md = "\n".join(lines)

    (output_dir / f"{SPLIT_KEY}_summary_table.md").write_text(table_md + "\n")
    summary_df.to_csv(output_dir / f"{SPLIT_KEY}_summary_table.csv", index=False)
    return table_md


def plot_scatter_by_run(df: pd.DataFrame, output_dir: Path, rot_col: str, rot_label: str, label_name: str = "seed",
                        x_col: str = "trans_l2_mm", x_label: str = "Test Trans L2 Norm (mm)",
                        file_suffix: str = "", y_hover_name: str = "rot", y_hover_unit: str = " deg"):
    if df.empty:
        return
    candidates = ordered_candidates(df["candidate"].tolist())
    fallback_seen = {}

    fig, ax = plt.subplots(figsize=(8, 6.5))
    for cand in candidates:
        cdf = df[df["candidate"] == cand]
        color = get_color(cand, fallback_seen)
        ax.scatter(
            cdf[x_col], cdf[rot_col],
            color=color, s=100, edgecolor="black", linewidth=0.6,
            alpha=0.85, label=cand,
        )
        for _, row in cdf.iterrows():
            seed_txt = str(int(row["seed"])) if pd.notna(row["seed"]) else "?"
            ax.annotate(seed_txt, (row[x_col], row[rot_col]),
                        xytext=(5, 5), textcoords="offset points", fontsize=8)

    ax.set_xlabel(x_label)
    ax.set_ylabel(f"{SPLIT_LABEL} {rot_label}")
    ax.set_title(f"{SPLIT_LABEL}-set error trade-off by fusion candidate\n(mean over {SPLIT_KEY} samples; labels = {label_name})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="fusion candidate", fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / f"scatter_{SPLIT_KEY}_by_run{file_suffix}.png", dpi=150)
    plt.close(fig)

    fig_plotly = go.Figure()
    for cand in candidates:
        cdf = df[df["candidate"] == cand]
        color = get_color(cand, fallback_seen)
        seed_labels = [str(int(s)) if pd.notna(s) else "?" for s in cdf["seed"]]
        fig_plotly.add_trace(go.Scatter(
            x=cdf[x_col], y=cdf[rot_col],
            mode="markers+text", name=cand,
            marker=dict(color=color, size=13, line=dict(width=1, color="black")),
            text=seed_labels, textposition="top center",
            hovertext=[f"{rn} ({label_name}={s}, n={n})" for rn, s, n in zip(cdf["run_name"], cdf["seed"], cdf["n_samples"])],
            hovertemplate="%{hovertext}<br>x=%{x:.4f}<br>" + y_hover_name + "=%{y:.4f}" + y_hover_unit + "<extra></extra>",
        ))
    fig_plotly.update_layout(
        title=f"{SPLIT_LABEL}-set error trade-off by fusion candidate (mean over {SPLIT_KEY} samples; labels = {label_name})",
        xaxis_title=x_label, yaxis_title=f"{SPLIT_LABEL} {rot_label}",
        template="plotly_white",
    )
    fig_plotly.write_html(output_dir / f"scatter_{SPLIT_KEY}_by_run{file_suffix}.html")


def plot_scatter_per_sample(df: pd.DataFrame, output_dir: Path, rot_col: str, rot_label: str, label_name: str = "seed",
                        x_col: str = "trans_l2_mm", x_label: str = "Test Trans L2 Norm (mm)",
                        file_suffix: str = "", y_hover_name: str = "rot", y_hover_unit: str = " deg"):
    if df.empty:
        return
    candidates = ordered_candidates(df["candidate"].tolist())
    fallback_seen = {}

    fig, ax = plt.subplots(figsize=(8, 6.5))
    for cand in candidates:
        cdf = df[df["candidate"] == cand]
        color = get_color(cand, fallback_seen)
        ax.scatter(
            cdf[x_col], cdf[rot_col],
            color=color, s=14, alpha=0.35, linewidth=0, label=cand,
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel(f"{SPLIT_LABEL} {rot_label}")
    ax.set_title(f"{SPLIT_LABEL}-set per-sample error by fusion candidate\n(every {SPLIT_KEY} sample, all runs pooled)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(title="fusion candidate", fontsize=8, loc="best", markerscale=3)
    fig.tight_layout()
    fig.savefig(output_dir / f"scatter_{SPLIT_KEY}_per_sample{file_suffix}.png", dpi=150)
    plt.close(fig)

    fig_plotly = go.Figure()
    for cand in candidates:
        cdf = df[df["candidate"] == cand]
        color = get_color(cand, fallback_seen)
        fig_plotly.add_trace(go.Scatter(
            x=cdf[x_col], y=cdf[rot_col],
            mode="markers", name=cand,
            marker=dict(color=color, size=6, opacity=0.4),
            hovertext=[f"{rn} ({label_name}={s}, sample={sid})" for rn, s, sid in zip(cdf["run_name"], cdf["seed"], cdf["sample_id"])] if "sample_id" in cdf.columns else None,
            hovertemplate="%{hovertext}<br>x=%{x:.4f}<br>" + y_hover_name + "=%{y:.4f}" + y_hover_unit + "<extra></extra>",
        ))
    fig_plotly.update_layout(
        title=f"{SPLIT_LABEL}-set per-sample error by fusion candidate (every {SPLIT_KEY} sample, all runs pooled)",
        xaxis_title=x_label, yaxis_title=f"{SPLIT_LABEL} {rot_label}",
        template="plotly_white",
    )
    fig_plotly.write_html(output_dir / f"scatter_{SPLIT_KEY}_per_sample{file_suffix}.html")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", nargs="+", required=True,
                         help="One or more run folders and/or container directories to scan (single run or multiple runs combined into one plot).")
    parser.add_argument("--output_dir", default="runs/scatter_test_comparison",
                         help="Directory to write CSVs and plots to.")
    parser.add_argument("--rot_metric", choices=["val_abs_rot_z", "val_abs_rot_x"], default="val_abs_rot_z",
                         help="Rotation error column to plot: val_abs_rot_z = quaternion geodesic distance (default, full rotation error), "
                              "val_abs_rot_x = roll-only error.")
    parser.add_argument("--include_other_candidates", action="store_true",
                         help="Include fusion variants outside the 7 named target candidates in the summary table "
                              "(they're always included in the scatter plots and test_comparison_by_run.csv).")
    args = parser.parse_args()

    rot_col = args.rot_metric
    rot_label = "Rot Error (geodesic, deg)" if rot_col == "val_abs_rot_z" else "Rot Error X / roll (deg)"
    rot_col_run = "rot_z_geodesic_deg" if rot_col == "val_abs_rot_z" else "rot_x_deg"

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    run_folders = discover_run_folders(args.run_dir)
    if not run_folders:
        print(f"No run folders with results/results_*.csv found under {args.run_dir}")
        return

    per_run_df, per_sample_df, excluded_df, configs = collect_runs(run_folders)

    if per_run_df.empty:
        print("No runs with usable test results were found.")
        if not excluded_df.empty:
            print(excluded_df.to_string(index=False))
        return

    per_run_df.to_csv(output_dir / "test_comparison_by_run.csv", index=False)
    excluded_df.to_csv(output_dir / "excluded_runs.csv", index=False)
    print(f"Included runs: {len(per_run_df)}  |  Excluded: {len(excluded_df)}")
    print(f"Candidates found: {sorted(per_run_df['candidate'].astype(str).unique().tolist())}")

    plot_scatter_by_run(per_run_df, output_dir, rot_col_run, rot_label)
    plot_scatter_per_sample(per_sample_df, output_dir, rot_col, rot_label)

    summary_df = build_summary_table(per_run_df, target_candidates_only=not args.include_other_candidates)
    summary_df = flag_comparability(summary_df)
    table_md = write_summary_table(summary_df, output_dir, args.run_dir)
    if table_md:
        print("\nTest-set summary (mean ± std across seeds/runs):\n")
        print(table_md)
        for _, row in summary_df.iterrows():
            for flag in row.get("flags", []):
                print(f"  ⚠️  {row['short_name']}: {flag}")

    # Runs found but not part of the clean 7-candidate table: either a non-target
    # fusion/dual_fusion combo, or tagged [+task_adapter] because a task-specific
    # adapter was enabled (see has_task_adapter()) -- not a fair comparison against
    # the plain fusion-only candidates even though it shares the same model.fusion.
    other_candidates = sorted(
        c for c in per_run_df["candidate"].astype(str).unique() if c not in CANDIDATE_ORDER
    )
    if other_candidates and not args.include_other_candidates:
        print("\nFound but NOT included in the table above (not comparable to the 7 target candidates):")
        for cand in other_candidates:
            cdf = per_run_df[per_run_df["candidate"].astype(str) == cand]
            reason = "task-specific adapter enabled" if "[+task_adapter]" in cand else "outside the 7 target fusion/dual_fusion combos"
            print(f"  - {cand} (n={len(cdf)}, {reason}): {sorted(cdf['run_folder'].apply(lambda p: Path(p).name).tolist())}")

    notes = compute_config_drift_notes(per_run_df, configs)
    if notes:
        lines = ["# Config drift within a candidate group\n",
                 "Runs grouped under the same candidate that differ on more than just "
                 "training.seed -- their averaged test error isn't a clean same-config, "
                 "different-seed comparison.\n"]
        for note in notes:
            lines.append(f"## {note['candidate']}\n")
            lines.append(f"Runs: {note['run_folders']}\n")
            for key, values in note["diffs"]:
                lines.append(f"- `{key}`: {values}")
            lines.append("")
        notes_text = "\n".join(lines)
        (output_dir / "config_drift_notes.md").write_text(notes_text + "\n")
        print(f"\n⚠️  Config drift detected within {len(notes)} candidate group(s) — see config_drift_notes.md")
        for note in notes:
            print(f"  - {note['candidate']}: differing keys = {[k for k, _ in note['diffs']]}")

    print(f"\nWrote comparison CSVs + plots to {output_dir}:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
