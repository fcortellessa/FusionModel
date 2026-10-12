#!/usr/bin/env python3
"""
Comparison of `model.pre_head_channel_divisor: 4` (runs/final_divisor4) vs. no bottleneck
(runs/no_divisor_final_ablation), across the 10 fusion-method candidates used in this ablation:

    concat | gated | gated_concat | dual_gated_concat | residual_gated_concat |
    se_channel_gated_concat | dual_se_channel_gated_concat | channelwise_gated_concat_local |
    channelwise_residual_gated_concat_local | se_gated_cross_attention

"dual_*" means config.model.fusion == <base> AND config.model.dual_fusion == true (see
CLAUDE.md's "Task-specific mechanisms" section) -- not a separate registry string.

Both run_dir arguments are directories of symlinks into other run folders elsewhere under
runs/; this script resolves symlinks and reads each target's config.yml + results/results_*.csv
(test-set per-sample errors) / results_val/results_val*.csv (validation). Gate behaviour
(deliverable 6) prefers real per-sample TEST-set inference written by
scripts/evaluate/plot_gate_distributions.py (--split test --rebuild_split) to
analysis_fusion/divisor_ablation/6_gate_behaviour/test_set_gate_inference/, falling back to each
run's gate_weights_best_epoch.pt (per-sample VAL gate weights at the best checkpoint) only for
runs that live-test pass didn't cover -- see each row's 'source' column.

Historical note: an earlier pass through this ablation found 33 runs under runs/final_divisor4
whose results/ existed but was empty (test.py had never been run on them), and had to source gate
behaviour from gate_weights_best_epoch.pt snapshots only because a training job was actively using
Data/Data_split at the time. Both gaps have since been filled (test.py was run on all 33 missing
runs; plot_gate_distributions.py --rebuild_split --split test was run for every gated-family run in
both variants) -- every deliverable below now uses the complete data and documents any genuinely
remaining gaps (e.g. runs still missing a checkpoint or predating gate checkpointing entirely)
rather than silently dropping them.

Usage:
    python scripts/evaluate/divisor_ablation_analysis.py
    (no arguments; paths are fixed to runs/final_divisor4 and runs/no_divisor_final_ablation,
    output to analysis_fusion/divisor_ablation/)
"""

import glob
import json
import os
import random
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import yaml
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

OUT_ROOT = REPO_ROOT / "analysis_fusion" / "divisor_ablation"

RUN_DIRS = {
    "divisor_4": REPO_ROOT / "runs" / "final_divisor4",
    "no_divisor": REPO_ROOT / "runs" / "no_divisor_final_ablation",
}

# ---------------------------------------------------------------------------------------------
# Fixed method order / color map (dataviz skill: assign categorical hues in a FIXED order, not
# cycled/data-dependent). tab10 is a standard validated-enough qualitative palette for this
# count; se_gated_cross_attention (a non-gated_concat-family attention fusion) gets the 10th slot.
# ---------------------------------------------------------------------------------------------
METHOD_ORDER = [
    "concat",
    "gated",
    "gated_concat",
    "dual_gated_concat",
    "residual_gated_concat",
    "se_channel_gated_concat",
    "dual_se_channel_gated_concat",
    "elementwise_gated_concat",
    "channelwise_residual_gated_concat_local",
    "se_gated_cross_attention",
]
_TAB10 = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
          "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
COLOR_MAP = {m: _TAB10[i] for i, m in enumerate(METHOD_ORDER)}
# requested swaps: elementwise_gated_concat <-> dual_gated_concat, residual_gated_concat <-> se_channel_gated_concat
COLOR_MAP["elementwise_gated_concat"], COLOR_MAP["dual_gated_concat"] = (
    COLOR_MAP["dual_gated_concat"], COLOR_MAP["elementwise_gated_concat"]
)
COLOR_MAP["residual_gated_concat"], COLOR_MAP["se_channel_gated_concat"] = (
    COLOR_MAP["se_channel_gated_concat"], COLOR_MAP["residual_gated_concat"]
)

VARIANT_ORDER = ["divisor_4", "no_divisor"]
VARIANT_LINESTYLE = {"divisor_4": "-", "no_divisor": "--"}
VARIANT_MARKER = {"divisor_4": "o", "no_divisor": "^"}
VARIANT_HATCH = {"divisor_4": None, "no_divisor": "///"}

# (fusion registry string, dual_fusion) -> method key
FUSION_DUAL_TO_METHOD = {
    ("concat", False): "concat",
    ("gated", False): "gated",
    ("gated_Softmax", False): "gated",
    ("gated_concat", False): "gated_concat",
    ("gated_concat", True): "dual_gated_concat",
    ("residual_gated_concat", False): "residual_gated_concat",
    ("se_channel_gated_concat", False): "se_channel_gated_concat",
    ("se_channel_gated_concat", True): "dual_se_channel_gated_concat",
    ("channelwise_gated_concat_local", False): "elementwise_gated_concat",
    ("channelwise_residual_gated_concat_local", False): "channelwise_residual_gated_concat_local",
    ("se_gated_cross_attention", False): "se_gated_cross_attention",
}


def method_of(fusion, dual_fusion):
    return FUSION_DUAL_TO_METHOD.get((fusion, bool(dual_fusion)))


def write_color_map():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "color_map.json").write_text(json.dumps(COLOR_MAP, indent=2) + "\n")


# ---------------------------------------------------------------------------------------------
# Export helpers (style from scripts/evaluate/results_analysis.py)
# ---------------------------------------------------------------------------------------------

def save_mpl_fig(fig, base_path: Path, dpi=150):
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{base_path}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_plotly_fig(fig, base_path: Path):
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(f"{base_path}.html", include_plotlyjs="cdn", full_html=True)


# ---------------------------------------------------------------------------------------------
# Deliverable 0: master per-sample dataframe
# ---------------------------------------------------------------------------------------------

def _latest(glob_pat):
    cands = sorted(glob.glob(glob_pat))
    return cands[-1] if cands else None


def discover_runs(container_dir: Path, variant: str):
    """Yields dicts describing every symlink/run folder under container_dir."""
    runs = []
    for name in sorted(os.listdir(container_dir)):
        link = container_dir / name
        target = link.resolve()
        cfg_path = target / "config.yml"
        if not cfg_path.exists():
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        model_cfg = cfg.get("model", {}) or {}
        data_cfg = cfg.get("data", {}) or {}
        fusion = model_cfg.get("fusion")
        dual_fusion = bool(model_cfg.get("dual_fusion", False))
        method = method_of(fusion, dual_fusion)
        runs.append(dict(
            symlink_name=name, run_folder=str(target), variant=variant,
            fusion=fusion, dual_fusion=dual_fusion, method=method,
            split_seed=data_cfg.get("split_seed"),
            training_seed=(cfg.get("training", {}) or {}).get("seed"),
            pre_head_channel_divisor=model_cfg.get("pre_head_channel_divisor"),
        ))
    return runs


REQUIRED_COLS = {"sample_id", "val_abs_y", "val_abs_z", "val_abs_rot_x", "val_abs_rot_z",
                  "val_signed_y", "val_signed_z", "val_signed_rot_x"}


def load_run_samples(run_info):
    """Returns (df_or_None, reason_if_skipped)."""
    rf = Path(run_info["run_folder"])
    csv_path = _latest(str(rf / "results" / "results_*.csv"))
    if csv_path is None:
        return None, "no results/results_*.csv (test.py likely never run on this run)"
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        return None, f"failed to read {csv_path}: {e}"
    if df.empty:
        return None, "results csv is empty"
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        return None, f"missing columns {missing}"

    out = pd.DataFrame({
        "run_dir": run_info["run_folder"],
        "symlink_name": run_info["symlink_name"],
        "fusion_method": run_info["method"],
        "variant": run_info["variant"],
        "split_seed": run_info["split_seed"],
        "dual_fusion": run_info["dual_fusion"],
        "sample_id": df["sample_id"].astype(int),
        "abs_y": df["val_abs_y"] * 1000.0,          # mm
        "abs_z": df["val_abs_z"] * 1000.0,          # mm
        "abs_rot_x": df["val_abs_rot_x"],           # deg (roll)
        "abs_rot_z_geodesic": df["val_abs_rot_z"],  # deg (full quaternion geodesic distance)
        "signed_y": df["val_signed_y"] * 1000.0,
        "signed_z": df["val_signed_z"] * 1000.0,
        "signed_rot_x": df["val_signed_rot_x"],
        "gt_trans_y": df["gt_trans_y"],
        "gt_trans_z": df["gt_trans_z"],
        "gt_roll_deg": df["gt_roll_deg"],
        "pred_roll_deg": df["pred_roll_deg"],
    })
    out["trans_L2"] = np.sqrt(out["abs_y"] ** 2 + out["abs_z"] ** 2)  # mm, y/z only -- see CLAUDE.md
    out["csv_path"] = csv_path
    return out, None


def build_master_df():
    all_runs = []
    for variant, container in RUN_DIRS.items():
        all_runs += discover_runs(container, variant)

    frames, inventory = [], []
    for r in all_runs:
        if r["method"] is None:
            inventory.append({**r, "status": "excluded",
                               "reason": f"(fusion={r['fusion']}, dual={r['dual_fusion']}) not in the 10 target methods"})
            continue
        df, reason = load_run_samples(r)
        if df is None:
            inventory.append({**r, "status": "missing_results", "reason": reason})
            continue
        frames.append(df)
        inventory.append({**r, "status": "ok", "reason": "", "n_samples": len(df)})

    master = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    inv_df = pd.DataFrame(inventory)
    return master, inv_df, all_runs


def print_availability(master, inv_df):
    print("\n=== Run inventory: usable vs missing ===")
    for variant in VARIANT_ORDER:
        print(f"\n-- {variant} --")
        v_inv = inv_df[inv_df["variant"] == variant]
        for method in METHOD_ORDER:
            m_inv = v_inv[v_inv["method"] == method]
            if m_inv.empty:
                continue
            ok = sorted(m_inv.loc[m_inv["status"] == "ok", "split_seed"].dropna().astype(int).tolist())
            missing = sorted(m_inv.loc[m_inv["status"] == "missing_results", "split_seed"].dropna().astype(int).tolist())
            print(f"  {method:42s} usable seeds={ok}" + (f"   MISSING RESULTS seeds={missing}" if missing else ""))
    excl = inv_df[inv_df["status"] == "excluded"]
    if not excl.empty:
        others = sorted(set(f"{f}{'+dual' if d else ''}" for f, d in zip(excl["fusion"], excl["dual_fusion"])))
        print(f"\nExcluded (not one of the 10 target methods): {others} -- n={len(excl)} runs")


def lcd_seeds_for(master, variant, methods):
    """Actual (not assumed) intersection of split_seeds for which every method in `methods`
    has usable results in `variant`."""
    sub = master[(master["variant"] == variant) & (master["fusion_method"].isin(methods))]
    per_method_seeds = {m: set(sub.loc[sub["fusion_method"] == m, "split_seed"].unique()) for m in methods}
    present_methods = [m for m, s in per_method_seeds.items() if s]
    if not present_methods:
        return [], per_method_seeds
    common = set.intersection(*(per_method_seeds[m] for m in present_methods))
    return sorted(int(s) for s in common), per_method_seeds


# ===============================================================================================
# Deliverable 1: scatter plots (per-sample), one figure per variant, 2 panels (rot_x vs trans_L2,
# y vs z), all methods overlaid with the fixed color map.
# ===============================================================================================

def deliverable_1_scatter(master, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant in VARIANT_ORDER:
        sub = master[master["variant"] == variant]
        if sub.empty:
            continue
        methods_present = [m for m in METHOD_ORDER if (sub["fusion_method"] == m).any()]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
        for m in methods_present:
            md = sub[sub["fusion_method"] == m]
            axes[0].scatter(md["trans_L2"], md["abs_rot_x"], s=8, alpha=0.35, color=COLOR_MAP[m],
                             linewidth=0, label=m)
            axes[1].scatter(md["abs_y"], md["abs_z"], s=8, alpha=0.35, color=COLOR_MAP[m],
                             linewidth=0, label=m)
        axes[0].set_xlabel("trans_L2 abs error (mm, y/z only)")
        axes[0].set_ylabel("rot_x (roll) abs error (deg)")
        axes[0].set_title("(a) rot_x vs trans_L2")
        axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel("y abs error (mm)")
        axes[1].set_ylabel("z abs error (mm)")
        axes[1].set_title("(b) trans_y vs trans_z")
        axes[1].grid(True, alpha=0.3)
        handles = [Patch(color=COLOR_MAP[m], label=m) for m in methods_present]
        fig.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)), fontsize=8,
                   bbox_to_anchor=(0.5, -0.08), frameon=False)
        fig.suptitle(f"Per-sample test error scatter -- variant: {variant} (all seeds pooled, n={len(sub)} samples)")
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        save_mpl_fig(fig, out_dir / f"scatter_{variant}")

        # interactive version
        fig_p = go.Figure()
        for m in methods_present:
            md = sub[sub["fusion_method"] == m]
            fig_p.add_trace(go.Scatter(x=md["trans_L2"], y=md["abs_rot_x"], mode="markers", name=m,
                                        marker=dict(color=COLOR_MAP[m], size=5, opacity=0.4),
                                        hovertext=[f"seed={s} id={i}" for s, i in zip(md["split_seed"], md["sample_id"])]))
        fig_p.update_layout(title=f"{variant}: rot_x vs trans_L2 (per sample)", xaxis_title="trans_L2 (mm)",
                            yaxis_title="rot_x abs error (deg)", template="plotly_white")
        save_plotly_fig(fig_p, out_dir / f"scatter_{variant}_rotx_vs_transL2")


# ===============================================================================================
# Deliverable 2: box plots, abs and signed metrics, grouped by method, side-by-side by variant.
# ===============================================================================================

def deliverable_2_boxplots(master, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    abs_metrics = ["abs_y", "abs_z", "abs_rot_x", "abs_rot_z_geodesic", "trans_L2"]
    signed_metrics = ["signed_y", "signed_z", "signed_rot_x"]

    def _plot(metrics, fname, title):
        n = len(metrics)
        fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 6))
        if n == 1:
            axes = [axes]
        methods_present = [m for m in METHOD_ORDER if (master["fusion_method"] == m).any()]
        seed_counts_note = []
        for ax, metric in zip(axes, metrics):
            positions, data, colors, hatches = [], [], [], []
            xticks, xticklabels = [], []
            pos = 0
            for m in methods_present:
                xticks.append(pos + 0.5)
                xticklabels.append(m)
                for variant in VARIANT_ORDER:
                    md = master[(master["fusion_method"] == m) & (master["variant"] == variant)]
                    if md.empty:
                        pos += 1
                        continue
                    positions.append(pos)
                    data.append(md[metric].to_numpy())
                    colors.append(COLOR_MAP[m])
                    hatches.append(VARIANT_HATCH[variant])
                    if metric == metrics[0]:
                        seed_counts_note.append(f"{m}/{variant}: n_seeds={md['split_seed'].nunique()}, n_samples={len(md)}")
                    pos += 1
                pos += 0.6
            bp = ax.boxplot(data, positions=positions, widths=0.8, patch_artist=True, showfliers=False)
            for patch, c, h in zip(bp["boxes"], colors, hatches):
                patch.set_facecolor(c)
                patch.set_alpha(0.75)
                if h:
                    patch.set_hatch(h)
                    patch.set_edgecolor("black")
            for med in bp["medians"]:
                med.set_color("black")
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=35, ha="right", fontsize=7)
            ax.set_title(metric, fontsize=10)
            ax.grid(True, alpha=0.3, axis="y")
        handles = [Patch(facecolor="white", edgecolor="black", hatch=VARIANT_HATCH[v], label=v) for v in VARIANT_ORDER]
        fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.1), frameon=False)
        fig.suptitle(title, y=1.17)
        fig.tight_layout()
        save_mpl_fig(fig, out_dir / fname)
        return seed_counts_note

    note_abs = _plot(abs_metrics, "boxplot_abs", "Absolute test error by fusion method (solid=divisor_4, hatched=no_divisor; all seeds pooled per method/variant)")
    note_signed = _plot(signed_metrics, "boxplot_signed", "Signed test error by fusion method (solid=divisor_4, hatched=no_divisor; all seeds pooled per method/variant)")

    with open(out_dir / "boxplot_seed_counts.txt", "w") as f:
        f.write("Seed/sample counts pooled into each box (abs-metric figure; identical pooling used for signed):\n")
        f.write("\n".join(note_abs))


# ===============================================================================================
# Deliverable 3: arrow plot (6 methods present in both variants), split_seed 36-40 only.
# ===============================================================================================

ARROW_METHODS = ["concat", "gated", "gated_concat", "se_channel_gated_concat",
                  "elementwise_gated_concat", "se_gated_cross_attention"]
ARROW_SEED_WINDOW = {36, 37, 38, 39, 40}


def deliverable_3_arrow(master, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    sub = master[(master["fusion_method"].isin(ARROW_METHODS)) & (master["split_seed"].isin(ARROW_SEED_WINDOW))]

    rows = []
    for m in ARROW_METHODS:
        per_seed = {}
        for variant in VARIANT_ORDER:
            md = sub[(sub["fusion_method"] == m) & (sub["variant"] == variant)]
            seed_means = md.groupby("split_seed").agg(trans_L2=("trans_L2", "mean"),
                                                        rot_z=("abs_rot_z_geodesic", "mean"))
            per_seed[variant] = seed_means
        rows.append((m, per_seed))

    fig, ax = plt.subplots(figsize=(9, 8))
    table_rows = []
    for m, per_seed in rows:
        nd, d4 = per_seed["no_divisor"], per_seed["divisor_4"]
        common_seeds = sorted(set(nd.index) & set(d4.index))
        if not common_seeds:
            print(f"  [arrow plot] {m}: NO overlapping seeds in {sorted(ARROW_SEED_WINDOW)} between variants -- skipped")
            continue
        nd_c, d4_c = nd.loc[common_seeds], d4.loc[common_seeds]
        # faint per-seed scatter behind the means (chosen over ellipses: simpler, shows the raw
        # seed-to-seed spread directly without a Gaussian assumption, and reads cleanly with only
        # up to 5 points per method/variant)
        ax.scatter(nd_c["trans_L2"], nd_c["rot_z"], s=22, color=COLOR_MAP[m], alpha=0.25, marker="^")
        ax.scatter(d4_c["trans_L2"], d4_c["rot_z"], s=22, color=COLOR_MAP[m], alpha=0.25, marker="o")
        x0, y0 = nd_c["trans_L2"].mean(), nd_c["rot_z"].mean()
        x1, y1 = d4_c["trans_L2"].mean(), d4_c["rot_z"].mean()
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=COLOR_MAP[m], lw=2.2, alpha=0.9))
        ax.scatter([x0], [y0], s=70, color=COLOR_MAP[m], marker="^", edgecolor="black", linewidth=0.8,
                   label=f"{m} (no_divisor -> divisor_4)")
        ax.scatter([x1], [y1], s=70, color=COLOR_MAP[m], marker="o", edgecolor="black", linewidth=0.8)
        table_rows.append(dict(method=m, n_common_seeds=len(common_seeds), seeds=common_seeds,
                                no_divisor_trans_L2_mean=x0, no_divisor_trans_L2_std=nd_c["trans_L2"].std(),
                                no_divisor_rot_z_mean=y0, no_divisor_rot_z_std=nd_c["rot_z"].std(),
                                divisor4_trans_L2_mean=x1, divisor4_trans_L2_std=d4_c["trans_L2"].std(),
                                divisor4_rot_z_mean=y1, divisor4_rot_z_std=d4_c["rot_z"].std(),
                                delta_trans_L2=x1 - x0, delta_rot_z=y1 - y0))

    ax.set_xlabel("mean trans_L2 abs error (mm)")
    ax.set_ylabel("mean abs_rot_z_geodesic error (deg)")
    ax.set_title(f"Divisor effect per method (triangle=no_divisor, circle=divisor_4)\n"
                 f"split_seed in {sorted(ARROW_SEED_WINDOW)}, only seeds common to both variants used per method; faint points = individual seeds")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    save_mpl_fig(fig, out_dir / "arrow_plot_trans_vs_rot")

    df = pd.DataFrame(table_rows)
    df.to_csv(out_dir / "arrow_plot_data.csv", index=False)

    # 1D supplementary versions
    for metric, label in (("trans_L2", "trans_L2 (mm)"), ("rot_z", "abs_rot_z_geodesic (deg)")):
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, row in df.iterrows():
            y0, y1 = row[f"no_divisor_{metric}_mean"], row[f"divisor4_{metric}_mean"]
            ax.annotate("", xy=(i, y1), xytext=(i, y0),
                        arrowprops=dict(arrowstyle="-|>", color=COLOR_MAP[row["method"]], lw=2.2))
            ax.scatter([i], [y0], color=COLOR_MAP[row["method"]], marker="^", s=50, edgecolor="black", linewidth=0.6)
            ax.scatter([i], [y1], color=COLOR_MAP[row["method"]], marker="o", s=50, edgecolor="black", linewidth=0.6)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["method"], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(f"Divisor effect: {label} (triangle=no_divisor, circle=divisor_4)")
        ax.grid(True, alpha=0.3, axis="y")
        save_mpl_fig(fig, out_dir / f"arrow_plot_1d_{metric}")
    return df


# ===============================================================================================
# Deliverable 4: summary tables (LCD seed set per variant), mean/median/std.
# ===============================================================================================

TABLE_METRICS = ["abs_y", "abs_z", "abs_rot_x", "abs_rot_z_geodesic", "trans_L2"]


def deliverable_4_tables(master, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for variant in VARIANT_ORDER:
        methods_present = sorted(master.loc[master["variant"] == variant, "fusion_method"].unique(),
                                  key=lambda m: METHOD_ORDER.index(m))
        lcd_seeds, per_method_seeds = lcd_seeds_for(master, variant, methods_present)
        print(f"\n[summary table] {variant}: true common seed set across ALL methods present = {lcd_seeds}")
        for m in methods_present:
            print(f"    {m}: seeds={sorted(per_method_seeds[m])}")

        # Use the true LCD across whichever methods actually have ANY data (methods missing
        # entirely from a variant are just absent rows, not grounds for shrinking the seed set).
        use_methods = [m for m in methods_present if per_method_seeds[m]]
        seeds_used, _ = lcd_seeds_for(master, variant, use_methods)
        if not seeds_used:
            # No seed is common to every available method (this happens for divisor_4, where
            # several methods have NO usable results at all -- see run_inventory.csv / the
            # printed availability warnings). Falling back to each method's own full available
            # seed set rather than producing an empty table; "seeds_used" per row then differs
            # across rows -- this is a real data-availability gap, not a modeling choice.
            print(f"    -> no seed common to all {len(use_methods)} available methods; "
                  f"falling back to each method's own available seeds for this table")
        rows = []
        for m in use_methods:
            seeds_for_m = seeds_used if seeds_used else sorted(per_method_seeds[m])
            md = master[(master["variant"] == variant) & (master["fusion_method"] == m) & (master["split_seed"].isin(seeds_for_m))]
            if md.empty:
                continue
            row = {"fusion_method": m, "n_seeds": md["split_seed"].nunique(), "n_samples": len(md),
                   "seeds_used": sorted(md["split_seed"].unique().tolist())}
            for metric in TABLE_METRICS:
                row[f"{metric}_mean"] = md[metric].mean()
                row[f"{metric}_median"] = md[metric].median()
                row[f"{metric}_std"] = md[metric].std()
            rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"summary_table_{variant}.csv", index=False)
        # for the gate-behaviour LCD restriction (deliverable 6): the union of seeds actually
        # used anywhere in this table (per-method, since there may be no single shared seed set)
        all_used_seeds = sorted(set(s for row in rows for s in row["seeds_used"]))
        results[variant] = (df, all_used_seeds)

        if df.empty:
            print(f"    -> {variant}: no usable rows for the summary table at all; skipping render")
            continue

        # Rendered PNG table
        fig, ax = plt.subplots(figsize=(1.6 + 1.6 * len(TABLE_METRICS) * 3, 0.5 + 0.45 * len(df)))
        ax.axis("off")
        cell_cols = ["fusion_method", "n_seeds"] + [f"{m}_{stat}" for m in TABLE_METRICS for stat in ("mean", "median", "std")]
        header = ["method", "n_seeds"] + [f"{m}\n{stat}" for m in TABLE_METRICS for stat in ("mean", "median", "std")]
        cell_text = [[f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cell_cols] for _, r in df.iterrows()]
        tbl = ax.table(cellText=cell_text, colLabels=header, loc="center", cellLoc="center")
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        for (row, col), cell in tbl.get_celld().items():
            if row == 0:
                cell.set_facecolor("#dddddd")
            elif col == 0 and row - 1 < len(df):
                cell.set_facecolor(COLOR_MAP.get(df.iloc[row - 1]["fusion_method"], "white"))
                cell.set_alpha(0.5)
        ax.set_title(f"Summary table -- {variant} (seeds used: {seeds_used})", fontsize=10)
        fig.tight_layout()
        save_mpl_fig(fig, out_dir / f"summary_table_{variant}", dpi=180)
    return results


# ===============================================================================================
# Deliverable 5a: split-effect removal (delta from same-split average over available methods).
# ===============================================================================================

def deliverable_5a_split_centered(master, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in VARIANT_ORDER:
        sub = master[master["variant"] == variant]
        for seed in sorted(sub["split_seed"].dropna().unique()):
            sd = sub[sub["split_seed"] == seed]
            methods_here = sd["fusion_method"].unique()
            split_avg = {metric: sd[metric].mean() for metric in TABLE_METRICS}
            for m in methods_here:
                md = sd[sd["fusion_method"] == m]
                row = {"variant": variant, "split_seed": int(seed), "fusion_method": m, "n_samples": len(md)}
                for metric in TABLE_METRICS:
                    row[f"{metric}_delta"] = md[metric].mean() - split_avg[metric]
                rows.append(row)
    delta_df = pd.DataFrame(rows)
    delta_df.to_csv(out_dir / "split_centered_deltas.csv", index=False)

    agg = (delta_df.groupby(["variant", "fusion_method"])
           .agg({f"{m}_delta": ["mean", "std"] for m in TABLE_METRICS}))
    agg.columns = ["_".join(c) for c in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(out_dir / "split_centered_deltas_summary.csv", index=False)

    fig, axes = plt.subplots(1, len(TABLE_METRICS), figsize=(5.2 * len(TABLE_METRICS), 6))
    for ax, metric in zip(axes, TABLE_METRICS):
        positions, data, colors, hatches, xticks, xticklabels = [], [], [], [], [], []
        pos = 0
        methods_present = [m for m in METHOD_ORDER if (delta_df["fusion_method"] == m).any()]
        for m in methods_present:
            xticks.append(pos + 0.5)
            xticklabels.append(m)
            for variant in VARIANT_ORDER:
                d = delta_df[(delta_df["fusion_method"] == m) & (delta_df["variant"] == variant)][f"{metric}_delta"]
                if d.empty:
                    pos += 1
                    continue
                positions.append(pos)
                data.append(d.to_numpy())
                colors.append(COLOR_MAP[m])
                hatches.append(VARIANT_HATCH[variant])
                pos += 1
            pos += 0.6
        bp = ax.boxplot(data, positions=positions, widths=0.8, patch_artist=True, showfliers=False)
        for patch, c, h in zip(bp["boxes"], colors, hatches):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
            if h:
                patch.set_hatch(h)
        ax.axhline(0, color="black", lw=1, ls=":")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=35, ha="right", fontsize=7)
        ax.set_title(f"{metric} delta vs split average", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
    handles = [Patch(facecolor="white", edgecolor="black", hatch=VARIANT_HATCH[v], label=v) for v in VARIANT_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.12), frameon=False)
    fig.suptitle("Per-method error delta from the average of all methods present in the same split_seed\n"
                 "(negative = better than that split's average)", y=1.2)
    fig.tight_layout()
    save_mpl_fig(fig, out_dir / "split_centered_deltas_boxplot")
    return agg


# ===============================================================================================
# Deliverable 5b: paired per-sample diff vs concat (same variant + split_seed).
# ===============================================================================================

def deliverable_5b_paired_vs_concat(master, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["abs_rot_z_geodesic", "trans_L2"]
    out_rows = []
    per_variant_box = {}
    for variant in VARIANT_ORDER:
        sub = master[master["variant"] == variant]
        concat_seeds = sorted(sub.loc[sub["fusion_method"] == "concat", "split_seed"].dropna().unique())
        if not concat_seeds:
            print(f"  [paired vs concat] {variant}: no concat runs with usable results at all -- skipped")
            continue
        methods = [m for m in METHOD_ORDER if m != "concat" and (sub["fusion_method"] == m).any()]
        box_data = {m: {metric: [] for metric in metrics} for m in methods}
        for seed in concat_seeds:
            concat_s = sub[(sub["fusion_method"] == "concat") & (sub["split_seed"] == seed)]
            if concat_s.empty:
                continue
            concat_s = concat_s.set_index("sample_id")
            for m in methods:
                md = sub[(sub["fusion_method"] == m) & (sub["split_seed"] == seed)]
                if md.empty:
                    continue
                md = md.set_index("sample_id")
                common = md.index.intersection(concat_s.index)
                if len(common) == 0:
                    continue
                for metric in metrics:
                    diff = (md.loc[common, metric] - concat_s.loc[common, metric]).to_numpy()
                    box_data[m][metric].append(diff)
                    for v in diff:
                        out_rows.append(dict(variant=variant, split_seed=int(seed), fusion_method=m, metric=metric, diff=v))
        per_variant_box[variant] = (box_data, methods)

    diff_df = pd.DataFrame(out_rows)
    diff_df.to_csv(out_dir / "paired_diff_per_sample.csv", index=False)
    summary = (diff_df.groupby(["variant", "fusion_method", "metric"])["diff"]
               .agg(["mean", "median", "std", "count"]).reset_index())
    summary.to_csv(out_dir / "paired_diff_summary.csv", index=False)

    for metric in metrics:
        fig, axes = plt.subplots(1, len(VARIANT_ORDER), figsize=(8 * len(VARIANT_ORDER), 6), sharey=True)
        for ax, variant in zip(axes, VARIANT_ORDER):
            if variant not in per_variant_box:
                ax.set_title(f"{variant}: no concat baseline available")
                continue
            box_data, methods = per_variant_box[variant]
            data, colors, labels = [], [], []
            for m in methods:
                chunks = box_data[m][metric]
                if not chunks:
                    continue
                data.append(np.concatenate(chunks))
                colors.append(COLOR_MAP[m])
                labels.append(m)
            if not data:
                continue
            bp = ax.boxplot(data, positions=range(len(data)), widths=0.6, patch_artist=True, showfliers=False)
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.75)
            ax.axhline(0, color="black", lw=1, ls=":")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
            ax.set_title(f"{variant} (concat seeds used: {sorted(sub_seeds for sub_seeds in master.loc[(master['variant']==variant)&(master['fusion_method']=='concat'),'split_seed'].unique())})", fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")
        fig.suptitle(f"Per-sample paired difference vs concat (same test samples, same split_seed): {metric}\n(+ = worse than concat)")
        fig.tight_layout()
        save_mpl_fig(fig, out_dir / f"paired_diff_vs_concat_{metric}")
    return summary


# ===============================================================================================
# Deliverable 6: gate behaviour, from gate_weights_best_epoch.pt (per-sample VAL gate weights at
# the best checkpoint) -- NO live inference, NO Data/Data_split access at all. See module
# docstring / final report for why: a training job is actively using Data/Data_split.
# ===============================================================================================

SCALE_CHANNELS_FULL = {"p3": 32, "p4": 64, "p5": 128}  # no pre_head_channel_divisor


def _scale_layout(dim, divisor):
    chans = {s: max(1, c // divisor) if divisor else c for s, c in SCALE_CHANNELS_FULL.items()}
    sizes = {"p3": chans["p3"] * 400, "p4": chans["p4"] * 100, "p5": chans["p5"] * 25}
    if sum(sizes.values()) != dim:
        return None
    bounds, off = {}, 0
    for s in ("p3", "p4", "p5"):
        bounds[s] = (off, off + sizes[s])
        off += sizes[s]
    return bounds


def load_gate_weights(run_folder):
    """Returns dict(branch -> (N, num_streams, dim) tensor) or None if the file doesn't exist /
    the fusion has no gate concept (plain concat, se_gated_cross_attention -- see report)."""
    p = Path(run_folder) / "gate_weights_best_epoch.pt"
    if not p.exists():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = torch.load(p, map_location="cpu", weights_only=False)
    def _to3d(w):
        return w if w.dim() == 3 else w.unsqueeze(-1)

    if "val_gate_weights" in d:
        return {"shared": _to3d(d["val_gate_weights"])}
    out = {}
    if "val_rotation_gate_weights" in d:
        out["rotation"] = _to3d(d["val_rotation_gate_weights"])
    if "val_translation_gate_weights" in d:
        out["translation"] = _to3d(d["val_translation_gate_weights"])
    return out or None


# Live TEST-set per-sample gate inference (scripts/evaluate/plot_gate_distributions.py
# --split test --rebuild_split), run once this ablation's Data/Data_split contention with a
# training job was resolved -- preferred over the VAL-at-best-epoch .pt snapshot above wherever
# a run has an entry here. Dual-fusion runs get one row per branch (rotation/translation); those
# are averaged together below exactly like the .pt path already averages the two branch tensors
# (see CLAUDE.md's dual_fusion note), just on the aggregated stats instead of raw tensors since
# that's what this CSV stores. No per-channel P3/P4/P5 breakdown is available from this source
# (plot_gate_distributions.py doesn't export it), so the scale-breakdown plot/CSV below still
# falls back to the .pt snapshot when present, and is simply skipped for runs that only have the
# live-test entry.
TEST_GATE_STATS_CSV = (REPO_ROOT / "analysis_fusion" / "divisor_ablation" / "6_gate_behaviour" /
                        "test_set_gate_inference" / "gate_distribution_stats_test.csv")


def load_test_gate_stats_table():
    if not TEST_GATE_STATS_CSV.exists():
        return None
    df = pd.read_csv(TEST_GATE_STATS_CSV)
    return df if not df.empty else None


def test_gate_stats_for_run(test_gate_df, run_dir):
    """Per-stream dict(stream -> dict(mean_gate, std_gate, frac_lt_0_1, frac_gt_0_9, n_samples))
    for one run, averaging rotation/translation branch rows (dual_fusion) if both are present.
    None if this run has no rows in the live-test table (plot_gate_distributions.py skipped it,
    or Task 2 simply hasn't covered it)."""
    if test_gate_df is None:
        return None
    name = os.path.basename(str(run_dir))
    sub = test_gate_df[test_gate_df["run"] == name]
    if sub.empty:
        return None
    out = {}
    for stream, grp in sub.groupby("stream"):
        out[stream] = dict(
            mean_gate=float(grp["mean"].mean()), std_gate=float(grp["std"].mean()),
            frac_lt_0_1=float(grp["frac_lt_0_1"].mean()), frac_gt_0_9=float(grp["frac_gt_0_9"].mean()),
            n_samples=int(grp["n_samples"].iloc[0]),
        )
    return out


def deliverable_6_gates(master, out_dir, lcd_seeds_by_variant):
    out_dir.mkdir(parents=True, exist_ok=True)
    test_gate_df = load_test_gate_stats_table()
    rows, scale_rows = [], []
    n_live, n_snapshot = 0, 0
    for variant in VARIANT_ORDER:
        seeds = lcd_seeds_by_variant.get(variant, [])
        sub = master[(master["variant"] == variant) & (master["split_seed"].isin(seeds))]
        run_lookup = sub[["run_dir", "fusion_method", "split_seed"]].drop_duplicates()
        for _, r in run_lookup.iterrows():
            method, seed, run_dir = r["fusion_method"], int(r["split_seed"]), r["run_dir"]

            live = test_gate_stats_for_run(test_gate_df, run_dir)
            if live is not None:
                # Preferred: real per-sample TEST-set inference (Task 2 / plot_gate_distributions.py
                # --split test --rebuild_split). No per-channel tensor here, so no scale breakdown.
                n_live += 1
                for stream_name, st in live.items():
                    rows.append(dict(
                        fusion_method=method, variant=variant, split_seed=seed, stream=stream_name,
                        n_samples=st["n_samples"], dim=None, mean_gate=st["mean_gate"], std_gate=st["std_gate"],
                        frac_lt_0_1=st["frac_lt_0_1"], frac_gt_0_9=st["frac_gt_0_9"], source="test_live_inference",
                    ))
                continue

            gw = load_gate_weights(run_dir)
            if gw is None:
                continue  # concat / se_gated_cross_attention: no last_fusion_weights -- skip gracefully
            n_snapshot += 1
            # average rotation/translation branches for dual_fusion runs (matches CLAUDE.md's
            # note that pooled gate diagnostics already average dual-fusion branches)
            branches = list(gw.values())
            w = torch.stack(branches, dim=0).mean(dim=0) if len(branches) > 1 else branches[0]
            w = w.clamp(0.0, 1.0)
            N, S, D = w.shape
            divisor = 4 if variant == "divisor_4" else None
            bounds = _scale_layout(D, divisor) if D > 1 else None
            for s in range(S):
                stream_name = "rgb" if s == 0 else "tactile"
                vals = w[:, s, :].numpy().ravel()
                rows.append(dict(
                    fusion_method=method, variant=variant, split_seed=seed, stream=stream_name,
                    n_samples=N, dim=D, mean_gate=float(vals.mean()), std_gate=float(vals.std()),
                    frac_lt_0_1=float((vals < 0.1).mean()), frac_gt_0_9=float((vals > 0.9).mean()),
                    source="val_best_epoch_snapshot",
                ))
                if bounds:
                    for scale, (lo, hi) in bounds.items():
                        scale_vals = w[:, s, lo:hi].numpy().ravel()
                        scale_rows.append(dict(fusion_method=method, variant=variant, split_seed=seed,
                                                stream=stream_name, scale=scale, mean_gate=float(scale_vals.mean())))

    print(f"  [gate behaviour] {n_live} run rows from live TEST-set inference, "
          f"{n_snapshot} from the VAL-best-epoch .pt snapshot fallback")
    gate_df = pd.DataFrame(rows)
    gate_df.to_csv(out_dir / "gate_table_per_run.csv", index=False)
    if gate_df.empty:
        print("  [gate behaviour] no gate data (live or snapshot) found for the LCD seed sets -- nothing to plot")
        return gate_df, pd.DataFrame()

    agg = (gate_df.groupby(["fusion_method", "variant", "stream"])
           .agg(mean_gate=("mean_gate", "mean"), gate_spread_std=("mean_gate", "std"),
                frac_lt_0_1=("frac_lt_0_1", "mean"), frac_gt_0_9=("frac_gt_0_9", "mean"),
                n_runs=("mean_gate", "count")).reset_index())
    pivot = agg.pivot_table(index=["fusion_method", "variant"], columns="stream",
                             values=["mean_gate", "frac_lt_0_1", "frac_gt_0_9", "gate_spread_std"])
    pivot.columns = [f"{a}_{b}" for a, b in pivot.columns]
    table = pivot.reset_index()
    table.to_csv(out_dir / "gate_behaviour_table.csv", index=False)

    scale_df = pd.DataFrame(scale_rows)
    if not scale_df.empty:
        scale_df.to_csv(out_dir / "gate_scale_breakdown.csv", index=False)
        scale_agg = scale_df.groupby(["fusion_method", "variant", "stream", "scale"])["mean_gate"].mean().reset_index()
        methods_present = [m for m in METHOD_ORDER if (scale_agg["fusion_method"] == m).any()]
        fig, axes = plt.subplots(1, len(methods_present), figsize=(4.2 * len(methods_present), 5), squeeze=False)
        for ax, m in zip(axes[0], methods_present):
            md = scale_agg[scale_agg["fusion_method"] == m]
            width = 0.35
            scales = ["p3", "p4", "p5"]
            x = np.arange(len(scales))
            for vi, variant in enumerate(VARIANT_ORDER):
                for si, stream in enumerate(["rgb", "tactile"]):
                    d = md[(md["variant"] == variant) & (md["stream"] == stream)].set_index("scale").reindex(scales)["mean_gate"]
                    offset = (vi * 2 + si - 1.5) * width / 2
                    ax.bar(x + offset, d.to_numpy(), width=width / 2, label=f"{variant}/{stream}",
                           color=COLOR_MAP[m], alpha=0.9 if variant == "divisor_4" else 0.45,
                           hatch=VARIANT_HATCH[variant], edgecolor="black" if stream == "tactile" else "none", linewidth=1.2)
            ax.set_xticks(x)
            ax.set_xticklabels(scales)
            ax.set_title(m, fontsize=8)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3, axis="y")
        axes[0][0].set_ylabel("mean gate value")
        fig.suptitle("Mean gate per pyramid level (P3/P4/P5) and stream (dark=divisor_4, light/hatched=no_divisor; edge=tactile)")
        fig.tight_layout()
        save_mpl_fig(fig, out_dir / "gate_per_scale_stream")

    skipped = sorted(set(master["fusion_method"].unique()) - set(gate_df["fusion_method"].unique()))
    with open(out_dir / "gate_behaviour_notes.txt", "w") as f:
        f.write(f"Gate behaviour: {n_live} run rows from real per-sample TEST-set inference "
                f"(scripts/evaluate/plot_gate_distributions.py --split test --rebuild_split, written to "
                f"{TEST_GATE_STATS_CSV.relative_to(REPO_ROOT)}), {n_snapshot} rows falling back to the older "
                "gate_weights_best_epoch.pt snapshot (per-sample VAL gate weights at the best checkpoint) "
                "where a run has no live-test entry. See gate_table_per_run.csv's 'source' column per row.\n\n"
                "The scale breakdown (gate_per_scale_stream / gate_scale_breakdown.csv) only covers rows "
                "sourced from the .pt snapshot -- the live-test CSV doesn't carry a per-channel tensor, "
                "only aggregated per-stream stats, so no P3/P4/P5 split is available for those rows.\n\n")
        f.write(f"Methods with NO gate data in the LCD seed sets (neither source -- no gate concept, or "
                f"the run predates both the .pt checkpointing feature and Task 2's live-test coverage): {skipped}\n")
    return table, scale_df


# ===============================================================================================
# Deliverable 7: data deliverables (hardest/typical montage + pose-space plot).
# ===============================================================================================

def _transf_matrix():
    import ast
    tree = ast.parse((REPO_ROOT / "data_processing" / "preprocess_data.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "transf_matrix" for t in node.targets):
            return ast.literal_eval(node.value)
    raise RuntimeError("transf_matrix not found")


def reconstruct_test_order(split_seed, data_root):
    """Reproduces test.py's sample_id -> timestamp mapping for a given split_seed, read-only
    (reads raw Data/ triplets directly; does NOT touch Data/Data_split). Adapted from
    scripts/evaluate/sample_level_error_analysis.py::build_sample_index."""
    from data_processing.GripFusionNet_split import group_triplets_by_pose, _assign_groups_to_ratios
    from data_processing.preprocess_data import _hub_frame_pose
    from utils.GripFusionNet_utils import get_triplets

    groups = group_triplets_by_pose(get_triplets(data_root), data_root)
    group_of = {t[0]: gi for gi, g in enumerate(groups) for t in g}
    _, _, test = _assign_groups_to_ratios(groups, 0.8, 0.15, 0.05, random.Random(int(split_seed)))
    stamps = [t[0] for t in test]
    scratch = REPO_ROOT / "runs" / "divisor_ablation_order_scratch" / f"seed_{split_seed}"
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        for ts in stamps:
            (scratch / f"{ts}.pt").touch()
        order = [f[:-3] for f in os.listdir(scratch) if f.endswith(".pt")]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    tm = _transf_matrix()
    poses = np.array([_hub_frame_pose(os.path.join(data_root, f"{ts}.json"), tm) for ts in order])
    return order, poses, group_of, groups


def _pair_image(ts, data_root, height=240):
    from PIL import Image
    ims = []
    for suffix in ("", "_digit"):
        im = Image.open(os.path.join(data_root, f"{ts}{suffix}.png")).convert("RGB")
        ims.append(im.resize((int(im.width * height / im.height), height)))
    canvas = Image.new("RGB", (ims[0].width + ims[1].width + 8, height), (255, 255, 255))
    canvas.paste(ims[0], (0, 0))
    canvas.paste(ims[1], (ims[0].width + 8, 0))
    return canvas


def deliverable_7_samples(master, out_dir, data_root):
    out_dir.mkdir(parents=True, exist_ok=True)
    # representative run per variant: concat at the lowest usable split_seed.
    chosen = {}
    for variant in VARIANT_ORDER:
        sub = master[(master["variant"] == variant) & (master["fusion_method"] == "concat")]
        if sub.empty:
            print(f"  [deliverable 7] {variant}: concat has no usable results; skipped for this variant")
            continue
        seed = int(sorted(sub["split_seed"].unique())[0])
        chosen[variant] = seed

    for variant, seed in chosen.items():
        sub = master[(master["variant"] == variant) & (master["fusion_method"] == "concat") & (master["split_seed"] == seed)].copy()
        order, poses, group_of, groups = reconstruct_test_order(seed, data_root)
        if len(order) != len(sub):
            print(f"  [deliverable 7] {variant} seed {seed}: reconstructed test set size {len(order)} != results size {len(sub)} -- skipped")
            continue
        ok = np.allclose(poses[:, 1], sub.sort_values("sample_id")["gt_trans_y"].to_numpy(), atol=1e-6)
        if not ok:
            print(f"  [deliverable 7] {variant} seed {seed}: reconstructed order DOES NOT MATCH ground truth -- skipped")
            continue
        sub = sub.sort_values("sample_id").reset_index(drop=True)
        sub["timestamp"] = order
        sub["pred_roll_deg"] = sub["pred_roll_deg"]

        hardest = sub.nlargest(3, "abs_rot_z_geodesic")
        lo, hi = sub["abs_rot_z_geodesic"].quantile([0.45, 0.55])
        typical_pool = sub[(sub["abs_rot_z_geodesic"] >= lo) & (sub["abs_rot_z_geodesic"] <= hi)]
        typical = typical_pool.sample(min(3, len(typical_pool)), random_state=0)

        def _montage(rows, name, title):
            n = len(rows)
            if n == 0:
                return
            fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), squeeze=False)
            for ax, (_, r) in zip(axes[0], rows.iterrows()):
                ax.imshow(_pair_image(r["timestamp"], data_root))
                ax.axis("off")
                ax.set_title(f"id {int(r['sample_id'])}: rot_z_geo {r['abs_rot_z_geodesic']:.2f}deg "
                             f"trans_L2 {r['trans_L2']:.2f}mm\ngt_roll {r['gt_roll_deg']:.1f} pred_roll {r['pred_roll_deg']:.1f}",
                             fontsize=8)
            fig.suptitle(f"{title} -- {variant}, concat, split_seed={seed}  (left RGB, right DIGIT)")
            fig.tight_layout()
            save_mpl_fig(fig, out_dir / f"{name}_{variant}")

        _montage(hardest, "montage_hardest", "Hardest test samples (top-3 abs_rot_z_geodesic)")
        _montage(typical, "montage_typical", "Typical test samples (45-55th percentile abs_rot_z_geodesic)")

        # pose-space plot
        fig, ax = plt.subplots(figsize=(7, 6.5))
        ax.scatter(sub["gt_trans_y"] * 1000, sub["gt_trans_z"] * 1000, s=14, color="#999999", alpha=0.5,
                   label="all test samples")
        ax.scatter(hardest["gt_trans_y"] * 1000, hardest["gt_trans_z"] * 1000, s=90, color="#d62728",
                   edgecolor="black", marker="*", label="hardest 3 (abs_rot_z_geodesic)")
        ax.set_xlabel("gt_trans_y (mm)")
        ax.set_ylabel("gt_trans_z (mm)")
        ax.set_title(f"Pose-space distribution of test samples -- {variant}, split_seed={seed}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        save_mpl_fig(fig, out_dir / f"pose_space_{variant}")

        hardest.assign(rgb_file=[os.path.join(data_root, f"{t}.png") for t in hardest["timestamp"]],
                        digit_file=[os.path.join(data_root, f"{t}_digit.png") for t in hardest["timestamp"]]
                        ).to_csv(out_dir / f"hardest_samples_{variant}.csv", index=False)


# ===============================================================================================
def main():
    write_color_map()
    master, inv_df, all_runs = build_master_df()
    print_availability(master, inv_df)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    master.to_csv(OUT_ROOT / "master_per_sample.csv", index=False)
    inv_df.to_csv(OUT_ROOT / "run_inventory.csv", index=False)
    print(f"\nMaster per-sample df: {len(master)} rows across {master['run_dir'].nunique()} usable runs")

    deliverable_1_scatter(master, OUT_ROOT / "1_scatter")
    deliverable_2_boxplots(master, OUT_ROOT / "2_boxplots")
    deliverable_3_arrow(master, OUT_ROOT / "3_arrow_plot")
    tables = deliverable_4_tables(master, OUT_ROOT / "4_summary_tables")
    deliverable_5a_split_centered(master, OUT_ROOT / "5_split_effect_removal")
    deliverable_5b_paired_vs_concat(master, OUT_ROOT / "5_split_effect_removal")

    lcd_by_variant = {v: tables[v][1] for v in VARIANT_ORDER if v in tables}
    deliverable_6_gates(master, OUT_ROOT / "6_gate_behaviour", lcd_by_variant)

    deliverable_7_samples(master, OUT_ROOT / "7_data_deliverables", str(REPO_ROOT / "Data"))

    print(f"\nDone. Outputs under {OUT_ROOT}")


if __name__ == "__main__":
    main()
