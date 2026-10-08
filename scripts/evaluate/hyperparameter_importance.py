#!/usr/bin/env python3
"""
Hyperparameter importance from existing ablation runs
======================================================

Two complementary analyses over the runs under ``runs/`` (default: last 30 days),
for the fusion groups scalar / channelwise / dual fusion:

1. ``paired``: matched-pair comparisons. Two sets of runs are compared when their
   configurations are identical in every tracked hyperparameter except one factor (for
   example grad_clip False vs True, rot_scaling_factor 1 vs 200, dual_fusion off vs on),
   whether the runs come from the same sweep or from two different sweeps. The result is the
   signed change of the metric (negative = lower error = better), drawn with an approximate
   95% seed-noise band. Fusion type is held fixed inside each match, never compared.
2. ``fanova``: exploratory fANOVA (Hutter et al., random-forest variance decomposition,
   via optuna's FanovaImportanceEvaluator) on the pooled runs, once for all groups
   (fusion group as a factor) and once per group. Repeated with several forest seeds
   and averaged, because the forest is random.

CAVEAT for (2): the runs come from one-factor ablations, not from a joint search, so most
factors vary in only one sweep and for one group. Importance from the pooled forest
partly reflects which sweeps exist. Treat it as exploratory and check it against (1).

Metrics: best (minimum) ``val_abs_trans`` and ``val_abs_rot_x`` per run. Translation is
stored in meters (raw pose units, un-normalized) and displayed in mm; rotation is in degrees.
``val_loss`` is not used: it mixes rotation/translation through rot_scaling_factor and
is not comparable across the scaling regimes present in the runs.

Usage:
    python scripts/evaluate/hyperparameter_importance.py --mode both
    python scripts/evaluate/hyperparameter_importance.py --mode paired --since 2026-08-22
"""

import argparse
import datetime as dt
import itertools
import logging
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Metric key -> (metrics.csv column, display label, scale factor to display unit, unit)
# val_abs_trans is stored in meters, displayed in mm.
METRICS = {
    "trans": ("val_abs_trans", "Translation error", 1e3, "mm"),
    "rot": ("val_abs_rot_x", "Rotation error", 1.0, "deg"),
}

# Factor name -> (dotted config path, kind, default when the key is absent).
# kind: "int" | "float" | "logfloat" | "cat". Derived factors are added in load_runs().
# use_batchnorm is deliberately not included.
# rot_scaling_factor: an absent key means the quaternion handler's built-in default (200).
FACTOR_SPECS = {
    "dual_fusion": ("model.dual_fusion", "cat", False),
    "mlp_quat_1": ("model.mlp_quat_1", "int", None),
    "mlp_quat_2": ("model.mlp_quat_2", "int", None),
    "mlp_quat_3": ("model.mlp_quat_3", "int", None),
    "mlp_trans_1": ("model.mlp_trans_1", "int", None),
    "mlp_trans_2": ("model.mlp_trans_2", "int", None),
    "mlp_trans_3": ("model.mlp_trans_3", "int", None),
    "backbone_trainable": ("model.backbone_trainable", "cat", False),
    "activation": ("model.activation", "cat", None),
    "pre_head_grid_size": ("model.pre_head_grid_size", "cat", "none"),
    "gate_init_prob": ("model.gate_init_prob", "float", 0.5),  # 0.5 = fusion classes' own default
    "lr": ("training.lr", "logfloat", None),
    "batch_size": ("training.batch_size", "int", None),
    "optimizer": ("training.optimizer", "cat", None),
    "weight_decay": ("training.weight_decay", "float", 0.0),
    "dropout_rot": ("training.dropout_rot", "float", None),
    "lr_schedule_type": ("training.lr_schedule.type", "cat", "none"),
    "head_lr_decay": ("training.head_lr_decay.enabled", "cat", False),
    "task_weighting": ("training.task_weighting.enabled", "cat", False),
    "grad_clip": ("training.grad_clip.enabled", "cat", False),
    "rot_scaling_factor": ("model.rot_scaling_factor", "cat", 200),
}
# Not used as comparison factors, but reported when two matched groups also differ in them.
CONTROL_SPECS = {
    "epochs": ("training.epochs", None),
    "early_stopping": ("training.early_stopping.enabled", False),
    "es_patience": ("training.early_stopping.patience", None),
    "es_min_delta": ("training.early_stopping.min_delta", None),
    "group_aware_split": ("group_aware_split", False),
    "lr_backbone": ("training.lr_backbone", None),
    "augmentation": ("data.augmentation.enabled", False),
    "head_type": ("model.head_type", None),
    "translation_head_mode": ("model.translation_head_mode", None),
    "grad_clip_max_norm": ("training.grad_clip.max_norm", None),
}
# Controls that change by construction when the factor is switched; not reported as "also differs".
DEPENDENT_CONTROLS = {"grad_clip": {"grad_clip_max_norm"}, "backbone_trainable": {"lr_backbone"}}
DERIVED_KINDS = {"fusion_group": "cat", "fusion_type": "cat"}
# fANOVA uses fusion_group / fusion_type for the fusion effect, so dual_fusion is left out there.
FANOVA_FACTORS = [f for f in FACTOR_SPECS if f != "dual_fusion"]

GROUPS = ["scalar", "channelwise", "dual"]
# Categorical slots 1-3 of the reference palette (light mode); all-pairs validated for <= 3 series.
GROUP_COLORS = {"scalar": "#2a78d6", "channelwise": "#eb6834", "dual": "#1baf7a"}
POOLED_COLOR = "#2a78d6"
BETTER_COLOR = "#2a78d6"  # diverging pair: blue = error decreased ...
WORSE_COLOR = "#e34948"   # ... red = error increased
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _get(cfg, dotted, default):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return default if node is None else node


def _fusion_group(fusion, dual):
    if dual:
        return "dual"
    return "channelwise" if str(fusion).startswith("channelwise_") else "scalar"


def _cat_str(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "unset"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and value == int(value):
        value = int(value)
    return str(value)


def load_runs(runs_dir, since):
    """One row per finished run: factors + controls + best metrics + provenance."""
    rows = []
    for cfg_path in sorted(Path(runs_dir).glob("*/*/config.yml")):
        run_dir = cfg_path.parent
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            continue
        try:
            cfg = yaml.safe_load(open(cfg_path))
            metrics = pd.read_csv(metrics_path)
        except Exception as exc:  # unreadable/partial run
            logger.warning("skipping %s: %s", run_dir, exc)
            continue
        if not cfg or not len(metrics):
            continue
        if any(METRICS[m][0] not in metrics for m in METRICS):
            continue

        started = str(metrics["timestamp"].iloc[0])[:10] if "timestamp" in metrics else ""
        if started < since:
            continue

        model = cfg.get("model", {})
        dual = bool(model.get("dual_fusion", False))
        fusion = model.get("fusion")
        row = {
            "sweep": run_dir.parent.name,
            "run": run_dir.name,
            "started": started,
            "fusion_group": _fusion_group(fusion, dual),
            "fusion_type": str(fusion),  # base fusion name; dual fusion is the dual_fusion factor
            "method": ("dual_" if dual else "") + str(fusion),
            "n_epochs": len(metrics),
        }
        for name, (dotted, _kind, default) in FACTOR_SPECS.items():
            row[name] = _get(cfg, dotted, default)
        for name, (dotted, default) in CONTROL_SPECS.items():
            row[f"ctl_{name}"] = _get(cfg, dotted, default)
        # A disabled schedule/decay means "none" regardless of a stale type key.
        if not _get(cfg, "training.lr_schedule.enabled", False):
            row["lr_schedule_type"] = "none"
        for key, (col, _label, scale, _unit) in METRICS.items():
            row[key] = float(metrics[col].min()) * scale
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No runs found under {runs_dir} since {since}.")
    for name, (_d, kind, _default) in FACTOR_SPECS.items():
        if kind == "cat":
            df[name] = df[name].map(_cat_str)
    for name in CONTROL_SPECS:
        df[f"ctl_{name}"] = df[f"ctl_{name}"].map(_cat_str)
    return df


def filter_runs(df, max_rot, max_trans):
    """Drop collapsed runs (translation/rotation stuck at the untrained level)."""
    collapsed = (df["rot"] > max_rot) | (df["trans"] > max_trans)
    logger.info("dropping %d collapsed runs (rot > %.1f deg or trans > %.1f mm)",
                int(collapsed.sum()), max_rot, max_trans)
    return df[~collapsed].reset_index(drop=True), df[collapsed]


def factor_kind(name):
    return FACTOR_SPECS[name][1] if name in FACTOR_SPECS else DERIVED_KINDS[name]


def varying_factors(df, candidates, min_level_runs=1):
    """Factors that were really changed: at least two levels with >= min_level_runs runs each.
    (A single stray run with a different value does not count as an ablation.)"""
    return [c for c in candidates
            if (df[c].value_counts() >= min_level_runs).sum() >= 2]


# ---------------------------------------------------------------------------
# Seed noise (replicate runs = identical factor vector)
# ---------------------------------------------------------------------------

def seed_noise(df, min_reps=3):
    """Typical seed noise: median coefficient of variation (%) over replicate sets, per fusion
    group and overall. The median (not an RMS) keeps a few unstable early runs, e.g. from the
    sweeps without gradient clipping, from inflating the noise level."""
    key_cols = list(FACTOR_SPECS) + ["fusion_type"]
    out = {}
    keyed = df.assign(_key=df[key_cols].astype(str).agg("|".join, axis=1))
    for scope, sub in [("all", keyed)] + [(g, keyed[keyed.fusion_group == g]) for g in GROUPS]:
        noise = {}
        for metric in METRICS:
            cvs = [rep[metric].std(ddof=1) / rep[metric].mean()
                   for _, rep in sub.groupby("_key")
                   if len(rep) >= min_reps and rep[metric].mean() > 0]
            noise[metric] = 100 * float(np.median(cvs)) if cvs else np.nan
        out[scope] = noise
    return out


# ---------------------------------------------------------------------------
# 1. Matched-pair comparisons
# ---------------------------------------------------------------------------

def _level_sort_key(level):
    try:
        return (0, float(level), "")
    except ValueError:
        return (1, 0.0, str(level))


def _sweep_names(text, keep=2):
    """Compact list of sweep names; long lists are cut after `keep` names."""
    names = [_short(n) for n in text.split(" + ")]
    return " + ".join(names) if len(names) <= keep else " + ".join(names[:keep]) + f" +{len(names) - keep} more"


def _short(text):
    """Compact names for axis labels (channelwise_ -> cw_, residual_ -> res_)."""
    return str(text).replace("channelwise_", "cw_").replace("residual_", "res_")


def paired_comparisons(df, min_cell_runs, noise):
    """Signed metric change between two levels of one factor with everything else identical."""
    controls = [f"ctl_{c}" for c in CONTROL_SPECS]
    cells = []
    for factor in FACTOR_SPECS:
        others = [f for f in FACTOR_SPECS if f != factor] + ["fusion_type"]
        keyed = df.assign(_key=df[others].astype(str).agg("|".join, axis=1))
        for _, cell in keyed.groupby("_key"):
            levels = {lv: sub for lv, sub in cell.groupby(factor) if len(sub) >= min_cell_runs}
            for a, b in itertools.combinations(sorted(levels, key=_level_sort_key), 2):
                ra, rb = levels[a], levels[b]
                skip = DEPENDENT_CONTROLS.get(factor, set())
                ctl_diff = [c[4:] for c in controls
                            if c[4:] not in skip and set(ra[c]) != set(rb[c])]
                cells.append({
                    "factor": factor, "from": a, "to": b,
                    "sweeps_from": " + ".join(sorted(ra["sweep"].unique())),
                    "sweeps_to": " + ".join(sorted(rb["sweep"].unique())),
                    "fusion_type": cell["fusion_type"].iloc[0],
                    "group": cell["fusion_group"].iloc[0],
                    "n_from": len(ra), "n_to": len(rb), "ctl_diff": ctl_diff,
                    **{f"{m}_from": ra[m].mean() for m in METRICS},
                    **{f"{m}_to": rb[m].mean() for m in METRICS},
                })
    if not cells:
        return pd.DataFrame()

    cells = pd.DataFrame(cells)
    rows = []
    for (factor, a, b, sw_a, sw_b), grp in cells.groupby(
            ["factor", "from", "to", "sweeps_from", "sweeps_to"], sort=False):
        groups = grp["group"].unique()
        scope = groups[0] if len(groups) == 1 else "all"
        row = {
            "factor": factor, "from": a, "to": b, "sweeps_from": sw_a, "sweeps_to": sw_b,
            "group": groups[0] if len(groups) == 1 else "mixed",
            "n_configs": len(grp), "fusion_types": ", ".join(sorted(grp["fusion_type"].unique())),
            "n_from": int(grp["n_from"].sum()), "n_to": int(grp["n_to"].sum()),
            "also_differs": ", ".join(sorted({c for cd in grp["ctl_diff"] for c in cd})),
        }
        for m in METRICS:
            change = 100 * (grp[f"{m}_to"] - grp[f"{m}_from"]) / grp[f"{m}_from"]
            cv = noise[scope][m]
            se_cell = cv * np.sqrt(1 / grp["n_from"] + 1 / grp["n_to"])
            se = float(np.sqrt((se_cell ** 2).sum()) / len(grp))
            row[f"change_{m}_pct"] = float(change.mean())
            row[f"band_{m}_pct"] = 1.96 * se
            row[f"clear_{m}"] = bool(abs(change.mean()) > 1.96 * se)
        rows.append(row)
    return pd.DataFrame(rows)


def _pair_label(row):
    where = (f"{_sweep_names(row['sweeps_from'])} vs {_sweep_names(row['sweeps_to'])}"
             if row["sweeps_from"] != row["sweeps_to"] else f"within {_sweep_names(row['sweeps_from'])}")
    head = f"[{row['group']}] {row['factor']}: {_short(row['from'])} → {_short(row['to'])}"
    n_cfg = row["n_configs"]
    detail = (f"{where}  ·  {n_cfg} config{'s' if n_cfg != 1 else ''}, "
              f"n = {row['n_from']} vs {row['n_to']}  ·  {_short(row['fusion_types'])}")
    lines = [head] + textwrap.wrap(detail, width=78)
    if row["also_differs"]:
        lines.append(f"also differs: {row['also_differs']}")
    return "\n".join(lines)


def plot_paired(table, path, max_rows):
    table = table.copy()
    table["_size"] = table[["change_trans_pct", "change_rot_pct"]].abs().max(axis=1)
    factor_order = {f: i for i, f in enumerate(FACTOR_SPECS)}
    table["_order"] = table["factor"].map(factor_order)
    table = (table.sort_values("_size", ascending=False).head(max_rows)
             .sort_values(["_order", "_size"], ascending=[True, False])
             .iloc[::-1].reset_index(drop=True))
    labels = [_pair_label(r) for _, r in table.iterrows()]
    y = np.arange(len(table))

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 0.95 * len(table) + 2.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, metric in zip(axes, METRICS):
        ax.set_facecolor(SURFACE)
        change = table[f"change_{metric}_pct"].to_numpy()
        band = table[f"band_{metric}_pct"].to_numpy()
        clear = table[f"clear_{metric}"].to_numpy()
        for yi, c, cl in zip(y, change, clear):
            ax.barh(yi, c, height=0.55, color=BETTER_COLOR if c < 0 else WORSE_COLOR,
                    alpha=1.0 if cl else 0.35, zorder=3)
        ax.errorbar(change, y, xerr=band, fmt="none", ecolor=INK, elinewidth=1.1, capsize=3, zorder=4)
        span = max(np.abs(change).max(), band.max())
        for yi, c, b in zip(y, change, band):
            side = 1 if c >= 0 else -1
            ax.text(c + side * (b + 0.025 * span), yi, f"{c:+.2f}%", va="center",
                    ha="left" if c >= 0 else "right", fontsize=9, color=INK_2)
        ax.axvline(0, color=INK_2, linewidth=1.0, zorder=5)
        ax.set_xlim(-span * 1.75, span * 1.55)
        ax.set_title(f"{METRICS[metric][1]} ({METRICS[metric][3]})", loc="left", fontsize=12, color=INK)
        ax.set_xlabel("Change in error when the factor is switched (%, negative = better)",
                      fontsize=9.5, color=INK_2)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="both", length=0, labelsize=8.5, colors=INK_2)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_ylim(-0.7, len(table) - 0.3)

    handles = [plt.Rectangle((0, 0), 1, 1, color=BETTER_COLOR),
               plt.Rectangle((0, 0), 1, 1, color=WORSE_COLOR),
               plt.Rectangle((0, 0), 1, 1, color=INK_2, alpha=0.35),
               plt.Line2D([], [], color=INK, marker="|", markersize=9, linestyle="-")]
    names = ["error decreased (better)", "error increased (worse)",
             "faded = within seed noise", "±1.96 SE seed-noise band (approx. 95%)"]
    fig.legend(handles, names, loc="lower center", ncol=4, frameon=False, fontsize=9.5,
               labelcolor=INK_2, bbox_to_anchor=(0.5, 0.03))
    fig.suptitle("Matched-pair comparisons: one factor switched, everything else identical",
                 x=0.01, ha="left", fontsize=13.5, color=INK)
    fig.text(0.01, 0.005,
             "Change = (mean of 'to' − mean of 'from') / mean of 'from', averaged over matched configurations. "
             "Grouped by factor, then by largest absolute change. cw_ = channelwise_, res_ = residual_. "
             "'also differs' lists other settings (not compared here) that differ between the two run sets.",
             fontsize=8.3, color=INK_2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Pooled / per-group fANOVA
# ---------------------------------------------------------------------------

def _distribution(optuna, df, name):
    kind = factor_kind(name)
    values = df[name]
    if kind == "int":
        return optuna.distributions.IntDistribution(int(values.min()), int(values.max()))
    if kind in ("float", "logfloat"):
        return optuna.distributions.FloatDistribution(
            float(values.min()), float(values.max()), log=(kind == "logfloat"))
    return optuna.distributions.CategoricalDistribution(sorted(values.astype(str).unique()))


def fanova_importance(df, factors, metric, repeats, n_trees=64, log_target=False):
    """Mean/std importance over `repeats` forest seeds. Returns DataFrame [factor, mean, std].
    log_target: decompose the variance of log(error), i.e. relative instead of absolute effects."""
    import optuna
    from optuna.importance import FanovaImportanceEvaluator, get_param_importances

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dists = {f: _distribution(optuna, df, f) for f in factors}
    study = optuna.create_study(direction="minimize")
    for _, row in df.iterrows():
        params = {}
        for f in factors:
            v = row[f]
            params[f] = str(v) if factor_kind(f) == "cat" else (int(v) if factor_kind(f) == "int" else float(v))
        value = float(np.log(row[metric])) if log_target else float(row[metric])
        study.add_trial(optuna.trial.create_trial(params=params, distributions=dists, value=value))
    draws = []
    for seed in range(repeats):
        evaluator = FanovaImportanceEvaluator(n_trees=n_trees, seed=seed)
        draws.append(get_param_importances(study, evaluator=evaluator, params=factors, normalize=True))
    frame = pd.DataFrame(draws)
    return pd.DataFrame({"factor": frame.columns, "mean": frame.mean().to_numpy(),
                         "std": frame.std(ddof=1).to_numpy()})


def run_fanova(df, repeats, min_runs, min_level_runs, min_method_runs, log_target=False):
    """Pooled (fusion_group as factor), per-group and per-fusion-method importances, per metric."""
    results = []
    scopes = [("pooled", df, varying_factors(df, ["fusion_group"] + FANOVA_FACTORS, min_level_runs))]
    for g in GROUPS:
        sub = df[df.fusion_group == g]
        scopes.append((g, sub, varying_factors(sub, ["fusion_type"] + FANOVA_FACTORS,
                                               min_level_runs)))
    for method, sub in df.groupby("method"):
        if len(sub) >= min_method_runs:
            scopes.append((f"method:{method}", sub,
                           varying_factors(sub, FANOVA_FACTORS, min_level_runs)))
    for scope, sub, factors in scopes:
        min_needed = min_method_runs if scope.startswith("method:") else min_runs
        if len(sub) < min_needed or len(factors) < 2:
            logger.warning("fANOVA %s skipped: %d runs, %d varying factors", scope, len(sub), len(factors))
            continue
        for metric in METRICS:
            imp = fanova_importance(sub, factors, metric, repeats, log_target=log_target)
            imp.insert(0, "n_runs", len(sub))
            imp.insert(0, "metric", metric)
            imp.insert(0, "scope", scope)
            results.append(imp)
    return pd.concat(results, ignore_index=True)


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, labelsize=9.5, colors=INK_2)


def plot_single(rows, path, title, color, note):
    """Importance bars (translation | rotation) for one fANOVA scope."""
    if rows.empty:
        return
    order = rows.groupby("factor")["mean"].mean().sort_values(ascending=False).index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.4 * len(order) + 2.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(order))
    for ax, metric in zip(axes, METRICS):
        sub = rows[rows.metric == metric].set_index("factor").reindex(order)
        _style(ax)
        ax.barh(y, sub["mean"], height=0.55, color=color, zorder=3,
                xerr=sub["std"], error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 2})
        top = (sub["mean"] + sub["std"].fillna(0)).max()
        for yi, m, sd in zip(y, sub["mean"], sub["std"].fillna(0)):
            ax.text(m + sd + 0.012 * top * 2, yi, f"{m:.3f}", va="center", fontsize=9, color=INK_2)
        ax.set_xlim(0, top * 1.18)
        ax.set_title(f"{METRICS[metric][1]} ({METRICS[metric][3]})", loc="left", fontsize=12, color=INK)
        ax.set_xlabel("Hyperparameter importance [share of variance]", fontsize=9.5, color=INK_2)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order)
    axes[0].invert_yaxis()
    fig.suptitle(title, x=0.01, ha="left", fontsize=13.5, color=INK)
    fig.text(0.01, 0.005, note, fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


def plot_pooled(imp, path, repeats):
    pooled = imp[imp.scope == "pooled"]
    if pooled.empty:
        return
    plot_single(
        pooled, path,
        f"Exploratory fANOVA over all groups pooled (n={int(pooled['n_runs'].iloc[0])} runs, "
        f"mean of {repeats} forest seeds)", POOLED_COLOR,
        "Runs come from one-factor ablations, not a joint search: importance partly reflects which "
        "sweeps exist.\nBars are mean \u00b1 sd over forest seeds; sorted by the mean of both metrics. "
        "Only factors changed in at least two settings with >= 3 runs each are shown.")


def plot_methods(imp, output_dir, repeats):
    """One plot per fusion method (dual fusion appears as dual_<fusion>)."""
    written = []
    for scope in imp.scope.unique():
        if not scope.startswith("method:"):
            continue
        method = scope[len("method:"):]
        rows = imp[imp.scope == scope]
        group = _fusion_group(method.replace("dual_", "", 1), method.startswith("dual_"))
        path = output_dir / f"fanova_method_{method}.png"
        plot_single(
            rows, path,
            f"Exploratory fANOVA: {method} (n={int(rows['n_runs'].iloc[0])} runs, "
            f"mean of {repeats} forest seeds)", GROUP_COLORS[group],
            "Only factors changed in at least two settings with >= 3 runs each are shown. Bars are "
            "mean \u00b1 sd over forest seeds; sorted by the mean of both metrics.")
        written.append(path.name)
    return written


def plot_by_group(imp, path, repeats):
    per_group = imp[imp.scope.isin(GROUPS)]
    if per_group.empty:
        return
    order = per_group.groupby("factor")["mean"].max().sort_values(ascending=False).index.tolist()
    present = [g for g in GROUPS if g in per_group.scope.unique()]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 0.62 * len(order) + 2.4), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(order))
    h = 0.24
    for ax, metric in zip(axes, METRICS):
        _style(ax)
        for i, g in enumerate(present):
            sub = per_group[(per_group.scope == g) & (per_group.metric == metric)].set_index("factor")
            sub = sub.reindex(order)
            offset = (i - (len(present) - 1) / 2) * (h + 0.03)
            mask = sub["mean"].notna().to_numpy()  # absent = factor not varied in this group
            ax.barh((y + offset)[mask], sub["mean"].to_numpy()[mask], height=h,
                    color=GROUP_COLORS[g], zorder=3, xerr=sub["std"].to_numpy()[mask],
                    error_kw={"ecolor": INK, "elinewidth": 0.8, "capsize": 1.5})
        ax.set_title(f"{METRICS[metric][1]} ({METRICS[metric][3]})", loc="left", fontsize=12, color=INK)
        ax.set_xlabel("Hyperparameter importance [share of variance]", fontsize=9.5, color=INK_2)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order)
    axes[0].set_ylim(len(order) - 0.5, -0.5)
    ns = per_group.groupby("scope")["n_runs"].first()
    names = {"scalar": "scalar", "channelwise": "channelwise", "dual": "dual fusion"}
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in present]
    fig.legend(handles, [f"{names[g]} (n={int(ns[g])})" for g in present], loc="lower center",
               ncol=len(present), frameon=False, fontsize=10, labelcolor=INK_2,
               bbox_to_anchor=(0.5, 0.025))
    fig.suptitle(f"Exploratory fANOVA per fusion group (mean of {repeats} forest seeds)",
                 x=0.01, ha="left", fontsize=13.5, color=INK)
    fig.text(0.01, 0.005, "A missing bar means the factor did not vary within that group. "
             "Sorted by the largest importance in any group.",
             fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


# Categorical slots 1-7 of the reference palette (light mode), assigned in fixed order.
METHOD_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]


def plot_by_method(imp, path, repeats):
    """Grouped bars: one bar per fusion method for every parameter (all methods in one figure)."""
    scopes = imp[imp.scope.str.startswith("method:")]
    if scopes.empty:
        return
    ns = scopes.groupby("scope")["n_runs"].first().sort_values(ascending=False)
    methods = [sc for sc in ns.index][: len(METHOD_COLORS)]
    order = (scopes[scopes.scope.isin(methods)].groupby("factor")["mean"].max()
             .sort_values(ascending=False).index.tolist())
    fig, axes = plt.subplots(1, 2, figsize=(14, 1.0 * len(order) + 2.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(order))
    h = 0.105
    for ax, metric in zip(axes, METRICS):
        _style(ax)
        for i, sc in enumerate(methods):
            sub = scopes[(scopes.scope == sc) & (scopes.metric == metric)].set_index("factor")
            sub = sub.reindex(order)
            offset = (i - (len(methods) - 1) / 2) * (h + 0.012)
            mask = sub["mean"].notna().to_numpy()  # absent = factor not varied for this method
            ax.barh((y + offset)[mask], sub["mean"].to_numpy()[mask], height=h,
                    color=METHOD_COLORS[i], zorder=3, xerr=sub["std"].to_numpy()[mask],
                    error_kw={"ecolor": INK, "elinewidth": 0.6, "capsize": 1})
        ax.set_title(f"{METRICS[metric][1]} ({METRICS[metric][3]})", loc="left", fontsize=12, color=INK)
        ax.set_xlabel("Hyperparameter importance [share of variance]", fontsize=9.5, color=INK_2)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order)
    axes[0].set_ylim(len(order) - 0.5, -0.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS[i]) for i in range(len(methods))]
    labels = [f"{_short(sc[len('method:'):])} (n={int(ns[sc])})" for sc in methods]
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9.5,
               labelcolor=INK_2, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle(f"Exploratory fANOVA per fusion method (mean of {repeats} forest seeds)",
                 x=0.01, ha="left", fontsize=13.5, color=INK)
    fig.text(0.01, 0.005, "A missing bar means the factor did not vary for that method. "
             "Sorted by the largest importance for any method. cw_ = channelwise_, res_ = residual_.",
             fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(path, dpi=180, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--since", default=(dt.date.today() - dt.timedelta(days=30)).isoformat(),
                        help="Only runs started on/after this date, YYYY-MM-DD (default: 30 days ago)")
    parser.add_argument("--output", type=Path, default=Path("results/hyperparameter_importance"))
    parser.add_argument("--mode", choices=["paired", "fanova", "both"], default="both")
    parser.add_argument("--repeats", type=int, default=10, help="fANOVA forest seeds to average")
    parser.add_argument("--collapse-rot", type=float, default=2.0,
                        help="Drop runs whose best val_abs_rot_x exceeds this (deg)")
    parser.add_argument("--collapse-trans", type=float, default=2.0,
                        help="Drop runs whose best val_abs_trans exceeds this (mm)")
    parser.add_argument("--min-cell-runs", type=int, default=1,
                        help="Paired comparison: minimum runs per side of a matched configuration "
                             "(all comparisons go to the CSV; thin ones are faded in the plot)")
    parser.add_argument("--plot-min-n", type=int, default=2,
                        help="Paired comparison plot: only rows with at least this many runs on "
                             "each side (summed over matched configurations)")
    parser.add_argument("--max-rows", type=int, default=40,
                        help="Paired comparison plot: show at most this many comparisons")
    parser.add_argument("--min-level-runs", type=int, default=3,
                        help="A factor counts as changed (and is shown/analysed in fANOVA) only if "
                             "at least two of its levels have this many runs")
    parser.add_argument("--min-runs", type=int, default=4,
                        help="Minimum runs per fusion group for fANOVA")
    parser.add_argument("--min-method-runs", type=int, default=12,
                        help="Minimum runs for a fusion method to get its own fANOVA plot")
    parser.add_argument("--log-target", action="store_true",
                        help="fANOVA on log(error): relative instead of absolute effects")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.output.mkdir(parents=True, exist_ok=True)
    runs, dropped = filter_runs(load_runs(args.runs_dir, args.since), args.collapse_rot,
                                args.collapse_trans)
    runs.to_csv(args.output / "runs_used.csv", index=False)
    dropped.to_csv(args.output / "runs_dropped_collapsed.csv", index=False)
    logger.info("%d runs since %s; per group: %s", len(runs), args.since,
                runs.fusion_group.value_counts().to_dict())

    if args.mode in ("paired", "both"):
        noise = seed_noise(runs)
        logger.info("seed noise CV (%%): %s",
                    {k: {m: round(v, 1) for m, v in d.items()} for k, d in noise.items()})
        table = paired_comparisons(runs, args.min_cell_runs, noise)
        table.to_csv(args.output / "paired_comparisons.csv", index=False)
        logger.info("%d matched-pair comparisons", len(table))
        shown = table[table[["n_from", "n_to"]].min(axis=1) >= args.plot_min_n]
        logger.info("%d of them have >= %d runs per side and are plotted", len(shown), args.plot_min_n)
        if not shown.empty:
            plot_paired(shown, args.output / "paired_comparisons.png", args.max_rows)

    if args.mode in ("fanova", "both"):
        imp = run_fanova(runs, args.repeats, args.min_runs, args.min_level_runs,
                         args.min_method_runs, args.log_target)
        imp.to_csv(args.output / "fanova_importance.csv", index=False)
        plot_pooled(imp, args.output / "fanova_pooled.png", args.repeats)
        plot_by_group(imp, args.output / "fanova_by_group.png", args.repeats)
        plot_by_method(imp, args.output / "fanova_by_method.png", args.repeats)
        logger.info("per-method plots: %s", plot_methods(imp, args.output, args.repeats))

    logger.info("wrote results to %s", args.output)


if __name__ == "__main__":
    main()
