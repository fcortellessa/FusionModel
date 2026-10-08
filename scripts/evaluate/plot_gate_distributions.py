#!/usr/bin/env python3
"""
Plot the distribution of the fusion gates of trained runs, and (dual_fusion) compare the
rotation-branch gates with the translation-branch gates.

For every run the best checkpoint is evaluated on a split and the raw per-sample gate values
(`model.last_fusion_weights`) are collected. Per run one figure is written with, per branch
(a single shared fusion, or rotation + translation for dual_fusion):

  A  all gate values         share of all gate values per bin (log y): is the gate saturated
                             (piled up below 0.1 / above 0.9) or in the interior?
  B  per-sample mean gate    variation ACROSS SAMPLES: a spike means the gate is one constant.
  C  per-channel mean gate   variation ACROSS CHANNELS (channel-wise gates only).

The legend of each panel carries the numbers: mu (mean), sigma (std of all values), the shares
below 0.1 / above 0.9. Dual-fusion runs get a second figure, `<run>_rot_vs_trans`, per stream:
overlaid distributions, per-sample and per-channel scatter of rotation gate vs translation gate
(y = x means identical), the Pearson r, the mean |g_rot - g_trans| over matched gates and the
Wasserstein-1 distance between the two distributions. An overview figure pools all runs by
fusion. A CSV with all numbers is written next to the figures.

Residual fusions apply (1 + gate) * feature: the plotted value is the gate w in (0, 1), not the
multiplier. Fusions without gate weights (concat) are skipped.

Usage:
    python scripts/evaluate/plot_gate_distributions.py --sweeps runs/final_ablation
    python scripts/evaluate/plot_gate_distributions.py --runs runs/final_ablation/<run_dir> --split test
    python scripts/evaluate/plot_gate_distributions.py --runs <run_dir> --max_batches 3   # smoke test

Data: the split of each run is defined by its data.split_seed, but Data/Data_split holds only ONE
split at a time. By default a run whose split is not the one currently on disk is SKIPPED (the
message names the seed). Pass --rebuild_split to rebuild Data/Data_split per run like
scripts/test.py does -- never while a training / evaluation job is reading that folder.
"""

import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import infer_split_gpu as I  # noqa: E402  (same folder; provides REPO, build_model)

sys.path.insert(0, I.REPO)
from data_processing.dataloader_preprocessed import PreprocessedTactileDataset, tactile_collate_fn  # noqa: E402
from data_processing.GripFusionNet_split import split_manifest_matches  # noqa: E402
from data_processing.preprocess_data import ensure_test_split_current  # noqa: E402
from utils.utils import (  # noqa: E402
    get_optimizer,
    load_checkpoint,
    load_config,
    resolve_data_root,
    resolve_group_aware_split,
    resolve_split_seed,
)

BINS = 100
EDGES = np.linspace(0.0, 1.0, BINS + 1)
CENTERS = 0.5 * (EDGES[:-1] + EDGES[1:])
BIN_W = 1.0 / BINS

# Categorical slots 1-3 of the reference palette (blue, orange, aqua): the first three are the
# documented all-pairs-safe set. One colour per STREAM in every figure.
STREAM_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#dcdbd6", "#fcfcfb"
STREAM_LABELS = {"rgb_image": "RGB", "digit_image": "Tactile"}
BRANCH_LABELS = {"shared": "Shared fusion", "rotation": "Rotation branch", "translation": "Translation branch"}


# --------------------------------------------------------------------------- accumulation
class GateAccumulator:
    """Streaming statistics of gate weights shaped (B, streams, D) over a whole split."""

    def __init__(self):
        self.hist = None
        self.n = 0
        self.out_of_range = 0
        self.sample_means = []

    def update(self, w):
        w = w.detach()
        B, S, D = w.shape
        if self.hist is None:
            self.S, self.D = S, D
            self.hist = torch.zeros(S, BINS, dtype=torch.float64, device=w.device)
            self.csum = torch.zeros(S, D, dtype=torch.float64, device=w.device)
            self.csumsq = torch.zeros(S, D, dtype=torch.float64, device=w.device)
        self.out_of_range += int(((w < 0) | (w > 1)).sum().item())
        wc = w.float().clamp(0.0, 1.0)
        for s in range(S):
            self.hist[s] += torch.histc(wc[:, s].flatten(), bins=BINS, min=0.0, max=1.0).double()
        wd = wc.double()
        self.csum += wd.sum(dim=0)
        self.csumsq += (wd ** 2).sum(dim=0)
        self.sample_means.append(wd.mean(dim=2).cpu())  # (B, S)
        self.n += B

    def finalize(self):
        hist = self.hist.cpu().numpy()
        total = hist.sum(axis=1, keepdims=True)
        share = hist / total
        csum, csumsq = self.csum.cpu().numpy(), self.csumsq.cpu().numpy()
        cmean = csum / self.n                                        # (S, D) mean over samples
        cvar = np.clip(csumsq / self.n - cmean ** 2, 0.0, None)       # (S, D) var over samples
        mean = cmean.mean(axis=1)
        var = np.clip(csumsq.sum(axis=1) / (self.n * self.D) - mean ** 2, 0.0, None)
        return dict(
            share=share, n_samples=self.n, D=self.D, S=self.S, mean=mean, std=np.sqrt(var),
            frac_lt01=share[:, :10].sum(axis=1), frac_gt09=share[:, 90:].sum(axis=1),
            frac_mid=share[:, 40:60].sum(axis=1),
            between_sample_std=np.sqrt(cvar).mean(axis=1),           # std over samples, mean over gates
            between_channel_std=cmean.std(axis=1),                   # std over channels of the per-channel mean
            channel_means=cmean, sample_means=torch.cat(self.sample_means).numpy(),
            out_of_range=self.out_of_range,
        )


def w1_distance(share_a, share_b):
    return float(np.abs(np.cumsum(share_a) - np.cumsum(share_b)).sum() * BIN_W)


def safe_corr(a, b):
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def collect_gates(model, loader, device, streams, max_batches=None):
    """Runs the split through the model; returns {branch: stats} and dual-pair statistics."""
    accs, absdiff, absdiff_n = {}, None, 0
    with torch.no_grad():
        for i, sample in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            inputs = {k: v.to(device) for k, v in sample.items() if k != "pose"}
            model(inputs)
            lfw = model.last_fusion_weights
            branches = lfw if isinstance(lfw, dict) else {"shared": lfw}
            prepared = {}
            for name, w in branches.items():
                if w is None:
                    continue
                w = w.reshape(w.shape[0], w.shape[1], -1)
                if w.shape[1] != len(streams):
                    raise RuntimeError(
                        f"gate weights of branch '{name}' have shape {tuple(w.shape)}; expected dim 1 = "
                        f"{len(streams)} streams -- this fusion's weights are not per-stream gates")
                prepared[name] = w
                accs.setdefault(name, GateAccumulator()).update(w)
            if "rotation" in prepared and "translation" in prepared:
                d = (prepared["rotation"].double() - prepared["translation"].double()).abs()
                s = d.sum(dim=(0, 2)).cpu().numpy()
                absdiff = s if absdiff is None else absdiff + s
                absdiff_n += d.shape[0] * d.shape[2]
    if not accs:
        return None, None
    stats = {name: acc.finalize() for name, acc in accs.items()}
    pair = None
    if "rotation" in stats and "translation" in stats:
        r, t = stats["rotation"], stats["translation"]
        pair = dict(
            mean_abs_diff=absdiff / absdiff_n,
            w1=np.array([w1_distance(r["share"][s], t["share"][s]) for s in range(r["S"])]),
            corr_sample=np.array([safe_corr(r["sample_means"][:, s], t["sample_means"][:, s]) for s in range(r["S"])]),
            corr_channel=np.array([safe_corr(r["channel_means"][s], t["channel_means"][s]) if r["D"] > 1 else np.nan
                                   for s in range(r["S"])]),
        )
    return stats, pair


# --------------------------------------------------------------------------- plotting
def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=8, length=3)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def stream_names(streams):
    return [STREAM_LABELS.get(s, s) for s in streams]


def stat_label(name, st, s):
    return (f"{name}  μ={st['mean'][s]:.3f}  σ={st['std'][s]:.3f}  "
            f"<0.1: {100 * st['frac_lt01'][s]:.0f}%  >0.9: {100 * st['frac_gt09'][s]:.0f}%")


def draw_hist(ax, values, color, label, log=False, ls="-"):
    v = np.asarray(values, dtype=float)
    if log:
        v = np.where(v > 0, v, np.nan)
    ax.stairs(v, EDGES, color=color, linewidth=1.6, linestyle=ls, label=label)


def finish_hist_axes(ax, title, ylabel, log, all_shares=None):
    ax.set_xlim(0, 1)
    ax.set_xlabel("gate value")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9, loc="left")
    if log:
        ax.set_yscale("log")
        pos = np.concatenate([a[a > 0] for a in all_shares]) if all_shares else np.array([1e-4])
        ax.set_ylim(max(min(pos.min() / 3.0, 1e-3), 1e-9) if pos.size else 1e-6, 1.5)
        ax.axvspan(0, 0.1, color=GRID, alpha=0.5, lw=0)
        ax.axvspan(0.9, 1.0, color=GRID, alpha=0.5, lw=0)
    style_axes(ax)
    leg = ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), labelcolor=INK_2)
    return leg


def plot_branch(axs, st, streams, prefix):
    names = stream_names(streams)
    S, D = st["S"], st["D"]
    shares = [st["share"][s] for s in range(S)]
    for s in range(S):
        draw_hist(axs[0], shares[s], STREAM_COLORS[s], stat_label(names[s], st, s), log=True)
    finish_hist_axes(axs[0], f"{prefix}A  all gate values", "share of gate values (log)", True, shares)

    for s in range(S):
        h, _ = np.histogram(st["sample_means"][:, s], bins=EDGES)
        sm = st["sample_means"][:, s]
        lab = f"{names[s]}  μ={sm.mean():.3f}  σ across samples={sm.std():.3f}"
        draw_hist(axs[1], h / max(h.sum(), 1), STREAM_COLORS[s], lab)
    finish_hist_axes(axs[1], f"{prefix}B  per-sample mean gate", "share of samples", False)

    if D > 1 and len(axs) > 2:
        for s in range(S):
            h, _ = np.histogram(st["channel_means"][s], bins=EDGES)
            lab = (f"{names[s]}  σ across channels={st['between_channel_std'][s]:.3f}  "
                   f"(D={D})")
            draw_hist(axs[2], h / max(h.sum(), 1), STREAM_COLORS[s], lab)
        finish_hist_axes(axs[2], f"{prefix}C  per-channel mean gate", "share of channels", False)


def plot_run(stats, streams, title, out_path, dpi):
    branches = list(stats)
    D = stats[branches[0]]["D"]
    ncols = 3 if D > 1 else 2
    fig, axes = plt.subplots(len(branches), ncols, figsize=(5.0 * ncols, 4.3 * len(branches)),
                             squeeze=False, facecolor=SURFACE, constrained_layout=True)
    for row, name in enumerate(branches):
        prefix = "" if name == "shared" else f"{BRANCH_LABELS[name]} — "
        plot_branch(axes[row], stats[name], streams, prefix)
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def plot_rot_vs_trans(stats, pair, streams, title, out_path, dpi):
    r, t = stats["rotation"], stats["translation"]
    S, D = r["S"], r["D"]
    names = stream_names(streams)
    ncols = 3 if D > 1 else 2
    fig, axes = plt.subplots(S, ncols, figsize=(5.0 * ncols, 4.3 * S), squeeze=False,
                             facecolor=SURFACE, constrained_layout=True)
    rng = np.random.default_rng(0)
    for s in range(S):
        c = STREAM_COLORS[s]
        ax = axes[s, 0]
        draw_hist(ax, r["share"][s], c, "Rotation  " + stat_label("", r, s).strip(), log=True, ls="-")
        draw_hist(ax, t["share"][s], c, "Translation  " + stat_label("", t, s).strip(), log=True, ls="--")
        finish_hist_axes(ax, f"{names[s]} — all gate values  (solid = rotation, dashed = translation)\n"
                             f"mean |g_rot − g_trans| = {pair['mean_abs_diff'][s]:.4f}   "
                             f"Wasserstein-1 = {pair['w1'][s]:.4f}",
                         "share of gate values (log)", True, [r["share"][s], t["share"][s]])

        ax = axes[s, 1]
        x, y = r["sample_means"][:, s], t["sample_means"][:, s]
        ax.scatter(x, y, s=10, color=c, alpha=0.6, linewidths=0)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        pad = max(0.02, 0.1 * (hi - lo))
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK_2, linewidth=0.8, linestyle=":")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("rotation-branch mean gate (per sample)")
        ax.set_ylabel("translation-branch mean gate")
        ax.set_title(f"{names[s]} — per-sample mean   r={pair['corr_sample'][s]:.2f}", fontsize=9, loc="left")
        style_axes(ax)

        if D > 1:
            ax = axes[s, 2]
            idx = rng.choice(D, size=min(D, 5000), replace=False)
            x, y = r["channel_means"][s][idx], t["channel_means"][s][idx]
            ax.scatter(x, y, s=4, color=c, alpha=0.35, linewidths=0)
            lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
            pad = max(0.02, 0.1 * (hi - lo))
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=INK_2, linewidth=0.8, linestyle=":")
            ax.set_xlim(lo - pad, hi + pad)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("rotation-branch mean gate (per channel)")
            ax.set_ylabel("translation-branch mean gate")
            ax.set_title(f"{names[s]} — per-channel mean   r={pair['corr_channel'][s]:.2f}", fontsize=9, loc="left")
            style_axes(ax)

    fig.suptitle(title + "  —  rotation vs translation gates", fontsize=10, color=INK, x=0.01, ha="left")
    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def plot_overview(groups, streams_by_group, out_path, dpi):
    """groups: {label: {'share': (S,BINS) summed, 'n': runs, 'mean': (S,) sum of run means}}"""
    labels = list(groups)
    ncols = 2 if len(labels) > 3 else 1
    nrows = int(np.ceil(len(labels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.6 * nrows), squeeze=False,
                             facecolor=SURFACE, constrained_layout=True)
    for ax in axes.flat[len(labels):]:
        ax.axis("off")
    for ax, label in zip(axes.flat, labels):
        g = groups[label]
        names = stream_names(streams_by_group[label])
        shares = [g["share"][s] / g["share"][s].sum() for s in range(len(names))]
        for s, name in enumerate(names):
            lab = (f"{name}  μ={g['mean'][s] / g['n']:.3f}  <0.1: {100 * shares[s][:10].sum():.0f}%  "
                   f">0.9: {100 * shares[s][90:].sum():.0f}%")
            draw_hist(ax, shares[s], STREAM_COLORS[s], lab, log=True)
        finish_hist_axes(ax, f"{label}  ({g['n']} run{'s' if g['n'] != 1 else ''} pooled)",
                         "share of gate values (log)", True, shares)
    fig.suptitle("Gate value distributions pooled by fusion", fontsize=10, color=INK, x=0.01, ha="left")
    fig.savefig(out_path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- per-run driver
def prepare_split(cfg, data_root, split, rebuild):
    split_root = os.path.join(data_root, "Data_split")
    seed = resolve_split_seed(cfg)
    matches = split_manifest_matches(data_root, split_root, seed=seed, group_aware=resolve_group_aware_split(cfg))
    folder = os.path.join(split_root, split, f"{split}_preprocessing")
    if matches and os.path.isdir(folder):
        return folder
    if not rebuild:
        return None
    ensure_test_split_current(cfg, data_root)
    return folder if os.path.isdir(folder) else None


def run_one(run_dir, args, device):
    cfg = load_config(os.path.join(run_dir, "config.yml"))
    fusion = cfg["model"]["fusion"]
    dual = bool(cfg["model"].get("dual_fusion", False))
    seed = resolve_split_seed(cfg)
    name = os.path.basename(run_dir)
    if fusion == "concat":
        print(f"skip (no gate): {name}")
        return None
    data_root = resolve_data_root(cfg, I.REPO)
    folder = prepare_split(cfg, data_root, args.split, args.rebuild_split)
    if folder is None:
        print(f"skip (split seed {seed} is not the split on disk; use --rebuild_split): {name}")
        return None

    loader = DataLoader(PreprocessedTactileDataset(folder), batch_size=16, shuffle=False, num_workers=2,
                        collate_fn=tactile_collate_fn)
    model = I.build_model(cfg)
    opt = get_optimizer(model_or_params=model, optimizer_name=cfg["training"]["optimizer"], lr=cfg["training"]["lr"])
    load_checkpoint(model, opt, os.path.join(run_dir, "checkpoint_best.pth"), device="cpu", load_optimizer=False)
    model.to(device).eval()

    streams = cfg["model"]["streams"]
    stats, pair = collect_gates(model, loader, device, streams, args.max_batches)
    if stats is None:
        print(f"skip (model returned no gate weights): {name}")
        return None
    for br, st in stats.items():
        if st["out_of_range"]:
            print(f"  warning: {st['out_of_range']} gate values of '{br}' outside [0, 1] were clamped for the plots")

    label = fusion + (" +dual" if dual else "")
    n = next(iter(stats.values()))["n_samples"]
    title = f"{label}  |  split seed {seed}  |  {args.split} (n={n})  |  {name}"
    if "residual" in fusion:
        title += "  |  plotted: gate w, multiplier = 1 + w"
    os.makedirs(args.out_dir, exist_ok=True)
    plot_run(stats, streams, title, os.path.join(args.out_dir, f"{name}.png"), args.dpi)
    if pair is not None:
        plot_rot_vs_trans(stats, pair, streams, title,
                          os.path.join(args.out_dir, f"{name}_rot_vs_trans.png"), args.dpi)

    rows = []
    for br, st in stats.items():
        for s, sname in enumerate(streams):
            row = dict(run=name, fusion=label, split_seed=seed, split=args.split, n_samples=st["n_samples"],
                       branch=br, stream=sname, n_channels=st["D"], mean=st["mean"][s], std=st["std"][s],
                       frac_lt_0_1=st["frac_lt01"][s], frac_gt_0_9=st["frac_gt09"][s],
                       frac_mid_0_4_0_6=st["frac_mid"][s], between_sample_std=st["between_sample_std"][s],
                       between_channel_std=st["between_channel_std"][s])
            if pair is not None:
                row.update(mean_abs_diff_rot_trans=pair["mean_abs_diff"][s], w1_rot_trans=pair["w1"][s],
                           corr_sample_means=pair["corr_sample"][s], corr_channel_means=pair["corr_channel"][s])
            rows.append(row)
    print(f"done: {label:42s} seed {seed}  -> {name}.png" + ("  (+ rot_vs_trans)" if pair is not None else ""))
    return dict(rows=rows, stats=stats, label=label, streams=streams)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", default=[], help="run directories")
    ap.add_argument("--sweeps", nargs="*", default=[], help="folders whose 2026* sub-directories are runs")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--out_dir", default=os.path.join(I.REPO, "results", "gate_distributions"))
    ap.add_argument("--rebuild_split", action="store_true",
                    help="rebuild Data/Data_split per run's split seed (see module docstring)")
    ap.add_argument("--max_batches", type=int, default=None, help="only use the first N batches (smoke test)")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    runs = [os.path.abspath(r) for r in args.runs]
    for s in args.sweeps:
        runs += [os.path.abspath(d) for d in sorted(glob.glob(os.path.join(s, "2026*")))]
    runs = [r for r in runs if os.path.exists(os.path.join(r, "checkpoint_best.pth"))]
    runs.sort(key=lambda r: str(resolve_split_seed(load_config(os.path.join(r, "config.yml")))))
    if not runs:
        sys.exit("no runs with a checkpoint_best.pth found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows, groups, streams_by_group = [], {}, {}
    for i, r in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] ", end="", flush=True)
        res = run_one(r, args, device)
        if res is None:
            continue
        all_rows += res["rows"]
        for br, st in res["stats"].items():
            key = res["label"] + ("" if br == "shared" else f" ({br})")
            g = groups.setdefault(key, dict(share=np.zeros_like(st["share"]), n=0, mean=np.zeros(st["S"])))
            g["share"] += st["share"]
            g["n"] += 1
            g["mean"] += st["mean"]
            streams_by_group[key] = res["streams"]
        pd.DataFrame(all_rows).to_csv(os.path.join(args.out_dir, f"gate_distribution_stats_{args.split}.csv"),
                                      index=False)
    if groups:
        plot_overview(groups, streams_by_group, os.path.join(args.out_dir, f"overview_{args.split}.png"), args.dpi)
        print(f"wrote overview + CSV to {args.out_dir}")


if __name__ == "__main__":
    main()
