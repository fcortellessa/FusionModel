"""Inventory every run under ``runs/`` and flag what is probably safe to delete.

Read-only: nothing is deleted or moved. Writes three files to ``--out-dir``:

* ``runs_inventory.csv``   one row per run (size, status, best metrics, tier, reason)
* ``cleanup_overview.md``  human-readable overview grouped by sweep folder and tier
* ``cleanup_<tier>.txt``   plain path lists per tier (one run dir per line)

Tiers (most to least certain):

* ``A_broken``       no metrics / no checkpoint / no results: nothing worth keeping
* ``B_incomplete``   training did not finish (crashed/killed) and was never evaluated
* ``C_duplicate``    identical config (incl. seed) re-run later; older copy is redundant
* ``D_ckpt_only``    finished + test results exist: the checkpoint (usually >90% of the
                     run's size) can go while metrics/results/config are kept
* ``R_recent``       would be tier A/B but was modified in the last ``RECENT_DAYS`` days
                     (possibly a sweep you still want to resume) - review by hand
* ``keep``           everything else

Usage (from repo root)::

    python scripts/evaluate/runs_inventory.py [--runs-dir runs] [--out-dir runs/_cleanup_overview]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

IGNORED_CONFIG_KEYS = {"root", "run_dir", "run_root_dir", "config_path", "output_dir", "results_dir"}
TIERS = ["A_broken", "B_incomplete", "C_duplicate", "D_ckpt_only", "R_recent"]
RECENT_DAYS = 3


def dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} PB"


def strip_keys(obj):
    if isinstance(obj, dict):
        return {k: strip_keys(v) for k, v in obj.items() if k not in IGNORED_CONFIG_KEYS}
    if isinstance(obj, list):
        return [strip_keys(v) for v in obj]
    return obj


def config_hash(cfg: dict) -> str:
    blob = json.dumps(strip_keys(cfg), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def read_metrics(path: Path):
    """Return (n_epochs, best_val_loss, best_rot_deg, best_trans_m) from metrics.csv."""
    if not path.is_file():
        return 0, None, None, None
    n, best_loss, best_rot, best_trans = 0, None, None, None

    def upd(cur, val):
        return val if cur is None or val < cur else cur

    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            n += 1
            for key, name in (("val_loss", "loss"), ("val_abs_rot_x", "rot"), ("val_abs_trans", "trans")):
                try:
                    v = float(row.get(key, ""))
                except (TypeError, ValueError):
                    continue
                if name == "loss":
                    best_loss = upd(best_loss, v)
                elif name == "rot":
                    best_rot = upd(best_rot, v)
                else:
                    best_trans = upd(best_trans, v)
    return n, best_loss, best_rot, best_trans


def newest_mtime(path: Path) -> float:
    newest = path.stat().st_mtime
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(root, f)))
            except OSError:
                pass
    return newest


def log_status(run: Path) -> tuple[bool, bool]:
    """(completed, early_stopped) from the newest train_*.log."""
    logs = sorted(run.glob("train_*.log"))
    if not logs:
        return False, False
    with open(logs[-1], errors="replace") as fh:
        text = fh.read()
    return "Training complete" in text, "Early stopping at epoch" in text


def inspect_run(run: Path, runs_dir: Path) -> dict:
    cfg = {}
    cfg_path = run / "config.yml"
    try:
        cfg = yaml.safe_load(cfg_path.read_text()) or {}
    except Exception:
        pass
    model, train, data = cfg.get("model", {}), cfg.get("training", {}), cfg.get("data", {})
    n_epochs, best_loss, best_rot, best_trans = read_metrics(run / "metrics.csv")
    completed, early = log_status(run)
    ckpt = run / "checkpoint_best.pth"
    ckpt_size = ckpt.stat().st_size if ckpt.is_file() else 0
    rel = run.relative_to(runs_dir)
    parts = rel.parts
    return {
        "path": str(rel),
        "group": "/".join(parts[:-1]) or ".",
        "name": parts[-1],
        "size": dir_size(run),
        "ckpt_size": ckpt_size,
        "mtime_ts": newest_mtime(run),
        "mtime": datetime.fromtimestamp(run.stat().st_mtime).strftime("%Y-%m-%d"),
        "fusion": model.get("fusion", "?"),
        "seed": train.get("seed", "?"),
        "split_seed": data.get("split_seed", "?"),
        "epochs_cfg": train.get("epochs", "?"),
        "epochs_done": n_epochs,
        "completed": completed,
        "early_stopped": early,
        "has_ckpt": ckpt_size > 0,
        "has_test": (run / "results").is_dir() and any((run / "results").iterdir()),
        "has_val_eval": (run / "results_val").is_dir(),
        "best_val_loss": best_loss,
        "best_val_rot_deg": best_rot,
        "best_val_trans_m": best_trans,
        "cfg_hash": config_hash(cfg) if cfg else "",
    }


def _as_int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 10**9  # unknown epoch budget: never count as "reached the end"


def classify(rows: list[dict]) -> None:
    for r in rows:
        r["tier"], r["reason"], r["reclaim"] = "keep", "", 0

    for r in rows:
        if r["epochs_done"] == 0 or (not r["has_ckpt"] and not r["has_test"]):
            r["tier"] = "A_broken"
            r["reason"] = f"{r['epochs_done']} epochs logged, no checkpoint/test results"
            r["reclaim"] = r["size"]
        elif not (r["completed"] or r["early_stopped"] or r["epochs_done"] >= _as_int(r["epochs_cfg"])) \
                and not r["has_test"]:
            r["tier"] = "B_incomplete"
            r["reason"] = f"no 'Training complete' in log ({r['epochs_done']}/{r['epochs_cfg']} epochs), never tested"
            r["reclaim"] = r["size"]

    cutoff = datetime.now().timestamp() - RECENT_DAYS * 86400
    for r in rows:
        if r["tier"] in ("A_broken", "B_incomplete") and r["mtime_ts"] > cutoff:
            r["reason"] = f"{r['tier']} but modified <{RECENT_DAYS} days ago: {r['reason']}"
            r["tier"], r["reclaim"] = "R_recent", 0

    # duplicates: same config hash + seed + split_seed, keep newest finished one
    by_hash = defaultdict(list)
    for r in rows:
        if r["tier"] == "keep" and r["cfg_hash"]:
            by_hash[r["cfg_hash"]].append(r)
    for same in by_hash.values():
        if len(same) < 2:
            continue
        same.sort(key=lambda x: (x["has_test"], x["has_ckpt"], x["completed"], x["name"]))
        newest = same[-1]
        for r in same[:-1]:
            r["tier"] = "C_duplicate"
            r["reason"] = f"same config+seed as {newest['path']}"
            r["reclaim"] = r["size"]

    for r in rows:
        if r["tier"] == "keep" and r["has_test"] and r["has_ckpt"]:
            r["tier"] = "D_ckpt_only"
            r["reason"] = "finished and tested; checkpoint only needed for re-evaluation/gate ablations"
            r["reclaim"] = r["ckpt_size"]


def fmt(v, spec=".3f"):
    return "" if v is None else format(v, spec)


def write_outputs(rows: list[dict], root_files: list[tuple[str, int]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = ["tier", "path", "size", "ckpt_size", "reclaim", "mtime", "fusion", "seed", "split_seed",
            "epochs_done", "epochs_cfg", "completed", "early_stopped", "has_ckpt", "has_test",
            "has_val_eval", "best_val_rot_deg", "best_val_trans_m", "best_val_loss", "reason"]
    with open(out_dir / "runs_inventory.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["tier"], x["path"])):
            w.writerow(r)

    for tier in TIERS:
        with open(out_dir / f"cleanup_{tier}.txt", "w") as fh:
            for r in sorted(rows, key=lambda x: x["path"]):
                if r["tier"] == tier:
                    fh.write(r["path"] + "\n")

    total = sum(r["size"] for r in rows)
    lines = [
        "# Runs cleanup overview",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/evaluate/runs_inventory.py` (read-only; nothing was deleted).",
        "",
        f"{len(rows)} runs, {human(total)} in run folders.",
        "",
        "## Summary by tier",
        "",
        "| Tier | Meaning | Runs | Reclaimable |",
        "|---|---|---:|---:|",
    ]
    meaning = {
        "A_broken": "no epochs/checkpoint/results - delete whole run dir",
        "B_incomplete": "training never finished, never tested - delete whole run dir",
        "C_duplicate": "same config+seed re-run later - delete older run dir",
        "D_ckpt_only": "finished + tested - delete only `checkpoint_best.pth`",
        "R_recent": f"unfinished but touched <{RECENT_DAYS} days ago - check whether you still want to resume",
    }
    for tier in TIERS:
        sel = [r for r in rows if r["tier"] == tier]
        lines.append(f"| {tier} | {meaning[tier]} | {len(sel)} | {human(sum(r['reclaim'] for r in sel))} |")
    lines += [
        "",
        "Tier D is the big win but is the only tier that costs you something: without the checkpoint you "
        "cannot re-run `test.py`, `gate_ablation_test.py` or `input_ablation_test.py` on that run. "
        "Keep checkpoints for runs you cite in the thesis/paper (e.g. the `final_ablation*` winners).",
        "",
        "## By sweep folder",
        "",
        "| Folder | Runs | Total | A | B | C | D reclaim | Keep |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    groups = defaultdict(list)
    for r in rows:
        groups[r["group"]].append(r)
    for g, rs in sorted(groups.items(), key=lambda kv: -sum(x["size"] for x in kv[1])):
        cnt = lambda t: sum(1 for x in rs if x["tier"] == t)  # noqa: E731
        d = sum(x["reclaim"] for x in rs if x["tier"] == "D_ckpt_only")
        lines.append(f"| `{g}` | {len(rs)} | {human(sum(x['size'] for x in rs))} | {cnt('A_broken')} | "
                     f"{cnt('B_incomplete')} | {cnt('C_duplicate')} | {human(d)} | {cnt('keep')} |")

    for tier in ("A_broken", "B_incomplete", "C_duplicate", "R_recent"):
        sel = sorted((r for r in rows if r["tier"] == tier), key=lambda x: -x["size"])
        lines += ["", f"## {tier} - whole run dirs ({len(sel)} runs, {human(sum(r['size'] for r in sel))})", ""]
        if not sel:
            lines.append("_none_")
            continue
        lines += ["| Run | Size | Epochs | Fusion | Seed | Reason |", "|---|---:|---:|---|---|---|"]
        for r in sel:
            lines.append(f"| `{r['path']}` | {human(r['size'])} | {r['epochs_done']}/{r['epochs_cfg']} | "
                         f"{r['fusion']} | {r['seed']} | {r['reason']} |")

    top_d = sorted((r for r in rows if r["tier"] == "D_ckpt_only"), key=lambda x: -x["ckpt_size"])[:25]
    lines += ["", "## D_ckpt_only - 25 largest checkpoints", "",
              "| Run | Checkpoint | Best val rot (deg) | Best val trans (m) |", "|---|---:|---:|---:|"]
    for r in top_d:
        lines.append(f"| `{r['path']}` | {human(r['ckpt_size'])} | {fmt(r['best_val_rot_deg'])} | "
                     f"{fmt(r['best_val_trans_m'], '.5f')} |")

    if root_files:
        lines += ["", "## Loose files directly in `runs/`", "", "| File | Size |", "|---|---:|"]
        lines += [f"| `{n}` | {human(s)} |" for n, s in sorted(root_files, key=lambda x: -x[1])]

    (out_dir / "cleanup_overview.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out-dir", default=None, help="default: <runs-dir>/_cleanup_overview")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    out_dir = Path(args.out_dir) if args.out_dir else runs_dir / "_cleanup_overview"
    if not runs_dir.is_dir():
        print(f"runs dir not found: {runs_dir}", file=sys.stderr)
        return 1

    run_dirs = sorted(
        p.parent for p in runs_dir.rglob("config.yml")
        if "_cleanup_overview" not in p.parts and "tensorboard" not in p.parts
    )
    # Optuna trial folders hold a config.yml plus the real run(s) nested below; skip such containers.
    run_dirs = [d for d in run_dirs if not any(o != d and d in o.parents for o in run_dirs)]
    rows = [inspect_run(p, runs_dir) for p in run_dirs]
    classify(rows)
    root_files = [(p.name, p.stat().st_size) for p in runs_dir.iterdir() if p.is_file()]
    write_outputs(rows, root_files, out_dir)

    for tier in TIERS:
        sel = [r for r in rows if r["tier"] == tier]
        print(f"{tier:14s} {len(sel):4d} runs  {human(sum(r['reclaim'] for r in sel))}")
    print(f"Wrote overview to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
