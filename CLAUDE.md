# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Research codebase for **multi-modal gated sensor fusion for planar in-hand object pose estimation**, combining a monocular RGB stream and a DIGIT visuo-tactile sensor stream. A shared YOLO26n backbone extracts per-stream features, a configurable **fusion module** combines the streams, and MLP heads regress translation (meters, y/z only — x is fixed to 0) and rotation (quaternion by default, several alternate parametrisations available). The core research questions this repo is built to answer:

- Which fusion strategy (concatenation, learned gating, attention, spatial gating) works best for combining RGB + tactile features?
- Do task-specific mechanisms (separate gating/attention adapters for the rotation vs. translation branches) help, given rotation and translation have different units/scales?
- How much does tactile data augmentation help, and which augmentation types matter?
- Are results stable across random seeds and against simple baselines?

Everything is driven by YAML configs (`configs/`) consumed by `scripts/train.py` / `scripts/test.py`; there is no experiment-tracking framework beyond per-run CSV/TensorBoard logs under `runs/`.

## Repository layout

| Path | Purpose |
|---|---|
| `base_model/` | `PoseEstimationNet` (the model), YOLO backbone/feature extraction (`yolo_trunk.py`), MLP/pose heads (`mlp_head.py`) |
| `fusion/` | Fusion-strategy registry (`registry.py`) and implementations; post-fusion task-specific adapters (`adapters.py`) |
| `losses/` | Base losses, per-rotation-parametrisation loss/metric handlers (`rotation_loss_handlers.py`), multi-task weighting (`uncertainty_weighting.py`, `task_weighting.py`, `gradnorm_pytorch.py`) |
| `data_processing/` | Dataset split (`GripFusionNet_split.py`), preprocessing/caching (`preprocess_data.py`), offline/online augmentation (`augment_data.py`), dataloaders (`dataloader_preprocessed.py`) |
| `configs/` | Hand-written experiment configs (`*.yml`) + `configs/generated/` (auto-generated ablation/sweep grids, do not hand-edit) |
| `scripts/train.py`, `scripts/test.py` | Single-run training and evaluation entry points |
| `scripts/train_multiple_runs.py`, `scripts/test_multiple_runs.py` | Batch launchers over multiple configs/run folders |
| `scripts/tactile_ablation/` | Config-grid generator (`generate_config_variants.py`), tactile augmentation ablation tooling |
| `scripts/evaluate/` | Post-hoc analysis/plotting over `metrics.csv` and test results (some files suffixed `_archived`/`_old` are superseded — check before using) |
| `utils/` | `train_utils.py` (training orchestration managers), `utils.py` (config/checkpoint I/O), `logging_utils.py` (CSV/TensorBoard logging), `normalization.py` (translation normalization constants) |
| `Data/` | Raw capture triplets + `Data_split/{train,val,test}` + cached `*_preprocessing/*.pt` tensors (large, data not code) |
| `runs/` | Per-run outputs: config snapshot, manifest, checkpoint, metrics CSV, TensorBoard logs, test results |
| `data_capture/` | **Separate** hardware data-collection tool (UR robot + RealSense + DIGIT), own `.py3_12_venv`; unrelated to model training |
| `yolo26_adaptive_input.py`, `yolo26n.pt` (repo root) | Legacy/dead reference file — imports non-existent `Dataloader`/`Loss` packages, not on any active import path. Do not use as a reference or "fix" its imports. |

## Architecture

### Data flow
1. Raw samples are triplets `{timestamp}.png` (RGB), `{timestamp}_digit.png` (tactile), `{timestamp}.json` (pose: position + quaternion in a robot/TCP frame).
2. `data_processing/GripFusionNet_split.py::split_dataset` shuffles triplets (via the global `random` module) into `Data_split/{train,val,test}` at an **80/15/5** ratio, recreating the split folders from scratch every time it runs.
3. `data_processing/preprocess_data.py::preprocess_data` computes RGB/tactile normalization stats from the train split, letterboxes + resizes + normalizes every image to 640×640, transforms the pose into the hub frame via a fixed calibration matrix, and caches each sample as `{split}/{split}_preprocessing/{timestamp}.pt` containing `{rgb, digit, pose}` (pose = `[x,y,z,qx,qy,qz,qw]`).
4. `data_processing/dataloader_preprocessed.py::get_dataloaders` reads the cached `.pt` files (`PreprocessedTactileDataset`). If `data.augmentation.enabled: true`, training instead uses `RawOnlineAugmentedTactileDataset`, which re-reads raw images and applies deterministic, per-sample-seeded photometric augmentation (`augment_data.py`) on the fly — val/test are never augmented.

### Model (`base_model/pose_estimation_model.py`)
- One deep-copied YOLO26n backbone per stream (`base_model/yolo_trunk.py`), extracting P3/P4/P5 feature maps while respecting YOLO's internal skip connections.
- Each scale is 1×1-conv channel-reduced and adaptive-avg-pooled to a fixed 10×10 grid.
- **1D pipeline** (default): P3/P4/P5 are flattened and concatenated per stream into one feature vector, then passed to the fusion module.
- **2D pipeline** (`model.fusion: spatial_gated_2d`): per-stream spatial maps (not pooled to vectors) are fused pixel-wise, then mixed with a post-fusion conv before flattening.
- Heads: separate rotation and translation MLP heads (`base_model/mlp_head.py`), selected by `model.head_type`: `quaternion` (default), `2D_unit_vector`, `angle`, `lie_algebra`, or `se3_direct` (single head jointly predicting a 6D SE(3) Lie-algebra vector instead of separate rotation/translation heads).
- Optimizer uses **named parameter groups** (`base`, `head_rot`, `head_y`, `head_z`) so head-specific LR scheduling/decay is possible.

### Fusion strategies (`fusion/registry.py` → `FUSION_REGISTRY`)
- **Weighted-sum** (collapse streams to one vector): `gated`/`gated_Softmax`, `simple_gated`, `gated_Sigmoid`, `ARGate_Fusion` (learns per-stream trust targets from auxiliary per-stream losses).
- **Concat-preserving** (retain every stream, useful when rotation needs modality-specific detail): `concat`, `gated_concat`, `residual_gated_concat`, `channelwise_residual_gated_concat`.
- **Attention**: `attention` (multi-head self-attention across stream tokens, mean-pooled).
- **Spatial**: `spatial_gated_2d` (per-pixel softmax gate across streams; only fusion that operates on 2D maps).
- New strategies must be added via the `@register_fusion(name)` decorator + `BaseFusion` subclass, not via ad hoc branching in the model.

### Task-specific mechanisms
Three independent, backward-compatible, off-by-default mechanisms — the current active research area (see branch `task-specific_fusion`). The first two are post-fusion feature *adapters* (`fusion/adapters.py`): a single shared `model.fusion` instance combines RGB+tactile into one fused vector, which each adapter then re-weights per-channel, separately for the rotation vs. translation branch, right before the heads:
- **Legacy**: `model.task_adapter: task_specific_residual_gate | task_specific_channel_attention` → `rot_adapter`/`trans_adapter`.
- **Current**: `model.task_feature_adapter: {enabled: true, type: channel_attention | residual_gate}` → `rotation_adapter`/`translation_adapter`, built from the same `TaskSpecificChannelAttention`/`TaskSpecificResidualGate` classes as the legacy mechanism (dispatched on `type`), with attentions optionally logged to TensorBoard.

The third is true **dual fusion** (`model.dual_fusion: true`, in `base_model/pose_estimation_model.py`): unlike the two adapters above, this builds two independent instances of the `model.fusion` module (own parameters each) — `self.fusion_rotation`/`self.fusion_translation` (plus, for `spatial_gated_2d`, independent `fusion_post_conv_rotation`/`fusion_post_conv_translation`) — so RGB+tactile are combined separately per task branch rather than sharing one fused vector. Composable with either adapter above (the adapters would then re-weight the already branch-specific fused vector). Incompatible with `head_type: se3_direct` (raises at construction) since that head needs one joint fused vector. Gate-weight diagnostics (`last_fusion_weights`/`last_fusion_targets`, TensorBoard `fusion/*`, `metrics.csv`'s `*_fusion_gate_*` columns) average the rotation/translation branches together when dual fusion is enabled — see `_reduce_dual_fusion_branches()` in `scripts/train.py` — so per-branch gate trust isn't separately visible yet.

### Unit balancing (rotation vs. translation)
Rotation loss is a quaternion geodesic distance (radians) multiplied by a fixed `rot_scaling_factor` (per-handler constant in `losses/rotation_loss_handlers.py`, e.g. 200 for quaternion/lie-algebra/angle, 2000 for `2D_unit_vector`) so it is numerically comparable to translation MSE (meters). Optionally, `training.task_weighting.type: uncertainty | gradnorm | dwa` replaces/augments this fixed scaling with a learned or dynamic balance between the rotation and translation loss terms. `losses/rotation_loss_handlers.py::get_rotation_handler()` is the factory dispatching on `model.head_type` — add new parametrisations there, not with inline conditionals in `train.py`.

## Development and experiment commands

Run everything from the repository root (scripts insert the repo root onto `sys.path` themselves; relative paths like `Data/`, `runs/` assume this).

```bash
# Editable install
pip install -e .

# Train a single run (defaults to configs/gated.yml)
python scripts/train.py --config configs/<name>.yml
python scripts/train.py --config configs/gated.yml --batch_size 32 --optimizer AdamW --force_preprocess

# Evaluate a run on the test split
python scripts/test.py --run_folder runs/<run_dir>        # uses runs/<run_dir>/manifest.json
python scripts/test.py --config <cfg.yml> --checkpoint <ckpt.pth> --results_dir <dir>

# Batch training over multiple configs/batch sizes
python scripts/train_multiple_runs.py --configs configs/gated.yml configs/concat.yml --batch_sizes 16 32
python scripts/train_multiple_runs.py --config_dir configs/generated/task_specific_fusion_ablation/

# Batch evaluation (edit RUN_FOLDERS_TO_BE_INFERENCED / ABLATION_FOLDER constants at top of file first — no CLI args)
python scripts/test_multiple_runs.py

# Generate a config sweep/grid from a template + spec (see configs/generated/*_spec.yml for spec format)
python scripts/tactile_ablation/generate_config_variants.py --template-config <base.yml> --spec <spec.yml> --output-dir configs/generated/<name>/ [--overwrite] [--dry-run]
```

Preprocessing (split + normalize + cache to `.pt`) runs automatically inside `train.py` the first time (or with `--force_preprocess`) via `data_processing/preprocess_data.py::preprocess_data`; there is no separate required preprocessing command. `scripts/data_preprocess_pipeline.py` is a **separate, unwired** standalone preprocessor with different output paths/naming (`{split}_preprocessed` at the `Data_split` root) — it is not called by `train.py`; don't confuse its cache with the real one at `Data_split/<split>/<split>_preprocessing/`.

**No test suite, linter, or formatter is configured** in this repo (no pytest/ruff/black/mypy config in `pyproject.toml`). `test_adapter.py` and `reproduce_issue.py` at the repo root are ad hoc, uncommitted debugging scripts, not a maintained test suite — don't assume `pytest` runs cleanly, and validate changes by running the actual training/eval script for a few steps/epochs or a small model-construction smoke test instead.

## Configuration system

- Configs are plain YAML with `model`, `training`, `data` blocks, loaded via `yaml.safe_load` (`utils/utils.py::load_config`) — **no schema validation**; a missing key raises a plain `KeyError` wherever it's first accessed. `configs/default.yml` documents every key inline and is the best reference for the full schema.
- `configs/generated/` holds auto-generated sweeps/ablations, produced by `scripts/tactile_ablation/generate_config_variants.py` from a **template config** + a **spec YAML** (`fixed_overrides`, `grid`, and/or explicit `variants`, all using dotted-path keys like `training.seed` or `task_feature_adapter.type`). Generated filenames encode an index + sanitized overrides + a hash for uniqueness. Regenerate rather than hand-editing files in this directory.
- **Seeds**: `training.seed` is a single integer; `scripts/train.py::set_global_seed` explicitly raises `TypeError` if given a list — sweep across seeds by generating one config per seed (see `configs/generated/task_specific_fusion_ablation/`, which grids `training.seed: [43, 44, 45]`).
- **Determinism**: `training.deterministic` (default `True`) seeds `random`/`numpy`/`torch`(+CUDA), sets `cudnn.deterministic=True`/`benchmark=False`, and enables `torch.use_deterministic_algorithms` (warn-only).
- Two config key names exist for the same feature across the repo's history: `training.task_weighting` (current) and `training.uncertainty_weighting` (legacy) — code reads both via `cfg.get("training", {}).get("task_weighting", cfg.get("training", {}).get("uncertainty_weighting", {}))`; preserve this fallback pattern if you touch it.
- **Outputs**: each run writes to `runs/<timestamp>_bs<batch_size>_<fusion>/` — `config.yml` (snapshot), `manifest.json`, `metrics.csv` (per-epoch), `checkpoint_best.pth` (overwritten whenever val loss improves — no periodic/last-epoch checkpoint), and `results/` (populated by `test.py`). TensorBoard logs go to `runs/tensorboard/<same-run-name>/`. (`scripts/train.py::main` also computes an unused nested `fusion/head_type/optimizer_timestamp` path for TensorBoard that gets overwritten by `RunSetupManager.setup_tensorboard` — the flat layout above is what's actually written; verify empirically rather than trusting that code path.)

## Experimental methodology

- **Split**: 80/15/5 train/val/test, reshuffled by `GripFusionNet_split.py` using the (seeded) global `random` module — there is no persisted split index. **Preprocessing is skipped if `Data_split/train/train_preprocessing` already exists**, so changing `training.seed` alone does *not* reshuffle the split on a subsequent run; pass `--force_preprocess` (or delete the cache) to actually get a new seed-driven split.
- **Ablations**: one varying dimension (or seed) per generated grid — see `configs/generated/*_ablation_spec.yml` for the `grid`/`fixed_overrides` convention. Don't vary multiple experimental variables in one ablation unless explicitly asked.
- **Baselines**: `configs/baseline.yml`, `configs/concat.yml`, and `configs/tactile_ablation_baseline.yml` (no augmentation) are the comparison points other fusion/adapter/augmentation configs are measured against.
- **Results**: per-epoch train/val metrics in each run's `metrics.csv`; test-time per-sample errors + summary stats + plots in `runs/<run>/results/`; cross-run aggregation via `scripts/evaluate/*.py` and `scripts/tactile_ablation/analyze_tactile_ablation.py`.

## Safe modification rules

- Don't change values in existing `configs/*.yml` (learning rate, `rot_scaling_factor`, dropout, epochs, split ratios, etc.) without explicit instruction — they are established experimental baselines other runs are compared against.
- Don't change the split ratios or split RNG behavior in `GripFusionNet_split.py` without explicit instruction — it silently invalidates comparisons with every prior run.
- Don't touch the `MEAN_Y`/`STD_Y`/`MEAN_Z`/`STD_Z` placeholders in `utils/normalization.py` without explicit instruction (see Pitfalls) — changing them changes what "meters" means for every existing checkpoint/metrics file.
- When reproducing an experiment, use the exact seed + config recorded in that run's `manifest.json`/`config.yml`; never substitute a different seed.
- Add new fusion strategies through `fusion/registry.py`'s `@register_fusion` + `FUSION_REGISTRY`, new rotation parametrisations through `losses/rotation_loss_handlers.py`'s `get_rotation_handler()` factory (+ a new head in `mlp_head.py`), and new task-weighting schemes through `TrainingConfigBuilder.build_uncertainty_module()`'s type dispatch — don't bypass these with inline conditionals.
- Generate new ablation/sweep configs via `scripts/tactile_ablation/generate_config_variants.py`, not by hand-copying YAML files.
- Preserve config-key backward compatibility (see the `task_weighting`/`uncertainty_weighting` fallback pattern above) when renaming/restructuring config sections.
- Vary exactly one experimental factor per ablation grid unless explicitly told otherwise.

## Common pitfalls

- **Preprocessing cache masks seed changes**: see "Experimental methodology" above — always pass `--force_preprocess` when a new seed must produce a new split.
- **Translation "normalization" is currently a no-op**: `utils/normalization.py`'s `MEAN_Y/STD_Y/MEAN_Z/STD_Z` are literally `0.0/1.0/0.0/1.0` placeholders ("Replace these with actual values from your dataset"). Don't assume cross-run translation metrics reflect real normalization.
- **Two unrelated preprocessing pipelines exist**: `data_processing/preprocess_data.py` (the real one, wired into `train.py`, caches to `Data_split/<split>/<split>_preprocessing/`) vs. `scripts/data_preprocess_pipeline.py` (orphaned standalone script, caches to `Data_split/<split>_preprocessed/`). Both cache directories currently exist on disk under `Data/Data_split/` — don't confuse them.
- **`yolo26_adaptive_input.py` at repo root is dead code** referencing non-existent `Dataloader`/`Loss` packages from an earlier layout; it's not imported by anything active.
- **`generate_config_variants.py`'s CLI defaults look swapped**: `DEFAULT_SPEC_PATH` points at a model config (`configs/gated_2d.yml`) and `DEFAULT_TEMPLATE_CONFIG` points at what looks like a spec file. Always pass `--template-config` and `--spec` explicitly rather than relying on the defaults.
- **`run_tactile_ablation_training.sh`'s seed loop is a documented no-op** — it doesn't pass `--seed` into `train.py`, so all "seeds" in that script train identically. Vary seeds via per-seed generated configs instead.
- **`data_capture/`** is a hardware capture tool (UR robot + RealSense + DIGIT) with its own `.py3_12_venv` and SDK dependencies (`pyrealsense2`, `rtde_control`, `digit_interface`) — unrelated to model training/eval; don't pull it into fusion/model work.
- **Only the best checkpoint is kept** (`checkpoint_best.pth`, overwritten on every val-loss improvement) — there's no way to recover final-epoch weights if they weren't the best.
- Repo enforces LF line endings for `.py`/`.sh`/`.json` via `.gitattributes`; expect (harmless) CRLF→LF conversion warnings from git on Windows.

## Agent workflow

1. Inspect the relevant files before changing anything — for model/fusion/loss changes, read `base_model/pose_estimation_model.py`'s `forward()` to see how your change is actually wired into the forward pass before editing a leaf module.
2. Trace the config path: find which `model.*`/`training.*`/`data.*` key controls the behavior you're changing, and check both `configs/default.yml` (documentation) and the specific config(s) the user is working with.
3. Identify whether a registry/factory mechanism already covers what you need (fusion registry, rotation-handler factory, task-weighting dispatch, config-grid generator) — use it instead of adding a new code path.
4. State the intended change briefly before making it, especially if it touches experimental defaults, seeds, or splits.
5. Make the smallest reasonable change.
6. Validate: for model/loss/fusion changes, prefer a quick forward-pass smoke test (construct `PoseEstimationNet` with the relevant config and run a dummy batch through it, e.g. in the style of `test_adapter.py`) over a full training run; for training-loop/config-plumbing changes, run `scripts/train.py` for a couple of epochs on a small batch size and confirm `metrics.csv`/TensorBoard output looks sane.
7. Report exactly what changed (files/lines) and what was run to validate it.
