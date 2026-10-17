# Analysis notes

Short, dated write-ups of explanatory analyses (what was checked, result, caveat). Newest first.
Full outputs live in `analysis*/` and `runs/`; scripts are named per entry.

## 2026-10-16 — Cross-attention fusions: where the error is, and is the SE block useful?

- **Error structure** (`scripts/evaluate/sample_level_error_analysis.py`, divisor-4 runs, splits 36-45): median error is
  ~0.2 mm / ~0.2 deg, but the top 5% of samples carry 33-64% of the mean error and are hard for every model
  (mirror-like roll swaps, ~56% of them). Fusion choice moves the typical error by only ~0.02 mm.
  `se_gated_cross_attention` is ~0.05-0.06 mm worse than concat on y/z across the whole distribution
  (paired, also without the hard samples), not on rotation. Caveat: CIs ignore split/seed variance.
- **`gated_cross_attention` vs `se_gated_cross_attention`** (divisor 1, splits 36-40): no difference (paired CIs
  centred on 0, sign flips between splits). Confounded: the 5 `gated_cross_attention` runs are the pre-LayerNorm
  code (no `vis_norm` in checkpoints); `se_gated_cross_attention` adds LayerNorm + positional encoder + SE block.
  One split-41 `se_gated_cross_attention` run collapsed (best epoch 13).
- **SE-block probe** (`scripts/evaluate/se_block_ablation_test.py`, split 36, 3 runs): SE weights are ~constant across
  samples (across-sample std 0.002 vs mean 0.26-0.39); replacing them by the mean or shuffling changes errors by
  <1e-3 mm / <1e-5 deg. It is a learned per-channel constant, not a per-sample gate. The token gate is also mostly
  static (except in the dual-fusion run). Caveat: old 1D module only; 1 split.
- **Consequence:** 2D-token variants in `fusion/gated_cross_attention_2d.py`; run the no-SE
  `gated_cross_attention_2d` first, `se_gated_cross_attention_2d` as its SE ablation
  (`configs/generation_specs/gated_cross_attention_2d_divisor4_spec.yml`).
