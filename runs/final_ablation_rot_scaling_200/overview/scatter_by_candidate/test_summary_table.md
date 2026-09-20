Run dirs scanned (`--run_dir`): ['runs/final_ablation_rot_scaling_200']

| fusion | n_seeds | run folders used | test rot_x (deg) | test trans (mm) | test y (mm) | test z (mm) | flags |
|---|---|---|---|---|---|---|---|
| concat | 4 | 20260913_174300_bs16_concat, 20260913_190826_bs16_concat, 20260914_155845_bs16_concat, 20260915_030826_bs16_concat | 0.452 ± 0.127 | 0.710 ± 0.102 | 0.553 ± 0.094 | 0.340 ± 0.044 | - |
| gated | 4 | 20260913_204322_bs16_gated, 20260914_062926_bs16_gated, 20260914_172605_bs16_gated, 20260915_043352_bs16_gated | 0.481 ± 0.151 | 0.601 ± 0.117 | 0.466 ± 0.064 | 0.283 ± 0.088 | - |
| gated_concat | 3 | 20260913_220921_bs16_gated_concat, 20260914_075604_bs16_gated_concat, 20260914_184842_bs16_gated_concat | 0.513 ± 0.098 | 0.630 ± 0.039 | 0.475 ± 0.008 | 0.311 ± 0.050 | only 3 seed(s) vs 4 for the others |
| residual_gated_concat | 3 | 20260913_233249_bs16_residual_gated_concat, 20260914_092021_bs16_residual_gated_concat, 20260914_201210_bs16_residual_gated_concat | 0.501 ± 0.077 | 0.762 ± 0.018 | 0.565 ± 0.033 | 0.393 ± 0.041 | only 3 seed(s) vs 4 for the others |
| channelwise_gated_concat_local | 3 | 20260914_005716_bs16_channelwise_gated_concat_local, 20260914_104456_bs16_channelwise_gated_concat_local, 20260914_213523_bs16_channelwise_gated_concat_local | 0.525 ± 0.083 | 0.580 ± 0.066 | 0.436 ± 0.034 | 0.292 ± 0.059 | only 3 seed(s) vs 4 for the others |
| channelwise_residual_gated_concat_local | 3 | 20260914_030342_bs16_channelwise_residual_gated_concat_local, 20260914_125001_bs16_channelwise_residual_gated_concat_local, 20260914_233923_bs16_channelwise_residual_gated_concat_local | 0.523 ± 0.103 | 0.631 ± 0.086 | 0.474 ± 0.048 | 0.317 ± 0.052 | only 3 seed(s) vs 4 for the others |
| dual_fusion (gated_concat) | 3 | 20260914_050606_bs16_gated_concat, 20260914_144105_bs16_gated_concat, 20260915_014518_bs16_gated_concat | 0.490 ± 0.078 | 0.622 ± 0.081 | 0.478 ± 0.061 | 0.301 ± 0.048 | only 3 seed(s) vs 4 for the others |
