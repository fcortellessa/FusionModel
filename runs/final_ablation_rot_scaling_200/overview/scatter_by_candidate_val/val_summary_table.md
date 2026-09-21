Run dirs scanned (`--run_dir`): ['runs/final_ablation_rot_scaling_200']

| fusion | n_seeds | run folders used | test rot_x (deg) | test trans (mm) | test y (mm) | test z (mm) | flags |
|---|---|---|---|---|---|---|---|
| concat | 4 | 20260913_174300_bs16_concat, 20260913_190826_bs16_concat, 20260914_155845_bs16_concat, 20260915_030826_bs16_concat | 0.491 ± 0.064 | 0.786 ± 0.050 | 0.628 ± 0.058 | 0.361 ± 0.015 | - |
| gated | 4 | 20260913_204322_bs16_gated, 20260914_062926_bs16_gated, 20260914_172605_bs16_gated, 20260915_043352_bs16_gated | 0.517 ± 0.049 | 0.686 ± 0.076 | 0.542 ± 0.070 | 0.324 ± 0.030 | - |
| gated_concat | 3 | 20260913_220921_bs16_gated_concat, 20260914_075604_bs16_gated_concat, 20260914_184842_bs16_gated_concat | 0.518 ± 0.082 | 0.664 ± 0.083 | 0.529 ± 0.082 | 0.307 ± 0.029 | only 3 seed(s) vs 4 for the others |
| residual_gated_concat | 3 | 20260913_233249_bs16_residual_gated_concat, 20260914_092021_bs16_residual_gated_concat, 20260914_201210_bs16_residual_gated_concat | 0.525 ± 0.102 | 0.795 ± 0.097 | 0.624 ± 0.094 | 0.376 ± 0.029 | only 3 seed(s) vs 4 for the others |
| channelwise_gated_concat_local | 3 | 20260914_005716_bs16_channelwise_gated_concat_local, 20260914_104456_bs16_channelwise_gated_concat_local, 20260914_213523_bs16_channelwise_gated_concat_local | 0.518 ± 0.071 | 0.642 ± 0.080 | 0.514 ± 0.075 | 0.294 ± 0.023 | only 3 seed(s) vs 4 for the others |
| channelwise_residual_gated_concat_local | 3 | 20260914_030342_bs16_channelwise_residual_gated_concat_local, 20260914_125001_bs16_channelwise_residual_gated_concat_local, 20260914_233923_bs16_channelwise_residual_gated_concat_local | 0.520 ± 0.085 | 0.667 ± 0.059 | 0.532 ± 0.061 | 0.309 ± 0.022 | only 3 seed(s) vs 4 for the others |
| dual_fusion (gated_concat) | 3 | 20260914_050606_bs16_gated_concat, 20260914_144105_bs16_gated_concat, 20260915_014518_bs16_gated_concat | 0.503 ± 0.093 | 0.661 ± 0.048 | 0.534 ± 0.051 | 0.298 ± 0.018 | only 3 seed(s) vs 4 for the others |
