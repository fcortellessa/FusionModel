Run dirs scanned (`--run_dir`): ['runs/final_ablation']

| fusion | n_seeds | run folders used | test rot_x (deg) | test trans (mm) | test y (mm) | test z (mm) | flags |
|---|---|---|---|---|---|---|---|
| concat | 4 | 20260916_062308_bs16_concat, 20260919_025906_bs16_concat, 20260919_042143_bs16_concat, 20260919_054549_bs16_concat | 0.593 ± 0.073 | 0.653 ± 0.034 | 0.500 ± 0.020 | 0.314 ± 0.037 | only 4 seed(s) vs 6 for the others |
| gated | 1 | 20260916_091204_bs16_gated | 0.622 | 0.666 | 0.527 | 0.287 | only 1 seed(s) vs 6 for the others |
| gated_concat | 6 | 20260916_074751_bs16_gated_concat, 20260917_090808_bs16_gated_concat, 20260918_162529_bs16_gated_concat, 20260919_070842_bs16_gated_concat, 20260919_124351_bs16_gated_concat, 20260919_173034_bs16_gated_concat | 0.520 ± 0.094 | 0.569 ± 0.102 | 0.437 ± 0.064 | 0.275 ± 0.070 | - |
| residual_gated_concat | 2 | 20260916_103922_bs16_residual_gated_concat, 20260917_124133_bs16_residual_gated_concat | 0.514 ± 0.117 | 0.581 ± 0.157 | 0.442 ± 0.130 | 0.282 ± 0.056 | only 2 seed(s) vs 6 for the others |
| channelwise_gated_concat_local | 5 | 20260916_120555_bs16_channelwise_gated_concat_local, 20260918_201049_bs16_channelwise_gated_concat_local, 20260919_092515_bs16_channelwise_gated_concat_local, 20260919_140226_bs16_channelwise_gated_concat_local, 20260919_185504_bs16_channelwise_gated_concat_local | 0.622 ± 0.122 | 0.556 ± 0.110 | 0.424 ± 0.057 | 0.269 ± 0.084 | only 5 seed(s) vs 6 for the others |
| channelwise_residual_gated_concat_local | 3 | 20260917_103320_bs16_channelwise_residual_gated_concat_local, 20260918_180846_bs16_channelwise_residual_gated_concat_local, 20260918_233548_bs16_channelwise_residual_gated_concat_local | 0.540 ± 0.082 | 0.482 ± 0.124 | 0.389 ± 0.074 | 0.205 ± 0.076 | only 3 seed(s) vs 6 for the others |
| dual_fusion (gated_concat) | 5 | 20260918_221139_bs16_gated_concat, 20260919_013711_bs16_gated_concat, 20260919_112652_bs16_gated_concat, 20260919_160703_bs16_gated_concat, 20260920_065132_bs16_gated_concat | 0.488 ± 0.113 | 0.584 ± 0.095 | 0.453 ± 0.062 | 0.273 ± 0.065 | only 5 seed(s) vs 6 for the others |
