# RoPE method screen

Paired baseline: `npe_native_pre_top2`. PPL is `exp(mean NLL)` across matched seeds.

## 8,192 tokens

| Variant | n | Gold PPL | Accuracy | Evidence recall | Evidence mass | ΔNLL vs `npe_native_pre_top2` [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| npe_native_pre_top2 | 8 | 2.334 | 62.5% | 14.2% | 5.391% | +0.000 [+0.000, +0.000] |
| rope_top2 | 8 | 2.377 | 62.5% | 32.2% | 5.193% | +0.018 [-0.168, +0.230] |
| npe_rollback_masspreserve_pre_top2 | 8 | 2.496 | 50.0% | 14.1% | 5.470% | +0.067 [-0.060, +0.195] |
| full_rope | 8 | 2.600 | 62.5% | 100.0% | 5.265% | +0.108 [-0.145, +0.391] |
| npe_random_matched_pre_top2 | 8 | 2.950 | 50.0% | 13.9% | 5.233% | +0.234 [+0.058, +0.423] |
| npe_rollback_pre_top2 | 8 | 3.044 | 62.5% | 13.8% | 5.949% | +0.265 [+0.034, +0.595] |
| npe_distance_clip_pre_top2 | 8 | 5.776 | 37.5% | 13.9% | 6.551% | +0.906 [+0.393, +1.581] |

## 32,768 tokens

| Variant | n | Gold PPL | Accuracy | Evidence recall | Evidence mass | ΔNLL vs `npe_native_pre_top2` [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| npe_native_pre_top2 | 8 | 3.958 | 37.5% | 47.2% | 5.097% | +0.000 [+0.000, +0.000] |
| npe_rollback_masspreserve_pre_top2 | 8 | 4.445 | 25.0% | 47.2% | 5.129% | +0.116 [+0.015, +0.219] |
| npe_random_matched_pre_top2 | 8 | 4.905 | 37.5% | 47.2% | 4.509% | +0.215 [-0.289, +0.769] |
| npe_rollback_pre_top2 | 8 | 4.949 | 12.5% | 46.6% | 4.972% | +0.223 [-0.497, +0.765] |
| rope_top2 | 8 | 10.472 | 37.5% | 39.3% | 4.788% | +0.973 [+0.269, +1.708] |
| npe_distance_clip_pre_top2 | 8 | 12.151 | 25.0% | 46.5% | 5.179% | +1.122 [+0.376, +1.964] |
| full_rope | 8 | 14.607 | 12.5% | 100.0% | 4.992% | +1.306 [+0.703, +1.931] |

## 65,536 tokens

| Variant | n | Gold PPL | Accuracy | Evidence recall | Evidence mass | ΔNLL vs `npe_native_pre_top2` [95% CI] |
|---|---:|---:|---:|---:|---:|---:|
| npe_native_pre_top2 | 8 | 4.449 | 50.0% | 44.8% | 4.012% | +0.000 [+0.000, +0.000] |
| npe_rollback_masspreserve_pre_top2 | 8 | 5.848 | 25.0% | 44.9% | 3.973% | +0.274 [-0.022, +0.632] |
| rope_top2 | 8 | 6.494 | 50.0% | 35.4% | 3.897% | +0.378 [+0.177, +0.582] |
| full_rope | 8 | 8.362 | 25.0% | 100.0% | 4.618% | +0.631 [+0.348, +0.942] |
| npe_random_matched_pre_top2 | 8 | 11.105 | 12.5% | 45.3% | 3.539% | +0.915 [+0.627, +1.198] |
| npe_rollback_pre_top2 | 8 | 21.069 | 0.0% | 44.9% | 3.503% | +1.555 [+0.781, +2.238] |
| npe_distance_clip_pre_top2 | 8 | 40.079 | 12.5% | 44.3% | 3.686% | +2.198 [+1.089, +3.125] |
