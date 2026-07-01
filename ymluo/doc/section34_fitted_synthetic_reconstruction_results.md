# Section 34: Fitted Synthetic KV Reconstruction Results

Date: 2026-06-29

## 0. Goal

After Section 33 showed that naive mean synthetic KV fails at high compression, this experiment tests whether fitted synthetic prototypes can close the gap.

New script:

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/analyze_fitted_synthetic_reconstruction.py
```

This is an offline reconstruction experiment, not a full PPL runtime yet.

It asks:

```text
Can a small number of synthetic K/V prototypes reconstruct the remote attention output
on held-out query tokens better than naive mean prototypes?
```

## 1. Setup

Server output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/r2h_fitted_synth_recon_codex_0629
```

Local copy:

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/r2h_fitted_synth_recon_codex_0629_local/
```

Model/data:

```text
model = /home/fdong/hrj/prove/Qwen3-0.6B
text = hard_topic_eval_v2.txt
prefill_tokens = 2048
calib_tokens = 32
heldout_tokens = 32
sink_tokens = 10
recent_tokens = 64
```

Layers tested:

```text
0, 4, 5, 7, 8, 13, 14, 16, 20, 27
```

All 16 heads were tested for each selected layer:

```text
160 heads total
```

Prototype counts:

```text
m = 4, 8, 16, 32
```

## 2. Methods Compared

### chunk_mean

Split remote KV into contiguous chunks and use mean K/V per chunk.

This is closest to the failed naive synthetic baseline.

### chunk_mean_ridge

Use chunk-mean K, but solve synthetic V by ridge regression on calibration queries:

```text
V_syn = (A^T A + λI)^-1 A^T Y_full
```

### topmass_ridge

Choose K prototypes from remote tokens with highest average calibration attention mass, then solve V by ridge regression.

This is still simple, but it is a real fitted synthetic baseline.

## 3. Aggregate Results

Mean held-out normalized MSE over 160 heads:

| Method | m | Heldout NMSE | Heldout Cosine |
| --- | ---: | ---: | ---: |
| chunk_mean | 4 | 0.5599 | 0.6806 |
| chunk_mean | 8 | 0.5516 | 0.6862 |
| chunk_mean | 16 | 0.5363 | 0.6962 |
| chunk_mean | 32 | 0.4820 | 0.7310 |
| chunk_mean_ridge | 4 | 0.4844 | 0.7733 |
| chunk_mean_ridge | 8 | 0.4639 | 0.7822 |
| chunk_mean_ridge | 16 | 0.5271 | 0.7807 |
| chunk_mean_ridge | 32 | 0.5061 | 0.7946 |
| topmass_ridge | 4 | 0.3995 | 0.8024 |
| topmass_ridge | 8 | 0.4061 | 0.8091 |
| topmass_ridge | 16 | 0.4886 | 0.8058 |
| topmass_ridge | 32 | 0.4550 | 0.8172 |

Best aggregate result:

```text
topmass_ridge m=4
heldout NMSE = 0.3995
heldout cosine = 0.8024
```

This is substantially better than naive chunk mean:

```text
chunk_mean m=8 heldout NMSE = 0.5516
topmass_ridge m=4 heldout NMSE = 0.3995
relative NMSE reduction ≈ 27.6%
```

## 4. Best-Per-Head Analysis

For each layer/head, choose the method/prototype count with the lowest heldout NMSE.

Overall:

```text
heads tested = 160
mean best heldout NMSE = 0.2945
heads with NMSE < 0.25 = 86
heads with NMSE < 0.40 = 118
heads with NMSE < 0.60 = 135
```

Layer summary:

| Layer | Mean Best NMSE | Heads < 0.25 | Heads < 0.40 |
| ---: | ---: | ---: | ---: |
| 0 | 0.2812 | 9 | 11 |
| 4 | 0.2375 | 10 | 14 |
| 5 | 0.2199 | 10 | 14 |
| 7 | 0.2371 | 8 | 16 |
| 8 | 0.3644 | 9 | 10 |
| 13 | 0.3537 | 5 | 11 |
| 14 | 0.3723 | 7 | 9 |
| 16 | 0.3941 | 6 | 8 |
| 20 | 0.4241 | 6 | 9 |
| 27 | 0.0610 | 16 | 16 |

Interpretation:

```text
Some layers/heads are very amenable to fitted synthetic KV.
Layer 27 is especially compressible in this reconstruction metric.
Layers 4,5,7 are also promising.
Layers 16,20 are harder and should remain conservative unless risk calibration says otherwise.
```

## 5. Key Finding

Fitted synthetic is clearly better than naive mean synthetic, but not yet enough to justify blanket 90% compression.

The right design is:

```text
Use fitted synthetic only for heads with good heldout reconstruction.
Use q-sketch/page routing or HYBRID_SAFE for medium heads.
Keep hard semantic heads FULL.
```

This matches the R2H-KV design:

```text
head-level profile -> backend routing
```

not:

```text
all selected layers -> same synthetic method
```

## 6. Next Experiment

Use the fitted reconstruction results to update `recommended_hybrid_policy_target90.json`:

```text
if best heldout NMSE < 0.25:
  SYNTH_M4/M8
elif best heldout NMSE < 0.40:
  QSKETCH_PAGE or HYBRID_SAFE
else:
  FULL/HYBRID_SAFE
```

Then run a PPL experiment at more realistic staged compression:

```text
50%, 70%, 80%, 90%
```

but only assign synthetic to heads/layers that passed reconstruction.

Current runtime still lacks true head-group synthetic, so the immediate next step is to implement a head-group synthetic runtime or run a layer approximation using only layers where most heads pass the threshold.

