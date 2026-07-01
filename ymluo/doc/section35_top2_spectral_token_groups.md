# Section 35: Top2 Token Spectral Group Analysis

Date: 2026-06-30

## 0. Goal

This experiment checks whether key token groups are represented mainly in a low-rank spectral subspace.

Question:

```text
For true full-QK top2-selected tokens, evidence tokens, sink tokens, and recent tokens,
how much token energy and q-k correlation contribution lie in the leading singular directions
of the per-layer/head K-cache matrix?
```

New files:

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/analyze_top2_spectral_token_groups.py
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_top2_spectral_token_groups_server.sh
```

## 1. Server Run

Server:

```text
fdong@10.176.37.31
```

Project path:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
```

Medium run output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_spectral_token_groups_medium_0630
```

Local copied output:

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_spectral_token_groups_medium_0630
```

Run command:

```bash
OUT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_spectral_token_groups_medium_0630 \
VARIANTS=compact_kv,json_kv,needle_sentence,topic_table \
TASKS_PER_VARIANT=2 \
LAYERS=0,4,8,13,20,27 \
HEADS=0,4,8,12 \
MAX_QUERY_TOKENS_PER_TASK=2 \
bash scripts/run_top2_spectral_token_groups_server.sh
```

Scale:

```text
tasks = 8
selected layers = 0,4,8,13,20,27
selected heads = 0,4,8,12
sampled query tokens per task = 2
observed SVD rows = 384
skipped SVD rows = 0
runtime = 62.2s
```

## 2. Metrics

For each sampled layer/head/query row:

1. Build the centered historical K matrix.
2. Run SVD on that K matrix.
3. Use full-QK score to select true top-fraction historical tokens with `top_fraction=0.02`.
4. For each token group, project token K vectors and query vectors onto the right-singular directions.
5. Aggregate:

```text
sv_energy_topK: singular-value energy CDF of the K matrix
token_energy_topK: token vector squared-norm fraction in first K singular directions
q_energy_topK: query vector squared-norm fraction in first K singular directions
abs_qk_contrib_topK: absolute q-k dot-product contribution fraction in first K singular directions
cosine_topK: cosine after projecting q and token to first K singular directions
```

Token groups:

```text
top2_selected
evidence_key
evidence_label
evidence_record
evidence_any
sink
recent
```

## 3. Overall K-Space Spectrum

Across 384 sampled layer/head/query rows:

| Metric | Value |
| --- | ---: |
| Effective rank | 26.96 |
| Rank for 50% energy | 5.70 |
| Rank for 80% energy | 18.28 |
| Rank for 90% energy | 30.94 |
| Rank for 95% energy | 44.63 |
| Top 1 energy | 29.79% |
| Top 2 energy | 38.83% |
| Top 4 energy | 50.52% |
| Top 8 energy | 64.28% |
| Top 16 energy | 78.00% |
| Top 32 energy | 89.58% |
| Top 64 energy | 97.29% |

Interpretation:

```text
The K-cache space is clearly low-rank-ish: top16 carries about 78% energy,
and top32 carries about 90%.
But it is not always an extreme top-2/top-4 phenomenon.
```

## 4. Layer Differences

| Layer | Effective rank | Rank80 | Rank90 | Top8 energy | Top16 energy | Top32 energy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.8 | 1.0 | 1.7 | 96.8% | 98.0% | 99.0% |
| 4 | 20.8 | 17.8 | 31.7 | 68.5% | 79.4% | 89.7% |
| 8 | 35.9 | 23.6 | 38.8 | 54.6% | 72.2% | 86.8% |
| 13 | 36.5 | 24.1 | 39.3 | 52.6% | 70.5% | 86.5% |
| 20 | 33.8 | 21.3 | 36.2 | 54.9% | 73.6% | 88.2% |
| 27 | 32.9 | 22.0 | 38.0 | 58.3% | 74.4% | 87.3% |

Interpretation:

```text
Layer 0 is almost rank-1/rank-2.
Layer 4 is still relatively concentrated.
Layers 8/13/20/27 are more distributed and need about 20-24 directions for 80% energy.
```

## 5. Token Group Projection and Q-K Contribution

Overall group statistics:

| Group | Cases | Full cosine | Attention mass | Token top8 | Token top16 | Token top32 | Abs q-k top8 | Abs q-k top16 | Abs q-k top32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top2_selected | 4800 | 0.3102 | 0.0639 | 60.9% | 74.5% | 86.4% | 60.4% | 72.5% | 82.8% |
| sink | 3840 | 0.2303 | 0.0463 | 63.4% | 75.7% | 87.3% | 62.9% | 73.8% | 83.7% |
| recent | 6144 | 0.0345 | 0.0017 | 63.5% | 78.2% | 89.7% | 61.3% | 74.2% | 84.6% |
| evidence_key | 4752 | -0.0069 | 0.0004 | 62.5% | 77.2% | 89.2% | 61.4% | 73.8% | 84.4% |
| evidence_label | 384 | 0.0292 | 0.0031 | 59.4% | 75.3% | 88.4% | 60.0% | 72.8% | 83.7% |
| evidence_any | 10608 | -0.0306 | 0.0004 | 61.5% | 76.3% | 88.7% | 60.3% | 73.0% | 83.9% |

Interpretation:

```text
top2_selected tokens have much higher full q-k cosine and attention mass,
but their spectral energy profile is not uniquely more low-rank than sink/recent/evidence.
The important difference is more visible in q-k alignment/attention mass than in raw token-vector energy.
```

## 6. Top2 By Layer

| Layer | Cases | Full cosine | Attention mass | Token top8 | Token top16 | Abs q-k top8 | Abs q-k top16 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 800 | 0.619 | 0.0444 | 94.8% | 96.6% | 98.7% | 99.0% |
| 4 | 800 | 0.211 | 0.0624 | 64.6% | 75.1% | 75.7% | 82.2% |
| 8 | 800 | 0.284 | 0.0705 | 49.4% | 66.1% | 51.6% | 66.1% |
| 13 | 800 | 0.263 | 0.0622 | 48.6% | 65.7% | 36.8% | 52.6% |
| 20 | 800 | 0.220 | 0.0678 | 51.1% | 69.8% | 44.1% | 63.5% |
| 27 | 800 | 0.263 | 0.0764 | 57.1% | 73.8% | 55.2% | 71.7% |

Interpretation:

```text
The low-rank explanation is strongest in early layers, especially layer 0.
For middle/deep layers, top2-selected q-k contribution is still biased toward leading directions,
but not enough to say only the first few directions are sufficient.
```

## 7. Current Conclusion

This run supports the mentor's hypothesis in a moderate form:

```text
Key-space and top2 q-k behavior are substantially concentrated in low-rank directions.
Top16/top32 directions explain most of the energy/contribution.
However, except for layer 0, the useful subspace is closer to tens of directions
than to only the top 2-4 singular directions.
```

Practical implication:

```text
A low-rank classifier/probe is reasonable, but the first experiment should sweep rank
instead of fixing an extremely small K.
Suggested ranks: 4, 8, 16, 32, 64.
```

Next experiment:

```text
Train/evaluate a low-rank top2 token classifier using rank sweep 4/8/16/32/64,
with per-layer and per-head reporting.
```
