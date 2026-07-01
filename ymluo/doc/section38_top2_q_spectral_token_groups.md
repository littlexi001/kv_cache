# Section 38: Top2 Q-Subspace Spectral Group Analysis

Date: 2026-06-30

## 0. Goal

This experiment mirrors Section 35, but uses the low-rank subspace of historical Q vectors instead of historical K vectors.

Question:

```text
For full-QK top2-selected tokens, evidence tokens, sink tokens, and recent tokens,
how much of the current query, token K projection, and q-k contribution can be explained
by the leading singular directions of the historical Q matrix?
```

New files:

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/analyze_top2_q_spectral_token_groups.py
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_top2_q_spectral_token_groups_server.sh
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

Final output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_q_spectral_token_groups_medium_0630_v3
```

Local copied output:

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_q_spectral_token_groups_medium_0630_v3
```

Run command:

```bash
OUT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_q_spectral_token_groups_medium_0630_v3 \
VARIANTS=compact_kv,json_kv,needle_sentence,topic_table \
TASKS_PER_VARIANT=2 \
LAYERS=0,4,8,13,20,27 \
HEADS=0,4,8,12 \
MAX_QUERY_TOKENS_PER_TASK=2 \
bash scripts/run_top2_q_spectral_token_groups_server.sh
```

Scale:

```text
tasks = 8
selected layers = 0,4,8,13,20,27
selected heads = 0,4,8,12
sampled query tokens per task = 2
observed SVD rows = 384
skipped SVD rows = 0
runtime = 164.5s
```

Important implementation detail:

```text
Transformers KV cache does not store Q, so the script records Q vectors during prefill/eval.
For each sampled query row, it builds an SVD basis from Q vectors before the current token.
The Q-SVD basis is fitted on centered historical Q.
Projection-energy metrics use the centered Q basis.
The lowrank top2-recall sanity check uses raw projected q/k dot scores; rank128 should recover full-QK.
```

## 2. Metrics

For each sampled layer/head/query row:

1. Build the centered historical Q matrix.
2. Run SVD on that Q matrix.
3. Use full-QK score to select true top-fraction historical tokens with `top_fraction=0.02`.
4. Project current q and token K vectors onto the Q right-singular directions.
5. Aggregate:

```text
qsvd_energy_topK: singular-value energy CDF of the historical Q matrix
current_q_energy_topK: current query squared-norm fraction in first K Q-SVD directions
token_k_energy_topK: token K squared-norm fraction in first K Q-SVD directions
abs_qk_contrib_topK: absolute q-k dot-product contribution fraction in first K Q-SVD directions
cosine_topK: cosine after projecting q and token K to first K Q-SVD directions
top2_recall: recall of full-QK top2 tokens using raw projected q-k dot in the Q-SVD basis
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

## 3. Overall Q-Space Spectrum

Across 384 sampled layer/head/query rows:

| Metric | Value |
| --- | ---: |
| Effective rank | 39.51 |
| Rank for 50% energy | 8.92 |
| Rank for 80% energy | 26.47 |
| Rank for 90% energy | 41.55 |
| Rank for 95% energy | 56.51 |
| Top 1 energy | 12.6% |
| Top 2 energy | 22.0% |
| Top 4 energy | 34.1% |
| Top 8 energy | 50.5% |
| Top 16 energy | 68.2% |
| Top 32 energy | 84.7% |
| Top 64 energy | 96.1% |
| Top 128 energy | 100.0% |

Interpretation:

```text
Historical Q is also low-rank-ish, but less concentrated than the K-space profile in Section 35.
Q needs about 26 directions for 80% energy and about 42 directions for 90% energy.
```

## 4. Layer Differences

| Layer | Effective rank | Rank80 | Rank90 | Top8 | Top16 | Top32 | Top64 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 44.05 | 31.92 | 50.03 | 48.1% | 64.1% | 80.3% | 94.5% |
| 4 | 31.76 | 20.83 | 35.08 | 58.9% | 74.8% | 88.1% | 97.6% |
| 8 | 43.95 | 28.59 | 43.89 | 44.9% | 64.4% | 83.5% | 95.8% |
| 13 | 31.67 | 20.31 | 33.73 | 57.3% | 75.1% | 89.4% | 97.4% |
| 20 | 36.42 | 25.47 | 40.86 | 54.2% | 71.2% | 85.7% | 95.8% |
| 27 | 49.20 | 31.69 | 45.72 | 39.5% | 59.6% | 80.9% | 95.9% |

Interpretation:

```text
Layer 4 and layer 13 have the most concentrated Q spectra.
Layer 0 and layer 27 are more distributed in Q space, unlike the K-space result where layer 0 was almost rank-1/rank-2.
```

## 5. Token Group Projection and Q-K Contribution

Overall group statistics:

| Group | Cases | Full cosine | Attention mass | Q top16 | Q top32 | K top16 | K top32 | Abs q-k top16 | Abs q-k top32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top2_selected | 4800 | 0.1043 | 0.0639 | 53.4% | 70.7% | 12.6% | 20.0% | 37.7% | 54.1% |
| sink | 3840 | 0.0275 | 0.0462 | 53.5% | 70.6% | 14.4% | 22.2% | 37.5% | 54.0% |
| recent | 6144 | 0.0305 | 0.0016 | 53.5% | 70.6% | 11.1% | 18.0% | 36.3% | 52.4% |
| evidence_key | 4992 | 0.0060 | 0.0003 | 53.5% | 70.6% | 11.9% | 19.2% | 37.8% | 54.2% |
| evidence_label | 384 | 0.0417 | 0.0049 | 53.5% | 70.6% | 12.2% | 20.4% | 37.6% | 54.5% |
| evidence_any | 10608 | 0.0089 | 0.0004 | 53.3% | 70.6% | 12.0% | 19.3% | 37.8% | 54.2% |

Interpretation:

```text
Current q itself is moderately concentrated in the Q subspace: top32 explains about 71%.
But historical K vectors are weakly represented in the same Q leading directions: top32 explains only about 20% for top2 tokens.
Even so, the absolute q-k contribution is more concentrated than raw K energy: top32 explains about 54%.
```

This means the Q subspace captures part of the q-k interaction, but it is not as direct as the K-space basis.

## 6. Q-Basis Low-Rank Top2 Recall

Using raw projected q-k dot scores in the Q-SVD basis:

| Rank | Top2 recall | Top2 attention-mass recall |
| ---: | ---: | ---: |
| 1 | 8.7% | 14.9% |
| 2 | 9.6% | 15.3% |
| 4 | 13.8% | 23.6% |
| 8 | 19.0% | 30.1% |
| 16 | 29.8% | 35.6% |
| 32 | 44.1% | 50.7% |
| 64 | 60.7% | 69.2% |
| 128 | 99.9% | 100.0% |

Sanity check:

```text
Rank128 is full Q head dimension and recovers full-QK top2 almost exactly.
So the projected-dot recall code is aligned with the full-QK target.
```

Comparison with K-basis lowrank dot from Section 36:

| Basis | Rank32 top2 recall | Rank32 mass recall | Rank64 top2 recall | Rank64 mass recall |
| --- | ---: | ---: | ---: | ---: |
| K-SVD basis | 73.3% | 90.4% | 85.1% | 98.0% |
| Q-SVD basis | 44.1% | 50.7% | 60.7% | 69.2% |

Interpretation:

```text
Q-basis low-rank scoring is far above random 2%, but much weaker than K-basis scoring.
The top2 selection mechanism is therefore not equally explained by the leading Q subspace alone.
```

## 7. Top2 By Layer

Projection statistics for `top2_selected`:

| Layer | Cases | Full cosine | Attention mass | Q top32 | K top32 | Abs q-k top32 | Q top64 | K top64 | Abs q-k top64 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 800 | 0.0111 | 0.0440 | 64.3% | 0.7% | 51.8% | 81.5% | 0.9% | 67.6% |
| 4 | 800 | 0.0208 | 0.0624 | 78.6% | 7.4% | 53.9% | 91.8% | 11.7% | 71.6% |
| 8 | 800 | 0.1850 | 0.0707 | 64.3% | 26.3% | 52.8% | 84.3% | 39.0% | 72.0% |
| 13 | 800 | 0.1976 | 0.0622 | 77.2% | 24.7% | 51.0% | 90.8% | 46.2% | 73.5% |
| 20 | 800 | 0.1473 | 0.0675 | 70.9% | 28.5% | 54.8% | 86.5% | 42.1% | 72.5% |
| 27 | 800 | 0.0642 | 0.0764 | 69.2% | 32.2% | 60.0% | 89.0% | 43.0% | 80.7% |

Q-basis recall by layer:

| Rank | Layer | Top2 recall | Top2 attention-mass recall |
| ---: | ---: | ---: | ---: |
| 32 | 0 | 44.9% | 64.9% |
| 32 | 4 | 48.8% | 66.1% |
| 32 | 8 | 39.7% | 34.9% |
| 32 | 13 | 47.0% | 46.6% |
| 32 | 20 | 34.1% | 22.9% |
| 32 | 27 | 50.1% | 68.7% |
| 64 | 0 | 55.0% | 72.5% |
| 64 | 4 | 68.9% | 91.1% |
| 64 | 8 | 55.5% | 56.9% |
| 64 | 13 | 66.2% | 59.0% |
| 64 | 20 | 47.7% | 40.9% |
| 64 | 27 | 71.1% | 95.0% |

Interpretation:

```text
Layer 4 and layer 27 have high rank64 attention-mass recall.
Layer 20 is weaker under Q-basis scoring.
This is different from the K-basis experiment, where rank64 was strong across all selected layers.
```

## 8. Current Conclusion

This experiment supports a weaker version of the low-rank hypothesis for Q:

```text
Historical Q vectors do have a low-rank-ish spectrum, and leading Q directions contain useful
information for top2 recovery.
However, the Q leading subspace alone is much less sufficient than the K leading subspace.
```

Most important takeaways:

```text
1. Q-space is less concentrated than K-space overall.
2. Current q energy is moderately concentrated in the Q leading directions.
3. Historical K vectors are not strongly concentrated in those same Q directions.
4. Q-basis lowrank dot at rank64 recovers only 60.7% top2 tokens and 69.2% attention mass,
   versus 85.1% and 98.0% for K-basis lowrank dot.
```

Practical implication:

```text
For top2 token selection, a low-rank classifier/probe should primarily use K-SVD or joint q-k features.
Q-SVD alone can be an auxiliary signal, but it is probably not the best standalone subspace.
```
