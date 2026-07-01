# Section 36: Low-Rank Top2 Classifier Probe

Date: 2026-06-30

## 0. Goal

This experiment follows Section 35 and tests the next hypothesis:

```text
If true top2 token selection is mainly determined in a low-rank spectral subspace,
then a classifier using only low-rank projected query-key features should recover
the key 2% tokens on held-out queries.
```

New files:

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/train_top2_lowrank_classifier.py
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_top2_lowrank_classifier_server.sh
```

## 1. Data Choice

No external large dataset was downloaded.

Reason:

```text
The server bandwidth is limited, and the existing project already has synthetic
key-value retrieval task generators that directly target key token retrieval.
```

Used task variants:

```text
compact_kv
json_kv
needle_sentence
topic_table
```

These cover compact key-value lines, JSON records, natural needle sentences, and table-style records.

## 2. Server Run

Server:

```text
fdong@10.176.37.31
```

Project:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_lowrank_classifier_0630_v3
```

Local copy:

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_lowrank_classifier_0630_v3
```

Run command:

```bash
OUT=/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_lowrank_classifier_0630_v3 \
TRAIN_TASKS_PER_VARIANT=8 \
EVAL_TASKS_PER_VARIANT=2 \
bash scripts/run_top2_lowrank_classifier_server.sh
```

Scale:

```text
train tasks = 32
eval tasks = 8
selected layers = 0,4,8,13,20,27
selected heads = 0,4,8,12
sampled query tokens per task = 2
train rows = 1536
eval rows = 384
skipped SVD rows = 0
runtime = 32.1s
```

## 3. Feature and Label

For each sampled layer/head/query row:

1. Use full-QK to label the true top 2% historical tokens.
2. Fit SVD on the centered historical K matrix.
3. Project query and historical keys onto the first `r` right-singular directions.
4. Build one feature per direction:

```text
feature_i = q_proj_i * k_proj_i
```

Rank sweep:

```text
r = 4, 8, 16, 32, 64
```

Two methods:

```text
lowrank_dot:
  score = sum_i feature_i

trained_linear:
  per-layer/head linear classifier trained on feature_i
```

Evaluation:

```text
For each held-out query row, score all historical tokens.
Select the same number of tokens as full-QK top2.
Report recall against the true full-QK top2 set.
```

Random recall at this budget is approximately:

```text
2%
```

## 4. Overall Results

| Method | Rank | Top2 recall | Top2 attention-mass recall |
| --- | ---: | ---: | ---: |
| lowrank_dot | 4 | 23.71% | 45.97% |
| trained_linear | 4 | 24.17% | 41.29% |
| lowrank_dot | 8 | 38.92% | 65.11% |
| trained_linear | 8 | 39.35% | 62.17% |
| lowrank_dot | 16 | 55.02% | 77.02% |
| trained_linear | 16 | 54.81% | 75.77% |
| lowrank_dot | 32 | 73.31% | 90.36% |
| trained_linear | 32 | 70.23% | 86.50% |
| lowrank_dot | 64 | 85.10% | 98.04% |
| trained_linear | 64 | 77.88% | 96.00% |

Interpretation:

```text
Low-rank features recover top2 selections far above the 2% random baseline.
Rank32 already recovers about 73% of selected tokens and 90% of selected attention mass.
Rank64 recovers about 85% of selected tokens and 98% of selected attention mass.
```

The untrained `lowrank_dot` is stronger than `trained_linear` at high rank.
This suggests the main signal is already the projected q-k dot product; the small learned classifier mostly learns direction weights and does not add much beyond the low-rank geometry.

## 5. Layer Results

### Rank32

| Method | Layer | Top2 recall | Top2 attention-mass recall |
| --- | ---: | ---: | ---: |
| lowrank_dot | 0 | 67.9% | 87.2% |
| lowrank_dot | 4 | 67.0% | 81.0% |
| lowrank_dot | 8 | 80.6% | 94.0% |
| lowrank_dot | 13 | 74.1% | 91.1% |
| lowrank_dot | 20 | 74.1% | 90.8% |
| lowrank_dot | 27 | 76.1% | 95.5% |
| trained_linear | 0 | 59.3% | 82.9% |
| trained_linear | 4 | 67.1% | 72.4% |
| trained_linear | 8 | 79.1% | 90.6% |
| trained_linear | 13 | 72.5% | 91.2% |
| trained_linear | 20 | 69.1% | 95.6% |
| trained_linear | 27 | 74.3% | 84.4% |

### Rank64

| Method | Layer | Top2 recall | Top2 attention-mass recall |
| --- | ---: | ---: | ---: |
| lowrank_dot | 0 | 82.9% | 94.7% |
| lowrank_dot | 4 | 83.4% | 96.7% |
| lowrank_dot | 8 | 88.1% | 99.3% |
| lowrank_dot | 13 | 86.3% | 97.1% |
| lowrank_dot | 20 | 85.1% | 98.8% |
| lowrank_dot | 27 | 84.9% | 99.9% |
| trained_linear | 0 | 68.4% | 88.9% |
| trained_linear | 4 | 77.9% | 92.6% |
| trained_linear | 8 | 83.5% | 98.9% |
| trained_linear | 13 | 79.6% | 95.0% |
| trained_linear | 20 | 75.0% | 97.4% |
| trained_linear | 27 | 82.9% | 99.9% |

## 6. Conclusion

This experiment supports the low-rank-selection hypothesis:

```text
The key 2% full-QK selected tokens can be largely recovered from low-rank
projected q-k features.
```

More precise conclusion:

```text
Rank4/8 is informative but insufficient.
Rank16 is already useful.
Rank32 is the first strong operating point.
Rank64 nearly recovers the selected attention mass.
```

So the current model does appear to rely heavily on a low-rank spectral subspace for top2 token selection, but the useful subspace is not always extremely tiny. A practical classifier/probe should sweep ranks and probably start with:

```text
rank16, rank32, rank64
```

## 7. Next Step

The immediate next experiment should turn this from an offline probe into a candidate selection method:

```text
Use rank32 or rank64 projected q-k score to produce candidate tokens,
then full-QK rerank only those candidates,
and compare recall/PPL/runtime against qabs candidate selection.
```

This would connect the representation finding directly to KV-cache acceleration.
