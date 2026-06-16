# First Training Experiment

## 1. Falsifiable Question

在真实 DCLM 数据上，head-level shared KV/expert bucket 能否仅靠 NTP：

1. 保持有限且持续下降的 training loss；
2. 避免所有 token 坍缩到同一 expert；
3. 把每层 attention candidate ratio 降到 full causal attention 的明显子集；
4. 让 router 从 NTP 获得非零梯度。

本轮只检验训练可行性，不声称已经节省物理 KV cache。

## 2. Fixed Setup

| Parameter | Value | Reason |
|---|---:|---|
| base config | Qwen3-0.6B | fits below the 2B limit and matches existing training stack |
| experts/buckets per head | 4 | first low-risk compression setting |
| router top-k | 1 | bucket id and expert id remain identical |
| ordinary expert intermediate size | 3072 | ordinary expert is `1024 -> 3072 -> 1024` |
| derived head expert intermediate size | 512 | shared expert is `64 -> 512 -> 1024`, derived by equal-parameter matching |
| sequence length | 1024 | existing DCLM training setting |
| loss | NTP only | isolates whether the structure learns without auxiliary objectives |

## 3. Minimal Runs

### Run 0: Equal-budget ordinary MoE

```text
architecture = ordinary_moe
attention = full causal attention
experts per layer = 4
expert shape = 1024 -> 3072 -> 1024
router input = post-attention residual
```

The full model has about `1.388888B` parameters. The shared-bucket model has about `1.389003B`, so this is the primary size-matched baseline.

### Run A: Main structure

```text
router_input = k
center_router_input = true
router_normalization = l2
local_window = 32
sink_tokens = 4
```

### Run B: Centering ablation

Same as A, except:

```text
center_router_input = false
```

### Run C: Router representation ablation

Run `layer_input`, `q`, and `v` after A passes the basic training checks.

## 4. Recorded Evidence

Each step is written to `run_dir/train_metrics.jsonl`:

| Metric | Meaning |
|---|---|
| `loss`, `perplexity` | NTP quality |
| `next_token_accuracy` | teacher-forced token accuracy |
| `gradient_norm` | overall optimization health |
| `candidate_ratio` | allowed attention pairs / full causal pairs |
| `router_max_probability` | routing confidence |
| `router_margin` | top-1 minus top-2 router probability |
| `router_token_entropy` | mean per-token routing entropy, normalized |
| `router_load_entropy` | aggregate expert-load entropy, normalized |
| `effective_experts` | `exp(load entropy)` |
| `max/min_expert_load` | collapse and dead-expert evidence |

## 5. Pass, Fail, and Insufficient Evidence

Pass for the implementation stage:

1. smoke test shows nonzero router gradient；
2. no NaN/Inf in loss or routing metrics；
3. all four experts remain active in most layers during early training；
4. candidate ratio is materially below 1；
5. loss decreases over a meaningful training interval。

Fail:

1. router gradient remains zero；
2. attention rows become fully masked；
3. one expert load approaches 1 and remains there；
4. loss does not improve relative to initialization；
5. model exceeds the 2B hard parameter limit。

Insufficient evidence:

1. only a few debug steps are available；
2. training loss decreases but no full-attention/dense control exists；
3. logical candidate ratio decreases but physical KV bytes and latency are not measured。

## 6. Required Next Controls

After the main run is stable, compare under the same data, token budget, and optimizer:

1. standard Qwen3 dense attention + dense FFN；
2. bucket attention + dense FFN；
3. full attention + head experts；
4. shared bucket attention + head experts；
5. separate attention bucket and expert routing IDs。
