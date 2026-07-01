# Section 38: Lowrank-Dot Sparse Attention PPL and Downstream Smoke

Date: 2026-06-30

## 0. Goal

After Section 36 showed that low-rank projected q-k features recover true top2 tokens offline, this experiment tests whether the same idea can be used as a sparse attention method:

```text
lrsvd32attn
lrsvd64attn
```

Question:

```text
Do rank32 / rank64 lowrank-dot token selectors preserve PPL on different topic texts?
Do they preserve downstream synthetic KV retrieval behavior?
What is the prototype speed?
```

New files:

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_lowrank_dot_ppl_downstream.py
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_lowrank_dot_ppl_downstream_server.sh
```

## 1. Method

For each decode query, layer, and attention head:

1. Take historical K.
2. Compute SVD of centered historical K.
3. Project q and K to the first `r` singular directions.
4. Compute:

```text
lowrank_score = sum_i (q_proj_i * k_proj_i)
```

5. Select top 2% historical tokens by lowrank score.
6. Always keep the current self token.
7. Compute exact full scores only over selected tokens.
8. Run softmax over selected tokens and reduce V.

Modes:

```text
lrsvd32attn: r = 32
lrsvd64attn: r = 64
```

Important speed note:

```text
This is a Python/SVD prototype.
It recomputes SVD per decode query/layer/head.
The measured speed is not representative of a cached/fused implementation.
```

## 2. Data

Because the server bandwidth is limited, the topic texts were copied from the local repo to:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/data/topic_texts
```

Topic files:

```text
finance
history
literature
science
software
mixed_qa
```

## 3. PPL Run

Server output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/lowrank_dot_topic_ppl_0630_v1
```

Local copy:

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/lowrank_dot_topic_ppl_0630_v1
```

Config:

```text
prefill_tokens = 512
eval_tokens = 32
top_fraction = 0.02
modes = baseline,lrsvd32attn,lrsvd64attn
```

The short eval length was chosen because the current Python/SVD prototype is very slow.

## 4. PPL Results

| Topic | Mode | PPL | PPL ratio | Eval seconds | Time ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| finance | baseline | 1.0007 | 1.000 | 1.1 | 1.0x |
| finance | lrsvd32attn | 1.5365 | 1.535 | 63.2 | 56.6x |
| finance | lrsvd64attn | 1.0129 | 1.012 | 63.2 | 56.6x |
| history | baseline | 1.0096 | 1.000 | 1.0 | 1.0x |
| history | lrsvd32attn | 1.1732 | 1.162 | 75.7 | 73.2x |
| history | lrsvd64attn | 1.0280 | 1.018 | 84.5 | 81.7x |
| literature | baseline | 1.0005 | 1.000 | 1.0 | 1.0x |
| literature | lrsvd32attn | 1.2515 | 1.251 | 84.1 | 80.2x |
| literature | lrsvd64attn | 1.0153 | 1.015 | 84.3 | 80.3x |
| science | baseline | 1.0157 | 1.000 | 1.1 | 1.0x |
| science | lrsvd32attn | 1.0776 | 1.061 | 84.3 | 80.1x |
| science | lrsvd64attn | 1.0571 | 1.041 | 84.4 | 80.2x |
| software | baseline | 1.0007 | 1.000 | 1.0 | 1.0x |
| software | lrsvd32attn | 1.1385 | 1.138 | 71.6 | 68.3x |
| software | lrsvd64attn | 1.0212 | 1.020 | 63.2 | 60.2x |
| mixed_qa | baseline | 1.0014 | 1.000 | 1.0 | 1.0x |
| mixed_qa | lrsvd32attn | 1.1696 | 1.168 | 67.2 | 64.9x |
| mixed_qa | lrsvd64attn | 1.0182 | 1.017 | 84.0 | 81.2x |

Aggregate:

| Mode | Mean PPL ratio | Max PPL ratio | Mean time ratio |
| --- | ---: | ---: | ---: |
| lrsvd32attn | 1.219 | 1.535 | 70.5x |
| lrsvd64attn | 1.021 | 1.041 | 73.4x |

Interpretation:

```text
rank64 is qualitatively strong on short topic PPL.
rank32 is not reliable enough as direct sparse attention.
```

The best quality signal is:

```text
lrsvd64attn stays within about +1% to +4% PPL ratio on all six topic texts.
```

But the current speed is unusable:

```text
about 56x-82x slower than dense baseline.
```

This is expected for the prototype because it recomputes SVD per layer/head/query in Python.

## 5. Downstream Smoke

Server output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/lowrank_dot_downstream_small_0630_v1
```

Config:

```text
variants = compact_kv,json_kv,needle_sentence,topic_table
tasks_per_variant = 2
records_per_task = 8
modes = baseline,lrsvd32attn,lrsvd64attn
```

Runtime:

```text
876.5s for only 24 scored rows
```

This is only a smoke test because the prototype is too slow for a larger downstream run.

| Variant | Mode | Accuracy | Delta vs baseline |
| --- | --- | ---: | ---: |
| compact_kv | baseline | 2/2 | 0 |
| compact_kv | lrsvd32attn | 0/2 | -2 |
| compact_kv | lrsvd64attn | 2/2 | 0 |
| json_kv | baseline | 2/2 | 0 |
| json_kv | lrsvd32attn | 2/2 | 0 |
| json_kv | lrsvd64attn | 1/2 | -1 |
| needle_sentence | baseline | 2/2 | 0 |
| needle_sentence | lrsvd32attn | 1/2 | -1 |
| needle_sentence | lrsvd64attn | 2/2 | 0 |
| topic_table | baseline | 2/2 | 0 |
| topic_table | lrsvd32attn | 2/2 | 0 |
| topic_table | lrsvd64attn | 2/2 | 0 |

Interpretation:

```text
The downstream sample is too small for a firm accuracy conclusion.
rank64 mostly preserves behavior, but json_kv has one failure.
rank32 is visibly less stable.
```

## 6. Conclusion

The direct sparse-attention test agrees with the offline classifier/probe:

```text
rank64 is the better operating point.
rank32 is useful for analysis but not reliable enough as direct sparse attention.
```

Quality:

```text
lrsvd64attn has strong short-topic PPL preservation.
```

Speed:

```text
The current implementation is far too slow because it recomputes SVD online.
```

Practical next step:

```text
Do not use per-step online SVD.
Instead, estimate V_r from a calibration window or prefill K cache,
cache V_r per layer/KV-head,
then use projected q-k scoring during decode.
```

That next version should test:

```text
calibrated-lrsvd64cand{2%,5%,8%}
full-QK rerank inside candidate set
PPL and downstream at larger sample sizes
```

This would keep the rank64 quality signal while avoiding the online SVD bottleneck.
