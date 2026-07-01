# Section 33: Target 90% Synthetic Compression Curve Results

Date: 2026-06-29

## 0. Scope

This section records staged experiments for aggressive R2H-KV compression targets:

```text
50%, 70%, 80%, 90% estimated KV compression
```

Script added:

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_r2h_synthetic_target_curve.py
```

Server paths:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/r2h_synthetic_target_curve_codex_0629
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/r2h_synthetic_target_curve_mean_codex_0629
```

Common config:

```text
model = /home/fdong/hrj/prove/Qwen3-0.6B
text = hard_topic_eval_v2.txt
prefill_tokens = 2048
eval_tokens = 64
synthetic_recent = 64
synthetic_prototypes = 8
protect_sink_tokens = 10
```

Layers were selected by the profile-derived layer risk order.

## 1. Important Method Note

Two synthetic methods were tested:

```text
method = mass
method = mean
```

The `mass` method is not a true physical KV compression benchmark in the current implementation.

Reason:

```text
method=mass computes query-dependent chunk scores over the remote tokens inside each synthetic chunk.
It still scans remote K/V at decode time.
```

So `mass` can be interpreted as:

```text
an oracle-ish chunk aggregation quality probe
```

not as:

```text
real stored-KV compression.
```

The `mean` method is closer to real synthetic KV:

```text
remote chunks are represented by mean K/V prototypes.
```

It is still implemented eagerly, but it is the more relevant quality signal for physical compression.

## 2. Mass Synthetic Results

Baseline:

```text
PPL = 5.2152
seconds = 2.1426
```

| Target | Synthetic Layers | Estimated Compression | PPL Ratio | Time Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 50% | 15 | 51.4% | 0.9987 | 2.20 |
| 70% | 21 | 72.0% | 1.0003 | 2.65 |
| 80% | 24 | 82.3% | 1.0004 | 2.87 |
| 90% | 26 | 89.1% | 0.9976 | 3.02 |

Interpretation:

```text
Mass aggregation shows that remote information can be summarized very compactly
if the summary is allowed to be query-dependent.
```

But because it still scans remote tokens, this is not a valid speed or memory compression result.

## 3. Mean Synthetic Results

Baseline:

```text
PPL = 5.2152
seconds = 2.1832
```

| Target | Synthetic Layers | Estimated Compression | PPL Ratio | Time Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 50% | 15 | 51.4% | 1.3751 | 1.33 |
| 70% | 21 | 72.0% | 6.3396 | 1.45 |
| 80% | 24 | 82.3% | 8.7898 | 1.52 |
| 90% | 26 | 89.1% | 9.0624 | 1.55 |

Interpretation:

```text
Naive mean synthetic KV fails badly.
Even 50% compression produces large PPL degradation.
At 70%-90%, PPL collapses.
```

This means:

```text
90% real KV compression is not achievable with simple mean prototypes.
```

## 4. Main Conclusion

The target-90 direction is still possible, but not with naive synthetic KV.

The experiment separates two facts:

1. Query-dependent remote aggregation can preserve quality at very high apparent compression.
2. Query-independent mean prototypes cannot.

Therefore the next method must approximate the useful behavior of `mass` without scanning all remote tokens.

That points to:

```text
learned / fitted synthetic KV prototypes
q-sketch page routing
page-level logsumexp/value summaries
counterfactual risk gate
```

## 5. Updated Plan for 90%

Do not jump directly to 90% mean/synthetic replacement.

Instead:

```text
1. Fit synthetic prototypes from calibration queries.
2. Validate held-out attention output MSE.
3. Only assign SYNTH_M8 to heads with low held-out MSE.
4. Use QSKETCH_PAGE for heads where routing is needed.
5. Keep high-risk heads FULL/HYBRID_SAFE.
```

The next experiment should compare:

```text
mean synthetic
mass oracle aggregation
fitted synthetic K/V
q-sketch routed page summaries
```

on the same staged curve.

