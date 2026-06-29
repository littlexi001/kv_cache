# Influence-Gated Hybrid Budget Experiment

Date: 2026-06-29

Server: `fdong@10.176.37.31`

Model: `/home/fdong/hrj/prove/Qwen3-0.6B`

Dataset: `/home/fdong/ymluo/projects/influence_bounded_synthetic_kv/data/topic_stress_eval.txt`

## Interpretation Of The Voice Input

The intended old strategy is interpreted as `qabs8cand3reuse`:

- select top-8 query dimensions per head;
- use those dimensions to retrieve a 3% candidate token set;
- combine three sets: current candidate, previous candidate, previous final selected set;
- rerank the union with exact scores and keep the final token budget.

For this experiment, both candidate fraction and final `top_fraction` were set to `0.03`.

## Implementation

Modified:

`src/evaluate_qwen3_top2_head_limit3_ppl.py`

Added layer-budget types:

- `qabs8cand3reuse`: per-layer QABS three-set reuse.
- `headmix_qabs_reuse`: selected heads use full attention, remaining heads use QABS three-set reuse.

Added experiment runner:

`src/run_influence_gated_hybrid_budget.py`

The runner loads the model once, reuses a shared prefill cache, writes layer-budget JSON maps, and evaluates multiple layer/head policies on the same topic-stress window.

## Standalone QABS Check

Output:

`/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/topic_stress_qabs8cand3reuse_tf3_v1`

Config:

- prefill: 1024
- eval: 128
- candidate fraction: 3%
- final top fraction: 3%
- protected sink/recent: 10/10

| Mode | PPL | Ratio |
| --- | ---: | ---: |
| baseline | 3.601 | 1.000x |
| qabs8cand3reuse | 3.762 | 1.045x |

Standalone QABS is reasonably close in quality but slower in the current Python/eager implementation. This makes it a candidate for selective layer/head use, not whole-model use.

## Hybrid Layer/Head Results

Output:

`/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/topic_stress_influence_gated_hybrid_p1024_e128_v1`

Config:

- prefill: 1024
- eval: 128
- protected sink/recent: 10/10
- landmark fallback: recent 512 + stride 64
- QABS: 8 dims, 3% candidate, 3% final
- headmix: 8 full heads + QABS on remaining heads

| Mode | Layers | Budget | PPL | Ratio |
| --- | --- | --- | ---: | ---: |
| baseline | none | full | 3.601 | 1.000x |
| pcic_0_6_landmark | 0,6 | landmark | 3.626 | 1.0069x |
| pcic_0_6_qabs3set | 0,6 | qabs | 3.644 | 1.0118x |
| pcic_0_6_headmix8 | 0,6 | headmix | 3.613 | 1.0034x |
| pcic_0_13_landmark | 0,13 | landmark | 3.612 | 1.0031x |
| pcic_0_13_qabs3set | 0,13 | qabs | 3.611 | 1.0028x |
| pcic_0_13_headmix8 | 0,13 | headmix | 3.615 | 1.0037x |
| safe_4_5_landmark | 4,5 | landmark | 3.603 | 1.0006x |
| safe_4_5_qabs3set | 4,5 | qabs | 3.605 | 1.0010x |
| safe_4_5_headmix8 | 4,5 | headmix | 3.601 | 0.9999x |
| auto_1_2_5_landmark | 1,2,5 | landmark | 3.596 | 0.9984x |
| auto_1_2_5_qabs3set | 1,2,5 | qabs | 3.709 | 1.0298x |
| auto_1_2_5_headmix8 | 1,2,5 | headmix | 3.672 | 1.0197x |
| mid_7_14_landmark | 7-14 | landmark | 3.683 | 1.0227x |
| mid_7_14_qabs3set | 7-14 | qabs | 3.601 | 0.9998x |
| mid_7_14_headmix8 | 7-14 | headmix | 3.621 | 1.0055x |

## Takeaways

1. The user's hybrid intuition is supported: uniform whole-model QABS loses about 4.5% PPL, while selective layer/head budgets can be nearly lossless.
2. The best larger compression candidate is `mid_7_14_qabs3set`: 8 layers use QABS three-set token selection with essentially no PPL loss on this topic-stress window.
3. Layer choice matters more than the compression primitive. `1,2,5` is good with landmark but bad with QABS, so the gate must choose both layer and strategy.
4. Headmix is useful as a stabilizer for some layer sets, e.g. `0,6`, but not universally better than pure QABS.
5. Current wall-clock speed is not meaningful as a final system result because the implementation is Python/eager and still keeps full KV cache. The result should be read as a quality and budget-policy experiment.

## Next Step

The next useful experiment is a strategy-search gate over `(layer, head, method)`:

- candidate methods: full, landmark, qabs3set, headmix-qabs;
- objective: minimize calibration loss gap under a target compressed-KV budget;
- validate on held-out topic-stress eval tokens.

This is a stronger paper direction than synthetic KV alone: compression is selected by measured layer/head influence and strategy compatibility.
