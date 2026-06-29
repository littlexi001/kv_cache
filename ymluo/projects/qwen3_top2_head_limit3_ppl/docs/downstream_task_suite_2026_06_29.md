# Multi-Task Downstream Suite for QABS KV Compression

Date: 2026-06-29

## Purpose

The previous key-value retrieval benchmark showed a large drop for `qabs8cand5reuse`, but dense baseline was also weak. This suite tests whether the downstream loss is tied to the task format.

Method under test:

- `qabs8cand5reuse`
- query top-8 channels
- candidate fraction 5%
- three-set reuse: current candidate + previous candidate + previous final
- exact rerank
- protected sink/recent tokens: 10/10

## Task variants

All tasks use A/B/C/D label scoring by next-token likelihood.

- `structured_noisy`: original noisy record format with topic, metric, checksum, and `ANSWER_LABEL`.
- `compact_kv`: compact `KEY => LABEL` rows.
- `natural_kv`: natural-language sentence per record.
- `json_kv`: JSON-like object per record.
- `needle_sentence`: filler sentence plus a needle fact per record.
- `topic_table`: table-like rows with topic/id/class/checksum.

## Long-ish context suite

Setting:

- 16 tasks per variant
- 64 records per task
- `top_fraction=0.05`
- Output: `/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/downstream_task_suite_qabs5_v1`

| Variant | Baseline | qabs5 | Delta |
|---|---:|---:|---:|
| structured_noisy | 6/16 = 37.5% | 1/16 = 6.3% | -31.3 pts |
| compact_kv | 5/16 = 31.3% | 6/16 = 37.5% | +6.3 pts |
| natural_kv | 6/16 = 37.5% | 5/16 = 31.3% | -6.3 pts |
| json_kv | 5/16 = 31.3% | 5/16 = 31.3% | 0.0 pts |
| needle_sentence | 8/16 = 50.0% | 8/16 = 50.0% | 0.0 pts |
| topic_table | 11/16 = 68.8% | 8/16 = 50.0% | -18.8 pts |

Interpretation: most 64-record variants are too hard for Qwen3-0.6B; dense baseline is often near random. These are useful stress tests, but not clean quality benchmarks.

## Short context suite

Setting:

- 32 tasks per variant
- 16 records per task
- Output 5%: `/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/downstream_task_suite_qabs5_shortctx_v2`
- Output 8%: `/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/downstream_task_suite_qabs8_shortctx_v3`

| Variant | Baseline | qabs5 | Delta qabs5 | qabs8 | Delta qabs8 |
|---|---:|---:|---:|---:|---:|
| structured_noisy | 19/32 = 59.4% | 15/32 = 46.9% | -12.5 pts | 12/32 = 37.5% | -21.9 pts |
| compact_kv | 29/32 = 90.6% | 18/32 = 56.3% | -34.4 pts | 21/32 = 65.6% | -25.0 pts |
| natural_kv | 18/32 = 56.3% | 13/32 = 40.6% | -15.6 pts | 15/32 = 46.9% | -9.4 pts |
| json_kv | 27/32 = 84.4% | 19/32 = 59.4% | -25.0 pts | 19/32 = 59.4% | -25.0 pts |
| needle_sentence | 18/32 = 56.3% | 17/32 = 53.1% | -3.1 pts | 17/32 = 53.1% | -3.1 pts |
| topic_table | 22/32 = 68.8% | 22/32 = 68.8% | 0.0 pts | 18/32 = 56.3% | -12.5 pts |

## Findings

The task format matters a lot.

- `topic_table` at 5% is the best case: no downstream drop.
- `needle_sentence` is also relatively robust: only 1/32 drop.
- `compact_kv` and `json_kv` are the clearest failure cases: dense baseline is strong, but qabs loses 8-11 tasks.
- Increasing uniform `top_fraction` from 5% to 8% is not reliably helpful. It improves compact/natural but hurts structured/topic and does not change json/needle.

The previous downstream loss is not only a data artifact. Some task formats expose a real retrieval failure, especially exact key-value mapping where the model must bind an arbitrary key to a short label.

## Implication

The method is still promising for PPL and for some downstream formats, but retrieval-preserving compression needs a different mechanism than uniform qabs token selection.

The next useful target is evidence-aware calibration:

1. For each downstream task, record target key span and target answer span.
2. Measure whether qabs candidate/final masks cover those spans.
3. Run an oracle span-retention variant.
4. If oracle recovers compact/json performance, build an evidence-gated hybrid that retains local spans only when the query appears to request exact lookup.

