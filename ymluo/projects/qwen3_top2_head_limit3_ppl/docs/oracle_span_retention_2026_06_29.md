# Oracle Span-Retention Diagnostic

Date: 2026-06-29

## Goal

Test whether QABS downstream retrieval failures can be recovered by forcing the target evidence span into the final retained KV set.

This is an oracle diagnostic, not a deployable method. It uses the known target key/label/record spans to answer:

> If the evidence span is present in the final retained tokens, does retrieval accuracy recover?

## Run

Output:

`/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/oracle_span_retention_qabs5_shortctx_v1`

Setting:

- Model: `/home/fdong/hrj/prove/Qwen3-0.6B`
- Variants: `compact_kv`, `json_kv`, `topic_table`, `needle_sentence`
- Tasks per variant: 16
- Records per task: 16
- Base sparse mode: `qabs8cand5reuse`
- `top_fraction=0.05`
- `protect_sink_tokens=10`
- `protect_recent_tokens=10`

Compared modes:

- `baseline`: dense attention
- `qabs8cand5reuse`: normal QABS
- `oracle_key_label`: normal QABS plus forced target key span and target label span in final mask
- `oracle_record`: normal QABS plus forced full target record line in final mask

## Results

| Variant | Baseline | qabs5 | Oracle key+label | Oracle record |
|---|---:|---:|---:|---:|
| compact_kv | 15/16 = 93.8% | 11/16 = 68.8% | 14/16 = 87.5% | 14/16 = 87.5% |
| json_kv | 9/16 = 56.3% | 10/16 = 62.5% | 11/16 = 68.8% | 10/16 = 62.5% |
| topic_table | 12/16 = 75.0% | 8/16 = 50.0% | 12/16 = 75.0% | 11/16 = 68.8% |
| needle_sentence | 9/16 = 56.3% | 7/16 = 43.8% | 12/16 = 75.0% | 11/16 = 68.8% |

## Interpretation

The oracle result strongly supports the evidence-retention hypothesis.

- `compact_kv`: qabs loses 4 tasks relative to baseline; forcing key+label recovers 3 of them.
- `topic_table`: qabs loses 4 tasks; forcing key+label fully recovers baseline.
- `needle_sentence`: qabs loses 2 tasks; forcing key+label exceeds dense baseline on this sample.
- `json_kv`: qabs already slightly beats baseline on this sample, and key+label oracle gives another small gain.

The key result is that `oracle_key_label` is consistently as good as or better than `oracle_record`. This means the useful rescue target is likely the compact evidence binding span, not necessarily the entire record.

## Implication

The retrieval loss is not primarily caused by model inability or task noise. It is largely caused by QABS not reliably retaining the key/value evidence span in the final KV set.

This points to a practical next method:

> Evidence-gated span rescue: keep QABS as the default compression path, but detect exact-lookup queries and allocate a tiny span-retention budget to candidate key/value bindings.

The deployable version cannot use oracle target spans. It needs an online proxy:

1. Detect lookup-like queries from query tokens or low-entropy key patterns.
2. Generate candidate evidence spans from key-like tokens, separators, JSON fields, table delimiters, or high key-token saliency.
3. Force a very small number of candidate spans into final KV.
4. Keep total retained KV under 10% by applying rescue only on retrieval-sensitive heads/layers or only when lookup confidence is high.

## Paper Direction

This is a stronger story than uniform qabs:

- PPL-preserving compression: QABS works well at around 6% retained KV.
- Retrieval-preserving compression: QABS needs evidence-aware span rescue.
- Oracle span-retention demonstrates an upper bound and validates that the missing mechanism is evidence retention.

