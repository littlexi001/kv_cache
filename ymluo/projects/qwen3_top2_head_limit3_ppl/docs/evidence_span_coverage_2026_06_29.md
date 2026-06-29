# Evidence Span Coverage Diagnostic

Date: 2026-06-29

## Goal

Test whether downstream retrieval loss comes from QABS failing to retain the actual evidence span.

The diagnostic locates three spans in each context:

- `key`: target lookup key
- `label`: target answer label token
- `record`: full target record line

Then it records whether QABS masks cover those spans:

- `current`: current query-channel candidate
- `union`: current candidate + previous candidate + previous final
- `final`: exact-reranked final retained tokens

Coverage is measured over all profiled layer/head/query decisions. `any` means at least one token in the span is retained; `all` means the full span is retained.

## Run

Output:

`/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/evidence_span_coverage_qabs5_shortctx_v2`

Setting:

- Model: `/home/fdong/hrj/prove/Qwen3-0.6B`
- Variants: `compact_kv`, `json_kv`, `topic_table`, `needle_sentence`
- Tasks per variant: 16
- Records per task: 16
- Mode: `qabs8cand5reuse`
- `top_fraction=0.05`
- `protect_sink_tokens=10`
- `protect_recent_tokens=10`

## Accuracy

| Variant | Baseline | qabs5 |
|---|---:|---:|
| compact_kv | 15/16 = 93.8% | 11/16 = 68.8% |
| json_kv | 9/16 = 56.3% | 10/16 = 62.5% |
| needle_sentence | 9/16 = 56.3% | 7/16 = 43.8% |
| topic_table | 12/16 = 75.0% | 8/16 = 50.0% |

## Overall Coverage

| Variant | Final key any | Final key all | Final label any | Final record any | Union key any | Union label any | Union record any |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact_kv | 23.5% | 0.1% | 10.5% | 31.4% | 40.5% | 18.3% | 50.9% |
| json_kv | 18.4% | 0.0% | 9.1% | 47.4% | 34.0% | 15.8% | 70.6% |
| needle_sentence | 19.2% | 0.0% | 14.5% | 63.7% | 32.0% | 21.0% | 79.4% |
| topic_table | 20.4% | 0.0% | 7.8% | 53.4% | 33.6% | 13.2% | 72.3% |

## Correct vs Incorrect

Final key/label `any` coverage:

| Variant | Wrong key | Correct key | Wrong label | Correct label |
|---|---:|---:|---:|---:|
| compact_kv | 18.3% | 25.9% | 10.2% | 10.6% |
| json_kv | 19.0% | 18.1% | 8.0% | 9.8% |
| needle_sentence | 20.3% | 17.8% | 16.8% | 11.5% |
| topic_table | 15.9% | 25.0% | 4.7% | 10.9% |

## Interpretation

The evidence coverage is low. Even after three-set reuse and exact rerank, the final retained set usually contains only a small part of the target evidence:

- Key `any` coverage is only about 18-24%.
- Full key-span coverage is almost zero.
- Label coverage is about 8-15%.
- Record `any` coverage can be higher, but full record coverage is essentially zero.

This supports the hypothesis that retrieval loss is not just a prompt artifact. The QABS final retained tokens often do not preserve the key/value evidence span.

However, correctness is not explained by this metric alone. For `compact_kv` and `topic_table`, correct samples have noticeably higher final key/label coverage. For `json_kv` and `needle_sentence`, the correlation is weak or reversed, which suggests additional factors:

- some heads/layers may need different evidence tokens than the literal span;
- the model can sometimes answer from partial evidence;
- the current coverage aggregation averages over all query tokens, layers, and heads, so it may dilute a small number of decisive attention decisions.

## Best Layer/Head Signals

The strongest final key-any coverage heads for `compact_kv` were:

| Layer | Head | Coverage |
|---:|---:|---:|
| 3 | 10 | 58.0% |
| 21 | 8 | 54.7% |
| 6 | 7 | 54.2% |
| 16 | 14 | 53.2% |
| 11 | 13 | 51.5% |
| 6 | 11 | 51.5% |
| 1 | 15 | 51.5% |
| 16 | 15 | 51.2% |

This is a better basis for influence-gated hybrid selection than hand-picking whole layers.

## Next Step

Run an oracle span-retention variant:

- keep normal `qabs8cand5reuse`;
- additionally force the target key+label or full target record span into the final retained set;
- compare accuracy to baseline and qabs.

If oracle span retention recovers `compact_kv` and `topic_table`, then the next method should be an evidence-gated span rescue mechanism, not uniform budget increase.

