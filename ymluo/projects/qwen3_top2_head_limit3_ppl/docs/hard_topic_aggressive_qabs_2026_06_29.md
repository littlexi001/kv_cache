# Hard-Topic Aggressive QABS Results

Date: 2026-06-29

Server: `fdong@10.176.37.31`

Model: `/home/fdong/hrj/prove/Qwen3-0.6B`

Dataset:

`/home/fdong/ymluo/projects/influence_bounded_synthetic_kv/data/hard_topic_eval_v2.txt`

Generator:

`/home/fdong/ymluo/projects/influence_bounded_synthetic_kv/src/build_hard_topic_eval_text.py`

This dataset is a harder topic-style evaluation set with mixed technical/legal/medical/system topics, random identifiers, numeric fields, constraints, and cross-reference bundles. It has about 88k Qwen tokens. The baseline PPL in this run is `4.615`, so it avoids the near-1 PPL problem.

## Strategy Name Mapping

| Name | Meaning |
| --- | --- |
| `qabs8cand2reuse_tf2` | Select top-8 query/key dimensions, retrieve 2% candidate tokens, union three sets, rerank exactly, keep final 2%. |
| `qabs8cand3reuse_tf3` | Select top-8 query/key dimensions, retrieve 3% candidate tokens, union three sets, rerank exactly, keep final 3%. |
| `qabs8cand5reuse_tf5` | Select top-8 query/key dimensions, retrieve 5% candidate tokens, union three sets, rerank exactly, keep final 5%. |

The three sets are:

1. current query candidate set,
2. previous query candidate set,
3. previous query final selected set.

All runs also keep:

- sink tokens: 10
- recent tokens: 10
- self token: enabled

## Compression Estimate

With `prefill=2048`, the approximate retained token fraction per head/layer is:

| Strategy | Final selected | Protected + self | Approx retained KV |
| --- | ---: | ---: | ---: |
| `qabs8cand2reuse_tf2` | 2% | about 21 tokens | about 3% |
| `qabs8cand3reuse_tf3` | 3% | about 21 tokens | about 4% |
| `qabs8cand5reuse_tf5` | 5% | about 21 tokens | about 6% |

These are all below the requested 10% average KV-retention target. The current evaluator still keeps full KV in memory for measurement convenience, so these are algorithmic budget estimates rather than actual measured memory savings.

## Results

Config:

- prefill tokens: 2048
- eval tokens: 256
- dtype: float16
- final top fraction: matched to candidate fraction
- candidate selection: top-k on 8 query dimensions
- reuse: current candidate + previous candidate + previous final

| Mode | PPL | PPL ratio | Approx retained KV |
| --- | ---: | ---: | ---: |
| baseline | 4.615 | 1.000x | 100% |
| `qabs8cand2reuse_tf2` | 5.135 | 1.113x | about 3% |
| `qabs8cand3reuse_tf3` | 4.936 | 1.070x | about 4% |
| `qabs8cand5reuse_tf5` | 4.720 | 1.023x | about 6% |

Output directories:

- `/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/hard_topic_v2_qabs8cand2reuse_tf2_p2048_e256`
- `/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/hard_topic_v2_qabs8cand3reuse_tf3_p2048_e256`
- `/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/hard_topic_v2_qabs8cand5reuse_tf5_p2048_e256`

## Takeaway

The aggressive setting is viable at the 5% final budget: it stays within a roughly 6% KV-retention budget and only increases PPL by about 2.3% on the harder topic data.

The 2% and 3% budgets are more aggressive but have noticeably larger quality loss. For paper-facing experiments, `qabs8cand5reuse_tf5` is the best current operating point under the 10% average budget requirement.

Next useful step: run the same 5% budget on multiple hard-topic seeds and then add a layer/head gate that uses 5% QABS for sensitive layers and 2-3% QABS for safer layers, keeping the average below 10%.
