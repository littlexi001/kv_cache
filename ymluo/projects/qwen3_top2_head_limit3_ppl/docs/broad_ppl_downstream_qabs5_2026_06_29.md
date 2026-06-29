# Broad PPL and Downstream Plan for QABS Three-Set Reuse

Date: 2026-06-29

## Method under test

`qabs8cand5reuse_tf5` means:

- Query-side saliency: for each head, use the top-8 absolute query channels to estimate token importance.
- Candidate budget: keep the top 5% candidate tokens from the fast query-channel score.
- Three-set reuse: union of current candidate tokens, previous-step candidate tokens, and previous-step final selected tokens.
- Exact rerank: recompute exact attention scores inside the union set and keep the final top 5%.
- Protected tokens: always keep 10 sink tokens and 10 most recent tokens.
- Decode setting: dense prefill, compressed decode attention.
- Approximate retained remote KV under `prefill_tokens=2048`: about 6% average retained tokens after protected sinks/recent tokens are included.

This is the current strongest operating point under the target of average KV retention below 10%.

## Broad PPL results

Model: `/home/fdong/hrj/prove/Qwen3-0.6B`

Evaluation setting:

- `prefill_tokens=2048`
- `eval_tokens=256`
- `dtype=float16`
- `attn_implementation=eager`
- Modes: `qabs8cand5reuse`, `baseline`

| Dataset | Baseline PPL | qabs8cand5reuse PPL | Ratio |
|---|---:|---:|---:|
| hard_topic_eval_v2 | 4.6147 | 4.7197 | 1.0228 |
| hard_topic_eval_v3 | 4.4129 | 4.5117 | 1.0224 |
| hard_topic_eval_v4 | 4.1257 | 4.2665 | 1.0341 |
| topic_stress_eval | 2.5155 | 2.6497 | 1.0534 |
| War and Peace | 34.2606 | 34.0770 | 0.9946 |
| Count of Monte Cristo | 32.1917 | 33.2323 | 1.0323 |

Mean PPL ratio across the six datasets is about `1.0266`, or a 2.7% relative PPL increase.

## Interpretation

The result is better than the earlier fully synthetic KV direction. On topic-like synthetic data, the method usually costs about 2-5% relative PPL. On public-book data, baseline PPL is around 32-34 rather than near 1, and the method is roughly neutral to mildly worse; War and Peace is slightly better than dense baseline in this short evaluation window.

The main positive signal is that the method reaches an aggressive effective KV budget, about 6%, while keeping broad PPL degradation modest. The main limitation is that this implementation is still a quality evaluator: it does not physically free KV memory or use a production sparse KV kernel, so speed and memory numbers should not be presented as realized system gains yet.

## Downstream evaluation results

Added script:

`/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_qabs_downstream_kv_retrieval.py`

Task design:

- Build long-context key-value retrieval prompts.
- Each task contains many structured records: key, topic, metric, checksum, and `ANSWER_LABEL`.
- The query asks for the single label corresponding to a target key.
- Score labels A/B/C/D by next-token likelihood rather than free-form generation.
- Compare `baseline` against `qabs8cand5reuse`.

The first attempted `48 x 160` and `32 x 96` configurations were stopped because they were too slow for repeated iteration on RTX 3090. The script was then optimized to prefill each task once and clone the dense context cache for baseline and compressed decode.

Completed downstream setting:

- `tasks=32`
- `records_per_task=64`
- Context length per task: about 3.7k tokens
- `chunk_size=256`
- Same seed for the 5% and 8% runs

| Run | top_fraction | Approx retained KV | Baseline accuracy | qabs accuracy | Delta |
|---|---:|---:|---:|---:|---:|
| `downstream_kv_retrieval_qabs8cand5_tf5_v4` | 0.05 | about 6% | 17/32 = 53.1% | 14/32 = 43.8% | -9.4 points |
| `downstream_kv_retrieval_qabs8cand8_tf8_v5` | 0.08 | about 9% | 17/32 = 53.1% | 14/32 = 43.8% | -9.4 points |

For the 5% run, overlap was:

- Both baseline and qabs correct: 12 tasks
- Baseline only correct: 5 tasks
- qabs only correct: 2 tasks
- Neither correct: 13 tasks

This suggests the method has a good broad-PPL profile, but the current query-channel selection is still weak for exact remote evidence retrieval. Raising `top_fraction` from 5% to 8% did not recover the lost downstream tasks under this small benchmark, so the next improvement should not be a simple uniform budget increase.

The more promising next variant is an influence-gated hybrid:

- Use qabs three-set reuse for ordinary heads/layers.
- Detect retrieval-sensitive heads/layers by calibration on key-value retrieval or by attention entropy/mass concentration.
- Assign full or larger-token budgets only to those heads/layers.
- Keep the average retained KV below 10% by compensating with more aggressive qabs budgets on robust heads/layers.

Reference run command for the completed 8% downstream test:

```bash
source /home/fdong/miniconda3/bin/activate moe
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -u src/run_qabs_downstream_kv_retrieval.py \
    --model_name_or_path /home/fdong/hrj/prove/Qwen3-0.6B \
    --output_dir /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/downstream_kv_retrieval_qabs8cand8_tf8_v5 \
    --tasks 32 \
    --records_per_task 64 \
    --chunk_size 256 \
    --dtype float16 \
    --device cuda \
    --device_map auto \
    --attn_implementation eager \
    --top_fraction 0.08 \
    --protect_sink_tokens 10 \
    --protect_recent_tokens 10 \
    --log_every 4
```
