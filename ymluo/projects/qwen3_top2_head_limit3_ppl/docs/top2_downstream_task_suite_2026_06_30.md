# Top2 downstream task suite, 2026-06-30

## Purpose

This experiment tests the downstream task performance of the `top2` attention-retention mode.

Important caveat: this `top2` mode is not a cheap deployable method by itself. It uses the current query's full attention scores to decide which historical tokens to retain, so it is closer to an oracle/upper-bound retention test than to QABS-style approximate candidate selection.

## Setup

- Server: `fdong@10.176.37.31`
- Project: `/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl`
- Model: `/home/fdong/hrj/prove/Qwen3-0.6B`
- Output: `/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/downstream_task_suite_top2_shortctx_v1`
- Script: `scripts/run_downstream_top2_task_suite_server.sh`
- GPU: `CUDA_VISIBLE_DEVICES=5`
- Modes: `baseline,top2`
- `top_fraction=0.02`
- `protect_sink_tokens=10`
- `protect_recent_tokens=10`
- Task variants: `structured_noisy`, `compact_kv`, `natural_kv`, `json_kv`, `needle_sentence`, `topic_table`
- Tasks per variant: 32
- Records per task: 16

## Results

| Variant | Baseline | top2@2% | Delta |
|---|---:|---:|---:|
| structured_noisy | 19/32 = 59.4% | 15/32 = 46.9% | -12.5 pts |
| compact_kv | 29/32 = 90.6% | 27/32 = 84.4% | -6.3 pts |
| natural_kv | 18/32 = 56.3% | 22/32 = 68.8% | +12.5 pts |
| json_kv | 27/32 = 84.4% | 26/32 = 81.3% | -3.1 pts |
| needle_sentence | 18/32 = 56.3% | 22/32 = 68.8% | +12.5 pts |
| topic_table | 22/32 = 68.8% | 25/32 = 78.1% | +9.4 pts |
| **Total** | **133/192 = 69.3%** | **137/192 = 71.4%** | **+2.1 pts** |

For reference, the earlier short-context QABS results on the same 6-way suite were much lower overall:

| Method | Total |
|---|---:|
| baseline | 133/192 = 69.3% |
| qabs8cand5reuse, top_fraction=0.05 | 104/192 = 54.2% |
| qabs8cand5reuse, top_fraction=0.08 | 102/192 = 53.1% |
| top2, top_fraction=0.02 | 137/192 = 71.4% |

## Interpretation

The result is useful because `top2@2%` keeps a very small token budget but does not show the same downstream collapse as QABS. This suggests that the failure of QABS downstream retrieval is not simply caused by retaining too few tokens; the bigger issue is that the approximate query-channel candidate selection misses evidence-bearing tokens.

The mixed per-task behavior also matters:

- `structured_noisy` still drops substantially, so top attention mass alone is not a universal retrieval-preserving objective.
- `compact_kv` and `json_kv` are close to baseline, unlike QABS, which dropped heavily on these formats.
- `natural_kv`, `needle_sentence`, and `topic_table` improve over baseline in this small sample; this should be treated as variance/task-scoring sensitivity, not as proof that compression improves reasoning.

## Next implication

This supports using `top2` as an oracle reference for retrieval-preserving compression:

1. Use `top2` or oracle span retention to identify which evidence tokens must survive.
2. Train or calibrate cheaper QABS/evidence-gated rules to approximate those retained evidence tokens.
3. Evaluate the deployable method against both PPL and downstream exact retrieval, because PPL-preserving and retrieval-preserving compression are different objectives.
