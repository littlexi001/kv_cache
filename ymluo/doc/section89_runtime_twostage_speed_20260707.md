# Section 89: Runtime two-stage planner speed test（中文记录，2026-07-07）

## 本轮目标

把上一节的离线 replay planner 接到真实 runtime，并同时报告：

1. planner 子系统开销；
2. KV page materialization / RoPE repack / query / decode 分项开销；
3. warm-cache online speedup；
4. one-shot end-to-end speedup；
5. 多 query 复用 full-context cache 时的 amortized end-to-end speedup。

## 新增 runtime 方法

脚本：

- `ymluo/projects/learned_hierarchical_summary_memory/src/run_rope_aware_kv_repack_benchmark.py`

新增参数：

- `--two_stage_planner_path`
- `--two_stage_threshold_full`
- `--two_stage_threshold_k3`

新增方法行：

- `two_stage_calibrated_kv_planner`

运行时动作：

- `full`
- `k2_compact`
- `k3_compact`

阈值策略：

- 若 `p(full) >= 0.01`，选择 `full`
- 否则若 `p(k3_compact) >= 0.01`，选择 `k3_compact`
- 否则选择 `k2_compact`

## 真实 runtime：Qwen3-8B, 4k, 13 tasks, m=4

输出目录：

- `/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/rope_repack_runtime_twostage_qwen8b_m4_k2_20260707`

整体结果：

| method | score | KV ratio | online speedup | one-shot e2e | amortized16 e2e |
|---|---:|---:|---:|---:|---:|
| full_kv_cache | 69.23 | 100.00 | 1.000 | 1.000 | 1.000 |
| prompt_rebuild_selected_pages | 65.38 | 26.90 | 0.888 | 1.379 | 0.919 |
| RoPE compact k2 | 65.38 | 25.70 | 1.003 | 1.002 | 1.003 |
| two-stage calibrated | 69.23 | 28.96 | 0.997 | 0.998 | 0.998 |

动作分布：

- `k2_compact`: 49 / 52
- `k3_compact`: 1 / 52
- `full`: 2 / 52

分组：

| group | method | score | KV ratio | online speedup |
|---|---|---:|---:|---:|
| LongBench | full | 20.00 | 100.00 | 1.000 |
| LongBench | RoPE compact k2 | 10.00 | 25.00 | 1.008 |
| LongBench | two-stage | 20.00 | 33.12 | 1.001 |
| RULER 4096 | full | 100.00 | 100.00 | 1.000 |
| RULER 4096 | RoPE compact k2 | 100.00 | 26.17 | 1.001 |
| RULER 4096 | two-stage | 100.00 | 26.17 | 0.996 |

分项平均耗时：

| method | planner | repack | query | decode |
|---|---:|---:|---:|---:|
| full | 0.00 ms | 0.00 ms | 54.08 ms | 1930.13 ms |
| RoPE compact k2 | 0.00 ms | 8.52 ms | 45.75 ms | 1923.23 ms |
| two-stage | 5.58 ms | 8.22 ms | 46.43 ms | 1928.99 ms |

解释：

- two-stage 已经把 score 拉到 full 水平，同时 KV 只有 28.96%。
- 但 4k/HF/SDPA 下真实 online 没有加速，原因是 planner + repack 开销约 13.8 ms，而 query attention 只省约 7.6 ms；decode 的逐 token Python/HF 路径约 1.93 s，是主要瓶颈。

## 真实 runtime：Qwen3-8B, RULER 8k, m=1

输出目录：

- `/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/rope_repack_runtime_twostage_qwen8b_ruler8k_m1_k2_20260707`

整体结果：

| method | score | KV ratio | online speedup | one-shot e2e | amortized16 e2e |
|---|---:|---:|---:|---:|---:|
| full_kv_cache | 100.00 | 100.00 | 1.000 | 1.000 | 1.000 |
| prompt_rebuild_selected_pages | 100.00 | 13.65 | 0.937 | 1.923 | 0.999 |
| RoPE compact k2 | 100.00 | 13.14 | 1.053 | 1.025 | 1.049 |
| two-stage calibrated | 100.00 | 13.14 | 1.046 | 1.022 | 1.043 |

解释：

- 到 8k 后，真实 runtime 开始出现正向 online/e2e speedup。
- two-stage 与 compact k2 接近，因为 RULER 8k 全部选择 `k2_compact`。
- one-shot e2e 只有 1.022x，因为 cache-native 方法仍需 full-context prefill；多 query 复用时收益更合理。

## 16k 真实 runtime 状态

尝试运行：

- `/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/rope_repack_runtime_twostage_qwen8b_ruler16k_m1_k2_20260707`

结果：

- 第 1 个样本完成；
- 第 2 个样本 full prefill 阶段 OOM；
- 24GB 3090 + HF/SDPA 对 Qwen3-8B 16k 连续运行不稳定。

结论：

- 16k 端到端需要更好的 serving/runtime 支持，例如 chunked prefill、FlashAttention/分页 KV kernel、显存复用控制。

## 子系统 microbenchmark

脚本：

- `ymluo/projects/learned_hierarchical_summary_memory/src/run_attention_paging_amortized_timing.py`

### 4k full KV

输出目录：

- `/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/attention_paging_twostage_subsystem_qwen8b_4k_20260707`

| active KV | steps=1 | steps=16 | steps=64 | steps=256 |
|---|---:|---:|---:|---:|
| k2, 1024 KV | 0.57x | 1.47x | 1.60x | 1.62x |
| k3, 1536 KV | 0.44x | 1.39x | 1.58x | 1.62x |

### 16k full KV

输出目录：

- `/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/attention_paging_twostage_subsystem_qwen8b_16k_20260707`

| active KV | steps=1 | steps=16 | steps=64 | steps=256 |
|---|---:|---:|---:|---:|
| k2, 1024 KV | 1.13x | 4.49x | 5.31x | 5.58x |
| k3, 1536 KV | 0.81x | 4.09x | 5.16x | 5.48x |

子系统解释：

- page materialization overhead 在 4k 为约 4.0 ms / 6.0 ms，在 16k 为约 8.1 ms / 12.4 ms。
- 单 token 时 overhead 占比高，未必加速。
- decode token 数足够多、上下文足够长时，attention/paging 子系统收益明显。

## 当前论文叙述建议

可以主张：

1. 方法不是 RAG：runtime 操作 full-context prefill 后的 KV pages，并做 RoPE-aware repack。
2. two-stage calibrated planner 能在真实 runtime 里把精度恢复到 full，同时保持低 KV ratio。
3. 子系统 attention/paging 在长上下文上有 4-5x 级别加速。
4. 当前 HF/Python 端到端只在 8k 开始有 1.04x 左右正收益；4k 不加速。

不能夸大：

- 不能说当前 Python/HF prototype 已经有大端到端加速。
- 大端到端加速需要 kernel/serving 实现，把 page selection、KV materialization 和 decode attention 放进高效 runtime。

下一步：

1. 做 chunked/full prefill memory-safe 版本，支撑 16k/32k runtime。
2. 做 cached multi-query benchmark：同一 full-context cache 下 4/16/64 个 queries。
3. 设计 kernel-facing API：`select_pages -> gather/repack -> decode_attention`，把当前 Python overhead 移到 CUDA/serving 层。
