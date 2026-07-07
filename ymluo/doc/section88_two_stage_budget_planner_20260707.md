# Section 88: Two-stage budget planner（中文记录，2026-07-07）

## 当前目标

把 RoPE-aware KV repack 从“固定 top-k compact”推进到一个更像论文方法的策略：

1. 先做 cache-native page/action planning，而不是重建 prompt。
2. 在 `k2_compact / k3_compact / full` 三个动作之间选择。
3. 用风险阈值做保守校准：若 `p(full)` 或 `p(k3_compact)` 足够高才升级预算，否则默认 `k2_compact`。

这仍然是 KV-cache 方法，不是 RAG：候选动作操作的是 full-context prefill 后的 KV pages 和 RoPE-aware repack；`prompt_k2/prompt_k3` 只作为对照，不进入方法动作空间。

## 新增脚本

- `ymluo/projects/learned_hierarchical_summary_memory/src/run_two_stage_budget_planner_from_repack_results.py`
  - 输入成对 benchmark：`k2_dir + k3_dir`
  - 输出 two-stage planner：`k2_compact / k3_compact / full`
  - 特征包含：
    - 当前 router/layout features
    - k2/k3 page layout
    - k2/k3 retriever gap、top-k score、selected page score
    - task family / benchmark family
  - 支持 `best` 和 `safe_vs_full` 标签。

- `ymluo/projects/learned_hierarchical_summary_memory/scripts/run_rope_repack_longbench_m12_20260707.sh`
  - 用于 targeted LongBench worst-case 扩充标签。

## Benchmark 数据

已有全任务小样本：

- `rope_repack_benchmark_qwen8b_13tasks_m4_k2_20260707`
- `rope_repack_benchmark_qwen8b_13tasks_m4_k3_20260707`
- 13 tasks, 每任务 4 条，共 52 cases。

新增 targeted worst-case：

- `rope_repack_benchmark_qwen8b_longbench_m12_k2_20260707`
- `rope_repack_benchmark_qwen8b_longbench_m12_k3_20260707`
- LongBench 5 tasks：`hotpotqa, 2wikimqa, musique, passage_retrieval_en, passage_count`
- 每任务 12 条，共 60 cases。

## 关键原始结果

### 全任务 m=4

`k2`:

- full: 69.23%, KV 100%
- prompt rebuild: 65.38%, KV 26.90%
- RoPE compact: 65.38%, KV 25.70%
- oracle with full: 71.15%, KV 48.88%

`k3`:

- full: 69.23%, KV 100%
- prompt rebuild: 65.38%, KV 39.93%
- RoPE compact: 65.38%, KV 38.71%
- oracle with full: 69.23%, KV 57.82%

结论：普通混合任务上 `k2_compact` 是最划算的默认动作。

### Targeted LongBench m=12

`k2`:

- full: 20.00%, KV 100%
- prompt rebuild: 18.33%, KV 26.72%
- RoPE compact: 8.33%, KV 25.19%
- oracle with full: 25.00%, KV 47.87%

`k3`:

- full: 20.00%, KV 100%
- prompt rebuild: 20.00%, KV 39.33%
- RoPE compact: 16.67%, KV 37.79%
- oracle with full: 28.33%, KV 58.78%

结论：LongBench 多跳/检索是当前最坏区域，`k3_compact` 明显优于 `k2_compact`，但仍需要少量 fallback。

## Two-stage planner 结果

训练输入：

- m=4 全任务 k2/k3 paired results
- LongBench m=12 k2/k3 paired results
- case-level split，避免同一 case 的 k2/k3 泄漏到不同 split。

输出目录：

- `two_stage_budget_planner_qwen8b_m4_plus_longbench12_safe_20260707`
- `two_stage_budget_planner_qwen8b_m4_plus_longbench12_best_20260707`

### Safe label, argmax

测试集 38 paired examples：

- fixed `k2_compact`: 28.95%, KV 25.37%
- fixed `k3_compact`: 26.32%, KV 38.05%
- full: 34.21%, KV 100%
- prompt_k2: 31.58%, KV 26.83%
- prompt_k3: 34.21%, KV 39.54%
- learned planner: 31.58%, KV 25.69%
- oracle: 34.21%, KV 27.67%

这说明 two-stage learned planner 已经超过 fixed `k2_compact`，并且达到 `prompt_k2` 的 score，但 KV 更低。

### Safe label, calibrated threshold

阈值策略：

- 若 `p(full) >= 0.01`，选择 `full`
- 否则若 `p(k3_compact) >= 0.01`，选择 `k3_compact`
- 否则选择 `k2_compact`

阈值 sweep 输出：

- `threshold_sweep.csv`
- `threshold_summary.json`

测试结果：

- calibrated two-stage planner: 34.21%, KV 31.62%
- full: 34.21%, KV 100%
- prompt_k3: 34.21%, KV 39.54%
- oracle: 34.21%, KV 27.67%

这条是当前最好的可部署策略：达到 full/prompt_k3 的测试 score，但 KV 只有约 31.6%，并且不需要 prompt rebuild。

## 当前结论

当前最值得推进成 ICLR 方法的版本是：

**Risk-calibrated two-stage KV budget planner + RoPE-aware KV repack**

默认动作是 `k2_compact`，只在风险概率触发时升级到 `k3_compact` 或 `full`。这个形式比“固定 compact”更有方法感，也比 prompt/RAG 边界更清晰。

## 仍然需要补的点

1. 现在 planner 是基于 replay results 训练和评估，下一步要集成到实际 benchmark runtime。
2. LongBench worst-case 上 full 本身只有 20%，说明部分 case 是模型能力/4k context 限制，不应把所有 0 分都归咎于 cache method。
3. 需要更大的 heldout：
   - LongBench m=20 或 m=50
   - RULER 8k/16k
   - 至少再加 1 个模型或 1 个 context length。
4. 需要把 calibrated policy 固化成脚本参数，而不是只在离线 threshold sweep 中使用。
