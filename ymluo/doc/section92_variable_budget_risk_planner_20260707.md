# Section 92: 风险约束可变 KV 预算规划器（2026-07-07）

## 当前判断

如果只写 `two-stage calibrated`，创新点偏薄：它更像一个有效的三动作 router。

现在更适合作为主方法的是：

**risk-constrained variable-budget KV planner + RoPE-aware KV repack**

核心变化是把动作空间从 `k2/k3/full` 扩展到：

`k1_compact / k2_compact / k3_compact / k4_compact / k6_compact / k8_compact / full`

planner 的目标不再是固定压缩，而是学习“在风险可控下的最小安全 KV 预算”。这比 two-stage 更有方法感，也更容易和 RAG/prompt rebuild 区分：主方法只操作 full-context prefill 后的 KV pages，并做 RoPE-aware repack。

## 新增实现

脚本：

- `ymluo/projects/learned_hierarchical_summary_memory/src/run_variable_budget_planner_from_repack_results.py`

新增能力：

- 支持 tail-risk 校准策略：
  - 对动作概率按预算排序；
  - 选择最小动作，使“需要更大预算”的尾部概率不超过阈值；
  - 输出 `risk_threshold_sweep.csv` 和 `risk_threshold_summary.json`。
- 支持 `--holdout_tasks` / `--holdout_benchmarks`，用于 leave-task-out 泛化评估。
- 支持 `--benchmark_groups`，可以把 m4 混合任务和 LongBench m12 worst-case 按 group 独立取交集后合并训练。

汇总脚本：

- `ymluo/projects/learned_hierarchical_summary_memory/src/summarize_variable_budget_multiseed.py`

## 关键结果

### 1. 13-task m4 小集合，多 seed

输出目录：

- `/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/variable_budget_multiseed_qwen8b_m4_k1k2k3k4k6k8_20260707`

8 个 seed，best-label：

| policy | score | KV ratio | full-level runs |
|---|---:|---:|---:|
| fixed full / k8 | 64.42% | 100.00% | 8/8 |
| fixed k1 | 56.73% | 12.37% | 0/8 |
| fixed k2 | 64.42% | 25.39% | 8/8 |
| learned planner | 64.42% | 14.14% | 8/8 |
| oracle best | 64.42% | 13.40% | 8/8 |

结论：在随机 case split 上，variable-budget planner 稳定达到 full-level，同时把 KV 从 100% 降到约 14%。

### 2. 13-task m4，leave-task-out

输出目录：

- `/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/variable_budget_loto_qwen8b_m4_k1k2k3k4k6k8_best_20260707`

| policy | score | KV ratio | full-level tasks |
|---|---:|---:|---:|
| fixed full / k8 | 69.23% | 100.00% | 13/13 |
| learned argmax | 55.77% | 19.21% | 8/13 |
| tail-risk calibrated | 69.23% | 39.13% | 13/13 |
| oracle min-safe | 69.23% | 16.14% | 13/13 |
| oracle best | 71.15% | 17.34% | 13/13 |

结论：跨任务泛化时，argmax 不够安全；tail-risk 校准是必要组件。oracle 和 calibrated 之间还有很大 KV gap，这是下一步优化空间。

### 3. LongBench m12 worst-case

新补齐预算：

- k1: `/home/fdong/.../rope_repack_benchmark_qwen8b_longbench_m12_k1_20260707`
- k4: `/home/fdong/.../rope_repack_benchmark_qwen8b_longbench_m12_k4_20260707`
- k6: `/home/fdong/.../rope_repack_benchmark_qwen8b_longbench_m12_k6_20260707`
- k8: `/home/fdong/.../rope_repack_benchmark_qwen8b_longbench_m12_k8_20260707`

原始 RoPE compact：

| budget | score | KV ratio |
|---|---:|---:|
| full / k8 | 20.00% | 100.00% |
| k1 | 5.00% | 12.60% |
| k4 | 15.00% | 50.24% |
| k6 | 20.00% | 75.23% |

LongBench-only best-label planner：

| policy | score | KV ratio |
|---|---:|---:|
| fixed full | 15.79% | 100.00% |
| learned planner | 15.79% | 25.66% |
| oracle best | 21.05% | 17.11% |

结论：worst-case 上固定低预算不够，但 learned planner 已能用约 25.7% KV 达到 full split；oracle 说明还有显著提升空间。

### 4. m4 + LongBench m12 combined

单 seed 输出：

- `/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/variable_budget_planner_qwen8b_m4_plus_longbench12_k1k2k3k4k6k8_best_20260707`

| policy | score | KV ratio |
|---|---:|---:|
| fixed full / k8 | 34.29% | 100.00% |
| fixed k2 | 31.43% | 25.40% |
| fixed k6 | 34.29% | 76.19% |
| learned planner | 37.14% | 17.40% |
| oracle min-safe | 37.14% | 14.51% |
| oracle best | 42.86% | 18.08% |

5 seed combined best-label：

| policy | score mean | KV mean | full-level runs |
|---|---:|---:|---:|
| fixed full / k8 | 33.06% | 100.00% | 5/5 |
| fixed k1 | 24.50% | 12.66% | 0/5 |
| fixed k2 | 27.35% | 25.35% | 0/5 |
| learned planner | 33.05% | 22.85% | 4/5 |
| tail-risk calibrated | 33.62% | 34.78% | 5/5 |
| oracle min-safe | 34.25% | 15.21% | 5/5 |
| oracle best | 37.55% | 17.28% | 5/5 |

结论：combined 场景下，variable-budget 已经明显强于 fixed budget。argmax 省 KV 但还不够稳；tail-risk 版本更安全但 KV 偏高；oracle gap 说明值得继续做风险学习和校准。

## 对创新性的判断

现在的创新点比 two-stage 强很多，已经有一个可写成论文主方法的形状：

1. **Cache-native variable action space**：动作不是检索文本，而是选择 KV page budget。
2. **Risk-constrained budget planning**：不是固定压缩率，而是按风险升预算。
3. **RoPE-aware KV repack**：解决 KV page 重新排列后的位置信息问题。
4. **Oracle / worst-case distillation**：用 targeted benchmark 生成最小安全动作标签。
5. **Runtime-compatible path**：已有 two-stage runtime，variable-budget 还需接入。

但是现在还不能说“已经足够 ICML”。主要缺口：

- 数据规模仍小，尤其 combined 多 seed 只有 112 examples。
- variable-budget 还没有接入真实 runtime，只是 replay planner。
- tail-risk 校准能保 full-level，但 KV 从 oracle 的 15%-17% 升到约 35%-39%，说明风险模型还有明显误报。
- 端到端加速还没有在 variable-budget 上兑现；当前 runtime 加速证据主要来自 two-stage 和子系统 microbenchmark。

## 下一步

优先级从高到低：

1. 做 cost-sensitive / risk-aware loss，让 planner 直接优化“score 不低于 full，同时 KV 最小”，缩小 calibrated 和 oracle 的 KV gap。
2. 把 variable-budget planner 接入 `run_rope_aware_kv_repack_benchmark.py` 的真实 runtime。
3. 扩大 worst-case：LongBench m=30/m=50，至少补多 seed。
4. 做 8k/16k 的 memory-safe runtime，解决 16k full prefill OOM。
5. 加强论文 baseline：至少和固定 top-k、prompt rebuild、two-stage、oracle、以及典型 KV eviction/page selection baseline 对比。

当前最适合写进论文的主张：

**在混合任务 + LongBench worst-case 的 replay evaluation 中，risk-constrained variable-budget KV planner 能用约 23%-35% KV 达到或略超过 full-level 精度；oracle 显示约 15%-17% KV 即可达到同等或更高精度，说明该方向有明确上限和进一步优化空间。**
