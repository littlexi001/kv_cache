# Section 78：统一 KV Memory + Two-Stage Planner 实验结果

日期：2026-07-06

## 主要服务器输出路径

- 旧动作 + recent-plus 动作合并后的 two-stage planner：
  `/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_combined_full_two_stage_20260706_v2_taskfallback`
- 只使用 recent-plus 动作的 two-stage planner：
  `/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_recent_plus_full_two_stage_20260706_v1`
- 大规模 synthetic pairwise router：
  `/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_two_stage_synth_pairwise_big_20260706`
- Qwen8B unified KV smoke，two-hop：
  `/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/unified_kv_memory_smoke_20260706_gpu0_8b_twohop_7k_p512`
- Qwen8B unified KV smoke，decoy：
  `/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/unified_kv_memory_smoke_20260706_gpu1_8b_decoy_6k`

## 关键结果

### 合并 action space 的 full planner

这是目前最强的结果，因为它同时从旧动作和 recent-plus 动作里选择。

| 策略 | 相对 full 质量 | token 比例 | success |
| --- | ---: | ---: | ---: |
| `runtime_ranker` | 0.9718 | 0.1797 | 0.9302 |
| `ranker_or_rule_gap_fallback` | 1.0285 | 0.4159 | 0.9767 |
| `ranker_or_task_rule_gap_fallback` | 1.0569 | 0.2337 | 1.0000 |
| `oracle_budget_oracle_action` | 1.0570 | 0.1545 | 1.0000 |

解释：合并后的 action space 确实有明显上限。`ranker_or_task_rule_gap_fallback` 的质量几乎追平 oracle，同时只用约 23.4% token。纯 learned ranker 仍然会在少数 exact case 上过于激进，导致质量掉下来。

### 只用 recent-plus 的 planner

| 策略 | 相对 full 质量 | token 比例 | success |
| --- | ---: | ---: | ---: |
| `runtime_ranker` | 0.8832 | 0.2222 | 0.8636 |
| `ranker_or_rule_gap_fallback` | 1.0300 | 0.4057 | 0.9773 |
| `oracle_budget_oracle_action` | 1.0887 | 0.2421 | 1.0000 |

解释：recent-plus 本身是有用的，但不如完整合并 action space。原因是完整合并后可以在合适场景选择更便宜的旧动作，例如 `recent_only`、`static_hier`、`summary1_8`。

### 大规模 synthetic pairwise router

| 切分/策略 | 相对 full 质量 | token 比例 |
| --- | ---: | ---: |
| synthetic test `runtime_pairwise` | 0.9489 | 0.4048 |
| synthetic test `runtime_pairwise_fallback` | 0.9711 | 0.4147 |
| heldout `runtime_pairwise` | 0.7030 | 0.3387 |
| heldout `runtime_pairwise_lenaware_fallback` | 0.9892 | 0.4141 |

解释：synthetic 训练可以扩展，也能拟合 synthetic held-out split；但不加 fallback 时，迁移到真实 LongBench/RULER 仍然弱。下一版 router 应该从 benchmark-calibrated task rule 里蒸馏，或者单独训练 risk/confidence model，而不是继续只堆 synthetic pairwise 数据。

### Qwen8B unified KV smoke

Two-hop 7k，page size 512：

| 方法 | active KV | NLL | exact |
| --- | ---: | ---: | --- |
| full KV | 7000 | 3.3649 | true |
| arbitrary sparse KV | 1536 | 5.8286 | false |
| prefix span KV | 6144 | 3.2890 | true |
| typed summary KV | 43 | 1.4402 | true |
| typed summary + span KV | 2158 | 3.8755 | false |

Decoy 6k：

| 方法 | active KV | NLL | exact |
| --- | ---: | ---: | --- |
| full KV | 6000 | 1.8924 | false |
| arbitrary sparse KV | 1024 | 5.6389 | false |
| prefix span KV | 6000 | 1.8924 | false |
| typed summary KV | 41 | 0.7432 | true |
| typed summary + span KV | 985 | 4.5923 | false |

解释：typed summary KV 是目前最稳定的主记忆路径。arbitrary non-contiguous KV 和 naive summary+span KV 仍然不安全。prefix/raw span 可以作为 fallback，但不适合作为主方法。

## 当前技术结论

目前最适合写成论文主线的方向是：

1. 用 typed summary KV 作为主要压缩记忆。
2. raw/retrieval/recent span 只作为受控 fallback。
3. 使用 two-stage planner：
   - stage 1 预测 budget/risk；
   - stage 2 选择 resolution/action；
   - 加 task/risk fallback，避免 exact task 失败。
4. router 应该从 combined planner 的行为里训练或蒸馏，而不是只依赖 synthetic pairwise 数据。

目前最关键的剩余问题是 router 泛化。action space 已经足够强，但当前 learned ranker 不加 fallback 还不可靠。
