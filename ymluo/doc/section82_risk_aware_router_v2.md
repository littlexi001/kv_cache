# Section 82: risk-aware router v2 训练入口

日期：2026-07-06

## 目标

这一步不是再手写一个 fallback rule，而是把 combined benchmark 和 worst-case targeted benchmark 里的 oracle 行为蒸馏成可训练 router。

新增脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_risk_aware_router_v2.py
```

它同时训练两个模型：

1. `risk_model`：输入当前 router features、retriever gap/top-k stability、task/task family、候选 action 特征，输出当前 action 是否危险。
2. `safe_action_model`：输入当前 case 的 router features、retriever stability、task/task family，输出最小安全动作。

## 标签定义

exact / retrieval 任务：

```text
safety_threshold = full_raw_score
```

summary 任务：

```text
safety_threshold = full_raw_score - summary_rouge_slack
默认 summary_rouge_slack = 0.03
```

action 级标签：

```text
dangerous = score < safety_threshold
```

case 级标签：

```text
min_safe_action = 所有 non-dangerous action 中 token ratio 最小的 action
```

如果一个 case 没有任何安全 action，脚本会写：

```text
has_safe_action = 0
min_safe_action = full_raw 或当前最高分 fallback
```

这样不会把 full_raw 也失败的样本误标成“便宜动作安全”。

## 特征

基础特征复用现有 `router_features`，包括 query 关键词、长度、prefix/recent/block token、retriever top1/top2/top3 overlap、score gap、positive blocks、top block 位置等。

额外增加的 risk-aware 特征：

```text
retriever_top2_over_top1
retriever_top3_over_top1
retriever_gap_over_top1
retriever_positive_block_density
retriever_top1_top2_position_distance
retriever_no_gap
retriever_has_two_positive_blocks
retriever_has_three_positive_blocks
task one-hot
benchmark one-hot
task_family one-hot
action one-hot
action type meta
action token ratio
```

这正对应当前要解决的失败模式：`k1/k2/k3` 什么时候不稳、summary1_4 什么时候反而比 retrieval 更稳、task rule 什么时候会过拟合。

## 输入目录

脚本输入应该是 benchmark 原始输出目录，目录里必须有：

```text
summary.json
trials.csv
```

不要传 `planner_eval_v1` 这种 offline planner 输出目录，因为那里没有完整 trial 表。

推荐先用这几组数据训练：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_recent_plus_bench_m4_parallel_20260706/merged
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_worstcase_targeted_20260706/longbench_exact
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_worstcase_targeted_20260706/ruler4_8_hard_v2
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_worstcase_targeted_20260706/ruler16_hard_v2
```

如果还要把旧的 non-recent-plus combined action space 加进去，可以把之前 combined benchmark 的原始 benchmark shard 目录一并追加到 `--benchmark_output_dirs`，前提仍然是每个目录都有 `summary.json` 和 `trials.csv`。

## 服务器运行命令

```bash
cd /home/fdong
python ymluo/projects/learned_hierarchical_summary_memory/src/run_risk_aware_router_v2.py \
  --benchmark_output_dirs /home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_recent_plus_bench_m4_parallel_20260706/merged,/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_worstcase_targeted_20260706/longbench_exact,/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_worstcase_targeted_20260706/ruler4_8_hard_v2,/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_worstcase_targeted_20260706/ruler16_hard_v2 \
  --output_dir /home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_risk_aware_router_v2_20260706 \
  --epochs 1200 \
  --hidden_dim 128 \
  --risk_threshold 0.25
```

## 输出

```text
risk_router_v2.pt
case_labels.csv
action_labels.csv
predictions.csv
prediction_summary.csv
safe_action_history.csv
risk_history.csv
summary.json
```

最重要先看：

```text
prediction_summary.csv
```

重点比较 test split：

```text
oracle_min_safe
safe_action_classifier
risk_filtered_cheapest
safe_classifier_then_risk_filter
full_raw
```

理想目标：

```text
worst-case success >= 98%
relative_to_full >= 1.00
token ratio <= 25%-30%
```

如果 `risk_filtered_cheapest` 成功率高但 token 偏高，说明 risk threshold 太保守；如果 token 很低但 success 掉，说明 risk threshold 太激进，或者 worst-case 标签还不够。
