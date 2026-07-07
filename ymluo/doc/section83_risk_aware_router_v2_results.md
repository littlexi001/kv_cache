# Section 83: risk-aware router v2 训练结果

日期：2026-07-06

## 运行状态

已经在服务器跑完。

服务器：

```text
fdong@10.176.37.31
```

使用环境：

```text
/home/fdong/miniconda3/envs/moe/bin/python
torch 2.4.0
transformers 4.53.0
```

## 数据

训练输入：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_recent_plus_bench_m4_parallel_20260706/merged
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_worstcase_targeted_20260706/longbench_exact
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_worstcase_targeted_20260706/ruler4_8_hard_v2
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_worstcase_targeted_20260706/ruler16_hard_v2
```

注意：第一版我把 exact / retrieval 的安全阈值错误设成了 `score >= 1.0`。这不符合之前 planner 的定义。已经修正为：

```text
exact/retrieval: score >= full_raw_score
summary: score >= full_raw_score - 0.03
```

下面只看修正后的结果。

## 版本 A：full action space

输出目录：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_risk_aware_router_v2_fullthreshold_20260706
```

候选动作包含：

```text
full_raw
recent_only
static_hier
summary1_8 / summary1_4 / summary1_2
retrieval_raw_k1/k2/k3/k4/k8
recent_plus_*
```

test 结果：

| policy | success | token ratio | relative |
|---|---:|---:|---:|
| oracle_min_safe | 100.00% | 14.94% | 103.89% |
| risk_filtered_cheapest | 83.08% | 15.10% | 82.83% |
| safe_action_classifier | 87.69% | 15.67% | 88.56% |
| safe_classifier_then_risk_filter | 89.23% | 17.58% | 90.48% |
| full_raw | 100.00% | 100.00% | 100.00% |

主要问题：

```text
min_safe_action 里 recent_only 占 79/216。
router 被便宜的 recent_only/static/summary1_8 标签污染，test 时会过度选择这些动作。
```

因此 full action space 版不能作为最终 router。

## 版本 B：core action space

输出目录：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_risk_aware_router_v2_coreactions_20260706
```

候选动作限制为：

```text
full_raw
summary1_4
retrieval_raw_k1/k2/k3
recent_plus_summary1_4
recent_plus_retrieval_raw_k1/k2/k3
```

test 结果：

| policy | success | token ratio | relative |
|---|---:|---:|---:|
| oracle_min_safe | 100.00% | 26.25% | 105.65% |
| risk_filtered_cheapest | 90.77% | 32.13% | 94.37% |
| safe_action_classifier | 89.23% | 28.69% | 92.50% |
| safe_classifier_then_risk_filter | 92.31% | 34.04% | 96.25% |
| full_raw | 100.00% | 100.00% | 100.00% |

按 group 看 `safe_classifier_then_risk_filter`：

| group | success | token ratio | relative |
|---|---:|---:|---:|
| LongBench | 100.00% | 43.14% | 100.10% |
| RULER 16k | 94.44% | 21.30% | 113.33% |
| RULER 4k | 81.25% | 43.45% | 81.25% |
| RULER 8k | 94.12% | 31.17% | 94.12% |

结论：

```text
core action space 明显比 full action space 更健康。
但是 learned router 仍然没有达到 paper 主结果标准。
最弱点是 RULER 4k exact case，说明 risk classifier 对“什么时候不能用便宜 retrieval/summary”的边界还没学稳。
```

## threshold sweep

我额外跑了 `risk_threshold=0.10` 和 `risk_threshold=0.50`。

输出目录：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_risk_aware_router_v2_coreactions_thr010_20260706
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_risk_aware_router_v2_coreactions_thr050_20260706
```

`safe_classifier_then_risk_filter`：

| threshold | success | token ratio | relative |
|---:|---:|---:|---:|
| 0.10 | 92.31% | 34.14% | 96.25% |
| 0.25 | 92.31% | 34.04% | 96.25% |
| 0.50 | 92.31% | 32.57% | 96.25% |

阈值变化没有修复 success，说明现在主要瓶颈不是 risk threshold，而是：

```text
1. 标签太偏向“最便宜安全动作”，没有编码 worst-case robust margin；
2. exact task 的细粒度失败边界样本还太少；
3. 需要把 k1/k2/k3 的选择改成 risk-aware ladder，而不是直接多分类预测一个 action。
```

## 当前结论

这次训练完成了两个有用产物：

```text
case_labels.csv
action_labels.csv
risk_router_v2.pt
prediction_summary.csv
```

但是目前 learned router v2 还不能作为最终 paper 方法。当前最好的 learned 版本是：

```text
qwen8b_risk_aware_router_v2_coreactions_thr050_20260706
success = 92.31%
token ratio = 32.57%
relative_to_full = 96.25%
```

oracle 仍然很强：

```text
success = 100.00%
token ratio = 26.25%
relative_to_full = 105.65%
```

所以方法上界还在，问题是 router 没蒸馏好。

下一步应该改成：

```text
risk-aware retrieval ladder:
先判断 summary1_4 是否安全；
不安全则判断 retrieval_k1；
k1 不安全则升 k2；
k2 不安全则升 k3；
k3 仍不安全才 full_raw。
```

并且 label 不再只取“最便宜安全动作”，而是加入 worst-case margin：

```text
如果同一 task/family/context-length 下 k1 有失败样本，则训练时把 k1 标成高风险；
如果 k2 在 16k single/multi evidence 有失败样本，则允许升 k3；
LongBench multi-hop 单独保留 summary1_4 分支。
```
