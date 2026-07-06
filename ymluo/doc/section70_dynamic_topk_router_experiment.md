# Section 70: Dynamic Top-k Router 优化实验

## 目标

本节继续优化 router，重点验证：

```text
固定 retrieval_raw_k2 是否足够？
加入 k3/k4/k8 后，oracle 和 runtime router 能不能更接近 95%+ full_raw / 20%-30% active KV？
```

## 代码改动

新增离线 router policy 评估脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_router_policy_offline_eval.py
```

增强 synthetic router 蒸馏脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_synthetic_router_distillation.py
```

主要变化：

- candidate actions 增加：

```text
retrieval_raw_k3
retrieval_raw_k4
retrieval_raw_k8
```

- synthetic multi-answer / multi-block exact tasks 不再硬编码 k2，而是根据 retrieval 是否真的覆盖答案生成 oracle label。
- 保留多长度训练：

```text
4096, 8192, 16384 tokens
```

## 新增 Benchmark 输出

为了让 held-out benchmark 也有 k3/k4/k8 的真实输出，补跑了 Qwen3-8B adapter benchmark：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_icml_bench_extra_k348_20260705
```

方法：

```text
retrieval_raw_k3
retrieval_raw_k4
retrieval_raw_k8
```

这部分仍然只是 held-out 测试输出，不用于训练 router。

## Dynamic-k Synthetic Router

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_router_synth_dynamic_k_20260705
```

router checkpoint：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_router_synth_dynamic_k_20260705/router.pt
```

Synthetic test：

| split | group | samples | label acc | synthetic success | token ratio |
|---|---|---:|---:|---:|---:|
| test | overall | 358 | 90.22% | 94.97% | 39.07% |
| test | exact | 276 | 87.68% | 93.84% | 46.72% |
| test | generation | 82 | 98.78% | 98.78% | 13.35% |

相比上一版，dynamic-k 训练更难，因为 label space 更大，而且要区分 k1/k2/k3/k4/k8。

## 合并 Held-out Trials 后的 Oracle

合并两个 held-out 输出：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_icml_bench_adapter_router_20260705
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_icml_bench_extra_k348_20260705
```

离线评估输出：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_router_policy_dynamic_k_offline_20260705
```

加入 k3/k4/k8 后，oracle 上界进一步提升：

| policy | score | full_raw score | relative to full | token ratio |
|---|---:|---:|---:|---:|
| oracle_match_full | 0.8381 | 0.7916 | 105.88% | 17.85% |
| oracle_best_under_35pct | 0.7605 | 0.7916 | 96.08% | 12.63% |
| oracle_best_under_50pct | 0.8152 | 0.7916 | 102.99% | 15.12% |

之前 oracle_match_full 是约 `20.84%` active tokens。

现在扩展 dynamic top-k 后：

```text
oracle_match_full token ratio = 17.85%
```

说明 action space 变强了，k3/k4/k8 确实有价值。

## Runtime Router 结果

| policy | score | full_raw score | relative to full | token ratio |
|---|---:|---:|---:|---:|
| learned_router | 0.7596 | 0.7916 | 95.96% | 54.16% |
| learned_router_conservative | 0.8065 | 0.7916 | 101.88% | 49.60% |
| length_aware_rule | 0.8143 | 0.7916 | 102.87% | 40.94% |

观察：

- `learned_router` 达到约 `95.96% full_raw`，但 token ratio 高达 `54.16%`。
- 加 safety 后质量恢复到 full_raw 以上，但 token ratio 仍是 `49.60%`。
- 当前最强实用策略仍然是 `length_aware_rule`：

```text
score = 0.8143
relative = 102.87% full_raw
token ratio = 40.94%
```

## 关键发现

### 1. k3/k4/k8 提升了 oracle，但没有直接提升 runtime router

扩 action space 后，oracle 更接近目标：

```text
105.88% full_raw
17.85% active tokens
```

但 MLP router 学到的是偏保守策略：

```text
95.96% full_raw
54.16% active tokens
```

说明问题不是 action space，而是 router 学习方式。

### 2. 固定 action classification 不适合这个问题

现在 router 直接分类：

```text
summary1_8 / retrieval_raw_k1 / retrieval_raw_k2 / retrieval_raw_k3 / ...
```

这种做法会把质量和成本混在一个 label 里，导致：

- 为了保质量，容易选择过大的 k 或 full_raw。
- 为了省 token，容易误选 summary/recent。
- 很难稳定逼近 oracle 的 cost-quality frontier。

### 3. 需要 two-stage budget-aware router

下一步应该把 router 改成：

```text
stage 1: 是否需要 raw evidence？
stage 2: 如果需要 raw evidence，预测 raw page budget / threshold。
stage 3: 是否需要 full_raw fallback？
```

而不是一个 MLP 直接输出 action。

## 当前最佳策略

当前建议论文实验里的强 runtime policy 用：

```text
length_aware_rule
```

结果：

| group | score | full score | relative | token ratio |
|---|---:|---:|---:|---:|
| overall | 0.8143 | 0.7916 | 102.87% | 40.94% |
| LongBench | 0.3509 | 0.2913 | 120.44% | 24.80% |
| RULER 4096 | 1.0000 | 1.0000 | 100.00% | 77.00% |
| RULER 8192 | 0.9688 | 1.0000 | 96.88% | 40.69% |
| RULER 16384 | 0.9375 | 0.8750 | 107.14% | 21.29% |

这也符合之前的直觉：

```text
短文本保守；
长文本激进。
```

## 下一步

要真正接近 oracle，需要继续做：

1. 用 oracle policy 训练两个 head：

```text
need_raw_evidence
target_budget_ratio / raw_k
```

2. 换成 cost-aware loss：

```text
loss = quality_failure_loss + lambda(context_length) * token_cost
```

3. 对 retrieval score curve 建模：

```text
top1/top2/top4/top8 score
score entropy
positive block count
evidence position spread
recent score
```

4. 不再把 full_raw 当普通 label，而是作为 fallback head。

当前阶段的结论：

```text
dynamic top-k 证明 oracle 更强；
runtime router 仍未接近 oracle；
下一步应该从 action classifier 改为 two-stage budget-aware router。
```

## Length-aware Label 训练补充

进一步把长度感知写进 synthetic oracle label：

```text
<= 5k tokens: target budget 80%
<= 9k tokens: target budget 50%
<= 18k tokens: target budget 30%
> 18k tokens: target budget 22%
```

训练输出：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_router_synth_length_aware_dynamic_k_20260705/router.pt
```

离线评估输出：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_router_policy_length_aware_dynamic_k_offline_20260705
```

结果：

| policy | score | full_raw score | relative to full | token ratio |
|---|---:|---:|---:|---:|
| length-aware learned_router | 0.8065 | 0.7916 | 101.88% | 49.25% |
| length-aware learned_router_conservative | 0.8143 | 0.7916 | 102.87% | 46.82% |
| length_aware_rule | 0.8143 | 0.7916 | 102.87% | 40.94% |
| oracle_match_full | 0.8381 | 0.7916 | 105.88% | 17.85% |

分组结果：

| group | length-aware learned_router relative | token ratio |
|---|---:|---:|
| LongBench | 77.53% | 22.45% |
| RULER 4096 | 100.00% | 86.26% |
| RULER 8192 | 100.00% | 54.41% |
| RULER 16384 | 114.29% | 33.88% |

观察：

- length-aware label 让 pure learned router 的质量从 `95.96%` 提升到 `101.88%` full_raw。
- token ratio 从 `54.16%` 降到 `49.25%`，有改善但仍远高于 oracle。
- conservative 后质量与 rule 持平，但 token ratio `46.82%` 仍高于手写 `length_aware_rule` 的 `40.94%`。

结论：

```text
把长度写进 label 是有效的，主要提升了质量稳定性；
但 action-classification router 仍然过度选择大 k / full_raw，无法接近 17.85% oracle token ratio。
```
