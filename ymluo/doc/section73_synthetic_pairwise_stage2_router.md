# Section 73: Synthetic Pairwise Stage-2 Router 快速实验

## 目标

上一节 two-stage budget router 的主要问题是：

```text
budget prediction 有信号；
stage-2 action selection 太差。
```

本节尝试下一步优化：

```text
two-stage budget router
+ non-benchmark synthetic exact/retrieval data
+ stage-2 pairwise action ranker
+ confidence/safety fallback
```

要求是：训练不使用 LongBench/RULER benchmark labels，只用非 benchmark 文本构造 synthetic exact/retrieval 数据；LongBench/RULER 只做 held-out offline evaluation。

## 代码

新增脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_two_stage_synthetic_pairwise_router.py
```

主要结构：

```text
stage 1:
  BudgetClassifier(features) -> active KV budget bin

stage 2:
  PairwiseActionRanker(features, action, token_ratio) -> action utility

runtime:
  先预测 budget
  在 budget 内用 pairwise ranker 选择 action
  如果低置信或选到高风险动作，触发 fallback
```

训练数据：

```text
War and Peace
Count of Monte Cristo
```

测试数据：

```text
已有 Qwen3-8B LongBench/RULER held-out trials
```

## Tiny Quick Test

配置：

```text
cases_per_dataset = 40
length = 8192
epochs = 300
hidden_dim = 48
token_penalty = 0.25
```

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_two_stage_synth_pairwise_tiny_20260705
```

结果：

| policy | relative to full | token ratio |
|---|---:|---:|
| pred_budget_oracle_action | 93.05% | 12.69% |
| runtime_pairwise | 63.50% | 52.16% |
| runtime_pairwise_fallback | 78.31% | 60.75% |

synthetic 内部：

| split | policy | success | token ratio |
|---|---|---:|---:|
| synthetic_train | runtime_pairwise | 100.00% | 34.47% |
| synthetic_test | runtime_pairwise | 85.71% | 34.90% |

解释：

- 在 synthetic test 上可以工作。
- 迁移到 benchmark 后失败。
- 失败不是 budget，因为 `pred_budget_oracle_action` 仍有 93.05% full / 12.69% token。
- 失败点仍然是 stage-2 action ranker。

## Token Penalty Sweep

### token_penalty = 0.8

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_two_stage_synth_pairwise_tiny_pen08_20260705
```

| policy | relative to full | token ratio |
|---|---:|---:|
| runtime_pairwise | 58.57% | 28.52% |
| runtime_pairwise_fallback | 73.38% | 34.89% |

### token_penalty = 1.2

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_two_stage_synth_pairwise_tiny_pen12_20260705
```

| policy | relative to full | token ratio |
|---|---:|---:|
| runtime_pairwise | 64.50% | 32.22% |
| runtime_pairwise_fallback | 73.38% | 36.73% |

结论：

```text
提高 token penalty 可以压低 full_raw 选择，
但不能解决 action 质量问题。
```

## Length-aware Fallback

### 低置信 fallback

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_two_stage_synth_pairwise_lenawarefb_20260705
```

| policy | relative to full | token ratio |
|---|---:|---:|
| runtime_pairwise_lenaware_fallback | 102.87% | 40.94% |

这个结果和之前的 `length_aware_rule` 基本一致：

```text
length_aware_rule:
  relative = 102.87%
  token ratio = 40.94%
```

进一步分析发现：

```text
fallback changed 128 / 128 = 100%
```

也就是说该版本完全退化成 length-aware rule。

### risk-only fallback

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_two_stage_synth_pairwise_riskonlyfb_20260705
```

| policy | relative to full | token ratio |
|---|---:|---:|
| runtime_pairwise_lenaware_fallback | 97.07% | 54.42% |

质量恢复了，但 token 更高。原因是 LongBench exact 保留了很多 pairwise 选出的高成本/低质量动作。

### risk-only fallback v2

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_two_stage_synth_pairwise_riskonlyfb_v2_20260705
```

| policy | relative to full | token ratio |
|---|---:|---:|
| runtime_pairwise_lenaware_fallback | 97.07% | 53.89% |

仍然不好。LongBench action 分布显示：

```text
runtime_pairwise:
  retrieval_raw_k8: 19
  recent_only: 11
  full_raw: 2

runtime_pairwise_lenaware_fallback:
  retrieval_raw_k8: 19
  retrieval_raw_k1: 8
  recent_only: 4
  full_raw: 1
```

问题是：synthetic ranker 认为 `retrieval_raw_k8` 很稳，但在 LongBench held-out trials 上它反而低分且高成本。

## 当前结论

这次优化没有成功替代 length-aware rule。

最重要的结论：

```text
1. budget prediction 仍然有价值。
   pred_budget_oracle_action = 93.05% full_raw / 12.69% token。

2. synthetic pairwise ranker 可以拟合 synthetic 数据，
   但不能直接泛化到 LongBench/RULER action selection。

3. safety fallback 能恢复质量，
   但目前会退化成 length-aware rule 或引入过高 token ratio。
```

所以当前可用版本仍然是：

```text
length_aware_rule:
  relative = 102.87%
  token ratio = 40.94%
```

而不是新的 pairwise ranker。

## 为什么失败

主要原因不是模型结构，而是 synthetic 数据和 benchmark action 分布不匹配：

```text
synthetic exact:
  retrieval_raw_k 越大通常越稳。

LongBench/RULER held-out:
  更大的 k 不一定更好；
  retrieval_raw_k8 有时更差；
  recent_only / summary / retrieval 的边界依赖任务格式。
```

这说明 stage-2 训练数据需要更接近真实 benchmark 的失败模式，而不是只构造“找到证据就成功”的 synthetic retrieval。

## 下一步建议

下一步应该改 synthetic 数据生成，而不是继续调 ranker hidden_dim：

1. **加入 benchmark-like action failure simulation**

   对 synthetic label 不再只看“是否包含答案”，还要模拟：

   ```text
   过多 retrieved blocks 会稀释证据
   recent_only 对远处证据失败
   summary 对 exact 任务失败
   k8 可能比 k1/k2 更差
   ```

2. **训练 stage-2 预测 action family，而不是具体 action**

   先预测：

   ```text
   recent / retrieval / summary / full
   ```

   再由规则决定 k：

   ```text
   retrieval family -> length-aware k
   summary family -> ratio summary
   ```

3. **保留 length-aware fallback 作为安全下界**

   当前论文实验里可以报告：

   ```text
   learned/synthetic router 还未超过 length-aware rule；
   oracle 说明仍有 93%+ / 13% token 的空间。
   ```

4. **如果短期要写论文结果**

   用 `length_aware_rule` 做主 runtime policy；
   把 `pred_budget_oracle_action` 作为 oracle gap analysis；
   不要把当前 pairwise ranker 当最终方法。

## 简短结论

这一步验证了一个重要事实：

```text
router 的理论空间还在，
但 stage-2 action selector 不能靠简单 synthetic retrieval pairwise 训练解决。
```

要接近 oracle，需要更真实的 synthetic label，尤其要建模“更多 raw block 不一定更好”的真实失败模式。
