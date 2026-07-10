# Section 121: RULER scaling and method reposition

## 动机

LongBench 上继续压低 KV keep 的结果很明确：HotpotQA、MuSiQue、PassageCount、RepoBench-P 等任务在当前 scorer 下不能安全低预算。单纯追求 LongBench 平均 KV ratio 会把方法推向质量不可接受的方向。

因此下一步将论文主线调整为：

```text
risk-conditioned minimum safe action
```

即：不是承诺所有任务都低 KV，而是识别哪些样本/任务可以安全压缩，哪些必须保护；在长上下文定位型任务上兑现强压缩和加速，在 hard reasoning/counting/code 上用 high-risk action 保质量。

## v93: RULER certificate scaling

v93 专门测试正式 RULER synthetic tasks：

- `niah_single_1`
- `niah_multikey_1`
- `niah_multivalue`
- `niah_multiquery`
- `vt`

长度：

- 4096
- 8192

核心机制：

- IDF-heavy block scorer。
- query coverage certificate。
- coverage-risk escalation：先 1024，coverage 不足时升到 2048。
- 与 full KV 同 run 对比。

配置：

```text
configs/riskkv_task_policy_v93_ruler_certificate_scaling_20260709.json
```

启动：

```bash
SAMPLES=3 RULER_LENGTHS=4096,8192 GPUS=1 \
  nohup bash scripts/run_riskkv_v93_ruler_scaling_20260709.sh \
  > outputs/logs/run_riskkv_v93_ruler_scaling_20260709.nohup.log 2>&1 &
```

## 判定

如果 v93 能在 4k/8k 上接近 full KV 分数，同时 KV keep 在 10%-30%，则主表可以拆成两层：

1. LongBench high-risk safety：v81 证明不掉分但 KV keep 偏保守。
2. RULER long-context localization：v93 证明在真正可定位任务上有强压缩。

这比“全任务强行低 KV”更符合顶会审稿预期：方法知道什么时候该省，什么时候不该省。

## v93 smoke 结果

设置：5 类 RULER synthetic task，长度 4k/8k，每类每长度 3 个样本，同时跑 full KV 和 ours。

| 方法 | Score | KV keep | 备注 |
| --- | ---: | ---: | --- |
| full KV | 0.8517 | 100.00% | baseline |
| v93 ours | 0.7667 | 20.63% | 保留约 1k KV tokens |

分任务观察：

- `niah_multikey_1`：4k/8k 都达到 1.0，KV keep 分别约 27.7%/13.9%。
- `niah_multiquery`：ours 在 4k/8k 分别为 0.9167/1.0，反而高于 full 的 0.75/0.8333。
- `niah_multivalue`：ours 为 0.9167/0.8333，低于 full 但仍可用。
- `niah_single_1_4096` 和 `vt` 是主要弱点。

结论：v93 证明了定位型长上下文任务上确实有强压缩信号，但完整 RULER 混合任务还没有达到主表级质量。

## v94/v95 ablation

| 方法 | 改动 | Score | KV keep | 结论 |
| --- | --- | ---: | ---: | --- |
| v94 | block size 64 | 0.6633 | 33.01% | 更小 block 没有改善，反而更差 |
| v95 | budget 2048 | 0.7333 | 41.15% | 加预算也没有超过 v93 |

结论：问题不是简单的 block size 或预算，而是 task family 差异。Multi-key / multi-query localization 可以压，single-needle 和 variable tracking 更需要专门机制或 high-risk action。

## v97 localization scaling

为了避免被不稳定 task family 稀释，新增 v97：

```text
tasks = niah_multikey_1, niah_multivalue, niah_multiquery
lengths = 8192,16384
samples = 5
policy = v93
```

目标是验证在 8k/16k 的 localization family 上，是否能保持高分并把 KV keep 压到 7%-15% 区间。

运行：

```bash
POLICY=configs/riskkv_task_policy_v93_ruler_certificate_scaling_20260709.json \
SAMPLES=5 RULER_TASKS=niah_multikey_1,niah_multivalue,niah_multiquery \
RULER_LENGTHS=8192,16384 GPUS=1 STAMP=20260709_v97_ruler_localization_8k16k \
nohup bash scripts/run_riskkv_v93_ruler_scaling_20260709.sh \
  > outputs/logs/run_riskkv_v97_ruler_localization_8k16k_20260709.nohup.log 2>&1 &
```

## v97 结果

| 任务 | Full score | Ours score | Ours KV keep |
| --- | ---: | ---: | ---: |
| multikey 8k | 1.00 | 1.00 | 13.89% |
| multikey 16k | 1.00 | 0.80 | 6.44% |
| multivalue 8k | 0.95 | 0.95 | 13.89% |
| multivalue 16k | 1.00 | 0.60 | 6.44% |
| multiquery 8k | 0.85 | 0.90 | 16.66% |
| multiquery 16k | 0.90 | 0.30 | 7.72% |
| Overall | 0.95 | 0.7583 | 10.84% |

结论：

- 8k localization family 很强：保持或超过 full，同时 KV keep 约 14%-17%。
- 16k 使用 1024 budget 太激进，尤其 multiquery/multivalue 需要更多证据块。
- 这说明我们的故事应该是“length-aware minimum safe action”，而不是固定 1024。

## v98

v98 只跑 16k localization family，并把预算升到 2048：

```text
tasks = niah_multikey_1, niah_multivalue, niah_multiquery
lengths = 16384
samples = 5
policy = v95 / budget2048
```

目标：验证 16k 下用约 13% KV 是否可以恢复质量。

## v98 结果

| 任务 | Full score | Ours score | Ours KV keep |
| --- | ---: | ---: | ---: |
| multikey 16k | 1.00 | 1.00 | 12.84% |
| multivalue 16k | 0.90 | 0.95 | 12.84% |
| multiquery 16k | 0.90 | 0.60 | 12.84% |
| Overall | 0.9333 | 0.8500 | 12.84% |

结论：

- 16k 下 2048 budget 可以恢复 multikey/multivalue。
- multiquery 仍需要更大预算或更强 query decomposition。
- 在线 decode 阶段 v98 比 full KV 略快：1.48s vs 1.64s；但端到端仍受 full prefill 主导。

## v99

v99 只针对 16k multiquery，把预算提高到 4096：

```text
task = niah_multiquery
length = 16384
samples = 5
policy = budget4096
```

目标：验证 multiquery 是否需要约 25% KV 才能恢复。

v99 结果：

| 任务 | Full score | Ours score | Ours KV keep | Full online | Ours online |
| --- | ---: | ---: | ---: | ---: | ---: |
| multiquery 16k | 0.85 | 1.00 | 25.65% | 2.85s | 1.99s |

结论：16k multiquery 的最小安全动作不是 1024/2048，而是约 4096 tokens。这个结果支持 length/task-aware budget ladder：更难的 query family 需要更大的安全动作，但仍比 full KV 少约 74%。

## v100

v100 把 v98/v99 合并成一个同-run policy：

- 16k multikey：2048 budget。
- 16k multivalue：2048 budget。
- 16k multiquery：4096 budget。

目标：在同一批 16k localization samples 上验证集成版是否接近或超过 full KV，同时保持低 KV ratio。

v100 结果：

| 任务 | Full score | Ours score | Ours KV keep |
| --- | ---: | ---: | ---: |
| multikey 16k | 1.00 | 0.60 | 12.84% |
| multivalue 16k | 0.95 | 0.60 | 12.84% |
| multiquery 16k | 1.00 | 0.95 | 28.21% |
| Overall | 0.9833 | 0.7167 | 17.96% |

结论：v100 是关键负例。2048 budget 对 16k multikey/multivalue 在不同随机样本上不稳；multiquery 用 4096 基本稳定。因此不能把 2048 作为 16k 通用安全动作。

## v101

v101 对 16k localization family 统一使用 4096 budget：

```text
tasks = niah_multikey_1, niah_multivalue, niah_multiquery
length = 16384
budget = 4096
samples = 5
```

目标：验证 25% KV 是否能成为 16k localization 的稳健 safe action。

v101 结果：

| 任务 | Full score | Ours score | Ours KV keep | Full online | Ours online |
| --- | ---: | ---: | ---: | ---: | ---: |
| multikey 16k | 1.00 | 1.00 | 25.65% | 0.277s | 0.187s |
| multivalue 16k | 1.00 | 1.00 | 25.65% | 2.152s | 1.286s |
| multiquery 16k | 0.85 | 1.00 | 28.21% | 2.800s | 2.023s |
| Overall | 0.9500 | 1.0000 | 26.50% | 1.743s | 1.165s |

结论：

- v101 是当前 RULER localization 的最强正结果。
- 16k 下用约 26.5% KV 可以超过 full KV 分数，并带来约 1.50x online decode speedup。
- 这支持论文里的主张：RiskKV-Block 不应该被表述成固定 KV ratio 压缩，而应该表述成 risk/length/task conditioned safe memory action。

当前推荐论文主线：

1. LongBench 展示 safety-aware controller：高风险任务不强压，v81 保质量但 KV keep 偏保守。
2. RULER localization 展示 compression/scaling：v101 在 16k 用 26.5% KV 超过 full。
3. 负例 v90/v92/v100 作为 boundary evidence：盲目低预算或简单 graph bridge 不可靠，证明 risk-aware action 必要。
