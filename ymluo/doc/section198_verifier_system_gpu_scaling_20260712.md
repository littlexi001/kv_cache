# Section 198: 10M 最强 Verifier 系统的单请求多卡加速

更新时间：2026-07-12

## 1. 目标

本实验测量当前 10M 最强 verifier 系统在同一个请求内部使用不同 GPU 数量时的真实延迟，并回答：

1. 2/4/6 张 RTX 3090 相对 1 张卡分别能加速多少？
2. 多卡加速发生在哪个阶段？
3. 最低延迟与最高资源效率分别对应多少张卡？

## 2. 严格协议

- 模型：冻结 `Qwen/Qwen3-8B`，FP16，greedy decode，最多 24 个新 tokens。
- 数据：固定 test query 1502--1531，共 30 条；query 1500--1501 只用于每个卡数的预热，不计入统计。
- GPU 数：1、2、4、6 张 RTX 3090。
- 并行口径：同一个请求内并行，不是把不同 query 分到不同 GPU 的 batch throughput。
- 每题都真实重新执行：
  1. Top3 concat 的 8B bridge 生成；
  2. 16 个第二跳 block 的独立答案抽取；
  3. 16 个 `question + evidence + candidate` 的冻结 Yes/No verifier；
  4. 选择 support margin 最大的候选。
- 16 个答案分支与 16 个 verifier 分支按 branch index 分配到各张 GPU。
- bridge 只有一个生成分支，固定在 rank 0 执行，因此是所有 GPU 数量都必须支付的串行部分。
- 模型加载和一次性 BM25 建库不计入延迟。
- 10M 在线检索已经单独实测约 42.5ms/题；本实验把该值加回 `estimated online total`。
- routing trace 与第二跳 Top16 候选被冻结，保证不同 GPU 数量处理完全相同的问题和候选；所有模型生成与 verifier 前向仍重新执行。

四种卡数下 bridge replay 与冻结链的匹配率均为 100%，30 条上的答案正确率均为 50%。这个 50% 只是速度子集的准确率，不替代完整 500 条的 36.4% 主结果。

## 3. 端到端单题延迟

| GPU | mean | median | p95 | 相对 1 卡加速 | 并行效率 | GPU-seconds/题 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8.993s | 8.040s | 14.593s | 1.00x | 100.0% | **8.99** |
| 2 | 5.101s | 4.730s | 7.837s | **1.76x** | **88.1%** | 10.20 |
| 4 | 3.177s | 2.975s | 5.330s | **2.83x** | 70.8% | 12.71 |
| 6 | **2.598s** | **2.462s** | **4.081s** | **3.46x** | 57.7% | 15.59 |

结论：多卡并行确实显著降低了单请求延迟，但存在清晰的边际收益递减。

- 1 -> 2 卡：减少 3.89s，获得 1.76x，加速效率最好。
- 2 -> 4 卡：再减少 1.92s，累计达到 2.83x。
- 4 -> 6 卡：只再减少 0.58s；GPU 增加 50%，延迟仅改善约 18.2%。
- 1 -> 6 卡：mean 降低 71.1%，p95 降低 72.0%，但 GPU-seconds 增加 73.3%。

因此：

- 最低延迟配置：6 卡，约 2.60s/题。
- 最佳延迟/资源折中：2 卡，约 5.10s/题、88.1% 并行效率。
- 如果目标是把延迟压到约 3s：4 卡比 6 卡更合理。

## 4. 分阶段加速

### 4.1 平均时间

| GPU | Bridge | 16 路答案生成 | 16 路 Verifier |
|---:|---:|---:|---:|
| 1 | 0.487s | 7.032s | 1.432s |
| 2 | 0.483s | 3.872s | 0.703s |
| 4 | 0.485s | 2.268s | 0.382s |
| 6 | 0.489s | 1.777s | 0.289s |

### 4.2 相对 1 卡加速

| GPU | Bridge | 答案生成 | Verifier |
|---:|---:|---:|---:|
| 2 | 1.01x | 1.82x | 2.04x |
| 4 | 1.00x | 3.10x | 3.75x |
| 6 | 0.99x | 3.96x | 4.95x |

Verifier 接近线性加速；答案生成受到不同 branch 输出长度和负载不均衡影响，6 卡只达到 3.96x。Bridge 完全不随卡数加速，因此它从 1 卡时约 5.4% 的时间占比，提高到 6 卡时约 19.1%，成为新的串行瓶颈。

## 5. 为什么 6 卡没有达到 6 倍

1. Bridge 是固定串行部分，约 0.49s。
2. 16 个分支不能被 6 整除，负载为 `3/3/3/3/2/2`；4 卡恰好是每卡 4 个分支。
3. 不同答案分支生成 token 数不同，每题必须等待最慢 GPU。
4. 每题存在 barrier、对象聚合和调度开销。
5. 10M 检索的约 42.5ms 不随 GPU 数增加而下降。

这解释了并行效率从 2 卡的 88.1% 降到 6 卡的 57.7%。如果以后 8 卡全部空闲，16 个分支可以均匀分为每卡 2 个；8 卡可能比 6 卡有更好的 branch balance，但仍需要实测，不能从本结果直接外推。

## 6. 与 40K Full Attention 的正确比较方式

现在已经有了 verifier 系统的真实同请求延迟：

| 系统 | GPU/请求 | 实测 mean latency | GPU-seconds/题 |
|---|---:|---:|---:|
| 40K full attention | 2 卡模型并行 | 17.99s | 约 35.98 |
| 10M verifier | 1 卡 | 8.99s | 8.99 |
| 10M verifier | 2 卡分支并行 | 5.10s | 10.20 |
| 10M verifier | 4 卡分支并行 | 3.18s | 12.71 |
| 10M verifier | 6 卡分支并行 | 2.60s | 15.59 |

这张表比旧的“4 卡 batch 墙钟折算 10.6s/题”更严格，因为现在 verifier 的数字都是同一请求从 bridge 到最终选择的实际墙钟。不过两种系统的并行结构仍不同：40K 使用两卡承载一个模型，verifier 使用多个完整模型副本处理独立 branches。因此论文中必须同时报告 latency、GPU 数和 GPU-seconds，不能只比较秒数。

## 7. 工程结论

当前最推荐的在线配置是 2 或 4 卡，而不是默认 6 卡：

- 2 卡适合关注资源效率的服务：5.10s，1.76x，效率 88.1%。
- 4 卡适合关注交互延迟的服务：3.18s，2.83x。
- 6 卡适合只追求最低 latency 的实验演示：2.60s，但资源效率明显下降。

下一步真正有效的优化不是继续堆 GPU，而是：

1. 把 16 路 8B candidate extractor 蒸馏成小模型或批处理模型。
2. 把 Yes/No verifier 蒸馏成轻量分类头。
3. 用风险门控让高置信度问题只读 Top3，只有低置信度问题扩展到 Top16。
4. 将 bridge 与候选预取/检索做流水化，隐藏部分串行时间。

## 8. 复现与结果

代码：

```text
src/benchmark_verifier_system_same_query_scaling.py
src/analyze_verifier_system_scaling.py
scripts/run_verifier_system_scaling_1_2_4_6.sh
```

服务器结果：

```text
outputs/musique_verifier_system_scaling_30q_v1/world1/summary.json
outputs/musique_verifier_system_scaling_30q_v1/world2/summary.json
outputs/musique_verifier_system_scaling_30q_v1/world4/summary.json
outputs/musique_verifier_system_scaling_30q_v1/world6/summary.json
outputs/musique_verifier_system_scaling_30q_v1/scaling.json
```
