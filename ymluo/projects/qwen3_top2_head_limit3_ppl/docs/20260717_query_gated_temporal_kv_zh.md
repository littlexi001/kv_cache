# Query-Gated Temporal KV Retrieval

更新时间：2026-07-17

## 1. 核心问题

当前方法每个 decode step 都执行一次 PCA64+INT4 全局扫描。即使最终只有约 2% 的历史 token 参与 attention，检索索引仍需在每一步扫描完整历史，因此短上下文下检索开销高于 Full Attention。

已有 32K 实验显示：相邻 step 的候选 token 重叠约 66.9%，最近 4 步候选并集覆盖约 81.2%。这说明候选检索不是相互独立的，而是一个随生成过程缓慢变化的状态。

## 2. 方法

方法不训练 router，也不使用任务标签，只增加一个事件触发规则。

### Refresh step

执行当前完整方法：

1. PCA64+INT4 扫描历史 K；
2. 0.25% exact 样本校准 partition；
3. 从 0.5% 到 8% 的离散预算中选择最小预算；
4. exact rerank 后计算稀疏 attention；
5. 保存候选池、预算、exact-QK 读取数和投影 query。

### Reuse step

对每层计算当前投影 query 与上次 refresh query 的平均余弦相似度：

`similarity = mean_head(cos(PCA(q_current), PCA(q_refresh)))`

当 `similarity >= 0.88` 且距离上次 refresh 只有一个 step 时：

1. 不扫描完整 PCA64+INT4 索引；
2. 对缓存候选池使用当前 query 重新计算 exact QK；
3. 把 refresh 后新产生的 token 加入候选；
4. 沿用上次预算并重新排序；
5. 计算稀疏 attention。

否则立即 refresh。不同层可独立决定 refresh 或 reuse。

## 3. 32K PPL 结果

### 10-case 汇总

| 方法 | 几何平均 PPL | 相对原方法 | Attention links | Exact-QK | Layer-step reuse |
|---|---:|---:|---:|---:|---:|
| 全局一次预算 | 10.46234 | 0.00% | 约 2.11% | 约 2.90% | 0% |
| Query-gated temporal | 10.53241 | +0.67% | 2.10% | 2.88% | 44.07% |

逐 case 质量变化：

| Case | 原方法 PPL | Temporal PPL | 变化 |
|---|---:|---:|---:|
| sports w0 | 9.0662 | 9.0226 | -0.48% |
| medicine w0 | 10.1205 | 10.2350 | +1.13% |
| computer w0 | 14.1936 | 14.3402 | +1.03% |
| medicine w1 | 7.7541 | 7.8386 | +1.09% |
| medicine w2 | 8.5545 | 8.5595 | +0.06% |
| politics w0 | 14.5945 | 14.7120 | +0.81% |
| religion w0 | 12.1048 | 12.1842 | +0.66% |
| space w0 | 16.8244 | 16.9571 | +0.79% |
| sports w1 | 2.6013 | 2.6250 | +0.91% |
| sports w2 | 23.5269 | 23.6956 | +0.72% |

### 串行速度

| Case | 原方法 | Temporal | Temporal 加速 |
|---|---:|---:|---:|
| sports w0 | 54.61s | 48.86s | 1.118x |
| medicine w0 | 54.36s | 47.45s | 1.146x |

固定隔步复用更快，体育达到 1.289x，但医学 PPL 退化 2.77%。Query gate 将医学退化降到 1.13%，代价是加速下降到约 1.15x。

### Qwen3-4B 冻结配置迁移

在 Llama 实验之后冻结全部配置，直接迁移到 Qwen3-4B-Instruct。测试使用 32K 历史、5 个主题、每主题 128 个 target tokens；没有为 Qwen 重新选择阈值或预算。

| 方法 | 5 主题几何 PPL | 相对 Full | Attention links | Exact-QK | Layer-step reuse | 平均在线时间 |
|---|---:|---:|---:|---:|---:|---:|
| Full Attention | 14.23565 | 0.00% | 100% | 100% | 0% | 33.31s |
| 全局一次预算 | 14.95792 | +5.07% | 3.60% | 4.46% | 0% | 50.02s |
| Query-gated temporal | 14.85987 | +4.38% | 3.60% | 4.47% | 21.89% | 49.62s |

Qgate 相比原稀疏方法的几何 PPL改善 0.66%，但平均时间只减少 0.8%。640 个配对 token 上，平均 NLL 相对原稀疏方法降低 0.00658，最坏 Full-relative NLL 差值从 7.01 降到 6.15。该结果支持时间复用可以跨模型运行，但也说明固定 cosine 阈值的复用率和速度收益不稳定。

## 4. LongBench 自由生成 smoke test

上下文约 7.5K，每任务仅 1 个样本。

| Task | FullKV score | 原稀疏 score | Temporal score | 原稀疏时间 | Temporal 时间 | 相对原稀疏加速 |
|---|---:|---:|---:|---:|---:|---:|
| NarrativeQA | 0.4286 | 0.6154 | 0.6154 | 7.37s | 7.01s | 1.05x |
| HotpotQA | 0.3333 | 0.3333 | 0.3333 | 6.23s | 5.94s | 1.05x |
| PassageRetrieval-en | 0.0000 | 0.0000 | 0.0000 | 16.77s | 14.47s | 1.16x |
| LCC | 1.0000 | 1.0000 | 1.0000 | 4.17s | 3.84s | 1.08x |

这组实验只证明真实生成链路可运行且没有观察到额外质量下降。样本量不足以支持 LongBench 总体结论。

## 5. RULER 64K/128K smoke test

任务为冻结的 `niah_single_1`，每个长度 1 个样本，FullKV 与 qgate 使用相同 prompt。

| 长度 | Prompt tokens | FullKV score | Temporal score | Attention links | Exact-QK | Reuse | Full 在线时间 | Temporal 在线时间 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64K | 64,881 | 1.0 | 1.0 | 2.33% | 2.94% | 38.87% | 1.40s | 6.78s |
| 128K | 130,884 | 1.0 | 1.0 | 2.38% | 3.04% | 45.37% | 5.65s | 11.69s |

从 64K 到 128K，Temporal/Full 的时间比从 4.86x 降到 2.07x。趋势符合预期：Full Attention 随历史长度增长更快，而检索与稀疏 attention 的相对代价下降。但当前实现仍未超过 FullKV。

冻结 `0.88` 阈值后，另外运行了 64K `niah_single_1` 的 5 个新样本：

| 方法 | 正确样本 | 平均 score | 平均在线时间 |
|---|---:|---:|---:|
| FullKV | 5/5 | 1.0 | 1.395s |
| Query-gated temporal | 5/5 | 1.0 | 6.179s |

两种方法在 5 个样本上的答案逐一相同。该组分片并行执行，时间只用于确认当前实现仍慢于 FullKV，不作为严格的单流延迟结论；同时该轮关闭了逐层 diagnostics，连接率和复用率应引用上面的单样本 profile，而不能填成 0。

128K FullKV 加模型权重在两张 24GB 3090 上 OOM，四卡分片后成功。当前 qgate runner 仍保留完整 DynamicCache，因此没有解决物理 KV 容量问题。

## 6. 当前判断

1. 事件触发检索是一个简单且独立的创新方向：将 KV retrieval 从“每步重新搜索”改为“可复用的时间状态”。
2. `qgate088` 在 10-case 上以 0.67% PPL 代价减少约 44% 的 layer refresh，适合作为 latency mode，不应替代质量默认版。
3. 当前 Python eager 原型在 7.5K 下仍显著慢于优化后的 Full Attention。它只证明相对原稀疏方法减少了检索时间，不能宣称端到端超过 FullKV。
4. 当前仍保留完整 DynamicCache，`attention_link_ratio` 不是物理 GPU KV 占用率。
5. `0.88` 是观察 medicine w0 后选出的阈值，因此包含 medicine w0 的 10-case 不是完全独立测试。下一轮必须冻结阈值后使用新窗口、第二模型或公开 benchmark。
6. 冻结配置已在 Qwen3-4B 上完成 5 主题迁移：质量没有整体变差，但只有 21.9% layer-step reuse，尚不足以形成通用速度结论。

## 7. 下一步

1. 冻结 `0.88`，在未参与设计的第二模型上测试；
2. 将 query gate、候选复用和物理 KV gather 融合到同一 CUDA 路径，消除逐层 Python `.item()` 同步；
3. 分析 query drift 与候选集合变化的关系，为事件触发条件给出有限误差解释；
4. 扩大 RULER 任务类型，而不只测试 `niah_single_1`。

## 8. 代码与结果

- 方法实现：`src/run_head_top2_targeted_ppl_20260714.py`
- LongBench harness：`src/run_sample_calibrated_longbench_20260717.py`
- RULER harness：`src/run_sample_calibrated_ruler_20260717.py`
- 单元测试：`tests/test_sample_calibrated_budget.py`
- 10-case 结果：`results/20260717_partition_global_qgate088_twotheme_32k` 与 `results/20260717_partition_global_qgate088_independent_32k`
- LongBench smoke：`results/20260717_qgate088_longbench_smoke4`
- RULER 64K/128K smoke：`results/20260717_qgate088_ruler64k_smoke` 与 `results/20260717_qgate088_ruler128k_smoke_4gpu`
- RULER 64K 冻结 5 样本：`results/20260717_qgate088_ruler64k_m5_frozen`
- Qwen3-4B 冻结 5 主题：`results/20260717_qwen3_4b_independent5_32k_full`、`results/20260717_qwen3_4b_independent5_32k_global` 与 `results/20260717_qwen3_4b_independent5_32k_qgate088`
