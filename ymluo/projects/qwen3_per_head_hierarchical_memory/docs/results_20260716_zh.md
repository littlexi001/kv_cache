# 每个 Head 独立多级记忆：实现与实验结果

日期：2026-07-16  
模型：Qwen3-0.6B（28 层 × 16 query heads = 448 heads；8 KV heads）  
数据：War and Peace，16,384-token prefill

## 1. 研究问题

目标是用外部方法帮助不同功能的 head 找到各自需要的历史 token，而不是重新计算完整 attention。我们把用户提出的“持续轨迹/长期记忆”思想落实为推理时的持久分层记忆：每个 query head 有独立 L0/L1 状态，完整历史只保留在共享 L2 索引中。

本轮回答四个问题：

1. 能否为全部 448 个 head 维护独立状态，并保证最浅层 L0 不超过 500 token？
2. L0 中用多少 recent token、多少远程检索 token 更合理？
3. 按 head 功能路由外部检索，能否接近 attention Top-2% Oracle？
4. Oracle 召回的改善是否能转化为真实生成 PPL 改善？

## 2. 方法

| 层级 | 默认容量 | 内容 | 更新方式 |
|---|---:|---|---|
| L0 / hot | 500/head | sink、recent、从 L1 提升的位置 | 每个 query 更新，唯一进入稀疏 attention |
| L1 / warm | 4096/head | 每个 head 独立的候选 token 位置 | L2 候选打分 + retention bonus |
| L2 / cold | 全历史 | 64-token blocks、语义/词法/格式元数据 | 共享只读索引，每 head 短列 64 blocks |

外部打分由 position、lexical、semantic、format、repeat 五项组成；每个 head 的混合权重来自 448-head 功能图谱。检索过程不读取 QK 或 attention score。attention Top-2% 只在离线评估中作为 Oracle 标签。

策略：

- `sink_recent_500`：纯 recent 基线。
- `flat_function_500`：直接在完整历史上检索，作为外部计算上界参考。
- `hier_function_500`：L2 → L1 → L0 的实际分层方案。
- `confidence_gated`：只有指定功能且非低置信度的 head 能用 promotion slots；其余 head 自动保留 sink + recent-496。

## 3. 已验证的实现不变量

- 448/448 个 query heads 都有独立状态行。
- 所有远程实验观测到的 L0 最大值均为 500，未越界。
- L0 内没有重复位置，sink 计入 500 总预算。
- L1 状态跨 query 持续保留，不是每次重新初始化。
- 稀疏 PPL 路径只允许每个 head 的 L0 历史位置与当前 causal chunk。
- 8 个本地单元测试全部通过。

注意：当前质量验证仍物理保存完整 KV，仅在 attention 前注入 per-head mask。它证明选择质量和模型行为，不代表已经获得显存或速度收益。

## 4. Recent / retrieval 预算扫描

全部 head 都启用检索、L0 总容量固定为 500，测试 64 queries：

| resident recent | promotion 上限 | overall Top-2% recall | Oracle mass recall | remote Top-2% recall | GQA union / 单 head |
|---:|---:|---:|---:|---:|---:|
| 256 | 240 | 39.55% | 86.43% | 7.50% | 1.1017× |
| 384 | 112 | 43.83% | 87.79% | 4.47% | 1.0513× |
| 448 | 48 | 45.36% | 88.14% | 2.11% | 1.0220× |
| 480 | 16 | 45.81% | 88.25% | 0.75% | 1.0072× |
| recent-500 baseline | 0 | 46.49% | 88.10% | 0 | 1.0000× |

分层检索确实能找回远程 Oracle token，但 promotion 越多，损失的 recent Oracle token 越多。48 或 16 个 promotion slots 是较合理的 Pareto 区域；240 个明显过量。

## 5. 真实稀疏 PPL：全 head 检索失败

16K prefill 后测试 64 tokens：

| 策略 | mean NLL | PPL | 相对 full PPL |
|---|---:|---:|---:|
| full attention | 3.83619 | 46.3487 | 1.000× |
| recent-500 | 3.88525 | 48.6790 | 1.050× |
| flat external retrieval, all heads | 4.03040 | 56.2832 | 1.214× |
| hierarchical retrieval, all heads | 4.04487 | 57.1037 | 1.232× |

结论：attention mass recall 与输出质量并不等价。给所有 head 统一牺牲 48 个 recent 位置，即使 mass recall 接近，也会显著伤害 PPL。

## 6. 精确 Oracle Top-2% PPL

为直接验证最初观察，又补充了严格对齐实验：先用 full attention prefill 到第 16,448 个 token，再逐 token 解码；每层每个 query head 根据当次完整 QK score，只保留历史位置中最高的 2%，并始终保留当前 self token。不额外保护 sink 或 recent。

| 测试窗口 | Full mean NLL | Full PPL | Oracle Top-2% mean NLL | Oracle Top-2% PPL | PPL 相对变化 |
|---:|---:|---:|---:|---:|---:|
| 64 token | 3.835325 | 46.3085 | 3.777926 | 43.7253 | -5.58% |
| 512 token | 3.231982 | 25.3298 | 3.224325 | 25.1366 | -0.76% |

64-token 窗口中，每个 head 每次平均保留 330.125 个历史 token，范围为 329–331；512-token 窗口按 2% 预算平均为 334.56 个，范围为 329–340。长测仍确认 Oracle Top-2% 优于 full attention，但收益比 64-token 小样本弱得多。

这验证了“少量高注意力 token 可以缓解 attention dilution/context rot”的质量上界，同时也界定了问题：Oracle 必须先计算完整 QK 才知道 Top-2%，计算量本身没有省掉。外部检索研究的目标正是以更低成本预测这约 330 个位置。

原始输出：`outputs/oracle_top2_war16k_aligned_64q_20260716/` 与 `outputs/oracle_top2_war16k_aligned_512q_20260716/`。

## 7. 按功能门控的消融

只让单类 head 使用检索；高置信度 head 最多 48 slots，中置信度 20 slots：

| 激活功能 | 激活 heads | 64-token mean NLL | 64-token PPL |
|---|---:|---:|---:|
| semantic evidence | 9 | 3.87996 | 48.4223 |
| structural anchor | 15 | 3.88888 | 48.8560 |
| lexical copy | 12 | 3.93454 | 51.1389 |
| recent-500 reference | 0 | 3.88525 | 48.6790 |

语义组在 64-token 小样本上看似优于 recent-500，但差值很小，因此又进行了 512-token 确认实验：

| 策略 | test tokens | mean NLL | PPL |
|---|---:|---:|---:|
| full attention | 512 | 3.231969 | 25.3295 |
| recent-500 | 512 | 3.424284 | 30.7007 |
| 仅 9 个 semantic heads 使用分层记忆 | 512 | 3.424957 | 30.7213 |

长测中语义记忆相对 recent-500 的 ΔNLL = +0.000673、PPL = +0.0207，基本持平但没有胜出。64-token 的正收益不能视为稳定结论。

## 8. 为什么当前检索未胜出

在 512-token Oracle 实验中，9 个语义 head 都找回了一些 500-token 以外的位置，单 head remote recall 约 0.74%–1.55%；但它们在校准集和测试集的总体 position recall、mass recall 全部低于 recent-500。当前语义索引只是模型输入 token embedding 的 64-token mean pooling，不能可靠定位证据中的精确 token。

这给出清楚的失败边界：

- “head 功能门控”是必要的：全 head PPL 57.10，语义门控降到 48–49。
- 仅凭功能标签仍不充分：同一功能内部也需要 query-aware、head-specific 的检索器。
- 远程召回本身不够；候选必须至少补偿被移除的 recent token 的因果价值。
- `Oracle mass recall` 只能作为诊断指标，最终决策必须由 held-out NLL/PPL 验证。

## 9. 内存含义

Qwen3-0.6B 的 head_dim=128。16K、FP16 的完整物理 KV 约 1.75 GiB。若后续实现真正的 paged KV compaction：

- 固定 500 token/KV head 的理论 KV 约 54.69 MiB，即完整 16K KV 的 3.05%。
- 语义门控下，同一 GQA 组两个 query heads 的平均 union 为单 head 的 1.00347×，约 501.7 token/KV head、54.88 MiB。
- 全 head recent-448 检索时 union 为 1.0220×，约 511.0 token/KV head、55.89 MiB。
- 仅 L0/L1 位置索引约 0.85 MiB + 7.00 MiB，不含外部 embedding/index。

这些是按选择集合推导的理论值；当前实现没有物理释放完整 KV，因此实测显存不会按此下降。

## 10. 下一步实验优先级

1. 用独立句向量模型构建 sentence/paragraph L2 索引，再在命中块内用词法或轻量 token scorer 精排；替换当前 mean token embedding。
2. 用 train queries 学习“是否给某 head 开 promotion”与 promotion 数量，不按功能类别硬编码；在独立 test queries 验证。
3. 对检索命中的完整证据 span 做邻域扩展，避免只选块中任意 48 个 token。
4. 在合成人造证据任务上分别测试无冲突/有冲突，测 gold evidence recall、decoy attention、held-out NLL。
5. 质量达标后再实现 GQA union-aware paged KV compaction，报告显存、prefill/decode latency 和检索开销。

## 11. 产物

- 主实现：`src/run_per_head_hierarchical_memory.py`
- 稀疏 PPL：`src/run_sparse_memory_ppl.py`
- 汇总脚本：`src/summarize_results.py`
- 汇总表：`analysis/experiment_summary.csv`
- 9 个语义 head 的 train/test 召回差：`analysis/semantic_head_recall_deltas.csv`
- 全部原始结果：`outputs/`
