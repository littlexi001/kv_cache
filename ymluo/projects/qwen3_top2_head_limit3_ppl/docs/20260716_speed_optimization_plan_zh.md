# 层次化 KV 检索的速度瓶颈与验证计划（2026-07-16）

## 1. 已验证基线

当前 128K 开发基线为 Llama-3.1-8B-Instruct、RTX 3090、PCA64 INT4、per-query-head 1% 候选、stream group 2、3.2% exact hot cache。单窗口前 32 个 target token 的同步计时如下：

| 阶段 | Full KV | 层次化 KV |
|---|---:|---:|
| dense prefill | 219.954 s | 214.659 s |
| cache conversion | 0 s | 24.927 s |
| 287 次在线 forward | 94.054 s | 35.018 s |
| protocol total | 314.008 s | 274.604 s |

因此，已测 decode speedup 为 2.686x，而包含 prefill 和转换的 protocol speedup 为 1.143x。当前结果只来自一个 128K religion 窗口，不能作为最终论文主结果。

按每步平均在线时间外推，当前设置约在 96 个在线 step 后回本，约 3100 step 达到 2x protocol speedup，约 16700 step 达到 2.5x。由于当前 decode speedup 为 2.686x，不改变 decode 实现时，protocol speedup 的渐近上限也是 2.686x。

## 2. 当前瓶颈

独立 attention 子系统计时为：

| 组件 | 每层时间 | 占层次化子系统 |
|---|---:|---:|
| PCA64 scan + top-k | 0.212 ms | 21.1% |
| fused hash/LRU directory | 0.050 ms | 5.0% |
| mapped-host miss fill + sparse attention | 0.741 ms | 73.9% |
| 合计 | 1.003 ms | 100% |

主要瓶颈不是 directory，而是随机 PCIe miss fill 与稀疏 attention 的组合。逻辑 token ratio 很低并不代表时间按同样比例下降，因为每步仍要扫描压缩索引、执行 top-k、维护 residency，并搬运未命中的 exact K/V。

在 4K--8K，attention 节省不足以覆盖 Python 调度、小 kernel、top-k 和 PCIe 固定开销。当前全量 LongBench 的旧 sparse 路径在约 7.5K prompt 上经常慢于 Full，因此最终系统需要纯长度门控，而不能强制所有请求走 sparse。

## 3. 已实现、待物理验证的优化

### 3.1 异步单 token host append

旧实现每层分别阻塞写回新 K 和 V。新实现用当前 CUDA stream 将 GPU 到 pinned host 的写回异步排队；所有自定义 kernel 同样使用当前 stream，因此下一个 token 读取前的顺序由 stream 保证。提供 `--host_append_mode async|sync` 做严格消融。

### 3.2 多 GPU 异步 conversion

旧 conversion 每处理一层就等待整层 K/V 写回 host，使 Python 无法及时向其他 GPU 派发 PCA、INT4 和 offload 工作。新实现对各设备当前 stream 异步派发，阶段末每张 GPU 只同步一次。提供 `--conversion_mode async|sync` 做配对消融。

### 3.3 Temporal candidate refresh

新增 candidate refresh interval 2/4。非 refresh token 复用远端候选，并为最近 token 保留位置，避免每个 token 都执行 PCA 全扫描和 top-k。按组件时间、暂不考虑质量变化估算，interval 2/4 可把 attention 子系统从 2.62x 提升到约 2.93x/3.11x；该数字是估算，不是实验结果。

### 3.4 Full-prompt-then-compress

LongBench/RULER 新路径先对完整 prompt 做一次 dense prefill，保留准确的首个 answer logits，然后只在生成阶段使用层次化 cache。它消除了逐 token 重放 question suffix 的协议开销，并使 Question-aware 信息在压缩前进入模型状态。

### 3.5 16K 固定长度 gate

RULER 和 LongBench 同时保留 raw sparse 与 Full 的物理结果，汇总时合成固定策略：requested/prompt length 小于 16K 使用 Full，16K 及以上使用 sparse。该策略不读取任务标签、答案或测试分数。raw sparse 结果仍单独报告，不能用 fallback 隐藏短上下文低效。

### 3.6 严格同步计时

Full 与 sparse 现在都在计时区间前后同步所有可见 CUDA 设备；最后一个不会再被消费 logits 的 forward 已移除。128K speed Pareto 使用同一轮、同主题、同窗口的配对 Full 结果，不再把旧 Full 时间与新 sparse 实现混用。

### 3.7 候选地址排序探索

attention 微基准新增 random 与 token-address-sorted 两种候选顺序。二者的候选集合完全相同，只改变 pinned-host KV 的物理访问顺序，用于验证 PCIe 合并访问能否降低 miss-fill 时间。预排序结果只表示 PCIe 访问上界，不能直接作为实现收益；实验另外计入在线 `argsort + gather` 的完整成本，并测试只压紧、排序 directory miss token、保持 attention 索引顺序不变的实现。只有包含排序开销的微基准显示稳定收益后才会改主路径。

### 3.8 检索状态感知的因果 router

三动作 router 在上一 token 的 logits、词频和 token 类型之外，新增上一 token 已完成检索产生的候选分数跨度、top-16 候选稳定度、刷新比例和特征有效位。特征不执行第二次 KV 扫描；同一 GPU 的所有层先聚合为三维向量，每张 GPU 只回传一次。counterfactual low/mid/high probe 在同一检索诊断检查点上开始，防止后执行的动作污染下一训练样本。运行时若预算未变化，不再清空 temporal candidate cache，使动态 router 能与 refresh 2/4 组合。以上均为已实现机制，质量和速度仍需独立测试确认。

## 4. 已排队实验

1. LongBench 16 个英文任务、3750 个样本、Full/sparse 严格配对，共 7500 行。
2. LongBench full-prompt speed 验证，以及 temporal refresh 2/4 的速度质量消融。
3. 128K async/sync、PCA48/56、扩大 residency、candidate refresh 2/4 的配对速度 Pareto。
4. 128K attention 分项：host-direct、miss-fill-then-attention、pack-then-attention、resident gather + SDPA，并比较 random/address-sorted 候选；随后在 religion/computer 上记录两个 per-head stream 的真实候选并集与重复率，trace 计时不作为速度结果。
5. RULER 4K/8K/16K/32K/64K/128K，报告 raw sparse 和 16K gate；生成样本由包含模型、任务、长度和 seed 的 manifest 锁定。
6. 冻结 LongBench-v2 calibration split 的 14 条 Long In-context Learning 物理配对；paper-test 在 router 与安全动作冻结前不参与实验运行或方法选择。
7. 32K matched-control 消融：PCA16/32/48/56/64/96、INT4/INT8、shared-mean/shared-max/per-head、stream group 1/2/4、hot cache 1.1/2.1/3.2/4.1、固定 1/1.5/2%、sorted/fused directory。实现中的 shared-sum 在固定 query-head 数下与 shared-mean 排序等价。
8. 六主题、三个不重叠 128K 窗口，每窗口 256 个 target token；报告均值、最差窗口、bootstrap CI、逐位置 NLL gap、逻辑 ratio 和绝对 tensor bytes。
9. 三折严格主题 holdout 的 1/1.5/2% 因果动态预算 router：train 使用 w0、独立主题 calibration 使用 w1、未见主题 test 使用 w2；最小安全动作以同位置 Full KV NLL + 0.05 为阈值。校准同时约束 required-action recall ≥95% 和整体 Full-relative retention ≥95%，不再以 2% 动作为相对上界。同一冻结 router 在六个 test 主题上分别运行 refresh-1/2，共 12 个物理结果，用于检验动态预算与候选复用能否安全合并。
10. 独立 religion window3 的 128K + 2048 target token 配对长 decode，用于验证质量保持和固定成本摊销。
11. 128K 低峰值 offloaded-exact 与同配置 dynamic/Full 配对；CPU/PCIe/NUMA；batch 1/2/4 throughput。
12. Qwen3-4B 的物理 PPL、LongBench 和 RULER 多模型验证。
13. 队列末尾执行 21 项 completion audit；缺少文件、方法或严格样本数时整体失败，不以进程退出码替代实验完整性。

## 5. LongBench 16 任务最终结果

### 5.1 实验协议

本轮完整实验使用 Llama-3.1-8B-Instruct，在 LongBench 16 个英文任务上评测 3750 个样本。每个样本分别运行 Full KV 和层次化稀疏 KV，共 7500 条严格配对结果；8 个 shard 均产生 `summary.json` 后才执行最终合并。主要配置如下：

- 最大上下文长度：7500 token；
- PCA 索引：PCA64 INT4；
- 每个 query head 的候选及最终 attention 历史 token 比例：2.5%；
- exact GPU K/V cache：3.2%；
- candidate selection：`per_head_stream`，stream group 1；
- prompt wrapper：Llama-3；
- 原始主实验协议：`prefix_sparse_suffix`，即 prefix 转换后逐 token 重放 question suffix。

最终结果位于服务器：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/
  outputs/20260716_hierarchical_longbench_full_v1_merged/
    sample_results.csv
    summary_by_task.csv
    summary_overall.csv
    summary.json
```

### 5.2 总体质量与物理 KV

| 方法 | Macro score | 相对 Full 质量保持率 | GPU KV ratio |
|---|---:|---:|---:|
| Full KV | 0.376365 | 100.00% | 100.00% |
| Hierarchical PCA per-head | 0.363799 | 96.66% | 10.62% |

这里的 `GPU KV ratio` 是持久 GPU cache tensor bytes 与完整 FP16 K/V bytes 的比例，不是单步参加 attention 的 token 比例。约 10.62% 的组成是 PCA64 INT4 全局 K 索引及 scale 约 6.64%、完整 K/V exact cache 3.2%，以及约 0.78% 的 PCA basis、directory 和其他元数据。每个 query head 每步真正进行精确 attention 的历史 token 为：

\[
K_{\mathrm{attn}}=\lceil 0.025(N-1)\rceil+1,
\]

其中最后的 1 是当前 token。因此，本实验的逻辑 per-head attention ratio 约为 2.5%，而不是 10.62%；完整精确 K/V 仍保存在 CPU pinned memory 中。

### 5.3 分任务结果

| 任务 | Full KV | Sparse | 分数变化 | 保持率 | GPU KV ratio |
|---|---:|---:|---:|---:|---:|
| 2WikiMQA | 0.4677 | 0.4690 | +0.0013 | 100.27% | 10.67% |
| GovReport | 0.2106 | 0.2058 | -0.0048 | 97.73% | 10.39% |
| HotpotQA | 0.4855 | 0.4690 | -0.0165 | 96.59% | 10.48% |
| LCC | 0.6321 | 0.6832 | +0.0511 | 108.08% | 11.36% |
| MultiNews | 0.1601 | 0.1586 | -0.0015 | 99.04% | 11.40% |
| MultiFieldQA-en | 0.5596 | 0.5183 | -0.0413 | 92.62% | 10.76% |
| Musique | 0.2814 | 0.1967 | -0.0847 | 69.91% | 10.44% |
| NarrativeQA | 0.2378 | 0.2413 | +0.0035 | 101.48% | 10.52% |
| PassageCount | 0.0988 | 0.0648 | -0.0341 | 65.51% | 10.41% |
| PassageRetrieval-en | 0.7250 | 0.7200 | -0.0050 | 99.31% | 10.35% |
| Qasper | 0.4541 | 0.3972 | -0.0569 | 87.47% | 10.88% |
| QMSum | 0.1726 | 0.1707 | -0.0019 | 98.91% | 10.80% |
| RepoBench-P | 0.5127 | 0.5502 | +0.0375 | 107.32% | 10.05% |
| Samsum | 0.1432 | 0.1476 | +0.0044 | 103.08% | 10.45% |
| TREC | 0.7000 | 0.6550 | -0.0450 | 93.57% | 10.62% |
| TriviaQA | 0.1807 | 0.1735 | -0.0072 | 96.03% | 10.19% |
| **Macro** | **0.37636** | **0.36380** | **-0.01257** | **96.66%** | **10.62%** |

16 个任务中有 11 个达到至少 95% 的 Full-relative 质量。LCC、RepoBench-P、NarrativeQA 和 Samsum 的 sparse 分数高于 Full，但这些正增益可能包含生成扰动和评测方差，不能解释为压缩必然提升模型能力。

### 5.4 三个主要弱项

Musique、PassageCount 和 Qasper 合计贡献约 87.4% 的总体 Macro gap。排除这三个任务仅用于诊断，不能作为论文主表：

| 评测范围 | Full KV | Sparse | 绝对差值 | 质量保持率 | GPU KV ratio |
|---|---:|---:|---:|---:|---:|
| 全部 16 个任务 | 0.37636 | 0.36380 | -0.01257 | 96.66% | 10.62% |
| 其余 13 个任务 | 0.39904 | 0.39709 | -0.00195 | 99.51% | 10.63% |

样本级诊断如下：

- **Musique**：200 条中 33 条下降、11 条提升、156 条不变；26 条由 Full 有分变成 sparse 0 分。错误通常是输出语义合理但错误的人名或年份，符合多跳桥接证据漏检。失败样本与其他样本的 cache hit 几乎相同，说明问题在候选证据召回，而不是 GPU residency。
- **PassageCount**：200 条中仅 17 条下降、11 条提升、172 条不变，但 Full 本身只有 0.0988，导致相对保持率对少数 exact-count 成败很敏感。任务需要覆盖所有段落并全局去重，局部 top-attention 不满足其计算结构。Sparse 平均生成长度从 Full 的 4.33 token 增至 25.69 token，常出现冗长枚举后给出错误数字。
- **Qasper**：84 条下降、45 条提升、71 条不变，属于广泛而非少数灾难性退化。下降样本平均更长，cache hit 略低，并倾向生成更长、更保守的解释或错误判断 `unanswerable`。科学文档中的分散语义证据和 Yes/No/Unanswerable 类型判断需要更高覆盖率或风险 fallback。

上述现象支持三类安全动作：multi-hop coverage、global aggregation coverage，以及 scientific-QA uncertainty fallback。后续 router 必须在不读取 gold 的条件下识别这些风险，不能通过从主表删除困难任务获得更好的平均数。

### 5.5 速度结果与协议限制

原始 `prefix_sparse_suffix` 协议的最终平均时间如下：

| 方法 | Prefill | Conversion | Query replay | Decode | Online | Total |
|---|---:|---:|---:|---:|---:|---:|
| Full KV | 1.617 s | 0 s | 0.054 s | 3.015 s | 3.069 s | 4.686 s |
| Raw sparse | 1.641 s | 0.827 s | 21.021 s | 8.383 s | 30.231 s | 31.871 s |

因此，短上下文下 raw sparse 的 online speed 只有约 0.101x，protocol total speed 约 0.147x。该负结果必须保留；它证明约 7.5K 上下文不足以摊销全局索引扫描、top-k、PCIe miss fill 和逐 token question replay。

新的 `full_prompt_then_compress` 在每任务 10 条、共 160 条的验证中消除了 query replay：Full 为 0.371040，raw sparse 为 0.365805，即 98.59% 质量保持率和 10.66% GPU KV ratio；但 raw sparse online 仍只有约 0.283x。16K 固定长度 gate 因而在该批短上下文样本上全部选择 Full，并得到与 Full 完全相同的 0.371040。该 gate 只读取长度，不读取任务标签、答案或测试分数。

LongBench 结果证明了当前方法在约 10.6% GPU KV 下的质量潜力，但没有证明短上下文加速。论文中的加速主张必须来自 16K--128K 物理实验，并同时报告 raw sparse、长度 gate、decode speed 和 protocol/E2E speed。

## 6. 论文报告约束

- 必须同时报告逻辑 attention/token ratio 与真实 GPU tensor bytes ratio。
- decode speed、online speed 和 protocol/E2E speed 必须分列，不能混用。
- 所有新优化在完成 async/sync 数值一致性和物理计时前，只能称为实现或假设，不能称为加速结果。
- 128K 主结果必须来自未参与方法选择的多个主题和不重叠窗口；单窗口 32-token 结果只保留为开发阶段证据。
- 短上下文必须报告 raw sparse，长度 gate 作为系统策略另列。

## 7. K 的 SVD 低秩表征初步实验

### 7.1 数学关系与评测协议

当前 PCA 实现对每隔 32 个 token 采样的 K 构造未中心化二阶矩 `K_s^T K_s`，再取最大特征向量。它与同一 `K_s` 的未中心化 SVD 右奇异子空间数学等价。因此，单纯把 `eigh(K_s^T K_s)` 改成 `svd(K_s)` 不会形成新的表征空间；有意义的变量是是否使用全量 token、是否中心化，以及低秩索引是否量化。

初步实验复用两份真实 32K RoPE-aware Q/K trace：sports 和 medicine。每个主题包含 layer 0/8/16/24/31、全部 32 个 query heads；总计 320 个严格配对的 layer-head-query case。对每个 query head，以精确 QK logits 的历史 top-2% token 为 oracle，报告：

- `top2 recall`：近似索引选出的同预算 top-2% 与 oracle top-2% 的 token 交集比例；
- `selected attention mass`：近似选择集合在精确 full-softmax 下的绝对 attention mass；
- `mass recall`：selected attention mass / oracle top-2% attention mass；
- FP16 低秩与实际对称 INT4（每 token scale）分开报告。

比较五种空间：当前 stride-32 uncentered PCA、同样本 uncentered SVD、stride-32 centered SVD、全量 token uncentered SVD、全量 token centered SVD。sampled PCA 与 sampled uncentered SVD 的主角余弦为 1.0，实测检索指标也仅有约 `1e-5` 量级数值差异，验证了二者等价。

结果位于：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/
  results/20260716_svd_index_recall_32k/
    sports/
    medicine/
    combined/
```

### 7.2 Rank-64 总体结果

| 表征空间 | 精度 | Top-2% recall | Recall p10 | Recall min | 绝对 attention mass | Oracle mass | Mass recall | Mass recall p10 | Mass recall min |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 当前 sampled PCA64 | FP16 | 0.7615 | 0.6342 | 0.3719 | 0.8079 | 0.8154 | 0.9897 | 0.9706 | 0.8632 |
| 当前 sampled PCA64 | INT4 | 0.6811 | 0.5761 | 0.3453 | 0.8032 | 0.8154 | 0.9827 | 0.9581 | 0.8734 |
| sampled uncentered SVD64 | INT4 | 0.6811 | 0.5761 | 0.3453 | 0.8032 | 0.8154 | 0.9827 | 0.9581 | 0.8734 |
| full uncentered SVD64 | FP16 | 0.7745 | 0.6480 | 0.4781 | 0.8085 | 0.8154 | 0.9906 | 0.9742 | 0.8932 |
| **full uncentered SVD64** | **INT4** | **0.6896** | **0.5842** | **0.4203** | **0.8039** | **0.8154** | **0.9834** | **0.9562** | **0.8888** |
| sampled centered SVD64 | INT4 | 0.7016 | 0.5684 | 0.3375 | 0.7702 | 0.8154 | 0.9419 | 0.9516 | -- |
| full centered SVD64 | INT4 | 0.7054 | 0.5858 | 0.4078 | 0.7136 | 0.8154 | 0.8775 | 0.4038 | 0.0186 |

全量 uncentered SVD64 INT4 相比当前 PCA64 INT4：平均 top-2% recall 提升 0.85 个百分点，mass recall 提升 0.07 个百分点；最差 recall 从 34.53% 提升到 42.03%，最差 mass recall 从 87.34% 提升到 88.88%。160 个跨主题 layer-head 单元中，113 个 recall 提升、41 个下降、6 个基本不变；mass recall 为 109 个提升、51 个下降。因此增益较稳定但不是逐 head 单调支配。

分主题结果：

| 主题 | 当前 PCA64 INT4 recall / mass recall | Full SVD64 INT4 recall / mass recall |
|---|---:|---:|
| Medicine | 0.6650 / 0.9826 | 0.6728 / 0.9829 |
| Sports | 0.6972 / 0.9827 | 0.7064 / 0.9839 |

### 7.3 Rank 消融与结论

| Rank，INT4 | 当前 sampled PCA recall / mass recall | Full uncentered SVD recall / mass recall |
|---|---:|---:|
| 32 | 0.5528 / 0.9584 | 0.5633 / 0.9611 |
| 48 | 0.6344 / 0.9748 | 0.6467 / 0.9764 |
| 64 | 0.6811 / 0.9827 | 0.6896 / 0.9834 |

Full SVD 在 rank 32/48/64 均优于当前 stride-32 PCA，但 full SVD48 尚不能替代 PCA64：它节省索引空间的同时，recall 和 mass recall 都明显较低。Centered SVD 的 token-count recall 有时更高，却会漏掉少数 attention mass 极大的 token；full centered SVD64 INT4 的 mass recall 仅 87.75%，且最差 case 几乎失效。因此 centered SVD 不应进入主方法。

当前最合理的研究方向不是把 `eigh` API 改名为 `svd`，而是改善 basis 的估计样本。全量 uncentered SVD 的持久存储与 PCA64 INT4 相同，但直接 SVD conversion 成本较高。生产实现应尝试流式累计全量 `K^T K` 或更密集/分层采样，再做 128x128 `eigh`；这能得到同一 full-SVD 右子空间，同时避免构造完整 U。完成 conversion latency 与下游 PPL/LongBench 配对前，本结果只能表述为索引召回改善，不能表述为端到端方法提升。

### 7.4 INT2 索引量化实验

在相同 320 个真实 layer-head-query case 上继续测试 rank-64 INT2。比较三类编码：逐 token 三值量化 `{-1, 0, 1}`、逐 token 四级均匀量化 `{-1, -1/3, 1/3, 1}`，以及后者的 group-16/group-8 分组 scale。所有方法仍直接按近似索引选出最终 top-2%，没有扩大候选集，也没有额外 exact rerank。

当前 sampled PCA64 的结果如下：

| 索引精度 | Top-2% recall | Recall p10 | Recall min | Mass recall | Mass p10 | Mass min | 逻辑索引占 Full K+V |
|---|---:|---:|---:|---:|---:|---:|---:|
| INT4，每 token 1 个 FP16 scale | **0.6811** | **0.5761** | **0.3453** | **0.9827** | **0.9581** | **0.8734** | 6.64% |
| INT2 三值，每 token 1 个 scale | 0.2030 | 0.0672 | 0.0078 | 0.6629 | 0.2081 | 0.0015 | 3.52% |
| INT2 四级，每 token 1 个 scale | 0.3320 | 0.2266 | 0.1297 | 0.8853 | 0.6621 | 0.3270 | 3.52% |
| INT2 四级，group-16 scale | 0.4453 | 0.3123 | 0.1578 | 0.9319 | 0.8117 | 0.5162 | 4.69% |
| INT2 四级，group-8 scale | 0.5199 | 0.3747 | 0.1625 | 0.9462 | 0.8572 | 0.5560 | 6.25% |

Full uncentered SVD64 的 INT2 结果也基本相同，说明主要瓶颈是 2-bit 量化误差，而不是 PCA basis 的采样误差：

| 索引精度 | Top-2% recall | Recall p10 | Recall min | Mass recall | Mass p10 | Mass min |
|---|---:|---:|---:|---:|---:|---:|
| INT4 | **0.6896** | **0.5842** | **0.4203** | **0.9834** | **0.9562** | **0.8888** |
| INT2 四级，每 token 1 个 scale | 0.3341 | 0.2280 | 0.0688 | 0.8784 | 0.6585 | 0.1827 |
| INT2 四级，group-16 scale | 0.4490 | 0.3109 | 0.0422 | 0.9325 | 0.8165 | 0.2038 |
| INT2 四级，group-8 scale | 0.5279 | 0.3934 | 0.1813 | 0.9485 | 0.8520 | 0.5561 |

索引占比按 Full raw K+V 的 FP16 bytes 计算。以 rank-64 为例，INT4 为 `(64*4 + 16)/(2*128*16)=6.64%`；逐 token INT2 为 `(64*2 + 16)/(2*128*16)=3.52%`；group-16 和 group-8 因分别需要 4 个和 8 个 FP16 scale，实际为 4.69% 和 6.25%。因此 group-8 只比 INT4 节省 0.39 个百分点索引，却让平均 mass recall 下降 3.65 个百分点，不具备替代价值。

若保持当前精确常驻 cache 与其他元数据不变，以已测总 GPU KV ratio 10.62% 为基准，替换索引后的粗略总占比分别约为：逐 token INT2 7.50%、group-16 8.67%、group-8 10.23%。这些是存储结构估算，不是重新实测的 allocator bytes。

结论：**INT2 不能直接作为同预算最终排序器替代 INT4**。三值 INT2 完全不可用；group-16 虽恢复到 93.19% mass recall，但长尾最差 case 仍只有 51.62%，风险过高。下一步最值得测试的是把 group-16 INT2 仅作为第一阶段粗召回，扩大到 5%--10% candidate，再用原始 K 或 INT4 索引 exact rerank 到最终 2%；另一条路线是按 layer/head 风险混合使用 INT2 与 INT4，而不是全局统一降到 INT2。

最终结果位于：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/
  results/20260716_svd_int2_grouped_recall_32k/
    sports/
    medicine/
    combined/
```
