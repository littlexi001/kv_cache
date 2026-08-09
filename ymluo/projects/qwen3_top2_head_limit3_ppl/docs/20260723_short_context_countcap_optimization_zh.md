# CountCap 短序列质量与速度优化

更新时间：2026-07-23

## 1. 问题

当前完整 LongBench 使用 Llama-3.1-8B-Instruct、16 个英文任务、3750 个样本，并为每个样本严格配对 Full KV 与 CountCap。旧 CountCap 在约 7.5K prompt 上存在两个问题：

1. 问题或指令后缀被拆成单 token，逐步经过稀疏 attention；Full KV 则用一次批量 SDPA 处理后缀。
2. 第一个稀疏 token 需要建立 PCA basis、投影并量化整段 K；QK-metric 在收集 4 个 query 后还会重建索引。

截至 3397/7500 行时，已经完整结束的 8 个任务中，旧 CountCap 的宏平均质量保持率为 93.42%。约 7.5K 上的时间拟合为：

| 路径 | decode 单步中位数 | 一次性索引成本 |
|---|---:|---:|
| Full KV | 44.9 ms | 0 |
| 旧 CountCap | 60.7 ms | 约 1.60 s |

因此在这个长度上，稀疏路径不仅有固定成本，steady token 本身也慢于优化后的 Full SDPA。增加生成长度不能解决该交叉点以下的速度问题。

## 2. 已有负结果

以下方向已有配对实验，不再作为默认路径重复调参：

| 方向 | 已有结论 |
|---|---|
| 每 4 步复用候选 | candidate-8% recall 从约 98% 降至 88%--92%，输出误差扩大约 9--12 倍 |
| Query-gated temporal reuse | 质量风险减小，但当前实现速度收益不稳定，7.5K 仍显著慢于 Full |
| One-shot Risk | 固定 2% 的质量成立，但 attention 子系统从 2.685x 降至 1.861x |
| 提前冻结 PCA basis | 前 8K basis 在长历史上的候选召回只有约 84%，完整 basis 约 98% |
| 无条件 split attention | 32K/64K 的 reduction 开销大于收益，只有更长工作量适合 split |

已有系统 profile 同时表明：32K 位于速度交叉点附近，64K 和 128K 的 sparse steady decode 才分别达到约 1.95x 和 2.32x。因此短序列不能强制使用 CountCap。

## 3. 当前优化

### 3.1 Dense suffix

`countcap_fullprompt` 保留相同 2% attention 和 3%--6% candidate 配置，但将完整问题或指令后缀作为一个密集 SDPA segment 处理。这样做有两个目的：

- 问题 token 使用完整上下文，避免检索误差在回答前进入隐藏状态；
- 将几十次单 token Python/CUDA 调度合并成一次批量 forward。

### 3.2 Key-PCA 单次建表

`countcap_fullprompt_keypca` 在 dense suffix 基础上移除 QK-metric 的第二次索引重建，只保留由完整 K 统计得到的 PCA basis、INT4 索引、sampled-quantile candidate 和 exact rerank。

该消融直接检验：在问题编码已经完全密集的条件下，QK-metric 的质量收益是否值得二次全历史投影成本。

### 3.3 解析式成本门控

每个实测长度分别拟合：

```text
T_decode(N, G) = T_fixed(N) + (G - 1) * T_step(N)
```

其中 `N` 为历史长度，`G` 为预计生成 token 数。当前 1641 个配对样本上，Full 与旧 CountCap 的线性拟合 R2 分别约为 0.991 和 0.9997，说明该分解能够解释绝大部分 decode 时间变化。

`countcap_auto` 只在以下条件全部成立时选择稀疏路径：

1. 相邻长度测量点的保守质量下界不低于 Full 的 95%；
2. `T_step_sparse < T_step_full`；
3. 计入 `T_fixed` 后，预测 decode 加速不低于 1.03x。

否则直接执行 Full SDPA。门控只依赖历史长度、预计生成长度和同硬件实测成本，不训练 router，也不读取任务名称。

已有配对运行可以先给出跨模型趋势，但不能代替本轮 Llama 长度扫描：

| 模型与长度 | Full steady | Sparse steady | 一次性索引估计 | 粗略 break-even |
|---|---:|---:|---:|---:|
| Llama-3.1-8B，约 7.5K | 约 37.7--44.9 ms | 约 53.7--60.7 ms | 约 1.60 s | 不存在，Sparse 单步更慢 |
| Qwen3-4B，64K | 190.4 ms | 69.4 ms | 约 2.44 s | 约 21 个 forward |
| Qwen3-4B，128K | 365.6 ms | 100.8 ms | 约 2.29 s | 约 9 个 forward |

64K/128K 使用不同模型与设备映射，这张表只说明 `N` 增大后 dense 单步增长更快、固定建表成本更容易摊销。正式 Llama 门控必须使用同模型、同 GPU 的 2K--32K 扫描结果。

## 4. 实验流水线

服务器按以下顺序自动执行：

1. 完整 16-task LongBench：7500 行，得到旧 CountCap 的正式结果。
2. 16-task m20：比较 Full、dense-suffix QK-metric、dense-suffix Key-PCA，共 960 行。
3. 2K/4K/6K/8K/12K/16K/24K/32K：每个长度 6 任务、12 个配对样本，测质量与真实交叉点。
4. 独立 16-task m20：比较 Full 与 `countcap_auto`，验证实际执行路径、质量和端到端时间。

所有 runner 支持按 `(task, sample_id, method)` 跳过已完成行；新字段追加到旧 CSV 时会沿用原表头，避免异常恢复时破坏已有结果。

## 5. 当前判断

短序列优化的目标不是强行让稀疏 attention 在所有长度上获胜。当前最合理的统一系统是：

```text
密集 prompt/question 编码
-> 硬件校准的质量约束成本门控
-> 短序列使用 Full SDPA
-> 达到交叉点且能够摊销建表时使用 CountCap
```

如果 Key-PCA 质量合格，它将作为低固定成本稀疏路径；如果质量不合格，则保留 QK-metric，并把下一步系统优化集中在完整索引转换的并行化或与 prefill 重叠，而不是继续降低 PCA 维数或复用过期候选。

## 6. 完整 LongBench 运行中的新证据

旧 CountCap 的 15 个已完整任务上，宏平均从 Full KV 的 0.3673 变为 0.3521，质量保持率为 95.88%。该数字不能掩盖任务间差异：Musique 为 76.03%，PassageCount 为 52.45%，Qasper 为 88.16%；NarrativeQA、LCC 等任务则没有下降。

时间分解表明质量问题和速度问题必须分别处理：

| 观测 | 结论 |
|---|---|
| RepoBench-P 平均 suffix 约 1010 token，TriviaQA 约 715 token | 逐 token 稀疏编码问题后缀是旧实现的主要在线开销 |
| 各任务 query speed 只有 Full 的约 0.7%--1.8% | 必须使用 dense suffix，不能把已有问题逐 token 重放 |
| 7.5K 下稀疏 decode 仍只有 Full 的约 0.54--0.79x | 短序列上检索、top-k 和多 kernel 调度成本高于节省的 QK/V 计算 |
| Musique 与普通 QA 的 suffix 都约 50 token，但质量保持率只有 76.03% | 弱项更可能来自多跳证据分散，而不是问题长度 |
| PassageCount 的平均生成长度从 4.3 增至 27.9 token | 固定 2% 先导致答案轨迹错误，再进一步放大运行时间 |

因此下一轮诊断严格配对以下路径：Full、精确 QK top-2%、精确 attention-mass 自适应预算、近似检索 top-2%、近似 attention-mass 自适应预算。若精确 top-2% 也失败，应扩大实际 attention 预算；若仅近似 top-2% 失败，应优化 PCA-INT4 的边界召回，而不是继续扩大所有任务的预算。

本轮还修复了长 suffix 触发的 PCA-INT4 索引容量异常。索引不再依赖固定的 `history + 2048` 余量，而是在需要时按 2048 token 余量自动扩容，并保留已有量化索引与 PCA basis。修复已通过单元测试和真实 Llama-3.1-8B RepoBench-P 断点续跑。

### 6.1 Dense-suffix 运行时 smoke

在正式 m20 前，利用空闲 GPU 对 Musique 与 PassageCount 各 1 个样本进行了运行时 smoke。该实验只用于验证代码路径和时间结构，不用于报告质量：

| 样本 | 路径 | Query | Decode | Online |
|---|---|---:|---:|---:|
| Musique | Full | 0.065 s | 1.228 s / 32 token | 1.294 s |
| Musique | dense-suffix QK-metric | 0.058 s | 4.050 s / 19 token | 4.109 s |
| Musique | dense-suffix Key-PCA | 0.059 s | 1.854 s / 19 token | 1.914 s |
| PassageCount | Full | 0.058 s | 0.039 s / 1 token | 0.098 s |
| PassageCount | dense-suffix QK-metric | 0.059 s | 0.527 s / 1 token | 0.586 s |
| PassageCount | dense-suffix Key-PCA | 0.059 s | 0.522 s / 1 token | 0.581 s |

Dense suffix 已把旧 CountCap 的 query 阶段从逐 token 重放恢复为与 Full 相同量级。Key-PCA 在 Musique 上又将在线时间相对 QK-metric 降低约 53%，说明二次全历史投影确实是重要固定成本。与此同时，7.5K 上首次建表约 0.5 s，steady sparse decode 仍慢于 Full；因此短序列不能无条件启用稀疏路径，必须结合长度与预计生成时域进行成本门控。

### 6.2 32K 生成时域交叉点

从官方 GovReport 中选取长度 40508 的真实样本并截断到 32805 prompt token，在同一 GPU 上改变最大生成长度：

| 生成长度 | Full online | Key-PCA online | Online speed | 质量保持率 |
|---:|---:|---:|---:|---:|
| 8 | 0.900 s | 1.785 s | 0.504x | 100.00% |
| 32 | 3.314 s | 3.233 s | 1.025x | 94.29% |
| 64 | 6.545 s | 5.103 s | 1.283x | 96.63% |

用 32/64-token 两点分解 decode 时间，首次 Key-PCA 建表约为 1.26 s，Full steady 约为 100.9 ms/token，Key-PCA steady 约为 58.4 ms/token，对应 steady attention 路径约 1.73x。8-token 点与该模型一致，综合交叉点约为 29--30 个生成 token。

该结果给出一个比固定长度阈值更精确的系统策略：是否启用稀疏路径必须由 `(history length, expected generation horizon)` 联合决定。32K 的短答案仍应使用 Full；当预计生成超过约 30 token 时，Key-PCA 才开始兑现在线加速。上述为单样本机制探针，正式阈值仍由 2K--32K 多任务配对扫描校准。

## 7. 已校验的正式结果

### 7.1 评分与长度协议

LongBench 评分已与官方实现对齐：`TriviaQA` 和 `SAMSum` 在评分前只保留预测首行，`TREC` 与 `LSHT` 同样按官方分类任务规则处理。16K 实验对完整 prompt 使用与 AdaKV 相同类型的中间截断，保留文档开头、文档结尾和完整问题后缀。所有结果均按 `(task, sample_id)` 严格配对。

16K、每任务 20 个样本的协议校准结果如下：

| 方法 | Macro score | 相对 Full 质量 | Online speed | 全流程 speed |
|---|---:|---:|---:|---:|
| Full KV | 0.48674 | 100.00% | 1.000x | 1.000x |
| Dense-suffix QK-metric | 0.48450 | 99.54% | 0.522x | 0.689x |
| Dense-suffix Key-PCA | 0.48458 | 99.56% | 0.621x | 0.769x |

本地 Full 为 48.674 分，与 AdaKV 表中 49.20 分只差 0.526 分，说明数据、模型、prompt 截断和评分协议已经基本对齐。`m20` 只用于协议和机制校准，不能替代完整 3750 样本结果。

### 7.2 质量瓶颈诊断

16 任务、每任务 8 个样本的五路严格配对诊断如下：

| 方法 | Macro score | 相对 Full 质量 | 实际 attention links |
|---|---:|---:|---:|
| Full KV | 0.40692 | 100.00% | 100% |
| 精确 QK top-2% | 0.40321 | 99.09% | 约 2% |
| 精确 mass-adaptive | 0.40538 | 99.62% | 4.05% |
| PCA-INT4 Key-PCA top-2% | 0.40910 | 100.53% | 2.01% |
| PCA-INT4 mass-adaptive | 0.40396 | 99.27% | 4.14% |

这个诊断不支持“PCA-INT4 近似召回是当前主要质量瓶颈”的假设。固定 2% 已经接近安全区间，而简单扩大到动态 4% 没有稳定改善宏平均。后续质量优化应优先定位任务机制和生成轨迹分歧，而不是继续无条件提高 PCA 维数或预算。

### 7.3 短上下文成本门控

7.5K、16 任务 `m20` 上，Key-PCA 稀疏路径只有 `0.536x` online speed；解析式成本门控将全部 320 个样本选择为 Full：

| 方法 | Macro score | Online speed | 全流程 speed |
|---|---:|---:|---:|
| Full KV | 0.43775 | 1.000x | 1.000x |
| `countcap_auto` | 0.43775 | 1.0002x | 1.0021x |

因此短序列的当前解法不是让稀疏 kernel 勉强超过 Full，而是在确定无法摊销固定成本时直接走 Full。该门控只使用实测成本、历史长度和预计生成长度，不训练 router，也不读取任务名称。

### 7.4 真实长样本二维速度边界

在同一个官方 GovReport 长样本上，分别截断到 8K、16K、24K、32K，并生成 8、32、64 token。下表的 online speed 包含 dense suffix、Key-PCA 建表、INT4 检索、exact rerank 和生成开销：

| 实际 prompt | 8 token | 32 token | 64 token | 解析 break-even |
|---:|---:|---:|---:|---:|
| 8,192 | 0.247x | 0.463x | 0.553x | 当前 steady 仍慢于 Full |
| 16,000 | 0.364x | 0.699x | 0.829x | 约 395 token |
| 24,576 | 0.444x | 0.904x | 1.105x | 约 44 token |
| 32,768 | 0.473x | 1.090x | 1.338x | 约 29 token |

这些结果验证了二维门控的必要性：同样是 32K，8-token 短回答应走 Full，32-token 左右开始越过盈亏平衡，64-token 时可获得约 1.34x 全在线加速；16K 的常规 LongBench 生成时域通常不足以摊销当前建表和检索成本。

### 7.5 当前统一方法

```text
完整 prompt 与问题使用 dense SDPA 编码
-> 根据 (历史长度, 预计生成长度) 查询同硬件实测成本模型
-> 不满足质量下界或速度收益时执行 Full KV
-> 满足条件时建立 48 维 Key-PCA + INT4 全局索引
-> 每个 KV head 近似检索候选、exact rerank，并保留 top-2%（最多 1280 token）
-> 稀疏 decode
```

### 7.6 16K 完整长上下文诊断

16 任务、3750 样本、7500 个方法样本已经全部完成并通过独立校验：

- `full_kv=3750`；
- `countcap_fullprompt_keypca=3750`；
- 3750 个 `(task, sample_id)` 严格配对；
- 共 16 个任务；
- 最大 prompt 为 16000 token；
- 日志无 Traceback、OOM 或断点缺失。

| 方法 | Macro score | 相对 Full 质量 | 平均每 head attention budget | Online speed | 全流程 speed |
|---|---:|---:|---:|---:|---:|
| Full KV | 0.47129 | 100.00% | 约 8836 token | 1.000x | 1.000x |
| Key-PCA CountCap | 0.46925 | **99.57%** | **176.7 token（约 2%）** | 0.638x | 0.773x |

分任务相对 Full 保持率：

| 任务 | Full | Key-PCA | 保持率 |
|---|---:|---:|---:|
| NarrativeQA | 26.63 | 26.51 | 99.53% |
| Qasper | 46.24 | 44.93 | 97.17% |
| MultiFieldQA | 56.99 | 56.58 | 99.29% |
| HotpotQA | 57.38 | 57.94 | 100.97% |
| 2WikiMQA | 47.80 | 48.13 | 100.68% |
| Musique | 32.15 | 32.60 | 101.43% |
| QMSum | 18.49 | 18.23 | 98.59% |
| TREC | 72.50 | 72.50 | 100.00% |
| TriviaQA | 91.83 | 91.66 | 99.82% |
| SAMSum | 39.38 | 39.04 | 99.15% |
| PassageRetrieval | 99.50 | 99.50 | 100.00% |
| PassageCount | 8.81 | 8.38 | 95.08% |
| GovReport | 21.06 | 21.10 | 100.21% |
| MultiNews | 15.97 | 15.46 | 96.79% |
| LCC | 63.20 | 62.34 | 98.65% |
| RepoBench-P | 56.15 | 55.90 | 99.54% |

该运行使用了同一套自写 ROUGE-L 评分和 16K 总 prompt 上限，因此其最可靠结论是 **Key-PCA 相对同运行 Full 保留 99.57% 质量**。它不能直接与 AdaKV Table 5 的绝对分数比较：AdaKV 使用 `rouge==1.0.1`、7.5K prompt 协议和完整官方停止符；本运行的 CSV 还将预测压缩到前 500 字符，无法无损事后重评分。

为解决该口径问题，已修复 runner：

1. ROUGE-L 改为官方 `rouge==1.0.1`；
2. Llama-3.1 decode 同时停止于 `end_of_text/eom/eot`；
3. SAMSum 在第一个换行停止；
4. CSV 保存完整 prediction，不再替换换行或截断到 500 字符；
5. 新的正式实验使用 7.5K 总 prompt 上限。

正式 AdaKV Table 5 对齐运行位于：

```text
results/20260723_countcap_adakv_table5_official75k_full_8gpu
```

该运行完成前，不报告“超过 AdaKV”的绝对结论。
