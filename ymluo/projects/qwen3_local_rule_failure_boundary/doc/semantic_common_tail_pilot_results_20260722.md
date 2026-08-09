# Common 与 long-tail 语义针：Qwen3-8B 64K pilot

## 结论

这轮 pilot **没有支持“common 语义整体上比 long-tail 语义更容易从长文本中捞出”**。

- Common：候选与 Greedy 准确率均为 `100% (32/32)`。
- Tail：候选与 Greedy 准确率均为 `93.75% (30/32)`。
- Tail 的两次失败全部来自同一个概念 `scraper`，且只发生在 `plain filler + paraphrase query`；其余三个 tail 概念为 `24/24`。
- 从模型内部指标看，tail 的证据 entry attention mass、Top-20 head recall 和证据 log-odds 反而略高于 common。因此目前的 2 次失败不能解释为“tail 证据普遍没有被 attention 捞到”，更像是概念/问法/标签映射的个例交互。

真正稳定的结果是：**相对距离从 4K 增加到 16K 会明显恶化证据的 attention 位置，而间接 paraphrase 查询比直接 lemma 查询更难。**

## 实验设计

- 模型：Qwen3-8B。
- 单卡限制：只使用物理 GPU 7；8-bit 权重，KV 与计算为 FP16；YaRN/RoPE factor 为 2。
- 长度：64K body。
- 证据到查询的相对距离：4K、16K。
- Filler：普通文本、类别相关的 semantic filler。
- 查询：直接给概念 lemma，或给独立 paraphrase clue。
- 证据中只出现定义，不出现查询词；答案是无意义的单-token A–H 标签。
- 4 个 common 与 4 个 tail 按动物、工具、容器、物质配对；所有查询词在 Qwen tokenizer 中均严格为 1 token，定义长度为 11–15 tokens。
- 短上下文门槛：8 个概念 × 3 种问法，共 24 条；全部候选正确且 Greedy 正确后才进入 64K。

| 类别 | Common（Zipf） | Tail（Zipf） |
|---|---|---|
| 动物 | horse（4.76） | mongoose（2.67） |
| 工具 | sword（4.37） | scraper（2.76） |
| 容器 | bottle（4.57） | toolbox（2.99） |
| 物质 | salt（4.60） | gypsum（2.93） |

注意：`sword=4.37` 低于最初建议的 common 下界 4.5，但仍比配对 tail 高 1.61 Zipf。正式实验应扩大概念数并将频率作为连续变量，而不是依赖这个小样本二分。

## 主要结果

### Common 与 tail

| Bin | N | 候选准确率 | Greedy 准确率 | 几何平均 PPL | 证据 entry mass | Top-20 head recall | 证据 log-odds |
|---|---:|---:|---:|---:|---:|---:|---:|
| Common | 32 | 100.00% | 100.00% | 1.0004 | 0.02558 | 18.52% | -7.509 |
| Tail | 32 | 93.75% | 93.75% | 1.1455 | 0.02697 | 20.79% | -7.382 |

Tail 虽然出现两次输出失败，但内部 evidence 指标没有整体变差：entry mass 高约 5.4%，Top-20 head recall 高约 2.28 个百分点，log-odds 也略好 0.127 nat。四个概念的样本量太小，不能把 100% 对 93.75% 当成频率因果效应。

### 相对距离

| 距离 | N | 准确率 | 几何平均 PPL | 证据 entry mass | Definition mass | Top-20 head recall | Top-2% head recall | 证据 log-odds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K | 32 | 96.88% | 1.0655 | 0.02708 | 0.002111 | 20.88% | 92.55% | -6.578 |
| 16K | 32 | 96.88% | 1.0755 | 0.02547 | 0.001323 | 18.42% | 75.07% | -8.314 |

从 4K 到 16K：

- entry mass 下降约 5.9%；
- definition mass 下降约 37.3%；
- Top-20 head recall 下降 2.46 个百分点；
- log-odds 下降 1.736 nat，相当于 evidence/background odds 缩小约 `exp(1.736)=5.67` 倍。

准确率暂时没有继续下降，是因为大多数样本仍有很大的候选 margin；内部退化已经发生，但尚未普遍越过输出边界。

### 查询形式与 filler

| 对比 | N | 准确率 | 几何平均 PPL | 证据 entry mass | 候选 margin |
|---|---:|---:|---:|---:|---:|
| Lemma query | 32 | 100.00% | 1.0044 | 0.02817 | 12.680 |
| Paraphrase query | 32 | 93.75% | 1.1410 | 0.02439 | 9.177 |
| Plain filler | 32 | 93.75% | 1.1403 | 0.02627 | 11.629 |
| Semantic filler | 32 | 100.00% | 1.0050 | 0.02629 | 10.229 |

本轮构造的 semantic filler 没有表现出更强的 softmax 稀释，反而使两条失败恢复正确。因此不能直接声称“语义相似 filler 一定更危险”；它既可能增加竞争，也可能产生语义 priming，最终取决于具体 token 分布。

## 两次失败

两次失败均为 `scraper + paraphrase`：

| Filler | 距离 | Gold PPL | Gold label | 预测 label | entry mass |
|---|---:|---:|---|---|---:|
| Plain | 4K | 7.526 | D | G（toolbox） | 0.01071 |
| Plain | 16K | 8.830 | D | G（toolbox） | 0.01057 |

在 semantic filler 下，同一概念恢复正确：4K PPL 为 1.0069，16K PPL 为 1.0014。错误 label `G` 对应另一个工具类概念 `toolbox`，说明模型至少保留了粗类别相关性，但在具体 entry 之间发生了混淆。是否由标签先验造成，需要轮换 A–H 映射后才能判断。

## 下一步（不再做密集长度扫描）

1. 固定目前最有区分度的 `64K + paraphrase` 条件，只增加 4–8 组标签轮换与 3 个 filler seed，确认 `scraper` 失败不是 D/G 标签偶然性。
2. 扩展为每档 20–30 个概念，频率作为连续变量；用概念和 filler seed 做随机效应。
3. 同时回归 token 数、定义长度、类别和短上下文 margin；只有频率系数跨 seed 稳定后，才能回答 commonness 是否决定可捞性。
4. Pilot 使用 8-bit 权重是单张 3090 的显存折中；最终关键条件再用 FP16 多卡复验。

## 产物

- 远端目录：`/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/semantic_common_tail_pilot_qwen3_8b_20260722`
- `report.md`：自动汇总表。
- `long_rows.jsonl`：64 条长上下文结果，包含逐层逐 head 指标。
- `short_rows.jsonl`：24 条短上下文熟悉度校验。
- `design.json`：概念、频率、token 数、标签映射与实验矩阵。
- `model_metadata.json`：量化与 RoPE 配置。
