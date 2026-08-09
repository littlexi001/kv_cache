# `scraper` tail 失败的结构与中间数值分析

## 1. `scraper + paraphrase + plain filler` 是什么

- `scraper`：本实验中的 tail 概念，含义是“刮刀”。
- Evidence definition：`a hand tool used to remove material by rubbing a sharp edge`。
- Paraphrase query：不出现 `scraper` 这个词，而是问 `an implement for stripping paint or residue from a surface`。
- Plain filler：64K 上下文由普通档案、办公室、物流等无关文本填充。
- 正确标签：`D`，对应 scraper。
- 错误输出：`G`，对应同属工具大类的 toolbox。

该错误在证据距查询 4K 和 16K 时各出现一次；tail 的全部两次错误都来自它。

## 2. 核心结论

Tail 准确率从 100% 降到 93.75% 的直接原因，并不是“所有 tail 证据都因词频低而无法进入 attention”，而是一个更具体的计算链：

1. Paraphrase 只提供“去除表面材料的工具”这一语义，模型在中后层形成的检索 query 偏向粗粒度的“工具”类别；
2. attention 没有稳定地区分 scraper 与 toolbox，正确 entry 的路由减弱，竞争 entry 增强；
3. 第 24 层以后，晚层 attention 把残差方向推向错误标签 G；
4. 晚层 MLP 没有纠正这一偏差，并把已有的 D/G 差异进一步固化为最终输出。

因此更准确的表述是：**这是“长尾概念的间接语义解析 + 同类 entry 竞争”失败，不是已被证明的 tail 频率主效应。**

## 3. Evidence routing 的中间数值

### 同一 plain filler、4K 距离

| Query | 正确 | D−G 最终 logit | Scraper entry mass | Toolbox entry mass | Scraper label mass | Toolbox label mass |
|---|---:|---:|---:|---:|---:|---:|
| 直接 lemma `scraper` | 是 | +13.484 | 0.02196 | 0.00918 | 0.00726 | 0.00236 |
| Paraphrase | 否 | -1.688 | 0.01071 | 0.01265 | 0.00377 | 0.00471 |

从直接查询改为 paraphrase 后：

- scraper entry mass 下降 51.2%；
- scraper definition mass 下降 53.2%；
- scraper 标签 D 的 attention mass 下降 48.1%；
- toolbox entry mass 增加 37.8%；
- toolbox 标签 G 的 attention mass 增加约 100%。

Scraper/toolbox 的 entry-mass 比值从 `2.39` 翻转为 `0.85`；标签 mass 比值从 `3.08` 翻转为 `0.80`。模型不是完全没有读到 scraper，而是错误 entry 在最关键的 attention 路由中相对占优。

### 与其他 tail 概念比较

在相同的 `plain + 4K + paraphrase` 条件下：

| 指标 | Scraper | 其他三个 tail 均值 | Scraper 相对变化 |
|---|---:|---:|---:|
| Entry mass | 0.01071 | 0.03214 | -66.7% |
| Definition mass | 0.000557 | 0.003737 | -85.1% |
| Label mass | 0.003770 | 0.010901 | -65.4% |
| Evidence log-odds | -7.350 | -6.352 | -0.998 nat |
| Top-20 head recall | 11.11% | 24.05% | -12.94 pp |

这说明 tail 组的总体下降由一个显著的概念级异常驱动，而不是四个 tail 概念共同退化。

## 4. 错误从哪一层出现：Logit lens

对每层输出施加最终 RMSNorm 和 LM head，计算正确标签 D 与错误标签 G 的 logit 差：

| 层后 | Direct lemma D−G | Plain paraphrase D−G | Semantic paraphrase D−G |
|---:|---:|---:|---:|
| Embedding | +4.023 | +4.023 | +4.023 |
| 11 | +0.098 | -0.431 | -0.168 |
| 19 | +2.224 | +2.678 | +1.168 |
| 23 | +2.409 | +1.637 | +1.831 |
| 24 | +6.008 | -0.015 | +3.817 |
| 29 | +6.388 | -0.849 | +4.800 |
| 31 | +9.462 | +0.566 | +6.079 |
| 34 | +8.798 | -0.662 | +5.470 |
| 35 | +13.491 | -1.691 | +7.714 |

0–23 层虽然有波动，但三种条件仍处于相近范围。决定性分叉发生在第 24 层：

- 直接查询建立了明显的 D 方向；
- plain paraphrase 没有建立该方向；
- semantic filler 下的 paraphrase 恢复了大部分 D 方向。

所以这不是 embedding 层“不认识 tail 词”，也不是早层完全无法编码 query；主要故障位于模型中后段从语义检索到答案标签决策的转换。

## 5. Attention 与 MLP 各负责多少

用最终 RMSNorm 的固定缩放和 `W_D-W_G` unembedding 方向，对每层 attention output 与 MLP output 做直接 logit attribution。所有分量之和对真实 D−G logit 的重建误差小于 0.01。

| 条件 | Attention 总贡献 | MLP 总贡献 | 最终 D−G |
|---|---:|---:|---:|
| Plain + direct lemma | +1.494 | +11.990 | +13.484 |
| Plain + paraphrase | -3.563 | +1.863 | -1.688 |
| Semantic + paraphrase | +2.062 | +5.646 | +7.719 |

Direct lemma 相对失败 paraphrase 多出的约 15.17 logits 中：

- 5.06 logits（约 33%）来自 attention 分支；
- 10.13 logits（约 67%）来自 MLP 分支。

但这不表示 MLP 是独立根因。MLP 的输入已经包含 attention 取回的信息；attention 首先决定 residual 中有哪些证据，MLP 再把该 residual 映射、放大为标签方向。

### 分层贡献

| 条件 | 层 | Attention D−G | MLP D−G |
|---|---|---:|---:|
| Plain direct | 0–11 | +0.228 | -0.229 |
| Plain direct | 12–23 | +0.567 | -0.061 |
| Plain direct | 24–35 | +0.700 | +12.280 |
| Plain paraphrase | 0–11 | +0.196 | -0.237 |
| Plain paraphrase | 12–23 | +0.118 | +0.277 |
| Plain paraphrase | 24–35 | -3.877 | +1.822 |
| Semantic paraphrase | 0–11 | +0.216 | -0.237 |
| Semantic paraphrase | 12–23 | +0.265 | +0.136 |
| Semantic paraphrase | 24–35 | +1.581 | +5.746 |

失败样本最关键的结构特征是：**24–35 层 attention 对 D−G 的合计贡献为 -3.877，直接把残差推向 G；随后 MLP 只提供 +1.822，无法抵消。**

## 6. 为什么 semantic filler 反而恢复正确

在同一个 paraphrase query 下，从 plain filler 换成 semantic filler：

- scraper entry mass 增加 26.9%；
- scraper 标签 D mass 增加 41.1%；
- toolbox entry mass 下降 22.2%；
- toolbox definition mass 下降 49.1%；
- toolbox 标签 G mass 下降 34.8%；
- D−G 从 -1.688 恢复到 +7.719。

9.41 logits 的恢复可分解为：

- attention 改善约 +5.62 logits（约 60%）；
- MLP 改善约 +3.78 logits（约 40%）。

Semantic filler 中包含工具、切割、维修等类别词。一个合理但仍需干预实验确认的解释是：它改变了后层 query/residual 的语义状态，使模型更容易把“清除表面材料”解析为具体工具功能，而不是停留在“装工具的东西”这一粗类别。结果说明 filler 不只通过 softmax 分母起作用，它也会改变 Q、hidden state 和晚层决策方向。

## 7. 对 tail 准确率下降的最终判断

当前证据支持以下因果顺序：

```text
间接 paraphrase + 同类竞争 entry
        ↓
中后层 QK/attention 对 scraper 的相对路由不足
        ↓
24–35 层 residual 被推向 toolbox/G
        ↓
晚层 MLP 放大或无法纠正已有偏差
        ↓
最终 D−G = -1.688，输出 G
```

它不能支持“tail 词频低，所以模型必然检索失败”这一普遍结论。要检验 frequency 主效应，必须增加概念数并轮换标签。

## 8. 仍需补的因果实验

1. 轮换 D/G 标签：排除标签先验和一次映射偶然性。
2. 删除 toolbox entry：若 scraper 立即恢复，说明同类竞争是必要原因。
3. 将 toolbox 替换为不同类别 entry：区分总竞争与语义近邻竞争。
4. 对第 24–35 层做 attention patching：把 semantic-success 的 attention output 替换进 plain-failure，直接测能否恢复 D−G。
5. 扩展 20–30 个 common/tail 概念，用概念、标签映射和 filler seed 做随机效应。

注意：本轮使用 8-bit 权重以适配单张 24GB 3090；direct logit attribution 是精确的加性残差分解，但不是单模块干预，因果确认仍需要 patching/ablation。

## 9. 数据路径

- 远端：`/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/scraper_structural_probe_qwen3_8b_20260722`
- `rows.jsonl`：四个条件的全量逐层、逐 head、logit-lens 和残差分解。
- `report.md`：自动摘要。
- `design.json`：模型、标签与上下文设置。
