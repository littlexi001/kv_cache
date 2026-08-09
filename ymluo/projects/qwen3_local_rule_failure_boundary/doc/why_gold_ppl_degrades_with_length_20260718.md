# 为什么上下文变长后正确答案 PPL 变坏

## 实验口径

- 模型：Qwen3-8B。
- 条件：clean 两跳链，英文单-token 证据，证据位于中部，`full2` query。
- 长度：0–128K，每 500 tokens 一点，共 257 点。
- 正确答案：`basket`，严格为一个 tokenizer token。因此本实验中 `PPL = 1 / P(basket)`。
- 短上下文统计：`length <= 8K`，17 点；长上下文统计：`length >= 120K`，17 点。

## 核心结论

PPL 退化不是单一的“softmax 被更多 token 稀释”，也不是单一的“query 找错了方向”。两种机制同时发生，并且在 log-attention 口径下贡献接近各半：

1. **证据方向退化**：最终证据 `basket` 的平均 raw QK logit 从 3.832 降到 0.769，变化为 -3.063，相当于 softmax 分子只剩 4.67%。
2. **候选竞争增长**：平均 head logsumexp 从 14.482 升到 17.373，变化为 +2.891，相当于在证据 logit 不变时，证据概率只剩 5.55%。
3. 两项相乘后，每个 head 的证据 attention 几何均值只剩 0.260%，即约下降 385 倍。log 衰减中方向退化占 51.4%，竞争分母占 48.6%。

所以更准确的描述是：**长上下文同时降低了真实证据的相对匹配分数，并引入了近似随候选数增长的 softmax 分母；最终证据从“很多 head 都能稳定找到”变成“少数 head 偶尔还能找到”。**

## 短上下文与长上下文对比

| 指标 | `<=8K` | `>=120K` | 变化 |
|---|---:|---:|---:|
| Gold PPL 中位数 | 13.05 | 2382.37 | 182.6× 变坏 |
| Gold PPL 均值 | 26.27 | 4305.66 | 163.9× 变坏 |
| `basket` raw logit | 3.832 | 0.769 | -3.063 |
| `basket` Q/K cosine | 0.1246 | 0.0735 | -41.1% |
| Mean logsumexp | 14.482 | 17.373 | +2.891 |
| `basket` arithmetic attention mass | 0.5687% | 0.0995% | 只剩 17.5% |
| `basket` 进入 Top-100 的 head 比例 | 32.68% | 10.68% | 只剩 32.7% |
| Query norm | 17.30 | 18.26 | +5.6% |
| Attention effective tokens | 51.0 | 277.8 | 5.45× |

Query norm 并没有缩小，反而略有增加；因此 raw logit 下跌不能用“query 向量幅度变小”解释。与此同时 Q/K cosine 明显下降，更支持**方向匹配退化**。

## 为什么判断最终证据检索是主要瓶颈

在 257 个长度点上：

| 内部指标与 `log(PPL)` | 原始 Spearman | 去除长度三次趋势后 | 相邻 500-token 差分 |
|---|---:|---:|---:|
| `basket` attention mass | -0.913 | -0.574 | -0.652 |
| `basket` 进入 Top-100 的 head 比例 | -0.875 | -0.593 | -0.661 |
| `basket` Q/K cosine | -0.816 | -0.478 | -0.538 |
| `basket` raw logit | -0.810 | -0.382 | -0.431 |

即使把平滑的长度趋势去掉，或者只比较相邻长度点的变化，`basket` mass 和 Top-100 可见性仍然与答案 PPL 强相关。这说明并非只有“长度”和“PPL”共同上升造成的伪相关；同一长度区间内，模型越找不到最终证据，答案置信度也越差。

普通 filler token 并没有表现出同样关系。例如 `office` 的 token-type mass 与 `log(PPL)` 的 Spearman 只有 -0.054。退化不是所有 token 的 attention 都等比例缩小，而是**真实链条 token 失去了相对于 filler 的优势**。

## softmax 竞争项的来源

在 `length >= 8K` 的点上，mean logsumexp 对 `log(key_length)` 的线性拟合：

- 斜率：0.881。
- `R²`：0.966。

因此分母增长几乎是候选位置数增长的确定性结果。上下文越长，哪怕单个 filler 的得分分布没有明显改变，重复出现的大量 filler 位置及极值竞争也会把 logsumexp 推高。

但它主要解释**整体趋势**，不解释相邻长度间的 PPL 尖峰。去除长度趋势后，logsumexp 的局部波动与 PPL 不再同向；局部失败更接近证据 cosine、raw logit 和 Top-100 可见性的变化。

## 约 56K 后出现更快的方向退化

对 9 点中值平滑后的曲线做分段线性拟合：

- `basket` raw logit 的候选变化点约为 56K。
- 变化点前每增加 10K，raw logit 约下降 0.130；之后约下降 0.387。
- Q/K cosine 的候选变化点约为 56.5K；斜率从每 10K 下降 0.00233 变为下降 0.00742。

证据位于中部，因此 56K context 对应的证据到 query 距离大约为 28K。这个结果更像距离增大后的位置编码/隐藏态方向逐步失真，而不是一个严格的 64K 硬边界。由于目前只有一个样本和一个 seed，这个变化点应视为待复现实验的候选现象。

## head 与层上的退化

最终证据 attention 越来越集中到少数 head：

| 区间 | Top 1% heads 承担的 `basket` mass | mass > 0.1% 的 head 数 |
|---|---:|---:|
| `<=8K` | 55.3% | 220 |
| 64K–96K | 64.1% | 87.9 |
| `>=120K` | 72.0% | 68.1 |

这意味着长上下文不仅让平均 mass 下降，还减少了冗余：更多 head 完全退出证据检索，剩余 mass 被少数 head 垄断。任何少数关键 head 的局部失配都会直接造成 PPL 尖峰。

与 PPL 关系最强的 `basket` mass 主要出现在后段层：Layer 31、33、25、32、24 的 Spearman 分别约为 -0.907、-0.904、-0.898、-0.887、-0.849。特别是 Layer 31 的长/短 mass retention 只有 4.12%。这表明答案生成前的后段证据整合是最直接的故障位置。

## 对 PPL 曲线的正确解读

PPL 随长度的趋势很强（length 与 `log(PPL)` 的 Spearman 为 0.788），但不是单调曲线。分箱中位数为：

| 长度区间 | Gold PPL 中位数 |
|---|---:|
| 1K–8K | 12.88 |
| 8K–16K | 53.95 |
| 16K–32K | 58.92 |
| 32K–64K | 139.39 |
| 64K–96K | 2358.38 |
| 96K–120K | 4840.54 |
| 120K–128K | 2382.37 |

每 500 tokens 的点会剧烈波动，因此不能把单个点当成稳定边界。更可靠的是分箱中位数、证据 mass 的强相关，以及 raw-logit/logsumexp 的可分解趋势。

## 下一步最有判别力的实验

1. **attention-logit 因果恢复**：在 128K 只把 gold evidence 的 raw logit 恢复到短上下文分布，保持所有 filler logits 不变，测答案 PPL。它直接检验方向退化是否因果。
2. **分母匹配实验**：保持长上下文 hidden states，但只在与短上下文相同数量的候选位置上做 softmax，或从 filler 中分层采样固定数量，测 PPL。它直接检验竞争分母。
3. **固定总长度、移动证据位置**：在同一个 128K filler 中把证据从中部移动到 recent，区分总长度效应与相对距离效应。
4. **多 seed/多模板复现**：至少 16–64 条 clean 链，随机 filler 和位置；否则 56K 变化点和局部 PPL 尖峰仍可能是单样本特例。

## 产物

- `artifacts/20260718_length_ppl_mechanism_analysis/mechanism_summary.json`
- `artifacts/20260718_length_ppl_mechanism_analysis/length_mechanism_rows.csv`
- `artifacts/20260718_length_ppl_mechanism_analysis/hop2_result_layer_diagnostics.csv`
- `src/analyze_attention_length_ppl_mechanism.py`
