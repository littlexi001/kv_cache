# Qwen3-8B 严格单跳数据的失败边界与 Attention 机制

## 结论

对严格单规则单跳样例：

```text
VERIFIED RULE: river → window
Query: 从 river 开始，只执行一步
Gold answer: window
```

在 18--80 token 的逐 token 密集扫描中：

- 完整词表首 token 正确率为 **16/63**；
- 在 `river / window / basket` 三个合法候选中，模型 **63/63 都选择 `window`**；
- 第一次完整词表翻转发生在 **29→30**；
- 31--35 之间发生两次短暂恢复；
- 从 **36 token 开始到 80 token 持续失败**。

因此更准确的失败边界不是一个单点，而是：

> **29--36 是输出临界带；36 以后进入稳定的首 token 失败区。**

该边界首先是完整词表输出边界，而不是稳定的单跳语义错误边界。边界附近最强竞争 token 主要是提示中的数字 `1`，说明模型在低 margin 状态下会复制 “Apply exactly 1 step” 中的步数，而不是直接输出 `window`。

---

## 1. 实验设计

模型与两跳实验保持一致：

- 模型：Qwen3-8B；
- code：英文单 token；
- seed：0；
- filler：同一确定性英文 filler 流；
- placement：middle；
- prompt style：legacy；
- attention：36 层 × 32 Query heads；
- 长度：18--80，每个整数长度均测试。

与两跳实验的差异是：上下文中只放置第一条正确规则，不再放置 `window → basket`：

```text
river → window
```

分析中的 evidence mass 定义为两个原子 token：

```text
start_key river + hop1_result window
```

---

## 2. 总体统计

| 指标 | 结果 |
|---|---:|
| 完整词表首 token 正确 | 16/63 |
| 三候选正确 | 63/63 |
| 全局 evidence mass 与完整词表 margin | \(r=+0.748\) |
| 全局 result-token mass 与完整词表 margin | \(r=+0.761\) |
| L30--33 evidence mass 与完整词表 margin | **\(r=+0.926\)** |
| L30--33 result-token mass 与完整词表 margin | **\(r=+0.926\)** |
| L30--33 evidence mass 与候选 margin | \(r=+0.765\) |

正确点和错误点的平均值：

| 范围 | 正确点 evidence mass | 错误点 evidence mass | 错误点变化 |
|---|---:|---:|---:|
| 全模型 | 1.0187% | 0.7597% | **−25.4%** |
| L30--33 | 1.9122% | 1.1484% | **−39.9%** |

这再次说明：靠近输出端的晚层 evidence mass 比全模型平均 mass 更能解释输出 margin。

对 L30--33 evidence mass 做样例内线性回归：

$$
m_{\mathrm{window},c}
\approx
-5.302
+
322.13\,M_{\mathrm{L30-33}},
$$

其中 \(M\) 使用 0--1 比例。换成百分点：

> L30--33 evidence mass 每增加 1 个百分点，完整词表 margin 平均增加约 3.22。

该回归的相关系数为 \(r=0.926\)，\(R^2=0.858\)。

使用后验阈值 \(M_{\mathrm{L30-33}}\approx1.496\%\)，可在同一批数据上区分 61/63 个正确与错误点，ROC AUC 为 0.993。该阈值只用于描述当前样例，不能直接作为跨样例阈值。

---

## 3. 边界附近的逐长度变化

| 长度 | 首 token | Gold 概率 | 完整词表 margin | 候选 margin | 全局 evidence mass | L30--33 evidence mass |
|---:|:---:|---:|---:|---:|---:|---:|
| 24 | ✓ | 59.92% | +1.844 | +3.750 | 1.2101% | 2.3781% |
| 25 | ✓ | 57.45% | +1.812 | +3.594 | 1.1871% | 2.2834% |
| 26 | ✓ | 53.20% | +1.719 | +3.594 | 0.9466% | 1.9865% |
| 27 | ✓ | 50.73% | +1.703 | +3.547 | 0.9389% | 1.9434% |
| 28 | ✓ | 52.31% | +1.703 | +3.250 | 1.0464% | 1.9567% |
| 29 | ✓ | **61.55%** | **+2.063** | +3.484 | **1.0529%** | **2.0895%** |
| 30 | ✗ | **14.07%** | **−0.500** | +3.109 | **0.8878%** | **1.5436%** |
| 31 | ✓ | 25.22% | +0.641 | +2.344 | 0.9242% | 1.8984% |
| 32 | ✓ | 17.11% | +0.047 | +2.000 | 0.9213% | 1.8028% |
| 33 | ✗ | 16.94% | −0.047 | +1.844 | 0.9377% | 1.7168% |
| 34 | ✓ | 21.51% | +0.422 | +3.000 | 0.8429% | 1.6350% |
| 35 | ✓ | 19.83% | +0.422 | +2.703 | 0.8650% | 1.7958% |
| 36 | ✗ | 8.99% | −0.375 | +2.438 | 0.7743% | 1.3842% |
| 37 | ✗ | 8.12% | −0.750 | +1.969 | 0.7108% | 1.3022% |
| 38 | ✗ | 7.94% | −0.703 | +1.906 | 0.6952% | 1.2484% |
| 39 | ✗ | 4.32% | −1.344 | +1.469 | 0.6930% | 1.1904% |
| 40 | ✗ | 7.60% | −1.281 | +1.781 | 0.6919% | 1.2962% |

关键现象：

1. 29→30 是第一次强烈翻转；
2. 30→31 立即恢复，说明 30 不是硬容量阈值；
3. 32→35 在零 margin 附近振荡；
4. 36 以后不再恢复，才进入稳定失败区；
5. 候选 margin 在整个临界带始终为正，因此模型仍把 `window` 视为正确合法答案。

---

## 4. 第一次翻转：29→30

| 指标 | 29 | 30 | 变化 |
|---|---:|---:|---:|
| Gold `window` 概率 | 61.55% | 14.07% | **−77.1%** |
| 最强竞争 token | `1`，7.82% | `1`，23.20% | **+196.5%** |
| Gold PPL | 1.6248 | 7.1063 | **×4.37** |
| 完整词表 margin | +2.0625 | −0.5000 | **−2.5625** |
| 候选 margin | +3.4844 | +3.1094 | −0.3750 |
| 全局 evidence mass | 1.0529% | 0.8878% | **−15.7%** |
| 全局 result mass | 0.7607% | 0.6667% | **−12.4%** |
| L30--33 evidence mass | 2.0895% | 1.5436% | **−26.1%** |
| L30--33 result mass | 1.6229% | 1.2479% | **−23.1%** |

### 输出 margin 的精确分解

正确答案的 log probability 变化：

$$
\Delta\log p(\texttt{window})
=
\log\frac{0.1407}{0.6155}
=
-1.476.
$$

竞争 token `1` 的 log probability 变化：

$$
\Delta\log p(\texttt{1})
=
\log\frac{0.2320}{0.0782}
=
+1.087.
$$

所以：

$$
\Delta m_{\mathrm{window},1}
=
-1.476-1.087
=
-2.5625.
$$

margin 损失来源约为：

- 57.6%：`window` 自身变弱；
- 42.4%：竞争 token `1` 增强。

L30--33 mass 的回归关系对该转折预测：

$$
\widehat{\Delta m}
=
3.221\times(-0.5459)
\approx
-1.758.
$$

实际为 \(-2.5625\)。因此晚层 mass 的线性变化能够描述约 68.6% 的本次 margin 下降；剩余部分来自 Value/OV 方向、其他层和输出模式竞争。该比例仍是描述性结果，不是干预意义上的因果贡献率。

---

## 5. 第一次翻转的 QK 与 Softmax 分解

全模型平均 target-token 指标：

| 指标 | 29 | 30 | 变化 |
|---|---:|---:|---:|
| target QK logit | 5.0995 | 4.9518 | −0.1477 |
| target QK cosine | 0.16325 | 0.16336 | 近似不变 |
| Query norm | 16.4168 | 16.0830 | −2.03% |
| target Key norm | 30.5187 | 30.4659 | −0.17% |
| softmax logsumexp | 13.1347 | 12.7320 | **−0.4027** |

softmax 分母的对数没有增大，反而下降。因此 29→30 不是由“多一个 token 撑大分母”主导。

在 L30--33：

- evidence mass 下降 26.1%；
- result-token 平均 QK logit 从 5.7232 降至 5.1137；
- 平均 logsumexp 从 13.2251 降至 12.6513；
- 57.8% 的晚层 heads 出现 evidence mass 下降；
- 基线 evidence-mass 加权的证据 log-numerator 下降 1.567；
- 同样加权的 denominator 下降 1.217；
- 证据 numerator 下降得更多，因此相对 mass 净下降。

全模型只有 44.7% 的 heads 出现 mass 下降，但原本承担较大 evidence mass 的 heads 下降更强。这说明简单统计“有多少 heads 下降”不够，应按 head 的原始 mass 和输出功能加权。

---

## 6. 关键层和 Head

与完整词表 margin 正相关最高的 heads：

| Head | 全扫描相关性 | 29 mass | 30 mass | 变化 |
|---|---:|---:|---:|---:|
| L31H28 | +0.907 | 36.71% | 17.98% | **−51.0%** |
| L29H6 | +0.885 | 20.79% | 7.35% | **−64.7%** |
| L30H9 | +0.880 | 31.25% | 23.85% | **−23.7%** |
| L31H30 | +0.872 | 20.56% | 11.00% | **−46.5%** |
| L34H27 | +0.913 | 0.177% | 0.064% | −63.6% |

其中 L31H28、L29H6、L30H9 和 L31H30 同时满足：

1. 与 margin 高相关；
2. 29 token 时承担较大的 evidence mass；
3. 30 token 时出现明显下降。

它们是后续 attention patch / OV attribution 的优先干预对象。

---

## 7. 如何解释这次单 token 转折

29 和 30 的样本不是严格的“末尾追加一个 token”：

```text
29:
... room
[river → window]
names, review cadence,

30:
... room numbers
[river → window]
, review cadence, note
```

由于 middle placement 重新居中：

- 规则块和 Query 同时后移 1；
- evidence--query 相对距离保持不变；
- 证据前后局部 filler 组成发生变化。

因此，29→30 不是直接 RoPE 距离增加，也不是单纯 softmax 分母增大，而是局部上下文重排改变了后续 Query/residual 状态；若干关键晚层 heads 的 evidence numerator 相对下降，最终让 `window` 信号变弱、数字 `1` 的输出模式增强。

这也解释了为什么 31 会恢复：临界区对 filler 截断位置敏感，并非超过某个长度后永久失效。

---

## 8. 与原两跳密集扫描的对照

| 指标 | 严格单跳 | 原两跳 |
|---|---:|---:|
| 密集范围 | 18--80 | 34--100 |
| 第一次完整词表翻转 | 29→30 | 47→48 |
| 首次翻转的主要竞争 token | `1` | `Let` |
| 候选正确 | 63/63 | 67/67 |
| L30--33 mass 与完整词表 margin | \(r=0.926\) | \(r=0.920\) |
| 错误点相对正确点的 L30--33 mass | −39.9% | −27.2% |

两个实验的共同点是：

- 晚层 evidence mass 比全局平均 mass 更能解释完整词表 margin；
- 完整词表输出可以在合法候选仍正确时提前失败；
- 首次单点翻转都对 middle-placement 的局部 filler 重排敏感；
- 所以不能仅凭 greedy 首 token 把失败解释成语义检索已经丢失。

单跳更早出现完整词表失败，并不说明单跳检索比两跳更难。这里单跳最强竞争项是 prompt 中的数字 `1`，主要反映停止/格式/复制通路；两跳首次翻转的竞争项是自然语言前缀 `Let`。两者属于不同输出模式。

---

## 9. 当前最准确的结论

1. **单跳比两跳少一步，并不自动避免完整词表输出失败。**
2. **单跳的首次翻转在 29→30，稳定失败从 36 开始。**
3. **错误点的 L30--33 evidence mass 比正确点低 39.9%，并与 margin 高度相关。**
4. **29→30 的 softmax denominator 没有增大；关键是晚层 evidence numerator 和高质量 head mass 下降。**
5. **完整词表失败主要是 `window` 与数字 `1` 的输出竞争，不是合法候选中的语义选择错误。**
6. **严格的语义检索边界尚未在 18--80 内出现；需要以候选 margin 过零或 attention patch 后的因果恢复来定义。**

---

## 数据与分析产物

- [逐长度标量轨迹](../artifacts/20260724_candidate_margin_dense_single_rule_hop1_18_80/analysis/single_hop_trace.csv)
- [失败与恢复转折](../artifacts/20260724_candidate_margin_dense_single_rule_hop1_18_80/analysis/transition_stats.csv)
- [逐层统计](../artifacts/20260724_candidate_margin_dense_single_rule_hop1_18_80/analysis/layer_attention_margin_stats.csv)
- [逐 Head 统计](../artifacts/20260724_candidate_margin_dense_single_rule_hop1_18_80/analysis/head_attention_margin_stats.csv)
- [自动分析报告](../artifacts/20260724_candidate_margin_dense_single_rule_hop1_18_80/analysis/report.md)
- [原始逐长度 JSON](../artifacts/20260724_candidate_margin_dense_single_rule_hop1_18_80/data)
