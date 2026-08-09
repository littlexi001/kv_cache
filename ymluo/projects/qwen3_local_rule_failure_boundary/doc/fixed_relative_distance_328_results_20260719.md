# 固定证据—Query 相对距离的 0–128K filler 实验

## 实验设置

- 模型：Qwen3-8B；
- 数据：clean 两跳英文单-token 链 `river → window → basket`；
- Query：full2；
- seed：0；
- 可变项：证据链之前的 filler，`0:128000:500`，共 257 点；
- 证据链之后固定保留 256 个 filler token；
- 最终证据 `basket` 到最终 Query token 的相对距离：所有点均严格为 328；
- 总 body 长度：`filler + 34-token evidence block + 256 = filler + 290`。

257/257 点全部完成，相对距离唯一值校验为 `[328]`。

## 核心结论

固定相对距离后，随长度增长出现的系统性 Q/K 方向退化消失：

| 指标 | 短：filler ≤8K | 长：filler ≥120K | 变化 |
|---|---:|---:|---:|
| Gold PPL 中位数 | 6.932 | 40.075 | 变坏 5.78× |
| Gold 概率中位数 | 0.1443 | 0.0250 | 只剩 17.3% |
| evidence raw logit | 5.024 | 7.249 | **+2.226** |
| evidence Q/K cosine | 0.1499 | 0.1923 | **+0.0424** |
| evidence attention mass | 0.00723 | 0.00437 | 只剩 60.5% |
| mean head logsumexp | 14.486 | 17.048 | +2.562 |
| evidence rank | 432 | 2,116 | 绝对名次变大 |

因此，middle 实验中的方向下降不是“只要上下文变长就必然发生”的现象。只要把证据和
Query 的距离固定，raw logit 和 cosine 都随长度提高。

## 与原 middle 实验对照

原 middle 条件中，证据—Query 距离随长度增长到约 64K：

| 指标 | 固定距离 328 | middle 距离增长 |
|---|---:|---:|
| 短→长 raw-logit 变化 | **+2.226** | **−3.063** |
| 短→长 cosine | 0.1499→0.1923 | 0.1246→0.0735 |
| 短→长 PPL 中位数倍率 | 5.78× | 182.59× |
| 短→长几何 attention 倍率 | 0.714× | 0.00260× |

长区间（filler ≥120K）逐点配对比较：

- 固定距离 PPL / middle PPL 的中位数：`0.0272`，即固定距离约好 **36.8×**；
- 固定距离 evidence raw logit 平均高 `+6.481`；
- 固定距离 evidence cosine 平均高 `+0.1188`；
- 固定距离 evidence attention mass 中位数高 `5.63×`。

按长度区间看 Gold PPL 中位数：

| filler | 固定距离 328 | middle | 固定距离改善 |
|---|---:|---:|---:|
| 0–8K | 6.93 | 13.05 | 1.88× |
| 8–32K | 5.55 | 53.95 | 9.72× |
| 32–64K | 11.34 | 145.49 | 12.83× |
| 64–96K | 13.30 | 2,358.38 | 177.28× |
| 96–120K | 45.69 | 4,840.54 | 105.94× |
| 120–128K | 39.67 | 2,162.04 | 54.50× |

这进一步验证了 pre/post-RoPE 向量反事实实验的结论：middle 长度实验中的 Q/K 方向退化
主要由相对位置增长造成。

## 固定距离后为什么 PPL 仍然变坏 5.78×

### 1. softmax 分母仍增长

短→长：

- evidence raw logit 增加 `+2.226`，numerator 增加 `exp(2.226)=9.26×`；
- head logsumexp 增加 `+2.562`，competition factor 为 `exp(-2.562)=0.0771`，
  相当于约 12.97× 稀释；
- 两者合并后的几何 attention mass 为 `0.714×`。

`logsumexp` 对 `log(key length)` 的回归：

- slope：`0.841`；
- `R²=0.978`。

说明即使检索方向改善，更多候选 token 仍然稳定扩大 softmax 分母。

### 2. attention mass 下降小于答案概率下降

- arithmetic evidence mass：只下降到 60.5%，约 1.65× 损失；
- Gold 概率中位数：下降到 17.3%，约 5.78× 损失。

因此剩余 PPL 退化不能只归因于最终证据 attention mass。检索之后的 Value 汇聚、两跳状态
更新、跨层 residual 传递和 vocabulary readout 仍然在长上下文中退化。

### 3. 局部内容轨迹仍会造成 PPL 尖峰

固定长度趋势后，log(PPL) 与：

- evidence mass：Spearman `−0.757`；
- evidence cosine：`−0.669`；
- evidence raw logit：`−0.600`。

这意味着固定距离消除了系统性的方向下降，但不同 filler 截断点会改变 Query/Key 的
pre-RoPE 内容表示，局部的 Q/K 波动仍与 PPL 尖峰相关。

## 异常点与统计口径

filler=0 的 PPL 为 `2102.38`，而 filler=500 时恢复为 `6.93`。filler=0 时证据块从绝对位置
0 开始，之后才有 256 个 filler；它同时改变了绝对位置、前缀结构和证据的上下文化方式，
属于明显的 start-of-sequence/结构异常点。

因此短区间应使用中位数而不是均值：短区间 PPL 中位数为 6.93，但均值被该点拉到 131.26。
该异常点不影响“固定距离后长区间方向不再下降”的结论。

## 当前结论边界

当前仍是一条词链、一个 seed。能够支持的是机制结论：

> 在这条样本上，相对距离增长是原 middle 实验中系统性 Q/K 方向退化的必要因素；固定
> 距离后方向退化消失，但 softmax 竞争和 attention 之后的计算仍使答案 PPL 随长度上升。

下一步最有价值的是使用 8–16 条词链重复固定距离实验，并对长上下文做 evidence-Value
注入或 residual patching，以定位剩余约 3.5×（超过 attention-mass 损失部分）的下游退化。
