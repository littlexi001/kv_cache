# 长上下文中 Q/K“方向退化”的来源：Query、Key 还是相对位置？

## 结论

在当前 Qwen3-8B、clean 两跳英文单-token 链 `river → window → basket`、seed 0
实验中，随长度增加而观察到的 post-RoPE Q/K cosine 与 raw attention logit 下降，
主要不是证据 Key 的语义方向损坏，也不是 Query 的 pre-RoPE 语义方向损坏，而是
**Query–Key 相对距离增大后，RoPE 相对相位把原本仍然对齐的内容向量旋转成低点积方向**。

最直接的分解是 middle 8K→128K：

- 最终证据 raw logit 实际下降 `-3.910`；
- 相对位置/RoPE 贡献 `-4.630`；
- Query 内容变化贡献 `+0.349`；
- Key 内容变化贡献 `+0.371`。

位置项单独造成的损失比最终损失还大；Q、K 内容变化不是损失来源，反而合计补回约
`0.720` raw logit。

## 为什么不能直接问“Q 错了还是 K 错了”

设投影和归一化后的 pre-RoPE 内容向量为 `q`、`k`，位置为 `m`、`n`：

```text
score = (R(m)q)^T (R(n)k)
      = q^T R(n-m) k
```

因此 post-RoPE cosine 下降可以在坐标表示上描述为 Query 转了、Key 转了，
但真正与点积不变性有关的量是相对位置 `n-m`。更可靠的判别方式是分别保存
pre-RoPE Q/K，并在相同内容向量上反事实替换位置旋转。

## 现有位置实验的自然对照

同一模型、同一链、同一 query，在 8K、32K、64K、96K、128K 比较
prefix / middle / recent：

| 长度 | prefix 距离 / cosine | middle 距离 / cosine | recent 距离 / cosine |
|---:|---:|---:|---:|
| 8K | 7,782 / 0.1084 | 4,055 / 0.1222 | 328 / 0.1625 |
| 32K | 31,782 / 0.0988 | 16,055 / 0.1370 | 328 / 0.1798 |
| 64K | 63,782 / 0.0800 | 32,055 / 0.1255 | 328 / 0.1946 |
| 96K | 95,782 / 0.0506 | 48,055 / 0.1168 | 328 / 0.1924 |
| 128K | 127,782 / 0.0275 | 64,055 / 0.0541 | 328 / 0.1836 |

十五个位置点中，`log(relative distance)` 与 evidence cosine 的 Spearman 为
`-0.920`，与 raw logit 的 Spearman 为 `-0.935`。控制五个总长度的固定效应后，
`log(distance)` 对 cosine 的回归系数仍为 `-0.0208`，模型 `R²=0.919`。

### prefix：Key 完全不变

prefix 中证据 token 始终位于绝对位置 284。模型是 causal 的，所以追加在它后面的
filler 不可能改写该证据位置的 hidden state 或 Key。8K 与 128K 的向量探针进一步确认：

- pre-RoPE evidence Key 最大逐元素差异：`0`；
- pre-RoPE Key cosine：`1.000`；
- Key norm：始终 `30.7468`；
- Query norm：`17.3808 → 18.5268`，并没有缩小；
- post-RoPE Q/K cosine：`0.1084 → 0.0275`；
- raw logit：`2.9507 → -1.7797`。

因此 prefix 的方向下降不可能由证据 Key 漂移造成。

### recent：相对距离完全不变

recent 中证据与 Query 的相对距离始终为 328。8K→128K：

- post-RoPE cosine：`0.1625 → 0.1836`；
- raw logit：`5.6677 → 7.0202`；
- 位置项的反事实 raw-logit 贡献只有 `+0.0014`，基本为零。

如果 Query 的语义检索方向会仅仅因为上下文变长而普遍损坏，那么 recent 也应明显下降；
实际结果相反。这排除了“Query 内容向量全局变坏”作为主要解释。

## 新的 pre/post-RoPE 向量实验

重新运行以下六个点：

```text
length ∈ {8K, 128K}
placement ∈ {prefix, middle, recent}
```

对 36 层 × 32 query heads 保存：

- 最终 Query 的 pre-RoPE / post-RoPE 向量；
- 四种证据角色的 pre-RoPE / post-RoPE Key 向量；
- Query 和 Key 所在位置的 RoPE cos/sin；
- 重建 raw logit 与 cosine。

然后对两个端点构造完整 `2³` 反事实：Query 内容来自左/右端点、Key 内容来自左/右端点、
位置旋转来自左/右端点。使用 Shapley 值分摊交互项。

FP16 向量重建实际 raw logit 的平均绝对误差为 `0.0014–0.0032`，重建 cosine 的平均绝对
误差为 `3.0e-5–5.1e-5`。

## 反事实分解结果

### 长度增长：8K→128K

| placement | 总 raw-logit 变化 | Query 内容 | Key 内容 | 相对位置/RoPE |
|---|---:|---:|---:|---:|
| prefix | -4.730 | +0.202 | 0.000 | **-4.932** |
| middle | -3.910 | +0.349 | +0.371 | **-4.630** |
| recent | +1.353 | +0.670 | +0.681 | +0.001 |

对应 cosine 分解：

| placement | 总 cosine 变化 | Query 内容 | Key 内容 | 相对位置/RoPE |
|---|---:|---:|---:|---:|
| prefix | -0.08096 | +0.00032 | 0.00000 | **-0.08127** |
| middle | -0.06811 | +0.00335 | +0.00821 | **-0.07966** |
| recent | +0.02111 | +0.00677 | +0.01433 | +0.00001 |

middle 中 82.3% 的 layer-head 对得到负的 position raw-logit 贡献；prefix 中为 74.7%。

### 固定 128K，仅移动证据位置

middle→recent 将相对距离从 64,055 缩短到 328：

- raw logit 恢复 `+7.098`；
- position/RoPE 贡献 `+7.415`；
- Query 内容贡献 `-0.284`；
- Key 内容贡献 `-0.033`；
- 95.4% 的 layer-head 对得到正的 position 贡献。

pre-RoPE 向量在两个位置条件之间仍非常接近：

- Query cosine：`0.968`；
- evidence Key cosine：`0.993`。

因此这里的方向恢复几乎完全来自相对位置，而不是把证据换成了更好的 Key 或把 Query
换成了更好的语义向量。

## pre-RoPE 与 post-RoPE 的直接对照

| placement | pre-RoPE cosine 8K→128K | post-RoPE cosine 8K→128K |
|---|---:|---:|
| prefix | 0.1925 → 0.1911 | **0.1084 → 0.0275** |
| middle | 0.1997 → 0.2267 | **0.1222 → 0.0541** |
| recent | 0.2061 → 0.2280 | 0.1625 → 0.1836 |

middle 的 pre-RoPE 内容匹配实际上改善，但 post-RoPE 匹配下降。这是“语义方向没有坏，
相对位置旋转坏了”的最直观证据。

## 与答案 PPL 的关系

该结论解释的是 attention numerator 中的方向项，不代表 PPL 完全由这个方向项决定。
例如 recent 8K→128K 的 evidence raw logit 上升，但 PPL 仍从 `5.59` 上升到 `904.69`。
原因包括：

1. softmax 分母仍随 token 数增长；
2. 多跳状态更新和 Value 汇聚仍可能失败；
3. 后续层的 residual/readout 可能丢失已经检索到的信息；
4. 不同位置会改变 sink、局部规则绑定和跨层传递路径。

另外在 128K，prefix 的 PPL 比 recent 好，但 prefix 的平均 evidence QK cosine 更低，
也说明不能用单一平均 QK 指标预测所有位置条件下的最终生成质量。

## 当前结论的边界

- 模型：Qwen3-8B；
- 数据：一条 clean 两跳英文单-token 链；
- seed：0；
- Query：full2；
- RoPE 扩展 factor：4；
- 分解是保存向量上的精确代数反事实，不是重新执行三种混合 hidden-state 模型。

下一步若要给出跨任务结论，应至少扩展到 16 个 seed、多个普通英文词链，并比较
factor 1 / factor 2 / factor 4 的 position-only 曲线。
