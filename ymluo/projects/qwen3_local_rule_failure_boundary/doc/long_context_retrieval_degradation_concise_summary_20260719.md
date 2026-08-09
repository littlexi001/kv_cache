# Qwen3-8B 长上下文证据检索退化：当前结论

## 一句话结论

长上下文中的答案置信度下降不是单一问题，而是两个可分离的主要机制叠加：

1. **相对距离增长破坏 QK 检索方向。** 在证据位于文本中部、Query 位于末尾时，
   evidence–Query 距离随 filler 增长；RoPE 相对相位使原本仍对齐的语义 Q/K 旋转到
   更低点积的方向。
2. **候选 token 增长扩大 softmax 分母。** 即使证据方向不变或变好，更多 filler
   token 仍会分走 attention mass。

固定 evidence–Query 距离可以大幅缓解第一项，但不能消除第二项，也不能修复
attention 之后的 Value 汇聚、两跳状态绑定、residual 传递和答案 readout。

## 当前最重要的结论

### 1. 不是 Query 或证据 Key 的语义方向普遍损坏

在 middle 8K→128K 的向量反事实分解中，最终证据 raw logit 的变化为：

| 来源 | raw-logit 贡献 |
|---|---:|
| Query 内容变化 | +0.349 |
| Key 内容变化 | +0.371 |
| 相对位置 / RoPE | **−4.630** |
| 合计 | **−3.910** |

Query 和 Key 的 pre-RoPE 内容变化合计补回 `+0.720`，真正的主要负项是相对位置。
因此更准确的表述是：

> 语义 Q/K 仍然可以对齐，但不断增长的相对距离通过 RoPE 把它们旋转成更低点积的方向。

### 2. 固定相对距离后，系统性的 QK 方向退化消失

新增实验把 evidence–Query 距离严格固定为 328，同时将 prefix filler 从 0 扩展到
128K。短区间（≤8K）与长区间（≥120K）对比：

| 指标 | 短 | 长 | 变化 |
|---|---:|---:|---:|
| evidence raw logit | 5.024 | 7.249 | **+2.226** |
| evidence Q/K cosine | 0.1499 | 0.1923 | **+0.0424** |
| evidence attention mass | 0.00723 | 0.00437 | 降至 60.5% |
| head logsumexp | 14.486 | 17.048 | +2.562 |
| Gold PPL 中位数 | 6.932 | 40.075 | 变坏 5.78× |

这说明随机 filler 本身不足以造成总体 QK 方向下降。原 middle 实验中的方向下降，
主要来自 filler 使证据和 Query 的相对距离不断增大，而不是“上下文只要变长，Query
语义方向就必然损坏”。

### 3. 分母稀释与距离退化是相互独立的

固定距离后，证据分子因 raw logit 上升而增强 `9.26×`；但 softmax 分母仍增强约
`12.97×`。两者合并后，几何 attention-mass 代理为 `0.714×`，实际算术平均
evidence mass 降至 `0.605×`。

因此固定距离虽然保住了“瞄准证据的方向”，却不能阻止更多 token 分享 softmax
概率。固定距离实验中，`logsumexp` 对 `log(key length)` 的斜率为 `0.841`，
`R²=0.978`，说明竞争池增长是稳定的长度效应。

### 4. 对最终 PPL 的影响程度

| 结果 | middle 距离增长 | 固定距离 328 |
|---|---:|---:|
| 短→长 raw-logit 变化 | −3.063 | +2.226 |
| 短→长 PPL 倍率 | **182.59×** | **5.78×** |
| 实际平均 evidence-mass 倍率 | 0.175× | 0.605× |

在长区间逐点配对时，固定距离的 PPL 约为 middle 的 `0.0272`，即约好 `36.8×`。
若只把短→长 PPL 倍率换算到 log-PPL 尺度，固定距离约消除了 `66.3%` 的退化量。

这个 `66.3%` 是控制实验的效应量，不是严格的因果占比。固定距离仍有 `5.78×` PPL
恶化，且其 attention mass 只损失约 `1.65×`，说明剩余退化还包括 Value 汇聚、两跳
状态更新、跨层 residual 传递和最终词表 readout。

## 当前机制图景

```text
filler 增长
├─ evidence–Query 相对距离增长
│  └─ RoPE 相对相位变化
│     └─ 正确证据 QK logit / cosine 下降
│        └─ attention numerator 下降
│
├─ 候选 token 数增长
│  └─ logsumexp / 极值竞争增长
│     └─ softmax denominator 扩大
│        └─ 正确证据 attention mass 被稀释
│
└─ 更长的状态传播路径与更多 Value 噪声
   └─ 两跳绑定、residual 与 readout 进一步损失
      └─ Gold probability 下降、PPL 上升
```

## 结论边界

- 当前最强机制实验来自 Qwen3-8B、一条 clean 英文单-token 链
  `river → window → basket`、seed 0、full2 Query。
- “RoPE 是主要方向损失来源”已得到位置对照和 pre/post-RoPE 向量反事实的共同支持；
  但跨任务结论仍需更多词链和 filler seed。
- 平均 QK、attention mass 与 PPL 不是一一对应关系。PPL 同时衡量检索、任务路由、
  两跳组合和答案槽 readout。
- filler=0 在固定距离实验中是明显的起始结构异常点；短区间采用中位数，避免其影响。
- 若要把位置、分母和下游传播精确分摊到 PPL，下一步应在相同 hidden states 上做
  `RoPE-distance patch × competitor masking` 的 2×2 干预，并在 NLL 尺度做 Shapley 分解。

---

## 实验支撑

### A. Middle 0–128K 密集扫描

- 模型：Qwen3-8B，FP16；
- 数据：clean 两跳英文单-token 链；
- filler：0–128K，每 500 tokens 一个点，共 257 点；
- 证据：放在 filler 中部；Query 位于末尾；
- 记录：36 层 × 32 heads 的 evidence logit、cosine、mass、rank、logsumexp，以及
  Gold probability/PPL。

主要现象：随着长度增加，证据 QK 方向下降、softmax 分母增长、证据 rank 变差，
Gold PPL 中位数的短→长倍率为 `182.59×`。

### B. Prefix / Middle / Recent 位置对照

在 8K、32K、64K、96K、128K 对比三种位置。15 个位置点中：

- `log(relative distance)` 与 evidence cosine 的 Spearman：`−0.920`；
- 与 evidence raw logit 的 Spearman：`−0.935`；
- 控制总长度后，距离对 cosine 的回归系数仍为 `−0.0208`，`R²=0.919`。

Prefix 条件中，证据 Key 位于因果前缀中的固定绝对位置，8K 与 128K 的 pre-RoPE Key
逐元素最大差异为 0，但 post-RoPE cosine 从 `0.1084` 降到 `0.0275`。因此 Key 漂移
不是必要条件。

Recent 条件中相对距离恒为 328，8K→128K 的 raw logit 从 `5.6677` 上升到 `7.0202`，
position-only 贡献只有 `+0.0014`。因此 Query 内容随长度普遍损坏也不是主要解释。

### C. Pre/Post-RoPE 向量反事实

对 8K/128K × prefix/middle/recent 保存 Query、Key 和 RoPE cos/sin，并构造 Query
内容、Key 内容、相对位置三个因素的 `2³` 反事实。FP16 重建 raw logit 的平均绝对
误差为 `0.0014–0.0032`。

Middle 8K→128K：

- pre-RoPE cosine：`0.1997 → 0.2267`；
- post-RoPE cosine：`0.1222 → 0.0541`；
- position raw-logit Shapley 贡献：`−4.630`；
- 82.3% 的 layer-head 得到负的位置贡献。

在固定 128K 下，把证据从 middle 移到 recent，raw logit 恢复 `+7.098`，其中
position/RoPE 贡献为 `+7.415`；95.4% 的 layer-head 得到正的位置贡献。

### D. 固定相对距离 328 的 0–128K 扫描

- 在证据链之前增加可变 filler；
- 证据链之后固定保留 256 filler tokens；
- 最终证据到 Query 的距离始终为 328；
- 0–128K、每 500 tokens，共 257/257 点完成。

结果：raw logit 与 cosine 随长度总体上升，证明系统性方向退化被消除；但
logsumexp、证据 rank 和 PPL 仍上升，证明候选竞争和下游退化独立存在。

完整报告：

- [固定相对距离实验](fixed_relative_distance_328_results_20260719.md)
- [Q/K 方向与 RoPE 分解](qk_direction_relative_position_decomposition_20260719.md)
- [英文单-token 128K 机制研究](english_single_token_128k_length_failure_mechanism_20260718.md)

---

## 公式推导

### 1. RoPE 只通过相对位置进入 QK 点积

设 pre-RoPE 内容向量为 `q`、`k`，位置为 `m`、`n`：

\[
q_m=R(m)q,\qquad k_n=R(n)k.
\]

RoPE 旋转矩阵是正交矩阵，因此：

\[
q_m^\top k_n
=q^\top R(m)^\top R(n)k
=q^\top R(n-m)k.
\]

所以共同平移绝对位置不改变位置项：

\[
R(m+c)^\top R(n+c)=R(n-m).
\]

在第 `i` 个二维频率平面中，点积贡献可写为：

\[
s_i(\Delta)=A_i\cos(\Delta\omega_i)+B_i\sin(\Delta\omega_i),
\qquad \Delta=n-m.
\]

当相对距离 `Δ` 增长时，不同频率维度产生不同相位，原本的相长叠加可能变为相消叠加。
RoPE 不改变 Q、K 范数，因此 post-RoPE cosine 的下降直接反映相对旋转后的方向失配。

### 2. Softmax 分子—分母分解

正确证据 `e` 的 attention 为：

\[
a_e=\frac{\exp(s_e)}{\sum_{j=1}^{N}\exp(s_j)}
=\exp\left(s_e-\operatorname{LSE}(s_1,\ldots,s_N)\right).
\]

从短上下文到长上下文：

\[
\Delta\log a_e
=\Delta s_e-\Delta\operatorname{LSE}.
\]

其中：

- `Δs_e` 是证据 QK 方向与模长共同决定的 numerator 变化；
- `ΔLSE` 是候选数量和竞争强度共同决定的 denominator 变化。

固定距离实验中：

\[
\exp(\Delta s_e)=\exp(2.226)=9.26,
\]

\[
\exp(-\Delta\operatorname{LSE})=\exp(-2.562)=0.0771,
\]

\[
\exp(2.226-2.562)=0.714.
\]

因此方向改善抵消了大部分、但没有全部抵消分母竞争。

### 3. PPL 与内部 attention 不成线性比例

Gold PPL 定义为：

\[
\operatorname{PPL}=\exp\left(-\frac{1}{T}\sum_{t=1}^{T}\log p(y_t\mid y_{<t},x)\right).
\]

Attention mass 只影响一次层内 Value 加权：

\[
o=\sum_j a_jv_j.
\]

随后还要经过输出投影、residual、MLP、后续层和词表 readout。因此不能把
attention mass 下降 `x×` 直接解释成 PPL 上升 `x×`；严格归因应在 NLL 尺度上对
位置、竞争和下游传播分别做干预。
