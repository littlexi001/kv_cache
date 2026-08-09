# 长文本中“针的可捞性”与失败边界

## 1. 核心问题

主问题不是再次证明上下文增长会稀释 attention，而是回答：

> 一条独立存在时可被正确使用的证据，在多大背景中会消失；这个边界由证据本身的什么性质决定？

第一阶段比较 common、medium、long-tail 三类有效证据。第二阶段再比较普通 filler、语义相似干扰和冲突证据。

## 2. 可直接量化的 softmax 定义

设证据位置集合为 \(E\)，其余位置为 \(B\)，pre-softmax attention logit 为 \(s_i\)：

\[
S_E = \operatorname{logsumexp}_{i\in E}(s_i), \qquad
S_B = \operatorname{logsumexp}_{j\in B}(s_j).
\]

定义一根针在当前 head 上的可捞性为：

\[
R = S_E-S_B.
\]

证据获得的总 attention mass 恰好是：

\[
M_E = \frac{e^{S_E}}{e^{S_E}+e^{S_B}} = \sigma(R).
\]

因此，真正随 filler 增长而“填满 softmax”的量是背景分区函数 \(S_B\)，不是 token 数本身。若把可用证据质量阈值记为 \(\tau\)，softmax 意义下的失败边界满足：

\[
R < \log\frac{\tau}{1-\tau}.
\]

同样长度的语义相似或冲突 filler 若具有更高 attention logits，会比普通 filler 更快增大 \(S_B\)。

## 3. Common vs long-tail 配对实验

### 数据控制

- 用英文语料 Zipf frequency 定义 commonness，分 common / medium / tail 三档。
- 只使用 WordNet 中的具体名词。
- 每个词在 Qwen3-8B tokenizer 中严格是一个 leading-space token。
- 所有证据、查询和答案采用相同 token 边界：`FINAL CODE:<leading-space-token>`。
- 每档 8 个词，循环轮换作为两跳链的起点、中间点和最终答案，消除答案角色偏差。
- 相同样本编号的三档使用完全相同的 filler。

### 因果对照

每个样本、每个长度构造两份等长输入：

1. evidence：在固定位置覆盖两条真实规则；
2. matched no-evidence：恢复这些位置原有的 filler token。

二者长度、查询、答案边界和 filler 完全相同。定义证据增益：

\[
\Delta_{evidence} = \log p(y\mid x,E)-\log p(y\mid x,\varnothing).
\]

它可以把“模型本来更喜欢输出常见词”与“模型更容易取回常见证据”区分开。

### 主要指标

- 同频率档候选准确率：主要决策指标，避免常见词先验跨档竞争。
- greedy 首 token 准确率：稳定的自然输出指标。
- gold token PPL 与证据增益 \(\Delta_{evidence}\)。
- 证据规则 attention mass、证据 token mass、两条规则同时进入 Top-2% 的 head 比例。
- 去掉 Top-20 attention token 后的剩余 mass。
- 背景 `logsumexp`、背景最大 logit、证据 log-odds \(R\)。

### 长度与位置

softmax 主实验采用 `fixed_recent`：证据到查询的相对距离固定为常数，新增 filler 全部放在证据之前。这样首先隔离背景竞争，避免把 RoPE 相对距离变化混入 common-tail 结论。

粗扫长度：1K、4K、8K、16K、32K、48K、64K、80K、96K、112K、128K。发现每根针的转折区后，再以 500 或 1K token 间隔局部细扫。

## 4. 怎样判定“更容易捞”

- common 只有 raw PPL 更低，但证据增益和 \(R\) 没有更高：只是输出先验，不是检索更强。
- common 的证据增益、规则 mass 和 \(R\) 都更高：common 语义确实具有更晚的失败边界。
- tail 的规则 mass 或 \(R\) 更高：长尾词的独特性降低了背景碰撞，抵消甚至超过其较弱的学习强度。
- 不应预设 common 一定更好。最终控制量是 \(S_E-S_B\)：common 可能提高 \(S_E\)，tail 也可能降低与其竞争的有效 \(S_B\)。

## 5. 第一主线后续实验

### 5.1 失败是渐变还是突变

对每根针同时记录 candidate margin、PPL、证据 mass 和自然输出。粗扫后在 candidate margin 过零附近局部细扫，拟合 change point。由于模型可能出现恢复，另报告“首次失败”和“此后不再恢复的持续失败”。

### 5.2 多证据是否同步消失

构造 1、2、4 跳链，分别记录每条规则的 mass 和 Top-2% 命中：

- 是否总是最弱的一条先消失；
- 两条证据的失败边界是否相关；
- 最终回答失败能否由 `min(R_1, R_2)` 或链级组合量预测。

### 5.3 干扰稀释倍数

在完全相同长度下比较：普通 filler、主题相似 filler、错误但不冲突规则、直接冲突规则。定义相对普通 filler 的背景增长倍数和边界提前量：

\[
\rho = \frac{\exp(S_B^{type})}{\exp(S_B^{plain})}, \qquad
\Delta N_{fail}=N_{fail}^{plain}-N_{fail}^{type}.
\]

这能直接回答“冲突信息比普通文本强多少倍地稀释真实证据”。

### 5.4 模型内部来源

在 common-tail 结果明确后，再分解：

- RoPE 前 Q 与证据 K 的语义相似度；
- 证据 K 在每层 K 矩阵高奇异值主子空间与长尾子空间的投影；
- commonness、子空间投影、证据 logit 和失败边界之间的路径关系。

## 6. RoPE 独立支线

RoPE 支线的合理主张应是“位置分辨率需要随语义尺度变化”，而不是所有远程信息都不需要顺序。

需要两个相反任务共同验证：

1. 局部顺序敏感任务：短句语序、否定、局部依赖；
2. 远程语义检索任务：规则或证据的位置顺序不影响答案。

比较标准 RoPE、远程 RoPE-free、距离截断/冻结 RoPE，以及“局部标准 attention + 远程无位置语义记忆”双通道。除平均准确率外，重点报告不同证据距离下 PPL 的方差和最坏值，以验证导师强调的 consistency 与 predictability。

## 7. 优先级

1. 完成 fixed-relative common-tail 边界实验并得到 \(S_E-S_B\) 的定量关系。
2. 在每档转折区细扫，确认边界形态。
3. 跑普通/相似/冲突 filler，计算稀释倍数。
4. 跑多证据同步失效与链级预测器。
5. 再把 RoPE-free remote semantic path 作为独立方法线推进。
