# Introduction 大纲：RoPE 的相位敏感长程检索失效

> 目标：按照下面的段落逻辑起草 Introduction。当前只搭建论证骨架，不需要展开完整文字；方法和结果允许保留多个可能分支。

## 核心主张

即使输入仍处于模型训练或支持的上下文范围内，固定 evidence-query 的语义匹配也可能因 RoPE 相位而在部分相对位置受到抑制；这种早期扰动会在真实模型中经验性地累积为跨层检索退化，因此精确位置可能只应作用于局部范围，而远程检索需要一个弱化位置影响的语义通道。

## Paragraph 1：问题与现象

- 长上下文窗口变长，并不代表模型能够可靠使用其中的远程信息。
- 在 Needle-in-a-Haystack 类任务中，即使 evidence 和 query 内容完全不变，仅改变二者距离或在中间加入 filler，模型也可能从正确回答突然变为失败。
- （可以给出一个例子+图，说明突然的失败/周期性的失败）
- 引用 Lost in the Middle、RULER、Needle-in-a-Haystack 等长上下文利用失败工作。

## Paragraph 2：现有解释与核心 claim

- 既有研究常把长上下文失败归因于长度 OOD、RoPE 频率外推、softmax dilution 或注意力不确定性。
- 近期工作也指出 RoPE 会产生 position/token inversion 或 aliasing，如 **RoPE Distinguishes Neither Positions Nor Tokens in Long Contexts, Provably**。
- 但本文揭示一条更本质的因果链：

```text
固定 evidence-query 的位置变化
-> 第一层语义分数受到相位抑制
-> 扰动进入 residual
-> 后续表示与检索逐层分叉
-> 最终答案失败
```

## Paragraph 3：模型第一层的理论分析

- RoPE 将 Q/K 的每个二维分量按不同频率旋转，使固定内容的 QK score 成为相对距离的振荡函数。
- 因此，在足够广的相位范围内，必然存在部分距离使正确 evidence 的贡献减弱、归零或变为负贡献。
- 固定、位置无关的线性 Q/K 投影可以改变初始方向和各频率权重，但无法保证非零语义匹配在所有相位上始终保持稳定高分。
- 第一层实验保持 pre-RoPE Q/K 不变，仅改变相对位置，并验证 post-RoPE score 的变化可以由相位分解重构。

## Paragraph 4：跨层经验与因果证据

- 第一层 attention 变化会改变 Value 写入和 residual state。
- 后续层不是对同一个 Q/K 反复旋转，而是从已经改变的 hidden state 中生成不同的 pre-RoPE Q/K。
- 实验观察到 residual difference、pre-RoPE Query drift 和检索差异随深度总体增大。
- Activation patching 表明，把正常距离运行的中后层状态替换进失败运行可以恢复答案，且较深层 patch 的恢复率更高。
- 准确表述：
  - 这是“累计性计算退化”的经验和因果证据；
  - （现在我们似乎尚未理论证明每一层的 evidence relevance 必然单调下降？）

## Paragraph 5：设计启发、可能方法与效果

设计启发：

- 精确位置对局部语法、语序和 evidence block 内部结构重要。
- 对远程 evidence 是否相关的候选排序，细粒度距离可能是干扰变量，不应压倒语义相关性。

可能的方法分支可以同时保留：
1. **训练时方案**：只在局部窗口使用 RoPE；远程 attention 使用 NoPE、部分 NoPE 或可学习 gate，让模型在训练中适应局部位置与全局语义的分工。
2. **推理时方案**：冻结模型，局部保留标准 RoPE；远程 token 使用 pre-RoPE semantic score、弱位置分数或 pre/post-RoPE 混合进行召回。
3. **可选增强**：限制远程候选预算、按 head 校准两类分数，或在召回 evidence block 后修复其整体位置并保留内部顺序。

结果部分暂时留槽位：

- 在受控 phase-sensitive failure setting 中，方法将 `[主要检索指标]` 从 `[baseline]` 提升到 `[result]`。
- 在 `[模型/长度/任务]` 上减少了不利位置导致的失败，并改善 evidence recall、attention mass 或答案概率。
- 在短上下文语言建模和局部顺序任务上保持 `[基本不变/可接受退化]`。
- 方法增加的训练、延迟、显存或候选预算为 `[cost]`。

## Contributions 段（可选）

建议 Introduction 最后暂列三项贡献：

1. 定义并刻画一种 supported-context 内的 phase-sensitive retrieval failure：固定 evidence-query 内容也会因相对位置变化而发生检索退化。
2. 建立从第一层 RoPE 相位抑制到跨层表示漂移、再到最终答案失败的机制证据链，并通过 activation patching 检验其因果作用。
3. 基于“局部位置、远程语义”的原则提出一种或多种 attention 改进方案，并在与上述 failure regime 对齐的实验中验证有效性。

## 写作提醒

- 不要把“官方支持长度”直接等同于“in-distribution”；若无法确认真实训练长度，使用 supported-context。
- 不要写“远程位置信息完全无用”，应写“远程候选相关性不应被细粒度距离主导”。
- 不要写“相位逐层变差”，应写“由相位触发的表示与检索状态逐层分叉”。
- Introduction 中所有结果数字先用占位符，等最终方法和完整实验确定后再填写。
