# 三篇组内论文的 Introduction 风格提炼

参考材料：Metis、Multihead-MoE、Spectra。三篇文章技术主题不同，但
Introduction 的共同骨架高度一致。

## 共同结构

1. **先建立理论承诺或现实价值。** 读者先理解研究对象为什么重要。
2. **用一个强反例打破预期。** 问题段只放最有冲击力的关键数字，不堆
   完整实验表。
3. **尽早给中心答案。** 在问题之后直接说明故障发生在哪里，而不是让
   读者读到最后才知道论文主张。
4. **分析沿因果链递进。** 每段首句先给 observation，后面只负责机制、
   度量和证据。
5. **分析对象就是方法对象。** Metis 在谱域发现问题并在谱域干预；
   Multihead-MoE 以 head 定位故障并恢复 head-wise routing；Spectra 找到
   spike subspace 后只干预该 subspace。
6. **最后集中报告结果。** 先交代模型、数据、baseline，再使用
   `from ... to ...` 和互补指标报告效果及代价。

## 对当前 RoPE 论文的对应关系

```text
长窗口的承诺
→ 固定语义、只改距离也会突然失败
→ RoPE 在 softmax 前把语义匹配与相对相位耦合
→ QK 抑制 → evidence mass 变化 → residual 分叉 → margin 穿零
→ 保留局部 RoPE；远程采用 pre-RoPE semantic proposal
→ exact post-RoPE consumption + sparse softmax
```

稳定术语链建议保持不换同义词：

```text
phase-sensitive score suppression
→ evidence-mass reduction
→ residual-state divergence
→ answer failure
→ local position / global semantics
```

## 不应照搬

- 不复制参考论文的具体句子、观察数量或技术术语。
- 不把单模型案例写成 universal RoPE failure。
- 不把 correlation 写成 causation；因果措辞只交给 phase intervention 和
  activation patching 等真正干预实验。
- 不称远程位置完全无用；当前更窄的主张是细粒度距离不应主导远程候选
  的语义相关性。
- 没有 latency/memory 实测前不写 negligible overhead。
- Introduction 只保留一个核心公式和最关键数字，完整推导放正文分析和
  Appendix。

