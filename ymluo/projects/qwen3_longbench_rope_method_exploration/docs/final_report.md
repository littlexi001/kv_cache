# LongBench 上此前 RoPE 方法是否有提升？

## 一句话结论

**有一个小样本的任务分数提升，但还没有形成能支撑论文方法主张的稳定证据。**

在同一批 18 条严格对齐的 LongBench HotpotQA 样本、同样 2% support
预算下，`local_global_postscore` 相比 exact post-RoPE Top-2%：

- QA-F1：60.00 → **71.11**；
- EM：50.00 → **61.11**；
- 产生 2 个 rescue、0 个 harm；
- 但 Gold NLL：1.1386 → 1.1541，配对差值为 +0.0155，95% CI
  [-0.2324, +0.3697]，没有稳定改善；
- Gold evidence recall：4.76% → 3.24%；
- Gold evidence attention mass：0.598% → 0.490%。

它相对 Full attention 也高 5.56 个 QA-F1/EM 点，但只有 1 个净 rescue，
18 条样本不足以说明稳定优于 Full。

## 应该怎样理解

我们能写的最强结论是：

> “局部 RoPE + 远程 pre-RoPE proposal + 原位置 native-RoPE consumer”
> 在严格 LongBench 小样本上显示出 task-level rescue 信号。

现在不能写：

> 它通过召回更多真实证据，稳定改善了长上下文问答。

因为平均证据 recall 和 mass 都下降了，而且 NLL 的置信区间跨零。两个
EM rescue 也没有伴随全局平均 evidence mass 上升。可能的替代解释包括：
保留最近 128 token 与 sink 改善了问题/格式信息；不同 head 只需要证据的
少数 token；Hotpot 标注没有覆盖全部有用上下文；或者稀疏路径改变了生成
而不是证据读取。

## 两个被否定的后续方案

1. **Question-span 多 Query 召回。** Token-max 把 recall 稳定提高 2.03
   个百分点，却把首 token NLL 恶化 2.2797，95% CI
   [+0.3342, +4.7028]。召回更多并不等于模型能正确消费。
2. **在 consumer 中混入语义分数。** 单样本 smoke 继续恶化 NLL，因此
   没有扩大。此前直接相位修复也已因慢、对控制敏感且不能恢复正确率而
   判定 NO-GO。

## 对论文最稳妥的建议

当前论文应继续以**机制解释**为主，而不是把 RoPE 改进包装成成熟方法。
方法部分可保留为机制产生的可恢复性测试，并明确：合成任务在 16K–32K
有稳定收益；LongBench strict-18 只有 QA/EM rescue 信号，likelihood 与
evidence mediation 尚不稳定。

若要把方法提升为主贡献，下一步不应继续堆更多无训练启发式，而应直接
优化“证据是否对答案有正向 Value utility”：在 held-out 数据上学习或
校准 head-specific selector/consumer，并以 `evidence recall → attention
mass → directed utility → answer NLL` 的同样本链条作为验收条件。

## 产物

- 主结果：`outputs/hotpot_strict18_20260803/merged/`
- 多 Query 反例：`outputs/queryspan_strict18_20260803/merged/`
- 实验设计：`docs/design.md`、`docs/experiment_design.md`
- 完整图表解释：`docs/visualization_results.md`
