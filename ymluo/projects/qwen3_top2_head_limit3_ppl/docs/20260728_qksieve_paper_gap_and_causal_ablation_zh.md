# QKSieve 论文缺口与 32K 因果消融

## 1. 当前判断

方法、理论和论文主线已经足够完整，当前不应继续增加定理或新模块。
投稿前的主要风险是实验闭环不足，优先级如下：

1. 三模型完整 LongBench 与完整 RULER，验证同一冻结配置的泛化。
2. 同一路径整模型测速，报告 16K/32K/64K/128K、固定生成长度、
   index build、break-even、显存和全部额外开销。
3. 与 packed FIER、Quest、SparQ 和可运行官方系统做边界清晰的比较。
4. 拆分 QK-balanced 坐标、自适应 mixed-bit 和 Query-weighted allocation
   的因果收益。
5. 长输出和多轮场景的 Query 分布漂移。
6. A100/H100 最终系统复测。RTX 3090 结果只能作为实现和交叉点诊断。

正文不应再增加新的理论主张。最终需要增加的是一张质量-速度 Pareto 图、
一张主要消融表和一张完整系统延迟分解表。

## 2. 本次实验协议

结果目录：

```text
results/20260728_qksieve_ppl_causal_ablation_32k_gpu5/
```

汇总文件：

```text
results/20260728_qksieve_ppl_causal_ablation_32k_gpu5/ablation_summary.json
```

共同条件：

- 模型：Qwen3-4B-Instruct。
- 数据：sports、medicine，每类两个 held-out 窗口。
- 历史长度：32K；每窗口评价 128 个 token。
- 每个 query head 实际注意 1,280 个 token，即 4% 历史。
- 所有方法使用相同原始 FP16 K/V 和相同 exact sparse attention。
- 无 rerank、无 fallback、无 recent/sink 特判。
- 这里只是机制和 PPL 诊断，不是最终多模型 PPL 表。

## 3. 完整结果

| 方法 | 索引 bit/token/KV-head | PPL 保持率 | Top-1 一致率 | Full-to-sparse KL | 稳态 decode 加速 |
|---|---:|---:|---:|---:|---:|
| QKSieve：QK-balanced + qMSE allocation | 240 | 100.198% | 94.531% | 0.00952 | 1.448x |
| Key-PCA + uniform 1-bit | 256 | 82.424% | 75.977% | 0.23590 | 1.553x |
| QK-balanced + uniform 1-bit | 256 | 72.033% | 71.289% | 0.34436 | 1.536x |
| Random rotation + uniform 1-bit | 256 | 98.271% | 89.062% | 0.04313 | 1.566x |
| Key-PCA + Key-MSE allocation | 240 | 99.200% | 92.969% | 0.01909 | 1.525x |
| QK-balanced + Key-MSE allocation | 240 | 100.296% | 94.727% | 0.00945 | 1.526x |
| FIER RTN-1 g32 packed reproduction | 256 | 100.404% | 92.383% | 0.01657 | 1.697x |

共同 Full PPL 为 17.80095。PPL 保持率大于 100% 表示 sparse PPL 在这四个
窗口上略低于 Full，不代表方法在统计意义上优于 Full。

这里的速度只比较该 PPL harness 中的稳态 decode。FIER 的 index build
没有以和 QKSieve 完全等价的字段单独记账，因此不能把本表速度写成正式
系统对比。

## 4. 能支持的结论

1. **Uniform 1-bit 不能支持主方法。**
   QK-balanced 坐标配 uniform 1-bit 只有 72.03% PPL 保持率。
   因此 QK-balanced 不是一个可以脱离 rate allocation 单独使用的改进。

2. **自适应 mixed-bit 是必要组件。**
   从 QK-balanced uniform-1 的 72.03% 提升到两个 QK-balanced
   mixed-bit 版本的约 100%，说明主要恢复来自非均匀 bit 分配。

3. **QK-balanced 坐标在自适应分配下有可见收益。**
   Key-PCA + Key-MSE 为 99.20%，QK-balanced + Key-MSE 为 100.30%；
   Top-1 从 92.97% 提升到 94.73%，KL 从 0.01909 降到 0.00945。

4. **这组数据不能证明 qMSE allocation 优于 Key-MSE allocation。**
   QK-balanced + Key-MSE 与完整 QKSieve 基本相同，前者甚至有略低 PPL。
   qMSE 的论文主张必须由已排队的 LongBench m20 消融、检索 recall/mass
   和更广 PPL 数据共同决定，不能只引用理论目标更合理。

5. **FIER 在目标 PPL 上不弱，但输出分布更偏离 Full。**
   FIER 的四窗口 PPL 略优，但其 Top-1 一致率和 KL 均差于两个
   QK-balanced mixed-bit 版本。最终比较必须看完整 LongBench、RULER、
   多窗口 PPL和同路径系统速度。

## 5. 对论文写法的直接影响

当前可以保留的主故事是：

> QKSieve 以 QK-balanced 坐标描述 query-sensitive score geometry，并用
> request-local mixed-bit allocation 在固定小索引预算下恢复检索质量。

目前不能写成：

> QK-balanced 坐标在任意统一量化下都优于 PCA 或随机坐标。

目前也不能把 qMSE allocation 写成已被下游任务严格证明优于 Key-MSE。
完整 LongBench 消融出来前，应把它写成有理论依据的设计，实验结论保持
开放。

## 6. 已处理的工程问题

本次实验发现 parser、运行时 context 和 attention dispatch 对新增 Key-MSE
模式的白名单不一致。现已统一由 `_PACKED_QMSE_SCORE_MODES` 驱动，并增加：

- 两个 Key-MSE full-topk 模式的 context 回归测试；
- 两个模式真正进入 attention dispatch 的预算饱和测试。

相关本地测试共 43 项通过。完整七组实验最终生成 `ALL_COMPLETE`，
最终日志中无 Traceback、OOM、AssertionError 或 ValueError。
