# QKSieve 解码冗余操作裁剪：实验设计

## 研究问题

在公平 Full KV 基线下，哪些操作可以直接删除或缩小，同时保持 QKSieve 的输出质量？实验只改变阈值样本数、ValueSketch 层范围和 exact attention 分块数；模型、历史长度、候选上限、索引格式、完整 K/V 常驻方式保持不变。

## 数据与模型

- 模型：Qwen3-4B-Instruct-2507，FP16，36 层。
- 硬件：单张 RTX 3090；每个条件独占一张卡。
- 历史长度：65,536 token。
- 速度窗口：混合文本，16 个 teacher-forced token；Graph 预热 5 步、计时 32 步。
- 质量探针：LongBench 的 NarrativeQA、GovReport、Qasper、RepoBench-P 文本拼接为四条确定性 64K 流，每条评测 64 个 teacher-forced token。
- 随机种子：20260808。
- 未训练参数：本实验不训练 router 或新模块。

## 条件定义

- Full KV：原生 GQA、GPU 常驻、零复制的 dense attention。
- 完整 ValueSketch c64：约 3,328 个阈值样本，36 层尾部补偿。
- 完整 ValueSketch s512：512 个阈值样本，36 层尾部补偿。
- NoVS+s512：512 个样本，不做 ValueSketch。
- Mean-tail+s512：用均值尾部近似代替 ValueSketch。
- Early/Mid/Late 12：仅在 0–11、12–23 或 24–35 层做 ValueSketch。
- Mid+Late 24：仅在 12–35 层做 ValueSketch。
- Unsorted rank-16：保持全层 rank-16 ValueSketch，但关闭确定性候选排序。
- Split 4/2/1：NoVS+s512 下强制减少 exact attention 分块。

## 指标

- Graph 延迟：固定 Graph 重放的 wall time，单位 ms/token；越低越好。
- 公平速度比：25.891 / Graph 延迟；25.891 ms/token 是相同接口的 Full KV 基线。
- 几何 PPL：`exp(mean NLL)`；与 Full 越接近越好，轻微低于 Full 不代表普遍能力提升。
- top-1 一致率：稀疏 logits 与 Full logits 的 argmax 相同 token 的比例；越高越好。
- KL：Full 输出分布到稀疏输出分布的 KL；越低越好。
- 辅助索引比例：代理 Key、ValueSketch 等字节数除以完整 FP16 K/V 字节数。
- 实际候选数：每层每 head 真正进入 exact attention 的平均 token 数。

## 实验顺序与判定

1. 样本消融：c64 -> c32 -> s512 -> true-s512。若速度不降或质量明显下降，停止缩小。
2. Value 消融：完整 ValueSketch、NoVS、Mean-tail。若替代项在四主题 top-1 明显低于 99.6%，不作为保守主版本。
3. 层消融：Early/Mid/Late 12，再测试最有希望的 Mid+Late 24。
4. 分块消融：在 NoVS+s512 下测试自动、4、2、1。任何延迟上升即判定删除操作失败。
5. 正确性：Graph/Eager 贪心 token 必须一致；候选溢出必须为 0。

rank-12 排序版只作为实现可行性检查。当前 `sortcompact` CUDA kernel 的合同固定为 rank-16 和 256-token Value block；若触发该约束，记录为“现有 kernel 不支持”，不能解释为 rank-12 方法的质量失败。

充分证据：四主题 top-1 不低于 99.6%，速度在重复测试中稳定，且至少补充一个独立长度和模型。

证据不足：单窗口 PPL 更好但四主题 top-1 更差；这只能说明该窗口未受损，不能冻结算法。

## 脚本与结果

- 主脚本：`scripts/run_qksieve_growing_graph_128k_2gpu_smoke_20260808.sh`
- 核心实现：`src/run_head_top2_targeted_ppl_20260714.py`
- 变体注册：`src/run_qksieve_coldskip_longcontext_quality_20260730.py`
- 远端根目录：`/home/fdong/qksieve_iclr2027/experiments/frozen_c64_20260807/results`

Graph 组件使用 Nsight Systems 的 CUDA Graph node trace 测量：在 `QKSIEVE_GRAPH_NSYS_CAPTURE=1` 时仅包围正式 Graph replay，并使用 `nsys profile --cuda-graph-trace=node --capture-range=cudaProfilerApi`。记录 4 个 token，按 kernel 总时间除以 4；prefill、建索引、warmup 和正确性检查都不进入该区间。

限制：当前速度来自单窗口，未报告置信区间；四条质量流不是完整 LongBench 样本分布；PPL 与下游生成指标并不等价。
