# QKSieve 简要汇报

## 1. 方法

我们提出长上下文稀疏注意力方法 **QKSieve**：

- 为每层、每个 KV head 构建 QK-balanced 坐标，保持原始 QK 点积。
- 将 128 维 Key 划分为 8 个 16 维 band。
- 使用 Key-MSE 在 240 bit 预算下分配 0/1/2/4/8 bit。
- 部署时冻结模型级模板，结合 WMMA Query 投影、GQA-4 共享扫描和 sampled-quantile 检索候选。
- 最后在候选的原始 FP16 K/V 上执行精确稀疏 attention，不使用 router、重排或 Full fallback。

## 2. LongBench 质量

Llama-3.1-8B-Instruct，16 个英文任务、3,750 个样本：

| 方法 | Macro score | 相对 Full |
|---|---:|---:|
| Full Attention | 0.45940 | 100% |
| QKSieve | 0.45885 | **99.88%** |

## 3. 速度结果

RTX 3090 上的稳态解码加速：

| 序列长度 | 整模型 Decode 加速 |
|---:|---:|
| 32K | 1.90x |
| 64K | 3.36x |
| 120K | **4.57x** |

GQA-4 attention 子系统：

| 序列长度 | 子系统加速 |
|---:|---:|
| 64K | 2.73x |
| 128K | **4.16x** |

## 4. 当前结论

QKSieve 在基本保持 LongBench 质量的同时，序列越长，加速优势越明显。LongBench 来自质量参考配置，速度来自冻结模板部署配置，后续还需补充完全同路径评测。
