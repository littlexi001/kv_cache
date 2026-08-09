# 实验结果与结论

## 如何阅读

`QKSieve growing Graph` 是真实增长后缀的整模型 CUDA Graph 延迟，包含 K/V
写入、整模型 forward、greedy argmax 和位置递增。原表的 `Full Graph` 在
Native-GQA 零复制修复之前测得，仍包含 GQA K/V 展开或物化开销，不能作为当前
最强 Full 基线。原始延迟保留用于追溯，但原加速比已失效。

## 正确性

attention 层：

| 前缀 | 最大后缀历史 | 动态对切片误差 | Graph 对普通误差 | 结果 |
|---:|---:|---:|---:|---|
| 4K | 63 | 0.0 | 0.0 | 通过 |
| 64K | 255 | 0.0 | 0.0 | 通过 |

整模型：4K、8K、32K、64K 的测试步中，Graph 与普通增长执行的 token 序列完全
相同，逐步 logits 最大误差为 0。16K 的 token 序列也完全相同，但 logits 最大
差异为 11.515625；此前普通 eager 重复执行在该长度也存在同等级非确定性，因此
暂时不能声称 16K bitwise deterministic，只能声称 top-1 等价。

## 原始整模型速度表（Full 基线已失效）

| 前缀 | Full Graph ms/token | QKSieve growing ms/token | 加速 |
|---:|---:|---:|---:|
| 8K | 18.7003 | 18.0042 | 1.039x |
| 16K | 23.2647 | 19.7857 | 1.176x |
| 32K | 32.0083 | 21.4289 | 1.494x |
| 64K | 50.9169 | 24.6714 | 2.064x |

2026-08-08 使用 `scaled_dot_product_attention(enable_gqa=True)`、直接读取
strided 预分配 K/V，并在逐元素验证 decode mask 全零后删除 mask，重新测得：

| 64K 路径 | 延迟 | 相对零复制 Full |
|---|---:|---:|
| Native-GQA 零复制 Full fixed-position Graph | 25.8910 ms/token | 1.000x |
| QKSieve 真实增长 Graph | 24.6886 ms/token | **1.049x** |

两条路径使用同一模型、单张 RTX 3090 和同一代码版本。Full 固定位置 Graph
不包含 argmax、位置递增和增长后缀管理，因此仍是略偏向 Full 的基线。QKSieve
的候选、输出和普通增长执行逐元素一致。

4K QKSieve growing 延迟为 16.5321 ms/token。该长度的主要目的为验证增长实现，
没有把它与缺少同口径的 Full Graph 数字组成主结果。

64K 的设备时间为 24.6934 ms/token，和 wall time 24.6714 ms/token 接近，说明
Python 调度已不再是主要瓶颈。它也略快于此前固定位置 QKSieve 的
25.1616 ms/token，因此动态 K/V 写入和精确后缀追加没有造成可见回退。

## 质量探针

独立 64K、5-token teacher-forced 探针中，QKSieve 相对 Full 的 PPL 质量保持率为
99.7586%，top-1 agreement 为 100%。该探针样本量不足以替代 LongBench/RULER，
这里只用于确认新执行路径没有改变 frozen QKSieve 公式。

## 失败路径

Full 增长 Graph 使用动态非全零 attention mask 时为 263.64 ms/token。原因是当前
PyTorch SDPA 对该 mask/形状组合选择了慢 kernel。这个数字不能作为论文主
baseline，也不能用于声称 10.7x。修复后的主比较使用 25.8910 ms/token
零复制 Full 固定位置 Graph，只得到 1.049x。

一组 Full 与 QKSieve 连续执行的诊断曾让 Full graph-mode cache 污染随后 QKSieve
状态，导致无效的 94.47% 质量。已在 benchmark 返回前恢复 cache 模式；该组质量
不进入结论。

## 当前结论

CUDA Graph 证明大量小 kernel 的 host launch gap 可以被显著消除：64K QKSieve
从普通执行的约 54--68 ms/token 降至 24.69 ms/token。但使用零复制
Native-GQA Full Graph 后，公平整模型优势只有约 **1.05x**，而不是旧表中的
2.06x。

8K/16K/32K 的旧加速比同样使用了修复前 Full 基线，必须重新测量后才能进入
论文。当前只可主张：在单 batch、固定前缀复用、greedy decode 的 RTX 3090
环境中，QKSieve 64K Graph 路径略快于强 Full Graph；尚不能主张稳定、大幅的
整模型加速，也不能外推到连续 batching、采样、多 GPU、H100 或 128K 以上。

## 128K 补测与基线纠正（2026-08-08）

### Legacy HF Full 的重复 K/V 物化

Qwen3-4B-Instruct-2507 的 128K FP16 K/V 与模型权重无法同时放入一张 24GB
RTX 3090，因此整模型使用两张 RTX 3090、`device_map=auto`。最初得到 Full
299.853 ms/token、QKSieve 88.193 ms/token，即 3.400x。

该 Full 路径不能作为公平基线。Transformers 的 SDPA adapter 每层先用
`repeat_kv` 把 8 个 KV heads 展开成 32 个 query heads，随后对 K/V 调用
`.contiguous()`。128K 时展开后的 K+V 每层约为 2 GiB；36 层每个 decode
token 至少物化约 72 GiB 临时 K/V。3.400x 主要来自避免这个 Legacy Full
开销，正式作废。

失效结果保留在远端
`results/model_growing_eager_qk_128k_2gpu_20260808/summary.json`，仅用于说明
错误测速口径。

### Native-GQA 与零显式复制 Full

用 PyTorch `enable_gqa=True` 消除 `repeat_kv` 后，Full 约为 89 ms/token。
但第一版 Native-GQA adapter 仍对预分配 cache 的有效 strided 切片执行
`key.contiguous()` 和 `value.contiguous()`，因此仍然每层复制一次完整的
8-head K/V。

同一 target hash、同一双 RTX 3090、同一 16-token decode 的严格 A/B 为：

| 128K Full 路径 | 延迟 | PPL | 峰值单卡显存 |
|---|---:|---:|---:|
| Native-GQA + K/V contiguous copy | 89.176 ms/token | 63.836988 | 15.327 GB |
| Native-GQA 直接读取 strided K/V | **61.267 ms/token** | 63.836988 | **14.790 GB** |

零复制路径只在单 token decode 且 4D mask 已逐元素验证为全零时删除 mask，并
直接把 strided K/V 交给 `scaled_dot_product_attention(enable_gqa=True)`。4K
和 128K 的 PPL 均与 contiguous 路径一致。单层 128K A/B 也得到 0.670 ms
对 1.957 ms，输出最大绝对误差为 0。

当前 QKSieve 在相邻同配置运行中约为 88.7 ms/token。因此相对真正零显式复制
Full，当前整模型实现不是加速，而是约 0.69x，即约慢 1.45 倍。两个延迟来自
相邻运行而不是最终 paired 进程，论文前仍需补一次最终配对，但已足以否定
3.400x 主张。

### 单层强 native-GQA Graph 基线

单张 RTX 3090 上直接比较 native-GQA SDPA Graph 与完整 QKSieve Graph：

| 128K 单层路径 | Graph 延迟 | 加速 |
|---|---:|---:|
| Full native-GQA SDPA | 0.6438 ms | 1.000x |
| 完整 QKSieve | 0.4107 ms | **1.568x** |

QKSieve 的 candidate ID、candidate count、threshold 和最终输出均与其 eager
路径逐元素一致。该结果说明纯 attention 数值流水线仍有 1.57x 优势，但整模型
中的大量小 kernel、Python/HF 调度和多卡传输吞掉了收益。

原始结果：远端
`results/cudagraph_layer_128k_recheck_20260808/results.json`。

### 128K 整模型 Graph 的边界

两卡整模型增长 Graph 在 Accelerate 的 GPU0 到 GPU1 层间传输处报
`dependency created on uncaptured work in another stream`。因此当前没有合法的
128K 多卡整模型 CUDA Graph 数字。

允许的结论是：

- 强 native-GQA 单层 Graph 实测 1.568x attention 加速；
- Legacy HF 的 3.400x 整模型数字因重复物化 K/V 而失效；
- 128K 零显式复制 Full 约 61.27 ms/token，当前 QKSieve 约 88.7 ms/token；
- 当前不能主张公平 Full 基线下的 128K 整模型加速。

## 下一不确定性

下一步应在 Graph replay 内做 kernel 级 profiling，分别统计低比特扫描、精确
候选 QK/AV、ValueSketch、MLP 和 logits 的设备时间。只有确认最大剩余阶段后，
再决定做 scan/consumer 融合、稀疏 AV 优化或模型侧算子融合。
