# CPU 常驻完整 KV 的 QKSieve 检索设计

## 可证伪结论

在完整 FP16 KV 无法放入单卡显存时，GPU 常驻 240-bit QKSieve Key
索引，并从 CPU pinned memory 只取回候选的精确 K/V，应当比逐层取回全部
KV 的 Full-offload 明显降低单 token 延迟。

若在 128K、top-1280 下，包含候选 ID 回传、CPU gather、H2D 和精确稀疏
Attention 后仍不能达到 Full-offload 的 1.5 倍速度，则当前实现不足以支持该
系统结论。

## 物理先验

1. Qwen3-4B 每个历史 token 的完整 FP16 KV 为 144 KiB，128K 时为 18 GiB。
2. QKSieve 240-bit Key 索引是完整 FP16 K+V 的 5.859%。
3. top-1280 的精确 KV 流量不随历史长度继续增长，但 Full-offload 流量与历史
   长度线性增长。
4. CPU 随机 gather、GQA 四个 query head 的候选并集和逐层同步可能抵消理论
   PCIe 流量优势，因此必须分别计时，不能只用字节数推算。

## 数学模型

设历史长度为 `N`，层数为 `L`，KV head 数为 `H_kv`，head dimension 为
`D`，每个元素为 FP16。Full-offload 每步传输：

```text
Bytes_full = L * H_kv * N * D * 2(K,V) * 2 bytes.
```

每个 query head 选择 `B` 个 token。一个 KV head 对应四个 query head，候选
并集放大系数为 `u`，页化传输放大系数为 `p`：

```text
Bytes_retrieval = L * H_kv * B * u * p * D * 2(K,V) * 2 bytes.
```

其中 `1 <= u <= 4`。本实验显式控制 query-head 公共候选比例，并测量 `u`；
`p` 由 page size 和候选位置共同决定。

## 实现契约

输入：历史长度、每 query head 候选数、GQA 组内公共候选比例、page size。

步骤：

1. 构造公共比例可控的 per-query-head 候选。
2. 对每个 KV head 求候选并集，并可扩展到完整 page。
3. 将 fetch ID 从 GPU 回传到 pinned CPU buffer。
4. 从 CPU 完整 FP16 KV gather 到连续 pinned staging buffer。
5. 将 staging buffer 传到 GPU。
6. 在 GPU 上映射回每个 query head 的原始 top-`B`，执行精确 Attention。
7. Full-offload 使用同一 CPU KV，传输全部 K/V 后调用 native GQA SDPA。

输出：每个阶段的 p50/p95、每层字节数、36 层估算、直接子系统加速和保守
整模型 Decode 估算。

## 结论边界

该微基准回答 CPU-offload 数据路径是否值得接入完整模型。它不证明 LongBench
质量，也不包含完整 QKSieve selector；整模型估算使用此前实测 resident Decode
作为底座，最终论文数字仍需真实 autoregressive 集成验证。
