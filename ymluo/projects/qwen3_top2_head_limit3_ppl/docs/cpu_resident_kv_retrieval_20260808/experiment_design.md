# CPU KV 检索实验设计

## 研究问题

当完整 KV 位于 CPU 时，QKSieve 只取回精确候选 KV，能否在真实 RTX 3090
PCIe/NUMA 环境中显著快于 Full-offload？

## 固定条件

- GPU：RTX 3090 24GB。
- 模型结构：Qwen3-4B，36 层、32 query heads、8 KV heads、head dim 128。
- 数据类型：完整 K/V 均为 FP16。
- 候选预算：每 query head 1,280 token。
- QKSieve 索引：240 bit/token/KV-head；索引扫描时间由既有 resident Decode
  结果覆盖，不在 host-transfer 微基准中重复模拟。
- CPU KV 与 staging buffer：PyTorch pinned memory。
- Full Attention：PyTorch native `enable_gqa=True` SDPA，不允许物化32头 KV。

## 改变量

1. 历史长度：64K、128K、256K。
2. GQA组内公共候选比例：100%、50%、0%，分别对应并集系数约1、2.5、4。
3. Page size：1、16、32、64 token。

## 指标

- `candidate_ids_d2h`：fetch ID 从 GPU 到 CPU 的延迟。
- `cpu_exact_kv_gather`：CPU 随机读取并压紧精确 K/V 的延迟。
- `exact_kv_h2d`：压紧 K/V 的 H2D 延迟。
- `gpu_remap_and_sparse_attention`：恢复 per-query-head 候选并执行精确 Attention。
- `full_kv_h2d`：Full-offload 每层完整 K/V 的 H2D 延迟。
- `direct_subsystem_speedup`：Full 的 H2D+Attention 除以检索路径四阶段总和。
- `conservative_decode_estimate`：在此前正确 native-GQA resident Decode 上增加
  当前测得的 host 数据路径开销。

## 通过与失败条件

- 通过：128K 的直接子系统和保守 Decode 估算均至少1.5倍快于 Full-offload，
  且 GPU 持久索引不超过完整 KV 的7%。
- 失败：所有合理候选重叠条件下均小于1.2倍，或 CPU gather 使检索路径接近
  Full-offload。
- 证据不足：只有100%共享候选成功，而真实候选并集系数尚未测量。

## 复现入口

```text
src/benchmark_cpu_resident_kv_retrieval_20260808.py
```

结果写入：

```text
results/20260808_cpu_resident_kv_retrieval/
```

## 执行结果

- 128K真实候选、token级整行取回：直接子系统20.70x，保守Decode估算9.68x。
- 128K p95保守估算约9.22x。
- 64K真实候选：直接子系统10.11x，保守Decode估算7.05x。
- page=2/16均慢于token级取回。

因此本微基准通过预设的1.5x门槛。完整autoregressive集成和强offload基线尚未
完成，属于下一阶段，而不是本微基准已经证明的内容。
