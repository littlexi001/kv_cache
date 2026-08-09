# CPU KV 检索结果

## 实验设置

- 服务器：2 路 Intel Xeon Silver 4216，8 张 RTX 3090 24GB。
- GPU 0--3 绑定 NUMA 0，GPU 4--7 绑定 NUMA 1。
- 模型结构：Qwen3-4B，36 层、32 query heads、8 KV heads、head dim 128。
- 完整 K/V：CPU pinned FP16；QKSieve 候选 K/V 以 FP16 精确取回。
- 候选：每个 query head 最多 1,280 token。
- Full：每层从 CPU 传完整 compact-GQA K/V，再调用
  `scaled_dot_product_attention(enable_gqa=True)`，没有32头KV复制。
- CPU gather：将每个 token 的128维向量看成连续一行，用单次
  `index_select` 压紧，而不是逐元素 `torch.gather`。
- 每项2次 warmup、7次测量，表中使用 p50。

脚本：

```text
src/benchmark_cpu_resident_kv_retrieval_20260808.py
src/extract_qksieve_candidate_trace_20260808.py
```

结果：

```text
results/20260808_cpu_resident_kv_retrieval/
```

## 真实候选结构

先在冻结 QKSieve 上提取真实候选 ID。64K 记录4个生成步、144个
layer-step；128K 记录2个生成步、72个 layer-step。

| 指标 | 64K | 128K |
|---|---:|---:|
| 每 query head 平均候选 | 1,265.37 | 1,260.71 |
| 同一 KV head 的4个 query head 候选并集系数 | 2.265 | 2.344 |
| 4个 head 共同候选比例 | 23.94% | 21.43% |
| 相邻生成 token 候选 Jaccard | 0.418 | 0.363 |
| page=2 传输膨胀 | 1.552x | 1.581x |
| page=16 传输膨胀 | 5.137x | 5.693x |

真实候选既不是四个 head 完全共享，也不是完全独立。page=16 会取回大量不参加
Attention 的 token，因此不能假设页化天然更快。

## 分阶段测速

### 128K 真实候选，token 级取回

| 每层阶段 | p50 | p95 |
|---|---:|---:|
| 候选 ID D2H | 0.037 ms | 0.042 ms |
| CPU 精确 K/V `index_select` | 0.629 ms | 0.686 ms |
| 精确 K/V H2D | 1.179 ms | 1.346 ms |
| GPU remap + 稀疏 Attention | 0.261 ms | 0.285 ms |
| 检索路径合计 | **2.136 ms** | **2.326 ms** |
| Full K/V H2D + native-GQA Attention | **44.212 ms** | **44.254 ms** |

直接 attention/offload 子系统 p50 加速为 **20.70x**。每层 Full 搬运
512 MiB，真实候选平均搬运13.82 MiB，主机流量减少 **37.03x**。

### 长度结果

“保守 Decode 估算”把实测 host 路径加到此前实测 resident Decode 上，并且不
减去原 resident QKSieve 已有的 GPU 候选 gather，因此倾向于低估 QKSieve-Host。

| 长度 | Full KV | QKSieve索引 | 取回KV/层 | 子系统加速 | Full-offload估算 | QKSieve-Host估算 | 保守加速 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64K | 9 GiB | 0.527 GiB | 14.17 MiB | 10.11x | 837.18 ms/token | 118.75 ms/token | **7.05x** |
| 128K | 18 GiB | 1.055 GiB | 13.82 MiB | 20.70x | 1,626.23 ms/token | 168.07 ms/token | **9.68x** |

128K 使用更保守的 resident 底座：Full 60.964 ms/token，QKSieve
101.657 ms/token。按各阶段 p95 重新计算，128K 保守加速仍约 **9.22x**。

### 页大小消融

| 128K取回方式 | KV/层 | 主机流量降低 | 子系统加速 | 保守Decode加速 |
|---|---:|---:|---:|---:|
| token级，page=1 | 13.82 MiB | 37.03x | **20.70x** | **9.68x** |
| page=2 | 23.35 MiB | 21.92x | 12.36x | 7.38x |
| page=16 | 104.86 MiB | 4.88x | 2.89x | 2.53x |

当前候选不够连续，最优实现是 token 级 ID，但每个ID复制完整128维K/V行。

## CPU gather 优化

128K真实候选下，逐元素 `torch.gather` 的 p50 为4.01 ms/layer；按行
`index_select` 为0.629 ms/layer，快 **6.38x**。独立非恒定 K/V 校验确认两种
方法逐元素完全相同。这个结果说明瓶颈不是“CPU随机取回必然很慢”，而是必须按
KV的连续行布局复制，不能让通用逐元素 gather 处理128维重复索引。

## 允许得出的结论

1. 在 Full KV 必须位于 CPU 的容量约束场景，QKSieve 的低比特 GPU 索引具有
   明确系统价值。
2. 128K 真实候选下，CPU精确KV取回子系统比朴素 Full-offload 快20.70倍；
   保守整模型估算为9.68倍。
3. 240-bit Key索引占完整FP16 K+V的5.859%，层级 staging buffer约
   12.6--15.4 MiB，不需要把完整KV放入GPU。
4. page=2/16均不如token级整行取回；大页不是当前主方法。

## 尚不能声称的内容

1. 9.68x 不是完整 HF autoregressive CPU-cache 集成后的直接实测，而是由真实
   host阶段与独立resident Decode组成的保守估算。
2. 当前只比较了朴素 Full-offload，尚未公平比较 RetrievalAttention、PQCache、
   InfiniGen、ShadowKV 等强 CPU-offload 系统。
3. 候选追踪来自一个冻结文本流，仍需不同数据、模型和CPU平台复现。
4. 当前没有把 CPU gather、H2D 与其他层计算重叠；完整系统可能更快，也可能因
   每层调度和cache接口成本变慢。
