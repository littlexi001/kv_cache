# QKSieve 速度优化结果

## 1. 结论

当前应保留的速度主版本是：

1. 模型级冻结 QK-balanced 基底，不在每个请求上重新做协方差、特征分解或 bit allocation；
2. 每层、每个 KV head 使用冻结的混合位宽索引；
3. GQA4 共享读取 Key 索引，一次扫描同时处理四个 Query head；
4. sampled-quantile 单遍扫描直接产生候选，不物化完整 score tensor，也不做 exact rerank；
5. 每个 Query head 最多使用约 1,280 个原始 FP16 K/V token 做精确 attention；
6. 新增 WMMA Tensor Core Query 投影与量化；
7. 对常见 bit pattern 使用编译期专门化扫描 kernel；
8. 原始 FP16 K/V 常驻 GPU；没有 router、Full fallback 或按任务手工策略。

在 Qwen3-4B-Instruct、RTX 3090、120K history、64 个评测 token 上，
WMMA 主版本的三次交叉顺序重复结果为：

| 指标 | scalar Query | WMMA Query | WMMA 收益 |
|---|---:|---:|---:|
| steady decode | 62.544 +/- 0.222 ms/token | **61.130 +/- 0.084 ms/token** | **1.023x** |
| 64-token 在线总时间 | 4.667 +/- 0.064 s | **4.553 +/- 0.009 s** | **1.025x** |
| 几何平均 PPL | 3.6386 | **3.6271** | 无可见系统性退化 |

相对同一 harness 的参考结果：

| 120K 方法 | PPL | 索引 / Full FP16 K+V | Attention token/head | steady decode | 64-token 总时间 |
|---|---:|---:|---:|---:|---:|
| Full Attention | 3.7273 | 100% | 120K | 279.379 ms | 17.601 s |
| 本地审计版 FIER RTN-1 g32 | 3.6112 | 6.250% | 约 1,280 | 80.647 ms | 5.534 s |
| QKSieve WMMA | 3.6271 | **5.771%** | 约 1,282 | **61.130 ms** | **4.553 s** |

因此，在这套同模型、同硬件、同 runner 的 120K 对照中：

- QKSieve 相对 Full 的 steady decode 加速为 **4.57x**；
- QKSieve 相对 Full 的 64-token 在线总加速为 **3.87x**；
- QKSieve 相对本地 FIER 的 steady decode 加速为 **1.32x**；
- QKSieve 相对本地 FIER 的在线总加速为 **1.22x**；
- 相对 FIER，steady latency 降低约 **24.2%**。

这里的 FIER 是按论文公式实现并审计的本地复现，不是官方代码。FIER 论文
官方摘要报告的是 11% cache budget 和 1.2x--1.5x decoding latency
speedup，使用了不同硬件和运行时，不能与上表跨平台直接相除。论文来源：
[FIER, Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.515/)。

## 2. 主方法如何工作

### 2.1 离线冻结 QK-balanced 模板

使用 sports、medicine、mixed_a 三个彼此独立的 32K 校准文本，分别收集每层
Key 与 Query 的二阶矩。对三个来源的二阶矩求均值后，重新计算每层、每个
KV head 的 QK-balanced 左右投影因子：

```text
q_hat = q * B_q
k_hat = k * B_k
q_hat * k_hat^T approximately equals q * k^T
```

最终模板包含 36 层的 `B_q`、`B_k` 和每个 KV head 的混合位宽分配，大小约
19 MB。运行时直接加载，不再对当前请求做特征分解。测试的 `mixed_b` 不参与
模板构建。

### 2.2 混合位宽 Key 索引

每 16 个投影维度为一个 band。冻结模板在 288 个 KV head 上只产生八种常见
pattern，主要包括：

```text
(4,4,4)
(4,4,2,1)
(4,4,1,1)
(4,4,1,2)
(8,1,1,1)
(8,4)
```

索引平均占 Full FP16 K+V 的 **5.771%**。生成新 token 时只增量编码新 Key，
不重建历史索引。

### 2.3 候选选取

每个 decode step、每层执行：

1. 用 WMMA 将四个 GQA Query head 投影到冻结的 QK-balanced 空间；
2. 对每个 16-D band 做 INT8 Query 量化；
3. 从历史索引中均匀采样，估计目标分位数阈值；
4. GQA4 kernel 完整扫描低比特 Key 索引；
5. score 超过阈值的 token 直接写入候选数组；
6. 在候选对应的原始 FP16 K/V 上计算精确 QK、softmax 和 AV。

目标预算为：

```text
attention_tokens = min(1280, max(256, ceil(0.06 * history_length)))
```

120K 时平均实际候选数约 1,282 token/head，即历史的约 1.07%。sampled
threshold 会产生少量逐 head 波动，但不需要物化 `history_length` 大小的
score tensor，也不需要 `torch.topk`。

## 3. WMMA Query 投影

旧路径使用普通 CUDA 标量投影，再逐 band 量化。新路径把 Query 组织成
16x128 tile，使用 FP16 WMMA、FP32 accumulation，并在同一 kernel 内完成
band-wise INT8 量化。

36 层 Query 投影微基准：

| 指标 | scalar | WMMA |
|---|---:|---:|
| 36 层总时间 | 2.298 ms | **0.715 ms** |
| 算子加速 | 1.00x | **3.21x** |
| Query code 完全一致率 | - | **99.9974%** |
| 最大 code 差 | - | 1 |
| 最大 scale 绝对差 | - | 3.05e-5 |

120K 整模型阶段剖析：

| 阶段，36 层合计 | scalar | WMMA |
|---|---:|---:|
| Key index append | 2.428 ms | 2.410 ms |
| Query prepare | 2.459 ms | **1.819 ms** |
| index retrieval | 3.149 ms | 3.317 ms |
| exact sparse attention | 3.986 ms | 3.940 ms |
| 四阶段合计 | 12.022 ms | **11.486 ms** |

微基准的 3.21x 不会直接变成 3.21x 整模型加速，因为 Query prepare 之外仍有
MLP、QKV/O projection、LayerNorm、索引扫描、KV append 和精确 attention。
三次完整重复中，最终兑现为约 2.3% steady decode 收益。

32K 的六个窗口上，WMMA 相对 scalar 的几何平均 PPL 变化为 -0.30%；120K
三次重复中则为 +0.32%。两个方向相反且量级很小，因此当前证据支持把它视为
FP16 accumulation 的数值抖动，而不是系统性质量下降。

## 4. Attention 子系统与 FIER

在一层 Qwen-style GQA attention、128K 历史的独立 kernel 测量中，所有稀疏
方法均包含阈值估计、完整索引扫描、候选 compaction 和精确 sparse attention：

| 方法 | 单层完整 attention | 相对 Full | QKSieve 相对该方法 |
|---|---:|---:|---:|
| Full SDPA | 2.439 ms | 1.00x | - |
| 本地 FIER，equal active | 1.179 ms | 2.07x | - |
| QKSieve | **0.360 ms** | **6.78x** | **3.28x** |

QKSieve 的主要优势不是少一次很小的 allocation，而是：

- 混合位宽 QK-balanced index 比 FIER 的统一 1-bit 原坐标更适合近似 QK
  排序；
- GQA4 让四个 Query head 共享一次 Key index 读取；
- sampled threshold 避免物化完整 score 和通用 top-k；
- QKSieve 新 Key append 在整模型中约 2.4 ms，而本地 FIER 约 9.8 ms。

## 5. 长度行为

32K 六窗口平均结果：

| 方法 | 几何 PPL | steady decode | 相对 Full |
|---|---:|---:|---:|
| Full | 19.6910 | 88.282 ms | 1.00x |
| 本地 FIER | 19.5054 | **44.778 ms** | **1.97x** |
| QKSieve WMMA | 19.6337 | 46.539 ms | 1.90x |

QKSieve 在 32K 仍比本地 FIER 慢约 3.9%，但在 120K 快约 31.9%。这符合两个
方法的成本曲线：短上下文时固定 kernel/HF 开销占主导；长度增加后，FIER
统一 1-bit score 路径的扫描与增量编码成本增长更快，QKSieve 的低字节索引与
GQA4 共享读取开始占优。

64K sports 两窗口同卡顺序实验：

| 方法 | 几何 PPL | steady decode | 两窗口平均总时间 |
|---|---:|---:|---:|
| Full | 6.0606 | 159.178 ms | 20.216 s |
| 本地 FIER | **5.9426** | 51.949 ms | 6.900 s |
| QKSieve WMMA | 6.0089 | **47.330 ms** | **6.446 s** |

64K 时 QKSieve：

- PPL 仍优于 Full，质量保持率为 100.86%；
- steady decode 相对 Full 为 **3.36x**；
- steady decode 相对本地 FIER 为 **1.098x**；
- 在线总时间相对本地 FIER 为 **1.070x**。

因此，实测 crossover 位于 32K 与 64K 之间。用两个长度点做线性插值约为
41K，但该值混合了文本与运行时差异，只能作为后续密集长度扫描的采样建议，
不能当作已实测 crossover。

## 6. 低索引配置

额外测试了固定 `(4,4,1)` 配置。它保留相同的冻结 QK-balanced 基底，只减少
索引 bit：

| 配置 | 索引 / Full FP16 K+V | 32K 六窗口 PPL | 32K steady |
|---|---:|---:|---:|
| 主版本混合位宽 | 5.771% | 19.6337 | **46.539 ms** |
| 固定 `(4,4,1)` | **4.6875%** | **19.4724** | 47.166 ms |

120K 单窗口中，`(4,4,1)` 为 60.527 ms/token，比 WMMA 主版本三次平均快约
1.0%，PPL 仍优于 Full。但 32K 上它反而慢约 1.35%。原因是低比特 score
使 sampled threshold 的候选数尾部更宽，增加精确 sparse attention 的最慢
head，抵消索引带宽收益。

因此 `(4,4,1)` 应作为低显存配置保留，不能替换主速度配置。

## 7. 已验证但不保留的优化

| 尝试 | 结果 | 结论 |
|---|---:|---|
| 检索与 exact attention 单 kernel 强融合 | 0.67x--0.72x | occupancy 与局部候选缓冲使其更慢 |
| band-major index layout | 0.86x--0.90x | 额外地址计算超过访存收益 |
| metadata shared-memory broadcast | 约 1.00x | metadata 不是瓶颈 |
| sparse-attention workspace 预分配 | 约 1.00x | allocator 不是瓶颈 |
| 更紧 candidate capacity + split8 | E2E 基本持平 | 可省 workspace，不提供稳定速度 |
| 增大 quantile sample | 降低最坏候选数，但 E2E 持平 | 阈值估计成本抵消 attention 收益 |
| 跨 decode step 候选复用 | refresh=2 仅保留约 96% QKSieve mass | 质量风险大，且最多只省扫描小部分 |

可保留但收益较小：

- 常见 bit-pattern 专门化：候选完全一致，扫描 kernel 约 **1.019x**；
- warp-level sampled quantile selection：120K 扫描约 **1.044x**；
- 两者已进入当前 extension，不应单独夸大为整模型收益。

## 8. 剩余瓶颈和下一步

当前 120K WMMA steady decode 约 61.1 ms/token，而可直接归因于 QKSieve 的四个
阶段约 11.5 ms。其余约 49--50 ms 来自模型 projection、MLP、LayerNorm、
HF cache/control flow 与 kernel launch。

所以继续只优化 index scan 的上限很低。即使把 3.3 ms retrieval 完全删除，
整模型也只会再快约 5%。下一步优先级应是：

1. 把 Query 投影、RoPE 和 WMMA QKSieve 投影融合进 `q_proj/k_proj` epilogue，
   同时把 Key index append 融入 K projection，减少读写与 kernel launch；
2. 在静态预分配 cache 上实现整步 CUDA Graph 或 vLLM/FlashInfer backend，
   降低约 50 ms 非检索底座中的 launch/control-flow 成本；
3. 在 H100 上用同一个 backend、同一个模型、同一个预算运行 QKSieve 与 FIER，
   才能形成可投稿的正式系统速度结论；
4. 保持当前 1,280-token 精确 attention，不再通过盲目减预算追求小幅速度，
   因为当前主要瓶颈已经不在候选 K/V 数量。

## 9. 复现入口

主要实现：

```text
src/qksieve_query_cuda_20260728.py
src/mixedblock_spectral_cuda_20260729.py
src/run_head_top2_targeted_ppl_20260714.py
src/run_direct_countcap_denseprompt_ppl_20260725.py
src/build_global_qksieve_template_20260729.py
src/rewrite_qksieve_template_allocation_20260729.py
```

验证与运行：

```text
src/benchmark_qksieve_global_pattern_specialization_20260729.py
src/benchmark_qksieve_fused_select_attention_20260729.py
src/benchmark_qksieve_metadata_broadcast_20260729.py
src/benchmark_qksieve_warpselect_20260729.py
src/benchmark_qksieve_bandmajor_20260729.py
src/benchmark_preallocated_sparse_attention_20260729.py
scripts/run_qksieve_frozen_template_frontier_20260729.sh
```

远端结果根目录：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/results/
20260729_qksieve_frozen_template_frontier
```
