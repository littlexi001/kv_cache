# CountCap 时间复用实验结论

更新时间：2026-07-24

## 1. 目标

当前 CountCap 每个 decode step 使用 PCA48-INT4 sampled-quantile 扫描历史
K，选出约 6% 候选，再以精确 QK 重排并保留约 2% token 做 attention。

本轮希望利用相邻 decode step 的候选稳定性，跳过部分全历史 PCA 扫描，
同时保持当前 CountCap 的生成结果。

所有正式实验均使用 Llama-3.1-8B-Instruct、RTX 3090，并只使用 GPU 0--3。

## 2. 严格证书不可用

实现了基于 Cauchy 上界的 no-crossing certificate：

```text
若上一时刻候选边界 margin
  > 2 * ||q_t - q_(t-1)|| * max_i ||k_i||
则候选排序不会跨越边界，可以安全复用。
```

实测 8K、16K、32K 的严格通过率均为 0。sampled-quantile 使用
`score >= boundary`，至少一个候选分数紧贴边界，因此严格 margin 通常为 0。
该证书理论正确，但无法产生实际跳扫。

## 3. Sampled-mass 逐 head 复用

实现了真实生成 shadow path：

1. 保存每层每个 query head 的上一时刻候选集合。
2. 补入新产生的历史 token。
3. 对旧候选做当前 query 的精确 QK 重排。
4. 从候选外固定采样 256 个 token，估计旧候选覆盖的 softmax mass。
5. 通过阈值的 head 使用旧候选输出，否则使用当前新扫描输出。
6. 缓存按照每个 head 的实际动作独立演化，可检验连续复用误差。

`history < 8192` 时关闭时间复用。原因不是任务标签，而是实测成本交叉点：
短序列跳扫不能抵消门控和候选维护开销。

跨 NarrativeQA、HotpotQA、MultiNews、Qasper、PassageCount 和
RepoBench-P 的 12 个样本结果：

| 方法 | Macro score | 相对当前 CountCap | 逐字相同样本 |
|---|---:|---:|---:|
| 当前 qprojscan CountCap | 0.354116 | 100% | 12/12 |
| cost-gated mass 90% | 0.354116 | 100% | 10/12 |
| cost-gated mass 94% | 0.354116 | 100% | 11/12 |
| cost-gated mass 95% | 0.354116 | 100% | 11/12 |

在 `history >= 8K` 且生成超过 5 token 的样本中：

| 阈值 | 平均逐 head 复用率 | Macro score |
|---:|---:|---:|
| 90% | 53.07% | 与当前 CountCap 相同 |
| 94% | 40.58% | 与当前 CountCap 相同 |
| 95% | 36.44% | 与当前 CountCap 相同 |

这说明逐 head 候选复用在模型质量上可行，但还不能直接换算成扫描加速。

## 4. GQA 是实际部署约束

Llama-3.1-8B 的一个 KV head 对应 4 个 query heads。当前 PCA 扫描 kernel
只读取一次 K，并同时计算同组 4 个 query heads。只有 4 个 head 全部复用，
才能跳过该 KV 组的完整扫描。

为此实现了 GQA cooperative candidate union：

```text
同一 KV 组的 4 个 query heads
  -> 合并上一时刻候选
  -> 当前 query 精确重排
  -> 4 个 head 全部门控通过时整组复用
```

候选并集平均占历史 token 的约 11%--12%。shadow 结果：

| Prompt | GQA mass 94% 组复用率 | GQA mass 95% 组复用率 | 质量 |
|---:|---:|---:|---:|
| 8K | 30.28% | 25.05% | 与当前相同 |
| 16K | 33.62% | 28.16% | 与当前相同 |
| 32K | 32.01% | 27.34% | 0.21854 降至 0.19868 |

GQA 并集提高了候选真实覆盖，但 sampled-mass 门控仍无法同时达到
`>50%` 整组跳扫和 32K 质量保持。

## 5. 为什么不继续开发 masked-scan kernel

旧周期复用实验提供了实际时间上界：

| Prompt | 当前 CountCap online | reuse4 online | 扫描跳过率 |
|---:|---:|---:|---:|
| 8K | 2.892 s | 2.949 s | 73.33% |
| 16K | 2.970 s | 3.040 s | 73.33% |
| 32K | 3.874 s | 3.763 s | 73.33% |

32K 跳过 73.33% 扫描只减少约 2.9% online 时间。反推 sampled-quantile
全历史扫描约占整模 online 时间的 4%；8K/16K 还会被额外 launch 和维护开销
抵消。因此，即使做出理想 masked-scan kernel，也无法解决当前短上下文速度问题。

## 6. 结论与下一步

时间复用得到两个可发表的机制观察：

1. 逐 head 的数值门控可在保持任务分数时复用约一半候选。
2. GQA fused scan 把“逐 head 可复用”转化为“物理可跳扫”时存在明显结构损失。

但它不是当前主方法的下一条速度主线。下一步应优化真正占时较高的部分：

1. 保留当前 qprojscan CountCap 作为质量基线。
2. 32K 继续使用已验证的 length-aware split-QKV candidate consumer。
3. 8K/16K 优先测试 StaticCache + CUDA Graph，减少 32 层逐 token kernel launch
   和 Python/HuggingFace 调度开销。
4. 若 CUDA Graph 收益不足，再实现 sampled-quantile scan 与精确候选消费的
   cooperative persistent kernel；必须报告整模 online 时间，不只报告微内核。

## 7. 单 block 流式融合内核实验

进一步实现了一个不改变算法的 CUDA 原型。每个 query head 由一个 block 完成：

```text
query PCA 投影
  -> 256 点 sampled-quantile 边界
  -> PCA48-INT4 全历史扫描
  -> 候选精确 QK
  -> sparse softmax / AV
```

该原型把原来的阈值、候选扫描和候选 attention 合并为一次 kernel launch。
修正 block 内 boundary 的可见性后，候选集合、候选计数和 quantile boundary
与原实现完全相同；输出最大绝对误差不超过 `1.22e-4`。

微内核结果：

| 历史长度 | 原 qprojscan + QKV attention | 单 block 流式融合 | 相对速度 |
|---:|---:|---:|---:|
| 8K | 0.2557 ms | 0.2746 ms | 0.931x |
| 16K | 0.5168 ms | 0.5734 ms | 0.901x |
| 32K | 0.8114 ms | 0.9815 ms | 0.827x |

失败原因是 Llama GQA 的 4 个 query heads 共享一个 KV head。原扫描 kernel
只读取一次压缩 K 并同时计算 4 个 query heads；单 query-head block 虽然减少
launch，却重复读取同一份索引，并且长序列下重复读取成本随长度增长。

该实现不接入主方法。若后续继续做 persistent fusion，必须以一个 KV group
为计算单元，并用多 CTA cooperative reduction 保留索引共享，不能再按单
query head 拆分。
