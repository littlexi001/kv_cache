# QKSieve 质量与速度口径设计

## 问题

QKSieve 的核心目标是保持长上下文解码质量，同时把每个 head 的精确 attention 限制为至多 1,280 个历史 token。它有两类开销：

1. 与请求或 KV 生命周期相关的一次性索引开销；
2. 每生成一个 token 都会发生的在线检索、稀疏 attention 和 Value 尾部补偿开销。

LongBench 中很多问答任务只生成 3--20 个 token，因此一次性开销很难在单请求内摊薄。Agent 或多轮问答会复用同一长前缀，生命周期不同，不能只用 LongBench 的平均在线速度代表。

## 延迟模型

只比较 decode 时，记上下文长度为 `N`，生成长度为 `G`：

```text
T_full(N,G) = G * t_full(N)
T_warm(N,G) = T_request(N) + G * t_sparse(N)
T_cold(N,G) = T_resident(N) + T_request(N) + G * t_sparse(N)
```

- `T_resident`：只依赖模型和 KV 前缀、可随 KV cache 持久化的索引构建成本。
- `T_request`：依赖当前 query 的 QK-balanced factors、qMSE 位宽分配和 packed Key 成本。
- `t_sparse`：索引就绪后的逐 token QKSieve 延迟，包含检索、精确稀疏 attention、ValueSketch 补偿和增量更新。

当 `t_full > t_sparse` 时：

```text
G_warm = T_request / (t_full - t_sparse)
G_cold = (T_resident + T_request) / (t_full - t_sparse)
```

若 `t_full <= t_sparse`，则不存在有限 break-even；增加输出 token 不能让该长度下的纯稀疏路径变快。

若把原始文本 prefill 也计入总请求：

```text
T_raw = T_prefill + T_decode
```

128K 本轮实测 prefill 约 227.6 秒，而冷索引约 1.42 秒。此时索引只占完整冷请求的一小部分，但 prefill 会掩盖 decode 加速。已经缓存 KV 的 Agent 场景可以合理排除 prefill，但必须单独标注。

## 同一算法，不同工作负载

质量与速度实验必须使用同一数值方法：相同位宽分配、packed Key、候选规则、1,280-token 上限、精确稀疏 attention 和 ValueSketch。允许改变的是工作负载和计时边界：

- LongBench/RULER：评价任务质量和真实请求总延迟。
- steady decode：评价索引就绪后的逐 token 上限。
- warm Agent：评价同一 KV 前缀已经携带持久索引时的新 query。
- cold decode：评价 KV 已有但所有索引均需现场构建的请求。
- raw request：额外包含完整 prefill。

不能用省略补偿或预先注入 oracle 索引的快路径替代质量实现。预建索引只能作为明确标注的 warm 生命周期。

## 当前接受的无损工程改造

通用配置：

```text
QKSIEVE_QK_FACTOR_SOLVER=legacy
QKSIEVE_QMSE_RATE_ALLOCATOR=torch
QKSIEVE_PRELOAD_EXTENSIONS=1
QKSIEVE_PRELOAD_QMSE_RATE_TABLES=1
QKSIEVE_FUSED_WOMETRIC_VALUE_APPEND=1
QKSIEVE_BATCH_QMSE_ALLOCATION=1
QKSIEVE_RESIDENT_VALUE_ATTENTION_WORKSPACE=1
QKSIEVE_TILED_VALUE_ATTENTION=0
```

同前缀 Agent 场景可额外启用：

```text
QKSIEVE_RESIDENT_KEY_FACTORS=1
```

三项已经进入主路径的改造：

1. W_o-metric Value INT4 编码与 append 融合，27,648 个测试值逐位一致。
2. 36 层 qMSE allocation 跨层批处理，36/36 层 allocation 与 active packed-Key hash 一致。
3. 精确候选 attention 直接读取预分配 KV cache 的真实 stride，并复用每层输出与归约 workspace。它不再为每个 decode step 对整段 K/V 调用 `contiguous()`，同时保持原来的标量计算与归约顺序。

第三项的实现契约是：K/V 最内层维度连续、token 行宽等于 `head_dim`，batch/head stride 可大于逻辑历史长度。CUDA kernel 显式接收这些 stride；候选 ID、候选顺序、精确 QK、softmax、ValueSketch 分子和分母公式均不改变。

拒绝进入主路径的实验：批量小矩阵 QK solver 在 RTX 3090 上慢约 11 倍；一次性 host metadata 拷贝在真实 32K A/B 中约慢 1.2%；warp-tiled 候选 attention 虽然单独 kernel 快约 1.19--1.27 倍，但整模型没有稳定超过 stride-aware 标量版本，而且改变浮点归约顺序，因此只保留为消融实现。

## Agent 前缀复用：只常驻 Key 二阶矩

可证伪假设：对同一 KV 前缀，缓存各层 Key 的采样二阶矩

```text
C_k = K_sample^T K_sample / n
```

可以省去后续每个 query 对历史 K 的重复采样和矩阵乘法；若 request-local 路径仍以同一个 `C_k` 调用原始 QK-balanced 求解器，则 bit allocation、packed Key、候选集合和输出应与未缓存路径一致。

实现契约：

1. resident 阶段只保存 `key_mean` 和 `key_second_moment`，不保存或复用平方根、逆平方根与特征向量；
2. 每个 query 仍调用 `_qk_metric_projection_factors_with_key_spectrum`，不改求解器、收缩率或位宽分配；
3. cache 的历史长度、底层 K 指针、score mode 或中心化设置变化时必须拒绝复用；
4. 该优化只减少 cached-prefix warm 固定成本，不改善单次 cold 请求总计算量，也不改变 steady decode。

如果 active packed-Key hash、候选 ID 或 PPL 与基线不一致，则“完全无数值影响”假设失败，不进入冻结默认路径。

## 部署扩展

8K/16K 当前 `t_sparse >= t_full`。生产系统若要求所有长度都不减速，可以使用由实测成本公式决定的 dense/sparse 算子调度：

```text
choose sparse iff T_index / reuse_count + G * t_sparse < G * t_full
```

这不是基于质量风险的 Full fallback，也不改变论文的纯稀疏主方法；它等价于 GEMM/attention 库中的硬件 autotuning。论文核心结果仍需报告不回退的 QKSieve 曲线。
