# QKSieve 质量与速度结论

## 核心判断

质量和速度不应强行使用同一种样本分布，但必须使用同一算法路径。

- LongBench/RULER 回答“任务质量是否保持”和“真实短输出请求到底多快”。
- 固定长度系统实验回答“索引固定成本、逐 token 成本、上下文长度和输出长度如何共同决定加速”。
- Agent warm 实验回答“同一长前缀及其伴随索引被多轮复用时有多快”。

三者互补，不能互相替代。论文中应同时报告，且明确是否包含 prefill、resident index 和 request-local index。

## 当前结果

最终等价优化扫描表明：

- 8K：稳态 0.81x，不存在摊销回本点。
- 16K：稳态 0.95x，不存在摊销回本点。
- 32K：稳态 1.56x，warm/cold 分别约 10/22 token 回本。
- 64K：稳态中位数 2.35x，warm/cold 分别约 5/10 token 回本；存在较大并行测量方差。
- 128K：稳态 3.32x，warm/cold 分别约 3/7 token 回本。

128K PPL probe 保持 99.998%，但它只证明本轮工程改造没有破坏当前数值路径，不替代完整 LongBench/RULER。

在上述长度扫描之后，进一步发现旧精确候选 attention 会把预分配 KV cache 的逻辑切片整体复制为连续张量。改为直接按 stride 读取并复用 workspace 后，128K 双 RTX 3090 的配对 A/B 从 88.298 降至 80.178 ms/token，Full 对比从 3.391x 提升到 3.740x，两个路径的 PPL 质量保持均为 99.6315%。旧长度扫描尚未按新默认路径全部重测，因此 8K--128K 表仍保留原始可复核数字，不能把新的 128K 点直接拼成一条最终曲线。

## 还能否在不影响质量的情况下加速

可以，而且本轮已经兑现三项：fused W_o-metric Value append、跨层 batched qMSE allocation，以及 stride-aware 精确候选 attention。它们不改变 bit allocation、packed Key、候选集合或 Value 补偿公式。

后续按收益和风险排序：

1. **让索引成为 KV cache 的伴随状态。** 在 prefill 期间增量构建或异步构建 query-independent 索引，并在多轮 Agent 中持久化。它不改变任何数值，只隐藏或复用 `T_resident`。
2. **融合 packed-Key 投影、量化与写入。** 当前仍有独立张量分配和多次 kernel launch；需要 bulk WMMA/CUDA kernel，并用 36 层 active index hash 验证。
3. **CUDA Graph 或 persistent runtime。** 消除 Hugging Face/Python 的逐层调度与 allocator 开销，重点改善 32K 及以下。
4. **融合 proxy scan、候选压缩与候选消费。** 已测试的单 kernel 原型在 32K--128K 只有旧参考的 0.67--0.71x，因为局部候选 attention 降低 occupancy 并破坏访存合并。下一版只能在保持全局协作扫描和合并读取的前提下设计，不能简单把三个阶段塞进一个 kernel。
5. **硬件成本算子调度。** 生产部署在 8K/16K 选择 exact dense SDPA，在更长上下文选择 QKSieve。决策来自延迟公式，不依赖任务、router 或质量风险；论文纯稀疏曲线仍单独报告。

## 可达到的边界

旧长度扫描中 128K Full 为约 302 ms/token，QKSieve 为约 91 ms/token；最新配对 A/B 已把 QKSieve 降到约 80 ms/token。此前板块测量显示非检索模型底座约为 55--60 ms/token，因此即使把检索开销完全消除，整模型稳态上界也约为 5.0--5.5x。最新 3.74x 相对该上界仍有约 1.34--1.47x 的实现空间，但不会再出现仅靠一个小 kernel 获得数量级提升。

32K 的 Full 只有约 88 ms/token，已经接近模型底座，当前 1.56x 更接近该长度的整模型上限。8K/16K 若坚持纯稀疏路径，只能依靠更深的跨层融合或更低开销 runtime；索引摊销本身无法解决。

## 论文报告建议

主表分成两条证据链：

1. **质量链：** LongBench 16 个任务、RULER 多长度、多模型，使用完整 QKSieve 实现。
2. **系统链：** 8K--128K 的 steady/warm/cold/raw 延迟曲线，横轴同时覆盖上下文长度和生成长度。

LongBench 真实请求速度应如实报告。多数 QA 输出很短、上下文又集中在 8K--16K，当前实现可能不加速甚至变慢；这不是隐藏项，而是系统交叉点结论。QKSieve 的主要部署优势应定位为 32K 以上、缓存长前缀、多轮 Agent 或较长生成。

## 下一步

1. 用新的 stride-aware 默认路径在独占 GPU、锁定时钟下重测 8K--128K，消除 64K 方差并冻结系统表。
2. 冻结同一实现，补齐 LongBench/RULER 的任务质量与实际请求时延。
3. 做真实多轮 Agent cache 增量更新和跨请求复用实验。
4. 在 H100 80GB 上按完全相同口径复测，并与公开方法的官方 kernel 对齐。
