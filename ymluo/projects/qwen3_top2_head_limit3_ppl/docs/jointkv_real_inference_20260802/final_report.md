# JointKV 真实推理接入报告

## 结论

本轮把 JointKV 从随机索引 CUDA replay 接入了真实 Qwen3-0.6B 的 28 层 Hugging Face
自回归 decode。真实 K/V 被编码为每 token、每 KV head 18 Byte 的辅助索引，新的
selector 按 Query head 独立做全局 sampled-threshold 扫描，并在候选真实 K/V 上
执行精确 attention。

系统结果是：8K decode 已接近 Full 速度，32K 可达到约 1.35x 稳态加速；但是连续
64-token 的 32K 质量最高只有约 96%，没有达到 99.5% 目标。该结果证明真实接入和
长序列加速成立，同时否定了“selected-only 已足够作为最终方法”的假设。

## 1. 从失败得到的结构

最初的高速 kernel 每 32 个 token 固定保留若干候选，并让同一 KV head 下的 Query
heads 共用候选。真实 Layer-0 的平均注意力质量只有约 42%，而 per-query 全局 proxy
可恢复到约 74%。原因是高分 token 在真实文本中会集中于少数区间，不满足均匀局部
quota 假设；不同 Query head 的高分集合也不一致。

因此，本轮没有继续调局部 quota，而是删除该结构约束，使用 per-query 全局阈值压缩。

## 2. 方法

### 2.1 真实索引

每个 KV head 使用冻结的 Query/Key 二阶矩坐标与二值 codebook。每个前缀 token 保存
64-bit 主编码、48-bit 残差编码、6-bit Joint ID 和 8-bit risk byte。当前物理格式为
18 Byte；risk byte 在本版 selector 中未参与排序。

### 2.2 融合 query 编码

每个 decode Query head 需要 192 个 probe 和 14 个 256-entry byte LUT。通用 einsum
加独立 LUT kernel 在 8K 单层耗时约 0.081 ms，接近 Full attention 本身。新增 CUDA
kernel 用一个 block 完成 head-specific 128→192 投影和 LUT 构建，耗时降至 0.0104 ms。

### 2.3 全局候选与精确 attention

每个 Query head 从 512 个均匀位置估计分数分位点，再扫描整个紧凑索引并压缩高于
阈值的 token。已生成 suffix 直接保留，当前 token 作为 self token。最终 attention
读取候选的真实 FP16 K/V 并重新计算精确 QK 和 softmax。本版不使用 Full fallback，
也不使用 tail correction。

## 3. 实验

Full 和 Sparse 各自运行真实 HF forward；dense prefill、DynamicCache 更新和后续 K/V
均未旁路。8K 在两类文本上连续评测 128 token，32K 在生物医学文本上连续评测
64 token。所有正式条件 overflow 为 0。

### 3.1 8K

严格约 6% 前缀候选时，质量保持率为 94.38-95.32%，decode 为 0.987-0.990x。
约 12% 候选时，生物医学为 98.87%、编译器为 100.71%，decode 为 0.988-0.989x。
8K 的结论是速度基本持平，质量仍随文本变化。

### 3.2 32K

候选从约 1258 增到 5124 时，质量从 31.19% 升到 96.07%，Top-1 从 82.81% 升到
98.44%；稳态 decode 始终约 1.31-1.36x。3x 到 4x 候选几乎不再改善质量，表明遗漏
尾部 Value 合计贡献比候选数量本身更关键。

### 3.3 开销

辅助索引占 exact FP16 K/V 的 3.516%，但 exact K/V 仍驻留 GPU。当前 Python 构建
索引耗时 4.1-4.5 秒，使 8K×128 和 32K×64 的一次性在线速度分别只有约 0.53x 和
0.60-0.61x。32K 需要约 360-375 个生成 token 才能摊平该构建成本。对已经保存并
复用 KV cache 的 Agent 场景，稳态 1.35x 是更相关的数值；对一次性请求则不是。

## 4. 下一步可证伪问题

下一步不应继续增加候选比例。应实现正确归一化的 tail correction：同时估计未选
token 的 softmax 分母和 Value 加权分子，再与候选 exact numerator/denominator 合并。
最小通过条件是在不使用 Full fallback 的情况下，把 32K×64 质量从约 96% 提高到
99.5%，同时保持至少 1.2x 完整模型稳态 decode。

之后才值得进行第二模型、LongBench、RULER 和 128K 的正式实验。索引构建则应改为
prefill 期间增量或异步 GPU 编码，单独验证冷启动总延迟。

## 5. 结论边界

本轮支持的最强结论是：JointKV 的真实紧凑索引和 per-query CUDA selector 已能在
32K 真实模型 decode 中产生约 1.35x 稳态加速。它尚不是质量闭环的最终论文方法，
也不能用早期 8-token 或随机索引结果声称 99% 以上长程质量。
