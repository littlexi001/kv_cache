# 实验设计

## 数据

- 基准：lm-evaluation-harness 中的 13 个官方 RULER 子任务。
- 长度：32768 token。
- tokenizer：与被测模型相同的 Qwen3-8B tokenizer。
- seed：42。
- 实际 raw prompt 长度由官方任务生成器决定；本缓存范围为 28,768–32,656 token，均属于 `max_seq_length=32768` 的 RULER-32K 协议。
- smoke：`niah_single_2` 与 `qa_hotpot` 各 1 条，包含 dense replay 审计。
- pilot expansion：13 个子任务各 2 条；若耗时过高，先完成各 1 条并明确标记样本量。

## 模型与计算

- 模型：Qwen3-8B。
- GPU：服务器 GPU 6–7；两个样本 shard 各使用一张卡。
- 权重：NF4 4-bit，计算 dtype 为 BF16；所有方法完全相同。
- attention：SDPA 进行共享前缀 prefill；单 token 查询/解码使用可审计 eager attention patch。
- prefill chunk：256 token。
- 解码：greedy；每个任务沿用其 RULER 上限，最多 128 token。

## 变体

1. `native_full`：模型原生完整 KV。
2. `full_rope_replay`：只在 smoke 中使用，验证 patch 等价性。
3. `rope_top2`：exact post-RoPE Top-2%。
4. `local_global_postscore`：pre-RoPE 远程召回，native post-RoPE 消费。
5. `local_global_blend25`：同一召回支持集，25% 校准语义分数补偿。

## 指标

主指标是每个任务的官方 RULER score，以及 13 任务宏平均。辅助指标为 paired sample delta、bootstrap 95% CI、第一答案候选的 next-token NLL、可对齐答案证据 recall/mass、生成耗时和审计错误率。

“答案证据”是诊断代理：仅在 NIAH/QA 中，将答案字符串在 context 中的全部精确 token 出现位置作为 span。它不是 RULER 官方证据标注，因此只支持相对比较，不作为主结果。

## 预注册解释规则

- 方法分数更高且 paired CI 全部大于 0：支持 H1。
- 点估计更高但 CI 跨 0：仅记为值得扩样的信号。
- 官方分数提升、证据代理 recall 不升：不能宣称提升来自更好的 gold evidence recall。
- blend 只改善 NLL、不改善官方分数：记为校准信号，而非任务提升。
- Full 明显低于稀疏方法时，检查稀疏是否通过过滤竞争信息获益，而不把它解释成近似 Full。

## 运行阶段

1. 校验生成缓存：13 个任务、目标长度和每任务样本数。
2. 两任务 smoke：检查显存、dense replay、支持集预算、停止 token 和官方评分。
3. GPU6–7 分片 expansion。
4. 合并 raw rows、paired bootstrap、任务图和失败样例。
