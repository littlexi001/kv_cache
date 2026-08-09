# Section 197: 10K/20K/40K Full-Attention 长度基线

更新时间：2026-07-12

## 1. 目标

本实验回答两个问题：

1. 当两条正确证据都保证存在时，Qwen3-8B 直接读取 10K、20K、40K 上下文，能答对多少二跳 MuSiQue 问题？
2. 当前 10M 检索链的 36.4% 是否只是一个偏低的数字，还是已经优于模型直接读取远短于 10M 的完整上下文？

## 2. 严格协议

- 数据：`musique_official_10m_aligned_2000_v3` 的完整 500 条 test queries，即 query 1500--1999。
- 模型：冻结 `Qwen/Qwen3-8B`，FP16，关闭 thinking，greedy decode，最多生成 32 tokens。
- 任务：向模型直接提供原始二跳问题，不提供 oracle bridge，不做检索，也不做分步生成。
- 上下文：每条分别构造精确 10,000、20,000、40,000 个 context tokens。
- 正确证据：每个上下文都完整包含该问题的两条官方 gold support blocks。
- 干扰文本：从共享 10M MuSiQue 真实语料的其它 blocks 中确定性采样，不使用合成文本或合成向量。
- 嵌套控制：同一问题的 10K distractors 是 20K 的子集，20K 又是 40K 的子集；增加长度只会增加干扰。
- 位置控制：两条 gold blocks 均衡分布在六种有序位置组合：early/middle、early/late、middle/early、middle/late、late/early、late/middle。
- Prompt：要求使用 context 完成全部关系推理，只输出最短精确答案。

主要指标 `Answer Hit` 与现有 10M 检索链完全一致：标准化后 gold answer 必须完整出现在生成文本中。同时补充标准化 Exact Match 和 LongBench 风格 token F1。

## 3. 主要结果

| Context | Answer Hit | 95% CI | Exact Match | Token F1 | 单条生成均时 |
|---:|---:|---:|---:|---:|---:|
| 10K | **21.2%（106/500）** | [17.84%, 25.00%] | 15.8% | 30.84 | 3.113 s |
| 20K | **16.6%（83/500）** | [13.60%, 20.11%] | 11.6% | 25.39 | 7.070 s |
| 40K | **9.4%（47/500）** | [7.14%, 12.28%] | 5.4% | 16.17 | 17.988 s |

上下文越长，答案质量单调下降，同时 full-attention 计算时间快速增加。40K 的总 prompt 长度均值为 40,057 tokens，仍低于模型的 40,960 原生上限，因此不是截断造成的结果。

## 4. 同题配对长度退化

| 比较 | 长上下文独有正确 | 短上下文独有正确 | ties | McNemar exact p |
|---|---:|---:|---:|---:|
| 10K -> 20K | 19 | **42** | 439 | 0.00444 |
| 10K -> 40K | 15 | **74** | 411 | 1.53e-10 |
| 20K -> 40K | 18 | **54** | 428 | 2.57e-5 |

长度退化不是总体平均的随机波动。虽然新增文本偶尔提供替代证据，更多问题会因为新增 distractors 从正确变成错误。

## 5. Gold 位置控制

六种位置模式的 Answer Hit 范围：

| Context | 六种位置模式最低--最高 |
|---:|---:|
| 10K | 16.87%--26.19% |
| 20K | 8.43%--25.00% |
| 40K | 5.95%--12.05% |

位置确实影响结果，但 40K 在六种位置组合上都很低。总体下降不能归因于某一批 gold evidence 恰好全被放在中间；主要现象是证据稀释、竞争实体、关系绑定错误和多跳链路失败。

## 6. 与 10M 检索链严格配对

对同一 500 个 query IDs，把 full-context `Answer Hit` 与当前冻结的 10M Top16 candidate extraction + Yes/No support selector（36.4%，182/500）逐题配对：

| 对照 | Full context | 10M 检索 | 提升 | 检索胜/负 | McNemar exact p |
|---|---:|---:|---:|---:|---:|
| 10K full attention | 21.2% | **36.4%** | +15.2pp | 128 / 52 | 1.40e-8 |
| 20K full attention | 16.6% | **36.4%** | +19.8pp | 137 / 38 | 2.46e-14 |
| 40K full attention | 9.4% | **36.4%** | +27.0pp | 159 / 24 | 1.27e-25 |

因此当前 10M 系统的质量结果确实有意义：即使 full-attention baseline 被保证看到了两条正确证据，模型直接读取 10K--40K 仍明显更差。只给模型更多原文不能替代证据检索、分步状态传播和候选验证。

## 7. 正确结论边界

可以得出：

1. 当前 10M 稀疏迭代系统明显优于一次性读取 10K/20K/40K dense context。
2. 稀疏选择不是单纯为了绕过 10M context window；它还降低了真实的证据干扰和关系混淆。
3. 36.4% 是一个有竞争力的研究原型结果，而不是“模型只答对三分之一所以方法无效”。

不能得出：

1. 不能声称纯检索器本身带来全部 15--27pp 增益。10M 系统还使用显式二跳分解、8B bridge、逐 block candidate extraction 和 verifier，而 full-context baseline 是原始问题单次生成。
2. 不能声称 10M 检索已经近似无损。gold state + 唯一 gold paragraph 的局部 reader oracle 为 70.8%，当前 36.4% 只保留约 51.4%。
3. 不能把 40K 趋势直接当成真实 10M full attention 的实测值；Qwen3-8B 无法直接运行 10M full attention，这里只能证明 10K--40K 范围内存在显著单调退化。
4. 不能把当前 36.4% 当成可部署效率点。Top16 逐分支抽取与 8B verifier 计算量仍很高，需要蒸馏和门控。

## 8. 下一步最重要对照

为了把“检索收益”与“显式分解收益”分开，下一步应运行 matched stepwise full-attention baseline：

1. 在同一个 10K/20K/40K context 上用相同 atomic prompt 生成第一跳 bridge。
2. 把生成 bridge 写入第二跳 query，再读取同一个完整 context 生成 final answer。
3. 与 10M 检索链保持相同模型、两步 prompt、decode 长度和答案判定，仅把小块检索替换为 full context。

若 10M 检索在这个完全匹配的两步 baseline 上仍领先，才能把增益更严格地归因于 sparse retrieval，而不是任务分解。

## 9. 复现与结果

评测脚本：

```text
src/evaluate_full_context_length_baseline.py
src/compare_full_context_to_retrieval.py
```

服务器汇总：

```text
outputs/musique_fullcontext_10k20k40k_test500_v3/summary.json
outputs/musique_fullcontext_10k20k40k_test500_v3/paired_vs_10m_retrieval.json
```

10K/20K 使用四张 RTX 3090 做 query data parallel。40K 单卡 FP16 会因 KV cache 和 MLP 临时张量超过 24GB，因此每个 reader 使用两张 3090 做模型并行，并以三组双卡 reader 处理三个 query shards；这只改变权重放置，不改变模型数值精度或 attention 计算。
