# LongBench 普通文本 RAG 基线实验

**一句话结论：普通文本 RAG 仅读取约 1K/6K tokens，就在 LongBench M100 上取得 34.05 分，保留 Full Attention 36.58 分的 93.1%，并将平均单题总耗时从 4.78 秒降到 2.83 秒；它明显强于当前纯 KV 系统，但仍未达到无损。**

## 1. 实验要回答什么

本实验检验一个更简单的假设：对标准 LongBench，是否不需要复杂的 KV cache 检索，直接在原始文本上做字符串或普通 RAG 检索，就能得到足够高的任务分数。

这里测试的是普通文本检索，不读取模型的 Q/K/V，不使用 attention、SVD 或 KV cache 压缩分数。检索过程不使用参考答案。

## 2. 公平口径

- 模型：Llama-3.1-8B-Instruct，与已有 Full Attention 基线相同。
- 数据：LongBench 16 个任务；正式结果每任务前 100 条，共 1600 条。
- 原始上下文：最多 7500 tokens，实际平均 6079 tokens。
- 文本切块：每块 128 tokens。
- RAG 预算：最多 1024 个检索文本 tokens，实际平均 1002 tokens。
- 生成与评分：沿用已有受控脚本的官方任务 prompt、最大生成长度和 LongBench 指标。
- 样本对齐：RAG 与 Full Attention 使用完全相同的 1600 个 sample id。
- 并行：M100 使用 7 张空闲 RTX 3090 做样本并行；GPU 3 未占用。

## 3. 检索方法

### BM25-1024

用问题作为 query，对全部 128-token 文本块计算 BM25，取最高分文本块直到达到 1024-token 预算，再按原文位置排序后输入模型。

### E5-1024

使用 `intfloat/e5-base-v2` 分别编码问题和文本块，按余弦相似度选择最高分文本块，预算同样为 1024 tokens。

### Hybrid+Recent-1024

这是筛选实验中的最佳方法：

1. 固定保留约 512 tokens 的文档尾部最近文本。
2. BM25 与 E5 分别对其他文本块排序。
3. 使用 Reciprocal Rank Fusion（RRF，k=60）融合两个排名。
4. 从融合排名中再取约 512 tokens，与最近文本合并并恢复原文顺序。

它不是“BM25 粗召回 + SVD 精排”，也不使用当前项目的四通道 SVD32。

## 4. M20 方法筛选

| 方法 | LongBench score | 平均总耗时 | 平均检索耗时 | 读取文本 |
|---|---:|---:|---:|---:|
| Full Attention | 37.27 | 4.71 s | - | 6108 tokens |
| Hybrid+Recent-1024 | **33.43** | **2.83 s** | 0.108 s | 1000 tokens |
| E5-1024 | 31.93 | 2.82 s | 0.098 s | 1004 tokens |
| BM25-1024 | 30.86 | **2.75 s** | **0.048 s** | 1006 tokens |
| 当前纯 KV v440 | 26.04 | 3.98 s | - | 288 tokens |

M20 上，Hybrid+Recent 相对当前纯 KV v440 高 7.39 分，配对 95% CI 为 `[+3.83, +11.09]`，sign-flip `p=0.00115`。但两者文本/KV 预算不同，因此这不是严格的等预算 AdaKV 对比。

E5 比 BM25 高 1.08 分、Hybrid+Recent 比 E5 高 1.50 分，但两项 M20 差异的置信区间均跨 0。现阶段不能断言语义检索或 Recent 组件单独带来稳定提升，只能据筛选结果选择 Hybrid+Recent 做 M100。

## 5. M100 正式结果

| 指标 | Full Attention | Hybrid+Recent-1024 | 变化 |
|---|---:|---:|---:|
| LongBench score | **36.58** | 34.05 | -2.53 分 |
| 相对 Full 分数保留率 | 100% | **93.1%** | -6.9% |
| 平均读取原文 | 6079 tokens | **1002 tokens** | -83.5% |
| 平均 prefill | 1.681 s | **0.318 s** | 5.29x |
| 平均检索 | 0 s | 0.103 s | +0.103 s |
| 平均单题总耗时 | 4.780 s | **2.829 s** | 1.69x |

RAG 与 Full 的逐样本平均差为 -2.53 分，分层 bootstrap 95% CI 为 `[-3.85, -1.22]`，配对 sign-flip `p=0.0003`。1600 条样本中，RAG 胜 376 条、平 695 条、负 529 条。因此结果是“接近但有损”，不能写成与 Full Attention 无显著差异。

7 卡正式推理阶段最长 worker 用时 695.8 秒，即约 11.6 分钟；包含错峰模型加载和最终汇总检查的端到端运行约 13.2 分钟。

## 6. 分任务边界

| 任务 | Full | RAG | RAG-Full |
|---|---:|---:|---:|
| multifieldqa_en | 56.48 | **56.89** | +0.41 |
| samsum | 14.48 | **18.78** | +4.30 |
| triviaqa | 18.13 | **18.70** | +0.58 |
| passage_count | 9.60 | **11.06** | +1.46 |
| hotpotqa | **49.40** | 47.95 | -1.46 |
| 2wikimqa | **45.75** | 42.98 | -2.77 |
| qasper | **45.11** | 41.44 | -3.67 |
| musique | **25.30** | 21.28 | -4.01 |
| narrativeqa | **22.49** | 15.80 | -6.69 |
| lcc | **59.85** | 45.58 | -14.27 |
| repobench-p | **50.60** | 44.30 | -6.29 |

其余任务差距在约 0.5 至 3.5 分之间。最大损失出现在代码补全、叙事问答和部分多跳问答：固定 1K 文本预算会漏掉跨块依赖或连续局部上下文。摘要任务并没有一致崩溃，但 1K 输入对全局覆盖仍缺少保证；`passage_count` 的绝对分数很低，其小幅反超不能证明 RAG 擅长全文聚合。

## 7. 当前结论

1. **普通文本 RAG 是必须加入的强基线。** 在标准 LongBench 上，它比当前纯 KV 检索简单、分数更高、平均速度也更快。
2. **“RAG 足以完成全部任务”目前只成立到 93.1% Full 分数。** Full 仍显著高 2.53 分，不能称为无损替代。
3. **后续等预算实验表明 SnapKV/AdaKV 明显强于本 RAG。** 在同模型、同 1600 条样本和 1024-token KV 预算下，SnapKV 为 36.44、AdaKV 为 36.36，均与 Full 36.58 无显著差异，并显著高于 RAG 34.05；H2O 为 33.48，与 RAG 无显著差异。完整结果见 `longbench_b1024_adakv_snapkv_h2o_comparison_20260714.md`。
4. **LongBench 只能验证 7.5K 以内的质量与速度。** 它不能证明 RAG 在 10M 或 1B tokens 上的索引构建、存储和在线检索仍然高效。
5. **研究价值应放在 RAG 做不到的部分。** 后续 KV 方法至少要在等预算下超过 Hybrid+Recent，或证明其在隐式状态、生成中动态查询、无法可靠文本化的证据上有独立优势。

## 8. 产物

- 运行器：`ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_longbench_text_rag_baseline.py`
- M20：`outputs/longbench_text_rag_m20_v2`
- M100：`outputs/longbench_text_rag_hybrid_recent_m100_v1`
- M100 逐样本结果：`task_results.csv`
- M100 分任务汇总：`summary.csv`
