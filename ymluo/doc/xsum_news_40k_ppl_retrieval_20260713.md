# XSum 新闻 40K Context / 512-Token Retrieval PPL 实验

> **一句话结论：** 在 200 条真实 XSum 新闻样本上，512-token 检索在两种场景中都能达到或超过 40K full attention：自然连续新闻流的最佳方法为 `recent256 + Hybrid256`，PPL 从 **31.63 降至 30.21**；长程新闻恢复的最佳非 oracle 方法为 E5，PPL 从 **26.92 降至 21.33**。单独 K-mean/SVD32 QK 检索明显弱于 E5，但 SVD32 与 full128 QK 几乎无损等价。

**日期：** 2026-07-13  
**模型：** Qwen3-0.6B；E5-base-v2  
**硬件：** 7 x RTX 3090（GPU 0/1/2/4/5/6/7）  
**数据：** XSum test，真实 BBC 新闻文本，无合成文本  

## 1. 实验要回答的问题

1. 普通新闻数据上，从 40K 历史中只读取约 500 tokens，语言模型 PPL 会损失多少？
2. `q x K_mean`、full128 QK 和 SVD32 QK 能否作为通用新闻检索器？
3. 与 full attention、最近上下文、BM25 和标准 dense RAG 相比，质量和时间分别如何？

## 2. 严格因果协议

每条样本固定为：

```text
40,000-token history
+ 64-token visible query prefix
+ 128-token hidden target continuation
```

- history 被切成 625 个 64-token blocks。
- 检索器只看 history 和已经出现的 64-token query，不看 128-token target。
- 每次选择 8 个 blocks，即 512 tokens；送入 reader 前按原始位置重新排序。
- PPL 只计算最后 128 个 target tokens，越低越好。
- full-attention baseline 的模型输入为 40,192 tokens。
- retrieval reader 的模型输入通常为 704 tokens，即 `512 + 64 + 128`。

本实验包含两个互补协议，每个协议 100 条：

| 协议 | 构造 | 测量目标 |
|---|---|---|
| natural_stream | 连续拼接真实 XSum 新闻；query/target 紧随 40K history | 普通新闻语言建模与局部连续性 |
| delayed_article | 同一新闻的前 512 tokens 被放入 40K history，后接无关新闻；原文后续 64+128 tokens 作为 query/target | 长程相关信息能否从 40K 中找回 |

delayed_article 只改变真实新闻的排列，不生成合成句子；目标新闻被从 distractor stream 中完全排除。

## 3. 比较方法

| 方法 | 512-token 上下文来源 |
|---|---|
| Query only | 不读取 history |
| Recent512 | 最后 8 个 blocks |
| Random512 | 随机 8 个 blocks |
| BM25 | query 对 625 个 blocks 做词法检索 |
| E5 | E5-base-v2 dense embedding Top-8 |
| Hybrid RRF | BM25 与 E5 全排序做 RRF |
| Hybrid + Recent | 4 个最近 blocks + 4 个 Hybrid blocks |
| QK K-mean | 16 个 query Q 与每块 K-mean 做 centered cosine max |
| QK full128 | 4 个 layer/head 通道做 token-level exact max-QK |
| QK SVD32 | K 子空间 rank-32 投影后做相同 max-QK |
| Hybrid + QK | BM25、E5、SVD32 三路 RRF |
| Oracle512 | delayed_article 的真实前 512 tokens，仅作为上界 |
| Full40K | 完整 40K history full attention |

Q/K 通道固定为 `L3/H10, L21/H8, L6/H7, L16/H14`。Q/K 均来自真实 Qwen3-0.6B 前向；block 独立 profiling，使用 pre-RoPE 表示以避免绝对位置不一致。SVD basis 只由当前 history 的 K 拟合，不使用 target 或 gold block。

## 4. 主要质量结果

### 4.1 自然连续新闻流

| 方法 | PPL | 相对 Full40K 的平均 Delta NLL | 配对 bootstrap 95% CI |
|---|---:|---:|---:|
| **Hybrid + Recent** | **30.208** | **-0.0460** | `[-0.0717, -0.0213]` |
| Recent512 | 30.489 | -0.0367 | `[-0.0649, -0.0068]` |
| Hybrid RRF | 30.686 | -0.0303 | `[-0.0562, -0.0044]` |
| E5 | 30.955 | -0.0216 | `[-0.0457, +0.0049]` |
| **Full40K baseline** | **31.630** | 0 | - |
| BM25 | 31.781 | +0.0048 | `[-0.0262, +0.0382]` |
| Query only | 34.272 | +0.0802 | `[+0.0363, +0.1222]` |
| QK SVD32 | 34.889 | +0.0981 | `[+0.0583, +0.1378]` |
| QK full128 | 34.914 | +0.0988 | `[+0.0629, +0.1388]` |
| QK K-mean | 35.997 | +0.1293 | `[+0.0896, +0.1757]` |
| Random512 | 36.198 | +0.1349 | `[+0.0966, +0.1774]` |

自然新闻的主要条件是局部连续性。只保留最近 512 tokens 已经显著优于 full-40K；再把一半预算用于 Hybrid 远程检索，得到当前最低 PPL。纯 Q/K 检索丢掉最近上下文后明显退化。

### 4.2 延迟新闻长程恢复

| 方法 | PPL | 8 个 source blocks 平均召回 | 最后一个 source block 命中率 |
|---|---:|---:|---:|
| **Oracle512** | **20.693** | 100.0% | 100.0% |
| **E5** | **21.334** | **67.9%** | **83.0%** |
| Hybrid RRF | 21.571 | 62.1% | 75.0% |
| Hybrid + QK | 22.215 | 53.2% | 64.0% |
| BM25 | 22.801 | 42.6% | 55.0% |
| Hybrid + Recent | 23.206 | 39.9% | 53.0% |
| QK full128 | 25.795 | 11.5% | 14.0% |
| QK SVD32 | 25.799 | 12.2% | 11.0% |
| Query only | 26.106 | 0 | 0 |
| **Full40K baseline** | **26.917** | 100.0% | 100.0% |
| QK K-mean | 27.977 | 3.5% | 3.0% |
| Random512 | 28.399 | 0.7% | 0 |
| Recent512 | 28.842 | 0 | 0 |

E5 比 full-40K 降低 `0.2325` NLL，配对 95% CI 为 `[-0.2674, -0.1958]`。它与 oracle 仍相差 `+0.0305` NLL，说明剩余损失主要来自未完整召回 source。

full-40K 虽然包含全部 source，但 source 后有约 39.5K distractor tokens。Qwen3-0.6B 没有稳定利用远处信息；把正确 source 搬回 query 附近后，oracle PPL 从 26.92 降到 20.69。

## 5. Q/K 与 SVD32 的判断

1. **单个 K-mean 不是通用语义索引。** delayed_article 的平均 block recall 只有 3.5%，自然新闻 PPL 也显著差于 full attention。
2. **token-level QK 比 K-mean 强，但仍不如外部 embedding。** full128/SVD32 最后 source block 命中率仅 14%/11%，E5 为 83%。
3. **SVD32 压缩本身近似无损。** rank-32 平均保留 96.73% K residual 能量；与 full128 的 Top-8 重合率为 82.5% 至 83.0%，Top-1 一致率为 74% 至 77%。
4. **SVD32 与 full128 的下游 NLL 几乎相同。** delayed_article 的平均差仅 `+0.00018 NLL`，natural_stream 为 `-0.00071 NLL`。

因此，当前瓶颈不是 128 维压到 32 维，而是“未经训练的 Q/K 最大点积能否表达跨文档语义相关性”。SVD32 适合压缩或候选内 residual rerank，不适合单独替代 RAG seed。

## 6. 时间结果

以下为每题在线时间，包含 query 检索和 reader 前向；索引视为已预建。

| 方法 | natural_stream | delayed_article | 相对约 4 秒 Full40K |
|---|---:|---:|---:|
| Full40K | 4.000 s | 4.051 s | 1.0x |
| BM25 | 0.0341 s | 0.0336 s | 约 119x |
| E5 | 0.0435 s | 0.0405 s | 约 96x |
| Hybrid + Recent | 0.0450 s | 0.0419 s | 约 93x |
| QK SVD32 | 0.0685 s | 0.0675 s | 约 59x |

704-token reader 前向本身约 33 ms。额外在线检索开销约为：

- BM25：0.4 至 0.5 ms；
- E5 query + matrix search：7 至 10 ms；
- Q/K query capture：34 ms；
- SVD32 矩阵打分：约 0.34 ms。

这验证了“Q 与 K index 做矩阵乘法很快”：真正的 SVD32 打分不到 1 ms；当前 Q/K 在线成本主要来自重新获得 query Q，而不是 625-block 矩阵乘法。

### 6.1 离线建索引时间

| 40K 索引 | 平均时间 |
|---|---:|
| BM25 | 0.062 s |
| E5 passage embeddings | 0.196 s |
| Qwen block K profiling | 1.152 s |
| SVD32 basis + K projection | 0.012 s |

若索引只使用一次，冷启动总时间约为 BM25 0.096 s、E5 0.238 s、QK SVD32 1.232 s；若同一 40K/10M/1B context 服务许多生成步骤，离线索引会被摊薄，此时应比较在线列。

完整 200 条、所有方法的 7 卡实验墙钟为约 197 秒。该数字是批量吞吐时间，不等于单题延迟。

## 7. 当前结论与下一步

本实验支持：

```text
普通语言建模 = 强 recent prior + 少量远程语义检索
长程信息恢复 = dense semantic seed + 小预算 reader
Q/K SVD = 高效 residual feature，而不是独立全局语义入口
```

下一步不应继续把 K-mean 当作 E5 的直接替代。更合理的实验是固定 E5/BM25 seed，在相同 512-token 最终预算下，用 residual-K hierarchy 或 post-RoPE exact QK 替换一部分 RAG 候选，并测试它能否在 E5 漏召回的 17% “最后 source block 未命中”样本上带来净增益。

## 8. 代码与结果

- 数据构造：`ymluo/projects/parallel_block_retrieval/src/prepare_xsum_news_ppl.py`
- PPL 与时间评测：`ymluo/projects/parallel_block_retrieval/src/evaluate_xsum_news_ppl_retrieval.py`
- 服务器数据：`/home/fdong/ymluo/projects/parallel_block_retrieval/data/xsum_news_40k_ppl_v2`
- 服务器结果：`/home/fdong/ymluo/projects/parallel_block_retrieval/outputs/xsum_news_40k_ppl_v2/summary.json`

