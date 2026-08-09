# RiskKV-Block 多主题因果 PPL 实验（2026-07-14）

## 1. 目的

本实验测试当前 `2K page gather + original RoPE + LPCM` 方法在普通、无显式问题文本上的下一词预测能力。重点不是 LongBench 任务分数，而是回答：当前方法能否作为通用 KV 压缩方法，用于不同主题文本的自由续写。

## 2. 严格因果协议

- 模型：`Llama-3.1-8B-Instruct`。
- 数据：20 Newsgroups 真实文本，移除 header、footer 和 quote。
- 主题：计算机、体育、医学、空间、政治、宗教。
- 每个主题 3 个互不重叠窗口，共 18 个窗口、4,608 个评价 token。
- 每个窗口先观察 32,000 个历史 token，再预测后续 256 个 target token。
- 历史被拆成 31,744-token remote context 和最后 256-token causal query。
- selector 只能使用已经观察到的 query，绝不使用未来 target token。
- 当前方法从 remote context 中保留 2,048 token；query 回放后物理 KV 为 `2048 + 256 = 2304` token，即原始历史的 7.20%。
- block size 为 16 token，保留 sink/recent，使用当前 `hybrid_late_mmr_multiscale_idf_flow` selector。
- 稀疏 KV 使用原始逻辑位置、显式物理因果 mask 和 LPCM。

比较方法：

1. `full_kv`：保留全部 32K 历史。
2. `sink_recent_2k`：相同 2K remote 预算，只保留 sink 和最近 token。
3. `ours_2k_lpcm`：当前 query-conditioned block selector。

PPL 按全部 target token 的平均 NLL 计算。PPL 越低越好；`PPL/Full` 越接近 1 越好。

## 3. 总体结果

| 方法 | KV ratio | NLL | PPL | PPL / Full |
|---|---:|---:|---:|---:|
| Full KV | 100.00% | 2.3700 | 10.6971 | 1.0000 |
| Sink + recent 2K | 7.20% | 2.5073 | 12.2719 | 1.1472 |
| Current 2K LPCM | 7.20% | 2.5350 | 12.6170 | 1.1795 |

当前方法在 7.20% KV 下相对 Full 的 PPL 增加 17.95%。同预算 recent baseline 增加 14.72%。当前方法在 10/18 个窗口优于 recent，但只有 1/18 个窗口不差于 Full；少数高风险窗口主导了总体退化。

## 4. 分主题结果

| 主题 | Full PPL | Recent 2K PPL | Current 2K PPL | Current / Full |
|---|---:|---:|---:|---:|
| 计算机 | 17.2116 | 18.1125 | 17.3717 | 1.0093 |
| 体育 | 8.0679 | 10.8657 | 11.0729 | 1.3725 |
| 医学 | 8.7336 | 11.4697 | 14.0695 | 1.6110 |
| 空间 | 7.6933 | 8.0155 | 7.8808 | 1.0244 |
| 政治 | 10.8531 | 11.1974 | 11.5644 | 1.0655 |
| 宗教 | 14.7963 | 16.8592 | 16.3551 | 1.1054 |

计算机和空间主题接近 Full，说明 query-conditioned 检索在词项稳定、局部主题连续时有效。体育和医学存在明显失败窗口，说明普通续写需要的远程信息不一定与最后 256-token query 有直接词面重合；当前 selector 会错过对下一词分布有帮助的长程重复、实体、文体和主题先验。

## 5. 结论

当前方法不能在 2K 预算下直接宣称为通用 PPL 保真压缩方法。它目前更适合有显式 query、答案证据可定位的任务；普通自由续写属于不同 action family。

这不否定已有 LongBench/RULER 结果，而是给出了更清晰的适用边界：

- 显式 query / retrieval-like task：可以使用当前激进 block selector。
- queryless / free continuation：不能默认使用当前 2K action，应采用 recent-heavy hybrid、更高预算或专用语义 selector。
- router 需要显式加入 `queryless/free-continuation` 任务状态，并为其配置安全动作。

下一步优先测试 2K 固定预算下的 `recent + semantic retrieval` 混合比例，然后测试 4K/8K 安全预算，目标是在普通 PPL 上达到 `PPL/Full <= 1.05`，同时不改变已冻结的 paper-test。

## 6. 实现与原始结果

- Harness：`ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_multitopic_lpcm_ppl_20260714.py`
- 回归测试：`ymluo/projects/qwen3_top2_head_limit3_ppl/tests/test_multitopic_lpcm_ppl.py`
- 原始结果：`ymluo/projects/qwen3_top2_head_limit3_ppl/results/20260714_multitopic_lpcm_ppl_32k_w3/`

注意：20 Newsgroups 是同主题论坛文档拼接流，不是单篇 32K 连续专著；当前结果适合作为自由生成风险的 pilot 和方法边界证据，不能替代 WikiText、PG-19 或 SlimPajama 等标准语言建模语料上的完整 PPL 表。

## 7. 体育和医学退化诊断

### 7.1 退化集中在少数窗口

| 主题/窗口 | Full PPL | Recent 2K | Current 2K | Current - Full NLL |
|---|---:|---:|---:|---:|
| 体育 0 | 8.790 | 18.226 | 18.648 | +0.7521 |
| 体育 1 | 2.584 | 2.718 | 2.784 | +0.0746 |
| 体育 2 | 23.122 | 25.894 | 26.150 | +0.1231 |
| 医学 0 | 10.224 | 20.836 | 20.033 | +0.6727 |
| 医学 1 | 7.694 | 7.974 | 14.249 | +0.6162 |
| 医学 2 | 8.468 | 9.082 | 9.756 | +0.1416 |

体育窗口 0 贡献了该主题约 79% 的总 NLL gap；医学窗口 0 和 1 合计贡献约 90%。因此不是“体育和医学中的每段文本都不适合压缩”，而是少数重复模板或强局部连续性窗口造成了主要损失。

### 7.2 Full KV 利用了远程原样重复

体育窗口 0 的 target 有 16.8% token 在 remote history 中存在至少 32-token 的原样重复；医学窗口 0 和 1 分别为 20.3% 和 19.9%。重复内容主要是邮件签名、分隔线和固定模板：

- 体育：Paul Andresen 的签名和分隔线在约 15.3K token 之前出现过。
- 医学：Gordon Banks 的固定签名在约 30K token 之前出现过。

当前 selector 只能观察 query，而这些未来重复片段在 query 中没有 16-token 重复线索。诊断发现三个窗口中可重复 target 片段的保留率均为 0%。Full KV 可以直接利用历史副本，当前方法和 recent baseline 都不能。

有 32-token 原样重复的三个窗口，`Current - Full` 平均 NLL gap 为 0.680；无此重复的另外三个窗口平均只有 0.113，前者约为后者的 6 倍。因此体育和医学的主题级 PPL gap 被易记忆的重复模板显著放大。

### 7.3 当前 selector 牺牲了连续近邻

| 主题/窗口 | 当前 2K 中来自最近 2K 的 token | 占最近 2K | 连续片段数 | 中位片段长度 |
|---|---:|---:|---:|---:|
| 体育 0 | 240 | 11.7% | 90 | 16 |
| 体育 1 | 1,303 | 63.6% | 62 | 16 |
| 体育 2 | 128 | 6.2% | 90 | 16 |
| 医学 0 | 192 | 9.4% | 91 | 16 |
| 医学 1 | 512 | 25.0% | 90 | 16 |
| 医学 2 | 583 | 28.5% | 77 | 16 |

当前 2K KV 通常被分散成 77--91 个小片段，而不是一段连续历史。体育窗口 1 保留了 1,303 个近邻 token，所以只轻微退化。医学窗口 1 是连续的 MSG 长文，recent baseline 几乎等于 Full；当前 selector 只保留 512 个最近 token，并选入距 query 28,672 token 的旧 MSG 片段。该片段和 query 主题高度相关，但不是实际下一段文本，既丢失局部连续性，又造成 related-but-wrong continuation interference。

### 7.4 数据协议的影响

20 Newsgroups 窗口由多篇独立帖子用 `---` 拼接。部分 256-token target 跨越帖子边界，Full KV 因而能利用同主题语料中的历史签名、模板和写作习惯。这是有效的 corpus-memory 压力测试，但不是纯粹的单篇连续文本 PPL。

后续需要拆成两套协议：

1. 多文档 corpus-memory PPL：保留当前协议，专门衡量重复模板和跨文档记忆。
2. 单文档 continuation PPL：使用 PG-19、WikiText 长文章或同类连续文本，避免独立帖子边界和重复签名主导结果。

对方法的直接改进是为 queryless continuation 增加 recency floor：不能只保证 64 个 recent token。应先测试 1K recent + 1K retrieval、1.5K recent + 0.5K retrieval，以及连续 span repack，再决定是否需要 4K/8K 安全预算。

## 8. 通用 KV retrieval 的解决方向

### 8.1 固定 local/retrieval 比例不够

保持总 remote budget 为 2K 时，`1K local + 1K retrieval` 的六主题总体 `PPL/Full` 为 1.1858，原方法为 1.1795，没有总体改善。固定 50/50 分配只能改善部分窗口，不能成为通用策略。

`1.5K local + 0.5K retrieval` 在两个最差主题上的结果更有诊断价值：

- 医学窗口 1：PPL 从 14.249 降到 7.955，接近 recent 7.974 和 Full 7.694。
- 医学主题 `PPL/Full` 从 1.611 降到约 1.334。
- 体育主题 `PPL/Full` 从 1.373 降到约 1.352，仍受远程重复签名主导。

因此 1.5K local 可以解决连续性丢失，但不能解决 query 之后才显现的历史重复。静态一次检索在因果条件下无法预知未来签名，这是信息可见性问题，而不是继续调 scorer 权重可以解决的问题。

### 8.2 Causal Tri-Memory KV Retrieval

通用方案应把固定一次检索改成三通道因果记忆：

1. Local continuity memory：不可被检索器覆盖的连续 recent KV，保护语法、文体和短程状态。
2. Semantic evidence memory：当前方法的 query-conditioned remote KV，用于 QA、实体和远程证据。
3. Echo/recurrence memory：对已经观察到的新 token 做 rolling suffix/hash 匹配，命中历史重复后取回其后续 KV block。

总预算满足 `B = B_local + B_semantic + B_echo`。router 只使用当前可见特征分配预算：是否有显式 query、retrieval gap、top-k 稳定性、最近 block 集中度、rolling match 长度和生成熵。建议的初始动作是：

- 显式 QA：小 local floor，大 semantic budget。
- 自由续写：约 1.5K local，剩余预算做 semantic retrieval。
- rolling suffix 命中：从 semantic budget 中切出 256--512 token 给 echo continuation。
- 高风险或低置信：升到 4K/8K 或 fallback。

Echo 通道必须在生成过程中每 16--32 token 更新一次。PPL 评价时只允许使用已经观察到的 target prefix；实际生成时使用模型已经生成的 prefix，因此没有未来泄漏。实现上可以在 chunk boundary 重新 gather remote KV 并重放短 query/decode suffix；后续再用 paged KV swap 优化开销。

该方案仍然是 KV retrieval：检索对象是同一上下文已经计算出的 KV page，不访问外部知识库、不拼接检索文本，也不重新编码外部文档，因此与 RAG 的边界清楚。

### 8.3 Trigger-only causal echo 实测

已经实现 8-token rolling suffix、trigger-only cache rebuild 原型。每 8 个已观察 token 查询一次历史 token index；只有出现新的 echo episode 才替换 echo KV 并重放短 suffix。同一 episode 的后续命中复用第一次取回的 continuation page。

| 方法 | KV ratio | PPL | PPL / Full |
|---|---:|---:|---:|
| Full KV | 100.00% | 10.6971 | 1.0000 |
| Sink + recent 2K | 7.20% | 12.2719 | 1.1472 |
| Static 1.5K local + 0.5K retrieval | 7.20% | 12.2611 | 1.1462 |
| Trigger-only causal echo | 7.20% | 11.6985 | **1.0936** |

分主题 `PPL/Full`：

| 主题 | 原始 2K selector | Trigger-only echo |
|---|---:|---:|
| 计算机 | 1.009 | 1.029 |
| 体育 | 1.373 | **1.167** |
| 医学 | 1.611 | **1.165** |
| 空间 | 1.024 | 1.040 |
| 政治 | 1.066 | 1.035 |
| 宗教 | 1.105 | 1.135 |

18 个窗口中只有 6 个出现 rolling match，总共触发 7 次 cache rebuild。无匹配主题基本保持静态双记忆结果；体育最差窗口从 18.173 降到 11.696，医学最差窗口从 20.969 降到 14.339。这证明在线因果检索能够追回静态 query 无法预见的远程重复。

当前 PPL harness 中 echo 方法平均记录 2.09 秒，静态方法为 0.25 秒，但两者不能直接作为系统速度比较：echo 为保证因果评价按 8-token chunk 执行，静态 PPL 使用 64-token 并行 teacher forcing。真实生成本来就是逐 token decode。下一步必须在统一 decode harness 下分别统计 rolling-hash、KV gather、suffix replay 和 attention 时间，才能报告端到端开销。
