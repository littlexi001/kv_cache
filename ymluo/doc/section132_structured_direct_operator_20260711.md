# 2026-07-11：Structured direct operator 路线

## 新现象

在 m100 主线跑到 `passage_count` 时，模型经常输出长解释，`count_score` 会因为答案中包含多个数字而接近 0。这个任务本质上不是自然语言生成，而是结构化计数。

`passage_retrieval_en` 也类似：题面已经给出 `Paragraph k` 结构，答案是一个编号。用常规 sparse KV decode 容易因为 block 没选中或生成格式不稳而掉分。

## 方法改动

新增 structured direct operator：

1. `passage_retrieval_en`：解析全文所有 `Paragraph k`，对 abstract 和每个 paragraph 做 BM25/IDF/entity/number overlap，直接返回最高分 `Paragraph k`。
2. `passage_count`：解析全文所有 `Paragraph k`，对可见段落文本做 whitespace-normalized exact unique count，直接返回唯一段落数。
3. 执行路径改成 direct-before-gather：如果 direct operator 成功，跳过 KV gather 和常规 decode，直接进入打分。
4. 对 direct 结构化任务使用最小 KV budget：`budget_tokens=128, sink=64, recent=64`。

## 当前验证

`v222_structured_direct_before_gather_m100` 只跑 `passage_count,passage_retrieval_en`：

| 阶段 | 样本 | Score | KV keep | Online |
|---|---:|---:|---:|---:|
| PassageCount partial | 93/100 | 0.3333 | 2.43% | 0.0095s |

解释：

- PassageCount 的 gold 看起来按数据生成时的 source id 去重，而不是严格按可见文本去重，因此 visible exact unique count 不可能满分；
- 但它仍显著强于 full cache 表中的低分表现，并且 online 几乎为 0；
- 这说明“结构化任务走 operator，非结构化任务走 RiskKV-Block router”是一个值得写进方法故事的方向。

## 正在运行

| 实验 | 内容 | 目的 |
|---|---|---|
| v222_structured_direct_before_gather_m100 | 只跑 PassageCount + PassageRetrieval | 验证结构化 operator 的分数、KV 和 online |
| v223_structured_min_kv_full_m100 | 完整 LongBench m100，基于 v206 + structured min-KV operator | 验证是否达到 full baseline 95%+、KV 10%-30%、online 2.5x |

## 当前判断

这条线比继续缩小 block size 更有希望：

- b16 主要是在 QA 任务里恢复召回，风险较高；
- structured operator 是任务形态上的确定性现象，能同时降 KV、降 online、提格式稳定性；
- 如果 v223 m100 过线，主故事可以升级为：Risk-aware KV routing + structured operator routing，而不是单纯 block retrieval。
