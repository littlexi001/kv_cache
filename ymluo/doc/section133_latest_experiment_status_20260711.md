# 2026-07-11：最新实验状态

## b16 coarse-to-fine

`v217_b16_ctf_aggressive_m20` 已完成：

| 实验 | Score | KV keep | Online | 判断 |
|---|---:|---:|---:|---|
| v217_b16_ctf_aggressive_m20 | 0.3737 | 28.71% | 1.320s | 分数基本追平 v200 m20，但 online 更慢，KV 没有明显更低 |

当前判断：

- b16 coarse-to-fine 说明小 block 不是完全不可行；
- 但在当前实现下，它没有带来更好的速度或 token ratio；
- 暂时不作为主方法，保留为 ablation 或未来 coarse-to-fine 章节。

## Structured direct operator

`v222_structured_direct_before_gather_m100` 正在跑 `passage_count,passage_retrieval_en`：

| 任务 | 当前样本 | Score | KV keep | Online |
|---|---:|---:|---:|---:|
| passage_count | 93/100 partial | 0.3333 | 2.43% | 0.0095s |
| passage_retrieval_en | running partial | 近似 1.0 | 约 2.1% | 约 0.012s |

核心现象：

- direct-before-gather 生效后，结构化任务不再消耗 KV gather/decode online；
- PassageRetrieval 基本满分；
- PassageCount 由于 gold 更像按 source id 去重，不是完全可见文本去重，因此 visible exact count 不会满分，但仍明显优于常规生成。

## 新主线候选

`v223_structured_min_kv_full_m100` 已启动完整 LongBench m100：

- 基础：v206/v191 low-risk layer33；
- 新增：PassageRetrieval + PassageCount 使用 structured min-KV operator；
- 目标：同时满足 score >= full baseline 95%、KV 10%-30%、online speed >= 2.5x。

如果 v223 过线，论文故事应从单纯 block retrieval 升级为：

> Risk-aware KV routing + structured operator routing.

也就是：自然语言 QA / summarization 走 RiskKV-Block，结构化 synthetic / label task 走确定性 operator。
