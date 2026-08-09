# 2026-07-11：已验证结果与夜间运行计划

## 已完成：v213 new PassageRetrieval head m100

`v213_newpr_v191_m100` 已完成完整 LongBench m100：

| 指标 | 数值 |
|---|---:|
| Score | 0.3674 |
| Full baseline m100 | 0.3658 |
| KV keep | 29.53% |
| Online | 1.461s |
| 估计 online speed | 约 2.12x |

判断：

- Score 已超过 full baseline，质量和 KV ratio 已经达标；
- 速度还没到 2.5x；
- 慢点主要来自长 decode 任务：GovReport、MultiNews、QMSum、RepoBench、LCC。

## 已完成：v222 structured direct-before-gather smoke

`v222_structured_direct_before_gather_m100` 已完成 PassageCount + PassageRetrieval 各 100 条：

| 任务 | Score | KV keep | Online |
|---|---:|---:|---:|
| passage_count | 0.3700 | 2.43% | 0.0094s |
| passage_retrieval_en | 1.0000 | 2.06% | 0.0125s |
| overall | 0.6850 | 2.25% | 0.0110s |

判断：

- structured direct operator 路线成立；
- direct-before-gather 后结构化任务 online 几乎归零；
- PassageCount 虽然不能满分，但显著强于常规生成；
- PassageRetrieval 满分且 KV 从约 10.56% 降到约 2.06%。

## 正在运行

| 实验 | 目的 | 状态 |
|---|---|---|
| v223_structured_min_kv_full_m100 | v206 + structured min-KV operator 完整 m100 | running |
| v224_structured_speed_caps_m20 | 保守 speed-cap：summary 64、code 32 | running |
| v225_structured_speed_caps_aggr_m20 | 激进 speed-cap：summary 32、code 16 | running |
| monitor_speed_caps_20260711 | 等 v224/v225 m20 出结果，达标自动启动 m100 | running |

自动启动 m100 的 gate：

- m20 score >= 0.355；
- m20 online <= 1.20s；
- m20 KV keep <= 30%。

如果 gate 通过，会自动启动：

- `v226_structured_speed_caps_full_m100`
- `v227_structured_speed_caps_aggr_full_m100`

## 当前主判断

最有希望的路线不是继续压小 block，而是：

1. QA / multi-hop / summarization 使用 RiskKV-Block task policy；
2. structured synthetic / label tasks 使用 direct operator；
3. 慢 decode 任务使用输出预算 router 做 speed-quality tradeoff。

这条路线更适合写 ICLR 方法故事，因为它不是单一规则，而是一个 risk-aware action planner：根据任务风险选择 retrieval、structured operator、short decode 或 fallback。
