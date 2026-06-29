# Batch-row Budget Group 分析（2026-06-29）

## 目的

该分析不跑模型，只解析已有 `batch_maps/*.json`，量化 batch-row candidate gate 中哪些层真的需要 row-wise mixed sparse attention。

详细 CSV：`docs/pcic_batch_row_budget_groups_2026_06_29.csv`
汇总 CSV：`docs/pcic_batch_row_budget_groups_summary_2026_06_29.csv`

## 输出级汇总

| output | maps | row_counts | all-same layer frac | avg groups/layer | mixed layers |
| --- | ---: | --- | ---: | ---: | ---: |
| `server_pcic_hardtopic_b4_horizongate_condautoanchor_batched_optdispatch_eval128_seed64_eager` | 8 | `2,4,5,8` | 0.817 | 1.183 | 41 |
| `server_pcic_monte_b2_horizongate_condautoanchor_batched_optdispatch_seed64_eager` | 4 | `2,3,4` | 0.857 | 1.143 | 16 |
| `server_pcic_war_b2_horizongate_condautoanchor_batched_optdispatch_seed64_eager` | 2 | `4` | 0.857 | 1.143 | 8 |

## 解释

- `all-same layer` 表示该层所有 candidate rows 使用同一个 budget，可直接整批 forward。
- `mixed layer` 表示该层不同 rows 使用 full / landmark 等不同 budget，是真正需要 fused/tensorized row-wise sparse attention 的位置。
- 如果 all-same layer 占比高，当前 dispatch 优化是合理的；如果 optimized batched 仍慢，剩余瓶颈主要在 mixed layers 的候选维 cache 复制和 sparse attention 本体。
