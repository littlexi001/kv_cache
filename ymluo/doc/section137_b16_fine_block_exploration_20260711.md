# Section 137: b16 fine-block exploration

日期：2026-07-11

## 背景

当前主线方法在 LongBench 上已经通过结构化直答、extractive summary operator、code cap 等机制把整体质量和速度推到较好的 practical 区间。但 QA 类任务仍然主要依赖 query-aware block retrieval。用户提出：`block_size=16` 未必太碎，因为 token 本身就是逐 token 处理；如果 16-token block 能多选一些块，可能比 128-token block 更细粒度地命中证据。

这轮实验的目标不是直接证明 b16 一定更强，而是拆开两个问题：

1. 小块是否提高证据召回和答案质量。
2. 小块带来的 selector 扫描成本和碎片化是否会抵消收益。

## 代码更新

在 `src/run_controlled_public_kv_benchmark_v1.py` 中加入了 direct structured answer 的 prefill short-circuit：

- 对 `passage_retrieval_en`、`passage_count`、`trec`、`gov_report`、`multi_news`，如果 policy 启用 direct structured answer，并且没有 full fallback，则主循环一开始跳过完整 prefill。
- 如果 direct answer 没有产出，`evaluate_method` 会临时补做 prefill，然后回到普通 sparse decode，避免实验中断。
- CSV 增加标记：
  - `ours_direct_prefill_short_circuit_active`
  - `ours_late_prefill_after_direct_miss`

Smoke test：

| Setting | Samples | Score | Mean total | Mean online | Mean prefill |
|---|---:|---:|---:|---:|---:|
| `v234_direct_prefill_skip_smoke` on gov_report/multi_news/passage_count/passage_retrieval_en | 8 | 0.4764 | 0.0069s | 0.0069s | 0.0000s |

结论：prefill short-circuit 生效，没有触发 late prefill。

## 已启动实验

### v232: b16 + coarse-to-fine recall

Policy:

`configs/riskkv_task_policy_v232_b16_ctf_recall_extractive_codecap_20260711.json`

设计：

- QA 类任务使用 `page_tokens=16`。
- 使用 coarse-to-fine 先粗筛候选，再在候选中的 16-token block 上选择。
- 保持当前主线的结构化直答和 code cap。
- 目标：控制扫描开销，同时验证 b16 是否有质量收益。

后台任务：

`riskkv_v19_v232_b16_ctf_recall_m20_20260711_b16_ctf_m20_bDyn_pDyn`

### v233: b16 all-block scorer

Policy:

`configs/riskkv_task_policy_v233_b16_allblocks_extractive_codecap_20260711.json`

设计：

- QA 类任务使用 `page_tokens=16`。
- 不启用 coarse-to-fine，直接在所有 16-token block 上打分选择。
- 目标：测 b16 本身的质量上限，同时观察 selector 开销。

后台任务：

`riskkv_v19_v233_b16_allblocks_m20_20260711_b16_allblocks_m20_bDyn_pDyn`

### v234: b16 + span-merge recall

Policy:

`configs/riskkv_task_policy_v234_b16_spanmerge_recall_extractive_codecap_20260711.json`

设计：

- QA 类任务使用 `page_tokens=16`。
- QA 预算略放宽，例如 narrativeqa/multifieldqa_en 768，qasper 1024，musique 1536。
- 扩大 flow neighbor 半径和补块比例，让高分小块周围形成更完整的局部语义片段。
- 目标：验证“更小块 + 多点召回 + 局部 span 修复”是否比单纯 b16 更稳。

后台任务：

`riskkv_v19_v234_b16_spanmerge_recall_m20_20260711_b16_spanmerge_m20_bDyn_pDyn`

### v236: b16 anchor-window + capped risk

Policy:

`configs/riskkv_task_policy_v236_b16_anchor_window_cappedrisk_20260711.json`

Code change:

- Add `anchor_window` as a new evidence-packing action.
- `page_tokens=16` is used for fine-grained scoring and anchor localization.
- The actual KV kept around each selected fine block is a centered continuous evidence window.
- Current window sizes: 96 tokens for narrativeqa / multifieldqa_en / 2wikimqa / qasper, 128 tokens for hotpotqa / musique.
- QA score-risk budgets are capped instead of escalating to `999999` or full KV.

Motivation:

Early b16 logs showed `kept=3105/7533` on some QA samples. That means smaller blocks alone do not guarantee compression, because risk fallback can still expand the action close to full KV. v236 makes b16 a locator rather than the final memory unit: fine blocks find anchors, medium windows preserve coherent evidence, and capped-risk keeps the action inside the 10%-30% target band.

Background job:

`riskkv_v19_v236_b16_anchor_window_cappedrisk_m20_20260711_b16_anchor_window_m20_bDyn_pDyn`

Early log parse:

| Method | Segment | n | Score | Keep ratio | Online |
|---|---|---:|---:|---:|---:|
| v232 CTF | narrativeqa | 20 | 0.1399 | 7.48% | 0.197s |
| v233 all-block | narrativeqa | 20 | 0.1416 | 7.48% | 0.196s |
| v234 span-merge | narrativeqa | 20 | 0.1197 | 10.87% | 0.219s |
| v236 anchor-window | narrativeqa | 20 | 0.1389 | 10.87% | 0.191s |
| v232 CTF | qasper | 20 | 0.4194 | 40.31% | 0.432s |
| v233 all-block | qasper | 20 | 0.4194 | 40.31% | 0.434s |
| v234 span-merge | qasper | 20 | 0.4691 | 41.64% | 0.570s |
| v236 anchor-window | qasper | 20 | 0.5010 | 33.03% | 0.532s |

Early conclusion:

- On narrativeqa, anchor-window roughly matches b16 CTF quality but uses more KV, so it is not yet justified there.
- On qasper, anchor-window is a positive phenomenon: better quality and lower keep ratio than v232/v233/v234.
- multifieldqa_en needs a full 20-sample segment before judging; the first 10 samples looked lower, probably because capped risk is too strict for that task.

Completed M20 results:

| Method | Score | Keep ratio | Online | Main interpretation |
|---|---:|---:|---:|---|
| v232 b16 CTF | 0.3934 | 22.90% | 0.568s | Strong quality/speed baseline; some QA tasks still over-keep. |
| v233 b16 all-block | 0.3962 | 22.69% | 0.569s | Best global M20 among the first b16 variants. |
| v236 anchor-window capped-risk | 0.3850 | 15.52% | 0.600s | Much lower KV; task-local wins on qasper and 2wikimqa, but worse on multifieldqa_en. |

Important task-level observations:

| Task | v233 score / keep | v236 score / keep | Observation |
|---|---:|---:|---|
| qasper | 0.4195 / 40.31% | 0.5009 / 33.03% | Anchor-window clearly helps. |
| 2wikimqa | 0.3387 / 32.40% | 0.3687 / 32.90% | Anchor-window improves score with similar KV. |
| hotpotqa | 0.2658 / 44.59% | 0.2558 / 31.27% | Small score drop, large KV reduction. |
| musique | 0.2000 / 75.23% | 0.1833 / 25.59% | Large KV reduction, modest score drop. |
| multifieldqa_en | 0.5618 / 57.43% | 0.3967 / 21.37% | Capped-risk/window is too aggressive. |

### v240: task-router b16 all-block + anchor-window

Policy:

`configs/riskkv_task_policy_v240_task_router_b16_window_20260711.json`

Design:

- Default: use v233 all-block behavior, because it is the best global M20 baseline.
- Use v236 anchor-window only where it showed task-local advantage:
  - qasper
  - hotpotqa
  - 2wikimqa
  - musique
- Keep v233 high-quality behavior for multifieldqa_en and repobench-p, because compression-only variants hurt quality too much there.

Estimated from completed M20 component rows:

| Estimate | Score | Keep ratio | Online |
|---|---:|---:|---:|
| v240 component estimate | about 0.402 | about 18%-19% | about 0.56s |

Background jobs:

- M20: `riskkv_v19_v240_task_router_b16_window_m20_20260711_task_router_m20_bDyn_pDyn`
- M100: `riskkv_v19_v240_task_router_b16_window_full_m100_20260711_task_router_m100_bDyn_pDyn`

Multifield targeted ablations:

| Run | Purpose |
|---|---|
| v237 | anchor-window, 128-token window, risk cap 2048 |
| v238 | flow/span repair, risk cap 2048 |
| v239 | anchor-window all-block, risk cap 2048 |

Initial finding: v237/v239 reached only 0.4061 score at 28.40% keep on multifieldqa_en, far below v233's 0.5618. Therefore v240 keeps v233 behavior for multifieldqa_en.

### v235: v231 + prefill short-circuit full M100

Policy:

`configs/riskkv_task_policy_v231_extractive_codecap_noqmsumcap_20260711.json`

设计：

- 复跑当前 practical 主线 v231。
- 使用新代码中的 direct prefill short-circuit，得到更真实的 total/E2E timing。
- 质量理论上应接近旧 v231，主要差异应体现在 direct structured tasks 的总耗时。

后台任务：

`riskkv_v19_v235_v231_prefillskip_full_m100_20260711_prefillskip_m100_bDyn_pDyn`

## 下一步判据

M20 出结果后优先看：

| Method | Score | Token ratio | Online speed | Total/E2E speed | QA score delta | 结论 |
|---|---:|---:|---:|---:|---:|---|
| v232 b16 CTF | TBD | TBD | TBD | TBD | TBD | TBD |
| v233 b16 all-block | TBD | TBD | TBD | TBD | TBD | TBD |
| v234 b16 span-merge | TBD | TBD | TBD | TBD | TBD | TBD |
| v236 b16 anchor-window capped-risk | TBD | TBD | TBD | TBD | TBD | TBD |

判断规则：

- 如果 v234 明显优于 v232/v233，说明 b16 需要局部 span 修复，小块不能独立使用。
- 如果 v233 优于 v232，说明 coarse-to-fine 有候选召回损失，需要改粗筛器。
- 如果 v232/v233/v234 都没有超过当前 v231/v230，说明 LongBench QA 的瓶颈不是 block_size，而是 scorer/risk routing/答案生成本身。
- 如果 b16 方法只提高 QA 但速度下降，需要考虑把 b16 作为 router 的局部动作，而不是全局默认动作。
