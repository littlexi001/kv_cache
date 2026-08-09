# Section 156: BM25 bridge locator smoke

日期: 2026-07-11

## 背景

v309/v310 的 micro-span repack 以及 v312/v313 的 window-vote span repack 都在测试同一个假设: `block_size=16` 可以作为细粒度 locator, 但最终应保留连续 span。

目前已完成的 v309/v310 局部结果显示, narrativeqa/musique/hotpotqa 等 QA 任务仍然明显掉分。v312/v313 已经在跑, 但从早期日志看 musique 也没有立刻恢复。这说明问题很可能不只是“保留太碎”, 还包括 evidence scorer 对 LongBench QA 的定位不稳。

## 新发现

现有实现中 BM25 scorer 和 bridge scorer 是分离的:

- `hybrid_late_mmr_bm25_flow` 能使用 BM25 lexical component, 但不走 entity bridge。
- `hybrid_late_mmr_bridge_flow` 能走 bridge, 但 lexical component 仍是普通 overlap。

因此新增组合 scorer:

```text
hybrid_late_mmr_bm25_bridge_flow
hybrid_late_mmr_multiscale_bm25_bridge_flow
```

它们同时具备:

- late-interaction semantic score;
- BM25 / IDF-style lexical localization;
- entity bridge expansion;
- flow neighbor smoothing;
- MMR reranking。

## Smoke 实验

| Version | 设计 | 目的 |
|---|---|---|
| v314 | B=16 + window-vote span repack + BM25 bridge | 判断 B=16 是否能被更强 locator 救回来 |
| v315 | B=128 + BM25 bridge | 判断 coarser block + better scorer 是否比 B=16 更稳 |

样本数:

```text
M20 per task
```

任务:

```text
narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique
```

## 对比方式

新增脚本 `compare_smoke_to_baselines_20260711.py`, 按 `(task, sample_id)` 对齐 full KV 与 v300 的同一样本, 输出:

- `summary_table.csv`: overall smoke 结果, 供 dashboard 读取。
- `detail_table.csv`: per-task 细节, 用于判断哪个任务有正信号。

## 判据

如果 v314 明显强于 v312/v313 或 v309/v310, 说明 B=16 的主要瓶颈是 scorer/locator, 可以继续发展为“micro-block locator + BM25/bridge evidence composer”。

如果 v315 明显强于 v314, 说明 B=16 的细粒度虽然理论上可行, 但在当前 scorer 下噪声过大; 论文主线应回到 B=128/B256 的 robust action router, 把 B=16 作为 ablation 或特定任务加速分支。

如果两者都弱, 下一步应转向 learned evidence scorer/router, 用 targeted benchmark + v300/full/oracle 标签蒸馏一个真正的 risk-aware evidence selector。
