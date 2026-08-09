# Section 164: B16 compressed-mid smoke

## 动机

用户提出：`block_size=16` 不一定“太碎”，因为 token 本来就是逐个生成的；如果 16-token 小块可以多选一些，也许能比 128/256-token 大块更精细地定位证据。

这轮实验不改主算法，只验证一个问题：在相近压缩预算下，B16 细粒度定位是否能带来更好的质量/速度折中。

## 实验设计

新增两个 LongBench QA smoke 配置，均跑 `narrativeqa, qasper, multifieldqa_en, hotpotqa, 2wikimqa, musique`，每个任务 M20：

- `v327_b16_compressed_mid`：`page_tokens=16`，提高可选块数量，但保持中等预算。
- `v328_b16_window64_compressed_mid`：在 v327 基础上启用 `span_repack`，用 16-token 小块投票，再扩展成 64/96-token 连续窗口，验证碎片化修复是否有帮助。

核心预算：

| Task | v327/v328 budget | Block size | 备注 |
| --- | ---: | ---: | --- |
| narrativeqa | 1024 | 16 | 约 64 个小块 |
| qasper | 1536 | 16 | bridge scorer |
| multifieldqa_en | 1024 | 16 | 约 64 个小块 |
| hotpotqa | 2048 | 16 | 约 128 个小块 |
| 2wikimqa | 1536 | 16 | 保留 verifier/retry |
| musique | 2048 | 16 | bridge scorer |

## 首批结果

截至 2026-07-11 14:47，部分 M20 任务已经完成。结论偏负面：B16 中等预算不能作为全局主方法。

| Variant | Task | Score | KV keep | 现象 |
| --- | --- | ---: | ---: | --- |
| v327 | hotpotqa | 0.2458 | 31.27% | 明显低于 v300 smoke 的 0.3967 |
| v327 | multifieldqa_en | 0.4266 | 21.58% | KV 降低，但质量明显低于 v300 |
| v327 | qasper | 0.4457 | 35.68% | 明显低于 v300/qasper |
| v328 | narrativeqa | 0.0933 | 14.26% | 直接崩，说明 B16+窗口也不稳 |
| v328 | qasper | 0.4879 | 35.68% | 比 v327 好，但仍低于 v300 |
| v328 | 2wikimqa | 0.2692 | 39.12% | 低于 v300 smoke |
| v328 | musique | 0.2333 | 27.63% | 低于 v300/full smoke |

初步解释：

- B16 的问题不是简单“选块数量太少”。即使允许选 64 到 128 个小块，LongBench QA 的跨句、跨段证据仍然容易断裂。
- `span_repack` 能修复一部分碎片化，例如 qasper 从 0.4457 提到 0.4879，但还不足以替代 v300。
- B16 可以作为局部候选或诊断工具，但不适合作为现在的主线。

## 新正向信号

离线候选动作挖掘发现，`2wikimqa` 上 `B128 + BM25 bridge` 是更有希望的方向：

- M20：`v315_b128_bm25bridge_2wikimqa` 得分 0.3631，v300 同样本得分 0.3187。
- KV keep：34.90%，与 v300 同样本 34.45% 接近。
- Online speed：相对 v300 同样本约 1.22x。

因此已经启动 M100 验证：

- `scripts/launch_v329_2wiki_b128_bm25bridge_m100_20260711.sh`
- `scripts/watch_combine_v329_2wiki_b128_bm25bridge_m100_20260711.sh`

另一个非压缩但可能提升端到端速度的信号：

- `multifieldqa_en` 的 aggressive short-decode 在 M20 上分数不降，online 约 1.12x。
- 已启动 M100 验证：`v330_multifield_shortdecode`。

合成 watcher：

- `scripts/watch_combine_v331_v329_v330_m100_20260711.sh`
- 等 v329/v330 都完成后，自动合成 `v331` 全表。

## 当前判断

B16 细块路线目前的结论是“可做辅助，不做主方法”。下一步主线应转向：

1. 对 multi-hop QA 使用更强的 lexical/BM25 bridge，而不是更小 block。
2. 对高 KV 但生成较长的任务，优先测试输出长度控制和 verifier，而不是盲目降低 KV。
3. 把 B16 只作为 router 的候选动作，在 qasper 等局部任务上按样本特征选择，而不是全任务替换。
