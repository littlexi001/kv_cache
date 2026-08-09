# Section 151: micro-block locator + span repack

日期：2026-07-11

## 动机

用户指出 `block_size=16` 不一定太碎，因为 token 本身就是逐个处理；问题可能不是 16-token block 的粒度，而是最终 KV 保留是否过于碎片化。

当前 `anchor_window` 只是在命中一个 fine block 后扩一个固定窗口。它没有把多个 fine block 组织成连续 evidence span，也没有把 span 内的 fine pages 明确标记为已选 evidence，后续 MMR 仍可能围绕碎块继续选。

因此新增一个更明确的 memory action：

```text
16-token micro-block scoring -> dense evidence span repack -> regular MMR fill
```

这个模块把 16-token block 作为高分辨率 locator，但最终保留的是 96/128-token 连续 span。这样可以同时保留细粒度定位能力和局部语义连续性。

## 代码改动

新增参数：

| 参数 | 含义 |
|---|---|
| `span_repack` | task policy 中打开 micro-span repack |
| `span_repack_window_tokens` | 每个微块中心扩成的连续 span 长度 |
| `span_repack_budget_fraction` | sparse budget 中预留给 span repack 的比例 |
| `span_repack_top_pages` | 参与 span repack 的 top fine pages 数量 |
| `span_repack_min_score` | 可选最低 fine-page score 阈值 |

默认全部关闭，因此不改变 v300/v307/v308 的已有行为。

## 新实验

| Version | 设计 | 目标 |
|---|---|---|
| v309 | quality-oriented，16-token locator + 128-token span repack | 看连续 span 是否能修复 b16 多块选择的质量问题 |
| v310 | speed-oriented，16-token locator + 96-token span repack | 在更低 KV 下测试是否仍能接近 v300 |

覆盖任务：

```text
narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique
```

non-QA 任务继续沿用 v300，完成后合并成完整 M100 表。

## 输出位置

启动脚本：

```bash
scripts/launch_b16_microspan_sweep_20260711.sh
```

自动汇总脚本：

```bash
scripts/watch_combine_b16_microspan_sweep_20260711.sh
```

最终汇总表：

```text
outputs/riskkv_v19_v309_v310_b16_microspan_sweep_20260711/summary_table.csv
```

## 判据

- 如果 v309 明显高于 v307/v308，说明 b16 的主要问题是最终 KV 过碎，micro-block locator + span repack 可以成为论文主创新之一。
- 如果 v310 接近 v300 且 KV 更低，说明可以把它纳入 action router，作为 speed-oriented safe action。
- 如果 v309/v310 都不如 v300，说明当前 fine-block scorer 的问题更根本，下一步应训练 query-evidence scorer 或改成 learned evidence composer。
