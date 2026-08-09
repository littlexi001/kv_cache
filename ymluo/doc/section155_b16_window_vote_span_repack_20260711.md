# Section 155: B16 window-vote span repack

日期: 2026-07-11

## 背景

用户提出一个关键判断: `block_size=16` 本身不一定太碎, 因为模型最终也是 token 级处理; 问题可能在于我们之前把 16-token block 当成最终保留单元, 导致 KV 过于碎片化, 局部语义不连续。

已有 `v309/v310 microspan` 做了第一版验证: 用 16-token block 做 locator, 再把命中的 block 扩成 96/128-token 连续 span。当前部分结果显示 narrativeqa/musique 质量仍然偏弱, 说明只按单个 micro-block 分数选 span 还不够稳。

## 新假设

不要把单个 16-token block 的分数当成唯一排序依据, 而是让一个连续窗口内的多个 16-token block 共同投票:

```text
16-token micro blocks -> window-level vote score -> dense span repack -> regular MMR fill
```

这样测试的是:

- B=16 提供细粒度定位。
- 多个相邻/同窗 micro-block 共同确认 evidence span。
- 最终 KV 仍是连续 96/128-token span, 避免碎片化。

## 代码改动

在 `src/run_controlled_public_kv_benchmark_v1.py` 中给 `span_repack` 增加:

```text
ours_span_repack_score_mode
```

可选值:

| mode | 含义 |
|---|---|
| `center` | 旧行为, 只按中心 16-token block 分数排序 |
| `window_sum` | 用整个 span 内 micro-block 分数和排序 |
| `window_topk` | 用 span 内 top micro-block 投票排序, 降低噪声 block 影响 |

默认值是 `center`, 因此不改变 v300/v309/v310 旧实验的行为。

## 新实验

| Version | 设计 | 目的 |
|---|---|---|
| v312 | B=16 + `window_topk` + 128-token span + 高召回预算 | 优先验证质量能否恢复 |
| v313 | B=16 + `window_topk` + 96-token span + 较低预算 | 验证速度版本是否接近 v300 |

覆盖任务:

```text
narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique
```

非 QA 任务沿用 v300, 完成后合并成完整 M100 表。

## 输出位置

启动脚本:

```bash
scripts/launch_b16_windowvote_sweep_20260711.sh
```

自动合并脚本:

```bash
scripts/watch_combine_b16_windowvote_sweep_20260711.sh
```

最终汇总表:

```text
outputs/riskkv_v19_v312_v313_b16_windowvote_sweep_20260711/summary_table.csv
```

## 判据

如果 v312 明显优于 v309/v310, 说明 B=16 的问题不是粒度本身, 而是需要 window-level evidence composition。这个方向可以成为论文中比普通 block retrieval 更强的创新点。

如果 v312 仍然不如 v300, 则说明当前 query-block lexical scorer 对 LongBench QA 的 evidence 定位能力不足, 下一步应转向 learned evidence scorer/router, 而不是继续调 block size。
