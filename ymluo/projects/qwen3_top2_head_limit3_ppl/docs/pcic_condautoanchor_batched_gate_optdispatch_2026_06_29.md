# Conditional Auto-Anchor Batched Gate Dispatch 优化（2026-06-29）

## 目的

本轮只优化 batch-row dispatch 的工程开销，不改变 Pairwise-CIC / online blockwise selection / conditional rescue gate 的算法语义。

优化点：当某一层所有 batch rows 使用同一个 layer budget 时，直接走整批 forward，跳过 `index_select` / `index_copy_`；同时缓存 batch-row index tensor，减少每 token / layer 的小张量构造。

原始 CSV：`docs/pcic_condautoanchor_batched_gate_optdispatch_2026_06_29.csv`

## 结果表

| run | dataset | mode | avg_delta_ppl | method/base | gate_s | extended | early | batched | combos |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hard_serial | hard | serial | -0.049633 | 4.144 | 53.648 | 4 | 0 | 0 | `0,6/2,0,7,12/0,6/0,6` |
| hard_batched_before | hard | batched_before | -0.049778 | 6.254 | 79.524 | 4 | 0 | 1 | `0,6/2,0,7,12/0,6/0,6` |
| hard_batched_opt | hard | batched_opt | -0.049778 | 4.879 | 61.139 | 4 | 0 | 1 | `0,6/2,0,7,12/0,6/0,6` |
| war_serial | war | serial | -2.135311 | 2.603 | 6.659 | 0 | 2 | 0 | `0,7/0,7` |
| war_batched_before | war | batched_before | -2.133787 | 3.304 | 8.689 | 0 | 2 | 1 | `0,7/0,7` |
| war_batched_opt | war | batched_opt | -2.133787 | 2.892 | 7.462 | 0 | 2 | 1 | `0,7/0,7` |
| monte_serial | monte | serial | -0.219215 | 2.600 | 6.641 | 2 | 0 | 0 | `2,7/2,0` |
| monte_batched_before | monte | batched_before | -0.251805 | 3.376 | 8.466 | 2 | 0 | 1 | `2,7/2,0` |
| monte_batched_opt | monte | batched_opt | -0.251805 | 2.953 | 7.467 | 2 | 0 | 1 | `2,7/2,0` |

## Before vs Optimized

| dataset | old batched gate_s | opt batched gate_s | gate_s change | serial gate_s | opt same combos as serial |
| --- | ---: | ---: | ---: | ---: | --- |
| hard | 79.524 | 61.139 | -18.385 | 53.648 | True |
| war | 8.689 | 7.462 | -1.227 | 6.659 | True |
| monte | 8.466 | 7.467 | -0.999 | 6.641 | True |

## 结论

- 该优化是安全的 dispatch-level 增量：目标是减少无意义的整批分组搬运，不改变候选选择规则。
- 如果 optimized 仍慢于 serial，说明瓶颈不只是 Python 小开销，而是候选维复制 KV/cache 与 landmark/recent attention 本身需要 fused/tensorized 实现。
- paper 速度主张仍应写成：batch-row semantic prototype + dispatch optimization + fused candidate probe 是工程路线；真实 wall-clock speedup 需要下一阶段 kernel/tensorized probe 证明。
