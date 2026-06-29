# Mixed Full/Landmark Dense Batch-row Gate（2026-06-29）

## 目的

验证 Stage 1 mixed-budget tensorized path：在 mixed full/landmark layers 上用整批 dense mask forward 替代 row-wise 分组 dispatch。

原始 CSV：`docs/pcic_condautoanchor_batched_gate_mixeddense_2026_06_29.csv`

## 结果表

| run | dataset | mode | avg_delta_ppl | method/base | gate_s | extended | early | batched | combos |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hard_serial | hard | serial | -0.049633 | 4.144 | 53.648 | 4 | 0 | 0 | `0,6/2,0,7,12/0,6/0,6` |
| hard_optdispatch | hard | optdispatch | -0.049778 | 4.879 | 61.139 | 4 | 0 | 1 | `0,6/2,0,7,12/0,6/0,6` |
| hard_mixeddense | hard | mixeddense | -0.050170 | 4.861 | 61.700 | 4 | 0 | 1 | `0,6/2,0,7,12/0,6/0,6` |
| war_serial | war | serial | -2.135311 | 2.603 | 6.659 | 0 | 2 | 0 | `0,7/0,7` |
| war_optdispatch | war | optdispatch | -2.133787 | 2.892 | 7.462 | 0 | 2 | 1 | `0,7/0,7` |
| war_mixeddense | war | mixeddense | -2.144379 | 2.866 | 7.507 | 0 | 2 | 1 | `0,7/0,7` |
| monte_serial | monte | serial | -0.219215 | 2.600 | 6.641 | 2 | 0 | 0 | `2,7/2,0` |
| monte_optdispatch | monte | optdispatch | -0.251805 | 2.953 | 7.467 | 2 | 0 | 1 | `2,7/2,0` |
| monte_mixeddense | monte | mixeddense | -0.227903 | 2.904 | 7.217 | 2 | 0 | 1 | `2,7/2,0` |

## Mixed Dense vs Optimized Dispatch

| dataset | optdispatch gate_s | mixeddense gate_s | gate_s change | serial gate_s | mixeddense same combos as serial |
| --- | ---: | ---: | ---: | ---: | --- |
| hard | 61.139 | 61.700 | 0.561 | 53.648 | True |
| war | 7.462 | 7.507 | 0.045 | 6.659 | True |
| monte | 7.467 | 7.217 | -0.250 | 6.641 | True |

## 解释

- 如果 combos 保持一致，说明 dense mixed full/landmark path 没有改变 selector 语义。
- 如果 gate 下降，说明 row-wise group dispatch 是主要瓶颈之一。
- 如果 gate 上升，说明 dense full score 代价大于省掉的 dispatch，下一步应直接做 sparse gather/fused kernel。

## 当前默认

`LAYER_BUDGET_MIXED_DENSE` 默认保持关闭。原因是该路径只在 Monte 上小幅降低 gate，Hard / War 反而更慢；它证明了语义可行，但不能作为主方法速度实现。
