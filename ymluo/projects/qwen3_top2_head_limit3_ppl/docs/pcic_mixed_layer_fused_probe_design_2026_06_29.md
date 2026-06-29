# Mixed-Budget Layer Fused Probe 设计草案（2026-06-29）

## 背景

当前主线是：

```text
Pairwise-CIC
+ online blockwise selection
+ conditional validation-prior horizon-anchor rescue gate
```

已有 batch-row gate 证明：

1. `batch_maps` 可以表达同一 forward 中不同 candidate rows 的不同 layer budget；
2. optimized dispatch 能降低一部分 overhead；
3. 但当前 eager batch-row 仍慢于 serial。

`docs/pcic_batch_row_budget_groups_2026_06_29.md` 进一步说明：`81.7%–85.7%` 的层是 all-same budget，真正需要优化的是少数 mixed-budget layers。

## 目标

下一步不应重写全模型 attention，而应只针对 mixed-budget layers 做 fused/tensorized probe。

目标接口：

```text
mixed_layer_fused_probe(
    q: [C, H, 1, D],
    k: [C, H, T, D],
    v: [C, H, T, D],
    row_budget_type: [C],          # full / landmark / recent
    row_recent: [C],
    row_stride: [C],
    attention_mask,
) -> out: [C, 1, H, D]
```

其中 `C` 是 candidate rows。目标是在一个函数内处理 full rows 与 landmark rows，避免当前路径中的多次 `index_select` / `index_copy_` 和 Python 分组开销。

## 最小实现路径

### Stage 1：tensorized mixed full/landmark branch

适用当前实验，因为 mixed layers 基本是 `full + landmark`：

1. 计算所有 rows 的 full score：
   ```text
   score_full = q @ k^T
   ```
2. 为 landmark rows 构造 dense boolean keep mask：
   ```text
   keep = sink ∪ landmarks ∪ recent ∪ self
   ```
3. 对 landmark rows mask 掉非 keep token；full rows 保持全量。
4. 一次 softmax + matmul 得到所有 rows 输出。

优点：

- 代码实现最快；
- 消除 Python row grouping；
- 语义容易和当前 eager 路径对齐。

风险：

- landmark rows 仍计算 full score，理论 FLOPs 更高；
- 只能证明 dispatch/tensorization 上限，不是最终稀疏 kernel。

### Stage 2：candidate-level sparse gather kernel

对 landmark rows 只 gather keep indices：

```text
indices = sink_indices ∪ landmark_indices ∪ recent_indices ∪ self
score = q · gather(k, indices)
out = softmax(score) · gather(v, indices)
```

full rows 走 full attention，landmark rows 走 sparse gather。两类 rows 在同一 kernel 或同一 compiled function 内完成，输出按 row 写回。

优点：

- 避免 landmark rows 的 full score；
- 更接近 paper speed claim 所需实现；
- 和 Horizon-PCIC 的 candidate probe 结构匹配。

风险：

- 需要 Triton/CUDA 或 `torch.compile` 路径；
- 需要处理 ragged indices / padding；
- 需要验证和现有 eager landmark 语义一致。

## Paper 表述建议

如果 Stage 1 完成：

```text
We implement a tensorized mixed-budget probe that removes row-wise Python dispatch while preserving the semantics of the Horizon-PCIC candidate gate.
```

如果 Stage 2 完成并加速：

```text
We further implement a fused candidate probe for the few mixed-budget layers identified by our batch-row analysis, avoiding full KV duplication across candidate policies.
```

## 下一步实验

1. 在 Hard / War / Monte 上实现 Stage 1，比较：
   - serial gate；
   - optimized batch-row gate；
   - tensorized mixed-layer gate。
2. 检查 combos 是否完全一致。
3. 如果 Stage 1 仍慢，直接进入 Stage 2，不再继续做 Python dispatch micro-optimization。
4. Stage 2 成功后再扩展到 b8 和正式 LongBench / RULER 子集。

## Stage 1 实测补充

实现入口：

```text
LAYER_BUDGET_MIXED_DENSE=1
```

汇总文档：`docs/pcic_condautoanchor_batched_gate_mixeddense_2026_06_29.md`

结果：

| dataset | optdispatch gate_s | mixeddense gate_s | change | serial gate_s | same combos |
| --- | ---: | ---: | ---: | ---: | --- |
| hard | 61.139 | 61.700 | +0.561 | 53.648 | True |
| war | 7.462 | 7.507 | +0.045 | 6.659 | True |
| monte | 7.467 | 7.217 | -0.250 | 6.641 | True |

判断：

- Stage 1 保持选择语义；
- 但速度收益不稳定，只在 Monte 上小幅改善；
- dense full-score 的额外 FLOPs 抵消了省掉的 row-wise dispatch；
- 因此默认不启用 `LAYER_BUDGET_MIXED_DENSE`；
- 下一步应进入 Stage 2：sparse gather / fused kernel，而不是继续 dense mixed path。
