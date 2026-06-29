# Needle-Style Benchmark Smoke（2026-06-29）

## 目的

服务器上没有现成 LongBench / RULER / Needle 数据。为避免占用服务器网络，本轮做一个本地生成的 needle-style smoke：

1. 生成 validation 文本和 eval 文本；
2. 文本开头放入隐藏 code / city / animal；
3. 中间加入长 distractor；
4. eval 区域包含 retrieval question / answer；
5. 用 validation fixed-combo 结果自动选择 anchor；
6. 在 eval 文本上比较 top2 与 conditional auto-anchor。

注意：这不是正式 LongBench / RULER，只是无下载的标准方向 smoke。

## 实验入口

脚本：`scripts/run_pcic_needle_smoke_condautoanchor.sh`

生成文本：

- `data/pcic_needle_style_validation_2026_06_29.txt`
- `data/pcic_needle_style_eval_2026_06_29.txt`

validation prior 排序：`docs/pcic_needle_smoke_validation_prior_2026_06_29.md`

逐块结果：`docs/pcic_needle_smoke_blocks_2026_06_29.csv`

## Anchor 选择

validation prior 自动选出：

```text
2,0
```

排序前几项：

| combo | avg_delta_ppl | avg_delta_loss |
| --- | ---: | ---: |
| `2,0` | -0.000112 | -0.000111 |
| `0,13` | 0.000016 | 0.000016 |
| `2,0,7,12` | 0.000305 | 0.000303 |

## 结果表

| run | blocks | avg_delta_ppl | method/base | gate_s | extended | early | anchors | combos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| needle top2 | 4 | 0.000118 | 1.831 | 13.206 | 4 | 0 | `` | `2,0,7,12/2,0/2,0/2,0` |
| needle cond auto-anchor | 4 | -0.000166 | 3.401 | 39.909 | 4 | 0 | `2,0` | `2,0/2,0/2,0,7,12/2,0` |

## 结论

1. Needle-style smoke 上，top2 已经非常稳定：平均 ΔPPL = `0.000118`。
2. conditional auto-anchor 质量略好：平均 ΔPPL = `-0.000166`。
3. 但 conditional auto-anchor 代价明显更高：gate 从 `13.206s` 增到 `39.909s`。
4. 这说明在简单 needle-style 场景中，强 rescue 不是必要的；需要更严格的 conditional trigger 或 adaptive margin。

## 对 paper 的意义

该结果有两面：

- 正面：validation-prior auto-anchor 流程可以迁移到 needle-style smoke，并且不破坏 PPL；
- 负面：当前 conditional trigger 在非常稳定的 needle smoke 上仍然过度 extension，说明系统侧还需要更强 early-exit / adaptive margin。

因此 paper 里不能只说 rescue gate 总是更好；更准确的表述是：

```text
rescue gate is necessary for delayed-win / hard-topic regimes,
but should be adaptively skipped in easy retrieval regimes.
```

## 下一步

1. 加入 adaptive margin：对 early loss gap 极小但 top2 已稳定的场景减少 extension。
2. 找到或本地准备正式 LongBench / RULER 子集，再做真正标准 benchmark。
3. 如果继续无下载，可以增强 needle smoke：多 key、多 query、answer 不重复、干扰 code 更强。
