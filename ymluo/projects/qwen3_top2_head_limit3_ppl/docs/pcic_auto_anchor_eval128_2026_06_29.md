# Auto-Anchor Horizon Rescue 实验记录（2026-06-29）

## 目的

上一轮 delayed-rescue 使用手写 `anchor06`，可以证明 long-horizon anchor rescue 有效，但还不能作为 paper 方法。

本轮把 anchor 自动化为 validation-prior：

```text
在独立 validation setting 上跑 fixed-combo；
按 avg_delta_ppl 排序；
取 top-k combo 作为 horizon rescue anchor；
在目标 eval128 上做 64→128 top2 + anchor cascade。
```

这一步把方法从“手写补丁”推进到可写进论文的方法组件：

```text
Pairwise-CIC + online blockwise selection + validation-prior horizon-anchor rescue gate
```

## Anchor 选择

自动选择脚本：`scripts/select_pcic_validation_anchor.py`

运行逻辑：

```bash
python scripts/select_pcic_validation_anchor.py \
  --combos '0,6;0,7;0,13;7,6;2,0;2,7;2,0,7,12;7,13' \
  --fixed_pattern 'server_pcic_hardtopic_static_b4_eval64_{combo_tag}_eager' \
  --topk 1 \
  --score avg_delta_ppl
```

自动输出：

```text
0,6
```

validation prior 排序表：`docs/pcic_auto_anchor_validation_prior_2026_06_29.md`

## 结果表

| run | avg_delta_ppl | method/base | gate_s | extended | early | anchors | combos |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| top2 32→64 | 0.004371 | 2.648 | 27.085 | 1 | 3 | `` | `0,7;2,0,7,12;0,6;0,13` |
| manual anchor06 | -0.049633 | 4.158 | 53.710 | 4 | 0 | `0,6` | `0,6;2,0,7,12;0,6;0,6` |
| auto anchor | -0.049633 | 4.164 | 51.997 | 4 | 0 | `0,6` | `0,6;2,0,7,12;0,6;0,6` |

## 结论

1. auto-anchor 自动选出了 `0,6`，不是手写指定。
2. auto-anchor 达到 blockwise oracle：ΔPPL = `-0.049633`。
3. auto-anchor 修复了原 `top2 32→64` 的 eval128 失败例。
4. auto-anchor gate `51.997s`，略低于 manual anchor `53.710s`，且远低于 s128 all-candidate `123.331s`。
5. 这使 rescue gate 更接近可投稿方法，而不是 ad-hoc 失败修补。

## Paper 表述建议

建议把最终方法写成三层：

1. **Pairwise-CIC**：用 block-local counterfactual loss 比较候选 attention policy；
2. **Online blockwise selection**：每个 block 动态选择候选 policy；
3. **Validation-prior horizon-anchor rescue gate**：当短 horizon 不稳定时，把 validation 上长期稳健的 policy 作为 anchor，一起扩展到长 horizon。

核心卖点：

- 不提出另一个固定 sparse attention rule；
- 提出在线 policy selection 框架；
- 用 horizon-anchor rescue 解决短 prefix selection 的 delayed-win failure；
- anchor 来自 validation prior，而不是 test-time oracle 或手工指定。

## 仍需补强

- 当前 auto-anchor 只在 Hard-topic eval128 上验证；
- 需要把同样流程跑到 War/Monte，确认不会破坏已有结果；
- 需要更长 blocks 验证 `0,6` 不是小样本偶然；
- 需要标准 benchmark 和真实速度优化。
