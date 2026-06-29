# Hard-topic eval128 Delayed-Rescue 实验记录（2026-06-29）

## 目的

上一组 fixed / online / oracle 结果显示：

- Hard-topic eval128 的 blockwise oracle 是 `0,6;2,0,7,12;0,6;0,6`，平均 ΔPPL = `-0.049633`；
- 现有 `top2 32→64 cascade` 选择 `0,7;2,0,7,12;0,6;0,13`，平均 ΔPPL = `0.004371`；
- online-oracle gap = `0.054004`，主要来自 block0 和 block3；
- 失败模式不是 Pairwise-CIC 完全无信号，而是短 horizon sentinel 被前 32/64 token 误导，错过了 `0,6` 这种后半段才变好的候选。

因此本实验测试一个更强的 rescue gate：

```text
64-token all-candidate early probe
+ top2 early candidates
+ long-horizon anchor combo 0,6
+ extend selected candidates to 128-token horizon
```

该实验的目的不是把 `0,6` 写成最终方法，而是验证 paper 主线中的核心假设：

> 长程退化需要 horizon-aware rescue；只看短 prefix 会错过 delayed-win policy。

## 结果表

| run | avg_delta_ppl | method/base | gate_s | extended | early | avg_ext_cands | combos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| top2 32→64 | 0.004371 | 2.648 | 27.085 | 1 | 3 | 3.00 | `0,7;2,0,7,12;0,6;0,13` |
| s128 all-candidate | -0.049633 | 8.291 | 123.331 | 0 | 0 | 0.00 | `0,6;2,0,7,12;0,6;0,6` |
| 64→128 top2 + anchor06 forced | -0.049633 | 4.159 | 52.291 | 4 | 0 | 4.00 | `0,6;2,0,7,12;0,6;0,6` |
| 64→128 top2 + anchor06 m05 | -0.049633 | 4.158 | 53.710 | 4 | 0 | 4.00 | `0,6;2,0,7,12;0,6;0,6` |

原始 CSV：`docs/pcic_delayed_rescue_eval128_2026_06_29.csv`

## 关键结论

1. `s128 all-candidate` 达到 blockwise oracle，但代价过高：`method/base = 8.291`，`gate_s = 123.331`。
2. `64→128 top2 + anchor06` 同样达到 oracle：平均 ΔPPL 从 `0.004371` 改善到 `-0.049633`。
3. 相对 `s128 all-candidate`，anchor delayed-rescue 的 `method/base` 从 `8.291` 降到 `4.159`，约下降 `49.8%`。
4. 相对 `s128 all-candidate`，anchor delayed-rescue 的 gate 从 `123.331s` 降到 `52.291s`，约下降 `57.6%`。
5. 相对原 `top2 32→64`，anchor delayed-rescue 质量显著更好，但速度代价从 `2.648×` 增到 `4.159×`，还不能作为最终速度方案。

## 对 paper 主线的意义

这组实验直接支持 paper 中的 `rescue gate` 必要性：

- `top2 32→64` 代表短 horizon policy selection，会发生 delayed-win miss；
- `s128 all-candidate` 代表强 oracle-style horizon gate，质量最好但太贵；
- `64→128 top2 + anchor` 说明可以用较少候选扩展恢复长程质量，把全候选长 horizon gate 成本砍半。

因此，下一步 paper-worthy 方法应该命名为：

```text
Pairwise-CIC + online blockwise selection + horizon-anchor rescue gate
```

其中 anchor 不应是手写 `0,6`，而应发展为可泛化的 anchor selection：

1. validation prior anchor：从少量校准文本中学到长期稳健 combo；
2. diversity anchor：覆盖不同 layer family / retrieval-depth；
3. uncertainty anchor：当 early probe 与 calibration/risk memory 矛盾时加入；
4. delayed-win detector：检测 early prefix 排名和历史 long-horizon 排名不一致时扩大扩展集合。

## 当前风险

- `anchor06` 是针对 Hard-topic eval128 的手动 anchor，不能直接作为论文最终方法。
- 当前结果证明 “horizon anchor rescue 有用”，但还没有证明 “anchor 可以自动选择”。
- 速度仍未超过 baseline；现阶段只能声称 gate 成本相对 full-horizon all-candidate 大幅降低，不能声称端到端加速。

## 下一步实验

1. 自动 anchor 选择：用已有 static/oracle 结果选 validation prior anchor，而不是手写 `0,6`。
2. 跨数据验证：在 War/Monte 上加入同样 anchor 机制，确认不会破坏原有 oracle 质量。
3. 更长 blocks：Hard-topic eval128 从 4 blocks 扩到 8/16 blocks，确认 delayed-win 不是偶然。
4. 系统优化：把 64-token all-candidate probe 和 extension probe 改成 batched/fused，降低真实 wall-clock。

## 自动 anchor 后续结果

自动 anchor 已补，见：`docs/pcic_auto_anchor_eval128_2026_06_29.md`

关键结果：

- validation prior 自动选出 `0,6`；
- auto-anchor eval128 ΔPPL = `-0.049633`，达到 blockwise oracle；
- auto-anchor gate = `51.997s`，低于 manual anchor `53.710s`，远低于 s128 all-candidate `123.331s`。
