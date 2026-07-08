# Section 108: Calibrated Risk Floor 与方法创新升级（2026-07-08）

## 目的

Section 106 的 `router_blocksize_floor_v2` 很稳，但如果只写成手工规则，创新性会显得弱。
本节把它升级为：

```text
calibrated risk floor over a memory-action lattice
```

新增脚本：

```text
ymluo/projects/learned_hierarchical_summary_memory/src/run_blocksize_floor_calibration_from_sweeps.py
ymluo/projects/learned_hierarchical_summary_memory/scripts/calibrate_blocksize_floor_from_sweeps_20260708.sh
```

输出：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/blocksize_floor_calibration_m3_20260708
ymluo/projects/learned_hierarchical_summary_memory/outputs/blocksize_floor_calibration_m3_20260708
```

## 公式化

定义 memory action：

```text
a = (b, k)
```

其中：

```text
b = block size
k = number of selected evidence blocks
```

对应动作：

```text
recent_plus_b{b}_span_top{k}_b0_a0
```

对 calibration case `i`，已有 sweep 得到每个 action 的 score：

```text
s_i(a)
```

定义 case-level 安全阈值：

```text
tau_i =
  full_raw_score_i                 exact / retrieval tasks
  full_raw_score_i - delta_rouge   summary tasks
```

当前使用：

```text
delta_rouge = 0.03
```

动作 `a` 在 case `i` 上的失败指示：

```text
L_i(a) = 1[s_i(a) < tau_i]
```

对 group `g`，经验风险：

```text
R_hat_g(a) = mean_{i in g} L_i(a)
```

平均成本：

```text
C_hat_g(a) = mean token_ratio_i(a)
```

calibrated floor 定义为：

```text
phi(g) = argmin_a C_hat_g(a)
         subject to R_hat_g(a) <= alpha
```

如果没有 action 满足风险约束，则选：

```text
argmin_a (R_hat_g(a), C_hat_g(a))
```

推理时 raw router 给出动作 `a_raw` 后，当前稳定主版本把动作投影到校准出来的 group action：

```text
a_final = phi(g)
```

原因是 block size 不是严格单调安全变量：更大的 block 会改变 evidence scorer 的分块与排序，
因此 `b512 top3` 不一定支配 `b256 top3`。更准确地说，calibration 选择的是完整 action，
而不是只选择一个可以被任意更大 block 替代的下界。

后续如果要恢复动态 router，可以在校准安全集合里路由：

```text
A_safe(g) = {a | R_hat_g(a) <= alpha}
a_final = argmin_{a in A_safe(g)} C_hat_i(a)
```

但当前最稳版本先使用 `phi(g)`。

## 严格校准结果

使用：

```text
alpha = 0
quality_mode = best_or_full
summary_rouge_slack = 0.03
```

自动校准得到：

| group | calibrated floor | calibration success | failure | token |
|---|---|---:|---:|---:|
| longbench | b128 top12 | 0.8333 | 0.1667 | 21.72% |
| ruler_4096 | b256 top3 | 1.0000 | 0.0000 | 30.97% |
| ruler_8192 | b256 top3 | 1.0000 | 0.0000 | 15.47% |
| ruler_16384 | b512 top3 | 1.0000 | 0.0000 | 10.87% |

这里和 Section 106 的 `floor_v2` 完全一致：

```text
LongBench: b128 top12
RULER 4k: b256 top3
RULER 8k: b256 top3
RULER 16k: b512 top3
```

LongBench 的 calibration success 不是 1.0，原因是 LongBench 混合了 summary、multi-hop QA、counting 和 retrieval；
在 m3 sweep action set 里没有一个低成本动作能严格满足所有 LongBench case。
因此校准器按 fallback rule 选择了风险最低且成本较低的 `b128 top12`。

## m10 验证回看

calibrated floor 对应的真实 m10 结果：

| setting | full score | calibrated floor score | token | speed |
|---|---:|---:|---:|---:|
| LongBench m10 | 0.3463 | 0.3590 | 15.49% | 1.501x |
| RULER 4k m10 | 1.0000 | 1.0000 | 30.65% | 1.224x |
| RULER 8k m10 | 1.0000 | 1.0000 | 15.31% | 1.589x |
| RULER 16k m10 | 0.8750 | 1.0000 | 10.99% | 1.991x |

这说明 floor 不是事后拍脑袋：

```text
m3 calibration selects the same floor,
m10 heldout validates the same floor.
```

## 对论文创新性的提升

现在主方法可以写成：

```text
RiskKV-Block is a calibrated risk-constrained memory granularity router.
```

而不是：

```text
we use a heuristic fallback
```

关键创新点可以组织成四层：

1. **Memory-action lattice.**
   把 block size 和 evidence budget 组成 action lattice，而不是固定 KV pruning ratio。

2. **Risk-calibrated floor.**
   对每个 task/length group，在 calibration set 上选择满足经验风险约束的最小动作。

3. **Complete-action calibration.**
   calibration 选择完整 `(block size, topK)` 动作，而不是假设更大 block 一定更安全。

4. **Prompt-level evaluation / KV-native serving 解耦。**
   同一个 action 可以用 prompt selected spans 快速评估，也可以用 KV-native gather/repack 实现 serving。

## 为什么比 learned risk head v0 更适合主线

Section 107 的 learned risk head v0 在 m10 上没有追上 floor：

| setting | learned risk score | calibrated floor score |
|---|---:|---:|
| LongBench m10 | 0.3094 | 0.3590 |
| RULER 4k m10 | 0.8500 | 1.0000 |
| RULER 8k m10 | 0.8250 | 1.0000 |
| RULER 16k m10 | 0.7875 | 1.0000 |

所以当前 paper 主结果应该是 calibrated floor，而不是 learned danger head。
learned risk head 可以放在 appendix，作为“直接学习风险边界仍不够”的分析。

## 给 paper 的建议写法

Method 里建议增加一个小节：

```text
Risk-Calibrated Action Floor
```

Algorithm 可以写：

```text
Input: action lattice A, calibration set D_cal, risk level alpha
For each group g:
  compute empirical risk R_hat_g(a) for each a in A
  choose phi(g) = cheapest a with R_hat_g(a) <= alpha
At inference:
  g = group(x, q)
  a_final = phi(g)
```

主表里 `router_blocksize_floor_v2` 可以改名为：

```text
RiskKV-Block calibrated
```

或者：

```text
RiskKV-Block + calibrated floor
```

## 下一步

1. 把 paper draft 的 method section 改成 calibrated risk floor。
2. 把 `floor_v2` 名字统一为 `calibrated_floor`，避免看起来像临时版本号。
3. 做 m50/full heldout，验证 calibration floor 不是 m10 偶然结果。
4. 做 ablation：
   - free router
   - calibrated floor only
   - router + calibrated floor
   - learned risk head v0
   - fixed block512 top3
5. 如果要进一步增强创新性，再把 alpha 做成可调风险-成本曲线：

```text
alpha in {0, 0.05, 0.10, 0.20}
```

这样可以画出 risk-cost frontier。
