# Section 107: Learned Risk Head 初探与主线判断（2026-07-08）

## 目的

Section 106 已经说明，free small-block router 在 m10 上过激，而 `router_blocksize_floor_v2` 很稳。
这一步尝试把手工 floor 往 learned risk head 推进：

```text
case features -> raw minimal-safe action head
case features + candidate action features -> danger head
if raw action is predicted dangerous:
  search the action lattice for the cheapest action predicted safe
```

新增代码：

```text
ymluo/projects/learned_hierarchical_summary_memory/src/run_blocksize_risk_router_from_sweeps.py
ymluo/projects/learned_hierarchical_summary_memory/src/memory_policy_router_runtime.py
ymluo/projects/learned_hierarchical_summary_memory/scripts/train_blocksize_risk_router_from_sweeps_20260708.sh
```

训练输出：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/smallblock_risk_router_from_sweeps_m3_20260708_thr0.90/router.pt
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/smallblock_risk_router_from_sweeps_m3_20260708_thr0.90/risk_router.pt
```

`router.pt` 是普通 action-head 兼容格式。
`risk_router.pt` 是新增格式，runtime loader 会自动识别：

```text
router_kind = blocksize_risk_router
```

## m3 内部验证

用 m3 small-block sweep 生成 model-aware labels。阈值扫描后，`risk_threshold=0.90` 最稳。

| split | policy | safe rate | label acc | token |
|---|---|---:|---:|---:|
| test | raw action head | 1.0000 | 0.5263 | 10.70% |
| test | risk-gated | 1.0000 | 0.4737 | 11.18% |
| train | raw action head | 0.9870 | 0.9221 | 11.66% |
| train | risk-gated | 0.9870 | 0.9091 | 11.66% |

danger head 在 m3 test 上：

| split | accuracy | precision | recall | FNR |
|---|---:|---:|---:|---:|
| test overall | 0.8902 | 0.4464 | 0.5952 | 0.4048 |
| train overall | 0.9746 | 0.9945 | 0.8941 | 0.1059 |

解释：

```text
1. action head 能学到 m3 oracle 的大方向，但 exact label accuracy 不高。
2. danger head 在 train 上很强，在 test 上 recall 明显下降。
3. m3 test 很小，不能证明 learned gate 已经可靠。
```

## m10 真实验证

用 `risk_router.pt` 作为 `--router_path`，在真实 Qwen3-8B + LoRA m10 上跑：

```text
methods = full_raw, router_blocksize
router_path = smallblock_risk_router_from_sweeps_m3_20260708_thr0.90/risk_router.pt
```

结果：

| setting | full score | learned risk score | token | speed |
|---|---:|---:|---:|---:|
| LongBench m10 | 0.3463 | 0.3094 | 11.49% | 1.574x |
| RULER 4k m10 | 1.0000 | 0.8500 | 17.00% | 1.278x |
| RULER 8k m10 | 1.0000 | 0.8250 | 8.87% | 1.664x |
| RULER 16k m10 | 0.8750 | 0.7875 | 4.76% | 2.171x |

对比上一节：

| setting | free small-block | learned risk | floor v2 |
|---|---:|---:|---:|
| LongBench m10 | 0.2966 | 0.3094 | 0.3590 |
| RULER 4k m10 | 0.8750 | 0.8500 | 1.0000 |
| RULER 8k m10 | 0.7875 | 0.8250 | 1.0000 |
| RULER 16k m10 | 0.7750 | 0.7875 | 1.0000 |

token 对比：

| setting | free small-block | learned risk | floor v2 |
|---|---:|---:|---:|
| LongBench m10 | 8.38% | 11.49% | 15.49% |
| RULER 4k m10 | 17.72% | 17.00% | 30.65% |
| RULER 8k m10 | 8.61% | 8.87% | 15.31% |
| RULER 16k m10 | 4.75% | 4.76% | 10.99% |

结论：

```text
learned risk head v0 略微修复了 LongBench、RULER8k、RULER16k，
但没有接近 floor_v2 的可靠性，也不能作为当前主方法。
```

## 对创新故事的影响

这个结果不是坏消息。它说明现在的核心创新不应该写成：

```text
we train a router and it learns everything
```

而应该写成：

```text
RiskKV-Block defines a risk-constrained memory-action lattice.
The current calibrated floor is the reliable constraint.
The learned risk head is an optional approximation that needs stronger labels.
```

对 ICLR/ICML 叙事更有利的主线：

1. **Action lattice 是方法核心。**
   动作不是 flat policy，而是有偏序结构：更大 block / 更大 topK 通常更安全、更贵。

2. **Risk floor 是可靠性机制。**
   free small-block 提供低成本 candidate，但 floor 把失败区域切掉。

3. **Block size 是风险变量。**
   32/64/128 block 不是单纯压缩超参，而是风险-成本边界上的可控动作。

4. **Learned risk head 目前只能作为探索/消融。**
   从 m3 benchmark sweep 蒸馏的 danger labels 太少，LongBench danger recall 不足。

5. **下一步必须做 non-benchmark worst-case risk labels。**
   不能只从 m3 sweep 学 risk，需要用更丰富的非 benchmark synthetic/book QA/multi-evidence adversarial cases。

## 当前主方法判断

当前最适合写进 paper 主表的还是：

```text
router_blocksize_floor_v2
```

完整 m10：

| setting | full score | floor v2 score | token | speed |
|---|---:|---:|---:|---:|
| LongBench m10 | 0.3463 | 0.3590 | 15.49% | 1.501x |
| RULER 4k m10 | 1.0000 | 1.0000 | 30.65% | 1.224x |
| RULER 8k m10 | 1.0000 | 1.0000 | 15.31% | 1.589x |
| RULER 16k m10 | 0.8750 | 1.0000 | 10.99% | 1.991x |

这比 learned risk head v0 更适合支撑投稿。

## 下一步

1. 把 paper 主方法明确写成 calibrated risk-constrained action lattice，而不是 learned router。
2. 把 learned risk head v0 放到 appendix / future ablation：说明直接学习 danger 不够。
3. 构造非 benchmark worst-case labels：
   - single evidence
   - multi-evidence
   - repeated key / multi-value
   - distractor/conflict evidence
   - summary/global information
   - long-context book QA
4. 用这些 labels 重新训练 danger head，目标不是 exact action match，而是：

```text
minimize unsafe decisions under a token budget
```

5. 只有当 learned risk head 在 m10/m50 上接近 floor_v2，才把它升为主结果。
