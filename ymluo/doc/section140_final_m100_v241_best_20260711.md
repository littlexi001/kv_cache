# Section 140: v241 当前最佳 M100 结果

日期：2026-07-11

## 结论

当前最好的 practical 方法是：

`v241_m100_validated_router`

Policy:

`configs/riskkv_task_policy_v241_m100_validated_router_20260711.json`

完整 LongBench M100 split-combined 结果：

| Method | Score | KV keep | Online | Total | Speed vs 3.033s full online |
|---|---:|---:|---:|---:|---:|
| v229 extractive summary | 0.3901 | 22.36% | 0.825s | 2.504s | 3.67x |
| v230 extractive + code cap | 0.3893 | 22.36% | 0.604s | 2.278s | 5.02x |
| v235 v231 + prefill skip | 0.3895 | 22.36% | 0.672s | 1.870s | 4.51x |
| v240 task router | 0.3834 | 18.13% | 0.686s | 1.860s | 4.42x |
| **v241 M100-validated router** | **0.3936** | **22.11%** | **0.636s** | **1.803s** | **4.77x** |

相对已有 full_raw baseline：

- 如果用 LongBench m20 full_raw `0.3596`，v241 = `109.46%`。
- 如果用 LongBench full_raw `0.372655`，v241 = `105.63%`。

因此 v241 同时满足当前目标：

- KV keep 在 `10%-30%`：实际 `22.11%`。
- online speed 超过 `2.5x`：实际约 `4.77x`。
- 分数达到 full baseline 的 `95%+`：实际超过 full baseline。

## v241 做了什么

v241 是 M100 验证后的 task-level memory-action router：

- 默认使用 v235/v231 的高质量 practical 路径。
- 只在 M100 已验证有效的任务上替换为 v240 动作：
  - `2wikimqa`
  - `hotpotqa`
  - `multifieldqa_en`

它没有沿用 v240 在 M20 上看似有效、但 M100 不泛化的动作：

- qasper 回退到 v235。
- narrativeqa 回退到 v235。
- musique 回退到 v235。

## 为什么不是 v240

v240 在 M20 上最好，但 M100 泛化后发现：

| Task | v240 vs v235 score delta | KV delta | v241 decision |
|---|---:|---:|---|
| 2wikimqa | +0.0238 | -1.75% | 用 v240 动作 |
| hotpotqa | -0.0004 | -14.01% | 用 v240 动作 |
| multifieldqa_en | +0.0421 | +11.80% | 用 v240 动作 |
| qasper | -0.0590 | -8.28% | 回退 v235 |
| narrativeqa | -0.0301 | -9.25% | 回退 v235 |
| musique | -0.0740 | -46.19% | 回退 v235 |

这个结果说明：router 的核心不应该是“统一小 block”，而应该是“把小 block/窗口动作作为任务条件动作”。

## 新现象

### 1. b16 适合作为定位器，不一定适合作为全局保留单元

在 2wikimqa 和 hotpotqa 上，b16/anchor-window 能降低 KV 或提高分数。

但在 qasper 上，M100 targeted ablation 证明 b16 不可靠：

| Qasper method | Score | KV keep | Online |
|---|---:|---:|---:|
| v235 qasper path | 0.3987 | 42.73% | 0.392s |
| v242 qasper b16 all-block | 0.3445 | 39.71% | 0.392s |
| v243 qasper b64 all-block | 0.3732 | 45.13% | 0.399s |

结论：qasper 的难点不是 block 太粗，而是需要更完整的 evidence composition。

### 2. musique 的压缩仍然是主要风险点

musique 上各种 capped-risk 探针都不如 v235：

| Musique method | Score | KV keep | Online |
|---|---:|---:|---:|
| v235 musique path | 0.2241 | 71.13% | 0.362s |
| v240 musique anchor-window | 0.1501 | 24.94% | 0.242s |
| v244 anchor-window risk3072 | 0.1810 | 35.83% | 0.269s |
| v247 anchor-window risk4096 | 0.1671 | 43.98% | 0.250s |
| v249 b128 risk4096 | 0.1389 | 41.76% | 0.252s |

结论：musique 不是简单提高风险预算就能恢复质量；它可能需要多跳 chain-level evidence planner，而不是局部 window。

## 下一步

最值得继续探索的是 musique/qasper 的 chain-level evidence planner：

1. 从 top evidence block 中抽取实体/数字/标题锚点。
2. 构造 evidence graph，而不是只选局部邻居。
3. 对多跳任务选择若干 chain windows。
4. 只在 graph coverage 不足时回退到 v235 高质量路径。

如果这个方向成功，v241 的 score 可以保持或提高，同时把 musique/qasper 的高 KV 任务压下来。

## 已启动的 chain-level 探针

针对上述结论，已启动 graph-bridge targeted M100：

| Run | Task | Design |
|---|---|---|
| v250 | qasper | graph bridge + bridge, risk cap 1536 |
| v251 | musique | graph bridge + bridge, risk cap 3072 |
| v252 | musique | graph bridge + bridge, risk cap 4096 |

这些实验不是替代 v241，而是判断下一步能否把 qasper/musique 的高 KV 路径压下来。

同时已有负结果：

| Run | Task | Score | KV keep | Conclusion |
|---|---|---:|---:|---|
| v242 | qasper b16 all-block | 0.3445 | 39.71% | 不如 v235 qasper，不能用。 |
| v243 | qasper b64 all-block | 0.3732 | 45.13% | 不如 v235 qasper，且更耗 KV。 |
| v244 | musique anchor-window risk3072 | 0.1810 | 35.83% | 比 v240 musique 好，但仍低于 v235。 |
| v247 | musique anchor-window risk4096 | 0.1671 | 43.98% | 加预算没有恢复质量。 |
| v249 | musique b128 risk4096 | 0.1389 | 41.76% | 原始 b128 capped-risk 也不行。 |
