# Section 148: b16 group sweep 与 bounded fallback 新一轮探索

日期：2026-07-11

## 目标

本轮继续围绕 ICLR 主线优化：

- KV keep 控制在 10%-30%。
- 相对 full KV 的 online / E2E speed 保持 2.5x 以上。
- 分数达到或超过 full baseline 的 95%，最好继续高于 full。
- 不做盲目调参，而是根据当前失败/瓶颈设计新的 memory action。

## 当前判断

`v300` 是当前 validation-robust 主线，M100 为：

| Method | Score | KV keep | Online |
|---|---:|---:|---:|
| v300 validation-robust router | 0.439235 | 27.41% | 0.563s |

它已经满足 10%-30% KV 区间，并且在之前 full_kv M100 口径下速度明显超过 2.5x。下一步不是简单继续压全局 budget，而是看哪些高 KV 任务仍然可以安全下降。

高 KV / 高风险任务主要是：

- `multifieldqa_en`
- `musique`
- `hotpotqa`
- `repobench-p`
- `qmsum`
- 部分 `narrativeqa` / `2wikimqa`

这些任务的共同点是：当前方法为了保质量，经常触发 full fallback 或长输出 decode。继续全局压缩会直接伤分，应该加入中间动作。

## 实验线 A：b16 group-size sweep

用户提出合理假设：`block_size=16` 不一定太碎，因为 token 本来就是逐个处理；如果选更多小块，可能比大块更灵活。

已有结果显示：b16 多块在 M100 上整体没有超过 v300/v286，但之前的 b16 多块大多使用 `ours_multiscale_group_pages=8`，等价于 128-token coarse group，可能没有真正验证更细 group。

因此新增：

| Version | 设计 |
|---|---|
| v301 | b16 high-recall，multiscale/coarse group 从 8 降到 4 |
| v302 | b16 high-recall，multiscale/coarse group 从 8 降到 2 |
| v303 | 补齐旧 v289 high-recall 缺失的 hotpotqa，并组合成完整对照 |

启动脚本：

```bash
scripts/launch_b16_group_sweep_20260711.sh
```

自动合并脚本：

```bash
scripts/watch_combine_b16_group_sweep_20260711.sh
```

最终表：

```text
outputs/riskkv_v19_v301_v302_b16_group_sweep_20260711/summary_table.csv
```

判断标准：

- 如果 v301/v302 明显优于 v300，说明更细 coarse group 才是 b16 的关键。
- 如果仍低于 v300，基本可以停止把 b16 作为 LongBench QA 主线，只保留为特定任务 locator。

## 实验线 B：bounded fallback

这是本轮更重要的新方法。

当前 full fallback 的问题：它安全，但会把某些高风险样本直接拉到接近 full KV。真正需要的可能不是完整上下文，而是比 sparse action 更大的结构化证据预算。

新动作：

```text
risk trigger -> bounded evidence fallback
```

也就是风险触发后，不直接 full，而是升到 3k/4k token 的中间预算，再由原 selector 选择证据块。

实现改动：

- `apply_v293_action_router()` 现在尊重配置中的 `score_risk_budget_tokens` / `coverage_risk_budget_tokens`。
- 默认仍为 `999999`，所以 v300/v293 原行为不变。
- 新配置可以把 fallback cap 设置成 4096 或 3072。

新增配置：

| Version | 设计 |
|---|---|
| v304 | bounded fallback 4k + qmsum decode 64 |
| v305 | bounded fallback 3k + qmsum decode 64 |

受影响任务：

```text
narrativeqa,multifieldqa_en,hotpotqa,2wikimqa,musique,qmsum,repobench-p
```

启动脚本：

```bash
scripts/launch_bounded_fallback_sweep_20260711.sh
```

自动合并脚本：

```bash
scripts/watch_combine_bounded_fallback_20260711.sh
```

最终表：

```text
outputs/riskkv_v19_v304_v305_bounded_fallback_20260711/summary_table.csv
```

## 预期现象

最理想情况：

- v304 分数接近 v300，但 KV 和 online_seconds 更低。
- v305 如果掉分不明显，则成为更强的 speed-oriented Pareto 点。

如果 v304/v305 掉分明显：

- 说明 high-risk 样本确实需要 full fallback 或更强 evidence composer。
- 下一步应训练 sample-level confidence router，只在“bounded fallback 可信”的样本上使用 3k/4k，其余回到 v300 safe action。

## 当前服务器状态

已提交后台任务：

- b16 group sweep: v301/v302/v303。
- bounded fallback sweep: v304/v305。

两个 watcher 都已启动，会自动等待任务完成并生成合并 summary。

## 下一步

1. 等 `summary_table.csv` 出来后，先看 v304/v305 是否形成新 Pareto。
2. 如果 bounded fallback 有收益，继续在 M150 / extra50 上验证，防止 M100 过拟合。
3. 如果只有部分任务收益，训练 action router v3：输入 task、score gap、entropy、prefix length、coverage，输出 `sparse / bounded fallback / full fallback`。
4. 如果 b16 group sweep 没收益，论文主线不再押 b16，只作为 block-size-aware ablation。
