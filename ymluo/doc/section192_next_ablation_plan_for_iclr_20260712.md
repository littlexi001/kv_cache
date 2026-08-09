# Section 192: Next Ablation Plan for ICLR-Grade Evidence (2026-07-12)

## 当前主假设

如果 v430/v431 的真实 M100 结果接近预测，那么方法主线应升级为：

```text
Source-Preserving Frontier Routing + Constrained Source Composer
```

这条线比单纯 router 更像论文方法，因为它解决了一个清晰问题：

1. 单一 KV frontier 无法同时适配 QA、summary、code、synthetic retrieval；
2. naive parameter overlay 会破坏 reference/fallback 语义；
3. source-preserving composition 能保留每个 frontier 的完整动作语义；
4. constrained composer 在 KV/latency 预算下自动选 source。

## 明早优先检查

运行：

```bash
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
/home/fdong/miniconda3/envs/moe/bin/python scripts/summarize_lowkv_queue_20260712.py
```

如果统一脚本尚未同步成功，则先用：

```bash
/home/fdong/miniconda3/envs/moe/bin/python scripts/summarize_v427_v428_m200_validation_20260712.py
/home/fdong/miniconda3/envs/moe/bin/python scripts/summarize_v427_ruler_validation_20260712.py
```

重点不是只看平均分，而是按以下 gate 判断：

| Gate | 通过条件 |
|---|---|
| LongBench quality | score >= 95% full，同等 M200 baseline 出来后优先用 M200 full |
| KV ratio | 1%-10% 平均 KV |
| online speed | >=2.5x |
| stability | M100 和 M200 不出现明显反转 |
| RULER | 4k/8k/16k 至少能解释清楚 KV/speed/score tradeoff |

## 必做 ablation

如果 v430/v431 成功，下一轮要补以下对照：

| Ablation | 目的 |
|---|---|
| v427 source-preserving vs v426 naive overlay | 证明保留 source policy 语义必要 |
| v430/v431 composer vs v427/v428 manual source | 证明 constrained composer 有额外收益 |
| DP composer v433/v434 vs Lagrange composer v430/v431 | 证明选择过程不是调 penalty 的偶然 |
| single frontier v417/v421/v424/v397 vs composer | 证明多 frontier composition 必要 |
| RULER v427 wildcard vs v436 low-budget wildcard | 证明 RULER 需要单独的 synthetic retrieval budget policy |

## 如果 v430/v431 没兑现

优先不要继续扫参数，而是查 source transfer 失败：

1. 按任务比较 v430/v431 的真实任务分数和 composer 预测任务分数；
2. 找出 source policy 在组合后是否因为 parent/extends 解析变化导致动作语义变化；
3. 检查 `reference` / fallback 是否指向预期 base；
4. 检查 speed 预测偏差是否来自 online overhead 叠加，而不是 attention token 本身；
5. 如果失败集中在某些任务，改成 sample-level source composer，而不是全局任务级 composer。

## 论文图表候选

1. Pareto curve：full, v417, v427, v428, v430, v431；
2. Bar chart：KV ratio vs score retention；
3. Task heatmap：每个任务由哪个 source frontier 负责；
4. Ablation table：source-preserving / overlay / single frontier / composer；
5. RULER length table：4k/8k/16k 的 score, KV, speed。

## 当前风险

1. LongBench M100 已经很强，但 M200 尚未完成，稳定性还没有最终证据；
2. RULER 当前 v427 可能因为 wildcard 预算导致 4k KV > 10%，需要 v436 补测；
3. 所有 composer 都是基于已完成 M100 统计生成，必须补 M200 或 holdout 证明不是过拟合；
4. 多模型验证还没完成，这是 ICLR 投稿必须补的核心证据。
