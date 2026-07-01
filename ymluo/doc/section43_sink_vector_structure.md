# Section 43: sink token 向量结构诊断

日期：2026-07-01

## 1. 问题

这次实验回答：

```text
为什么 sink token 看起来很神奇？
为什么很多不同 q 向量都能和 sink token 的 k 向量有较高 cosine/logit？
```

核心假设有三个：

1. sink K norm 特别大，所以 logit 高；
2. q 向量本身有公共方向，sink K 对齐这个 common direction；
3. RoPE 起始位置相位把 sink K 调成更容易和 query 对齐。

## 2. 新增代码和输出

新增脚本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/analyze_sink_vector_structure.py
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_sink_vector_structure_server.sh
```

服务器输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/sink_vector_structure_0701_v1
```

本地输出：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/sink_vector_structure_0701_v1
```

规模：

| item | value |
|---|---:|
| model | Qwen3-0.6B |
| head_dim | 64 |
| selected layers | 0,4,8,13,20,27 |
| selected heads | all 16 heads |
| tasks | 16 |
| sampled layer/query rows | 192 |

## 3. 指标解释

| metric | 含义 |
|---|---|
| `full_cos_mean` | q 和该 group K 的原始平均 cosine |
| `residual_cos_mean` | 去掉 q 的平均公共方向后，再算 q/k residual cosine |
| `k_norm_mean` | K 向量 norm |
| `q_norm_mean` | Q 向量 norm |
| `k_cos_qmean` | K 和所有 query 平均方向 `q_mean` 的 cosine |
| `k_common_energy_frac` | K 在 `q_mean` 方向上的能量比例 |
| `logit_mean` | q-k dot / sqrt(head_dim) |
| `q_mean_norm` | 所有 query 单位向量平均后的 norm；越大说明 q 方向越集中 |
| `q_pc1_energy` | q 单位向量的第一主成分能量 |

这里最关键的是：

```text
如果 full_cos 高，但 residual_cos 接近 0，
说明 q-k cosine 主要来自公共方向，而不是 task-specific residual。
```

## 4. Overall 结果

### 4.1 Pre-RoPE 空间

| group | full cosine | residual cosine | K norm | K cos q_mean | K common energy | logit |
|---|---:|---:|---:|---:|---:|---:|
| sink_first2 | 0.2093 | -0.0037 | 59.49 | 0.2311 | 8.31% | 8.607 |
| sink_rest | 0.1031 | -0.0016 | 64.43 | 0.1145 | 2.24% | 5.664 |
| sink_all | 0.1164 | -0.0018 | 63.81 | 0.1291 | 3.00% | 6.032 |
| recent | 0.0985 | 0.0147 | 67.43 | 0.1009 | 1.94% | 5.653 |
| evidence_key | 0.0581 | 0.0165 | 68.21 | 0.0550 | 0.80% | 4.097 |
| evidence_label | 0.1102 | 0.0290 | 66.65 | 0.1070 | 2.01% | 6.237 |
| evidence_any | 0.0702 | 0.0140 | 69.20 | 0.0701 | 1.08% | 4.741 |
| other_sample | 0.0974 | 0.0016 | 65.03 | 0.1064 | 2.06% | 5.537 |

Pre-RoPE already shows:

```text
sink_first2 has the highest q-k cosine and highest K alignment to q_mean.
After removing q_mean, residual cosine becomes approximately 0.
```

### 4.2 Post-RoPE 空间

| group | full cosine | residual cosine | K norm | K cos q_mean | K common energy | logit |
|---|---:|---:|---:|---:|---:|---:|
| sink_first2 | 0.1812 | -0.0017 | 59.49 | 0.2341 | 10.55% | 5.566 |
| sink_rest | 0.0564 | -0.0009 | 64.43 | 0.0800 | 2.66% | 1.403 |
| sink_all | 0.0720 | -0.0010 | 63.81 | 0.0993 | 3.65% | 1.923 |
| recent | 0.0907 | 0.0439 | 67.43 | 0.0910 | 2.63% | 4.485 |
| evidence_key | 0.0179 | 0.0098 | 68.21 | 0.0177 | 0.92% | 0.124 |
| evidence_label | 0.0723 | 0.0218 | 66.65 | 0.0821 | 2.70% | 2.307 |
| evidence_any | 0.0271 | 0.0021 | 69.20 | 0.0374 | 1.30% | 0.424 |
| other_sample | 0.0494 | 0.0013 | 65.03 | 0.0692 | 2.57% | 1.071 |

Post-RoPE 结果更清楚：

1. `sink_first2` 的 full cosine `0.1812` 远高于 `sink_rest=0.0564`、`evidence_any=0.0271`、`other_sample=0.0494`。
2. `sink_first2` 的 residual cosine 是 `-0.0017`，基本为 0。
3. `sink_first2` 的 K norm 是 `59.49`，反而低于 evidence/recent/other，因此不是 K norm 大导致的。
4. `sink_first2` 在 q_mean 方向上的能量是 `10.55%`，明显高于其他组。

关键结论：

```text
sink_first2 的高 cosine 主要来自它对齐了所有 query 共享的 common direction。
去掉这个 common direction 后，sink 和 q 不再有特殊相似度。
```

## 5. Query 向量本身不是任意分散的

整体统计：

| space | q_mean_norm | q_pc1_energy |
|---|---:|---:|
| pre-RoPE | 0.921 | 44.6% |
| post-RoPE | 0.809 | 28.4% |

这说明 sampled query 的单位向量高度集中，并不是各向同性随机分布。

如果 q 是随机各向同性的，很多 q 单位向量平均后 norm 应该接近 0。但这里：

```text
pre-RoPE q_mean_norm = 0.921
post-RoPE q_mean_norm = 0.809
```

说明不同 query 都带有很强的公共分量。sink_first2 的 K 正好对齐这个公共分量，所以看起来像“无论 q 是什么，都能对上 sink”。

更准确的分解是：

```text
q = q_common + q_task_specific
k_sink = k_common_aligned + k_residual

cos(q, k_sink) 主要来自 q_common · k_common_aligned
```

而不是 sink 能理解所有 query 的 task-specific 语义。

## 6. First 2 tokens 和 sink_rest 的差异

Post-RoPE：

| group | full cosine | residual cosine | K cos q_mean | K common energy | logit |
|---|---:|---:|---:|---:|---:|
| sink_first2 | 0.1812 | -0.0017 | 0.2341 | 10.55% | 5.566 |
| sink_rest | 0.0564 | -0.0009 | 0.0800 | 2.66% | 1.403 |

`sink_first2` 相比 `sink_rest`：

1. full cosine 约 3.2 倍；
2. logit 约 4 倍；
3. common-direction energy 约 4 倍；
4. residual cosine 都接近 0。

所以前面 ablation 里“保留前 2 个 KV 就能恢复”的原因是：

```text
前 2 个 K 才真正承载了 query common direction anchor。
第 3-16 个 sink token 没有同等强的向量结构。
```

## 7. 分层结果

Post-RoPE layer aggregate：

| layer | sink_first2 full/res/common | sink_rest full/res/common | other full/res/common |
|---:|---|---|---|
| 0 | 0.015 / 0.000 / 0.4% | 0.015 / 0.000 / 0.3% | 0.014 / 0.000 / 0.3% |
| 4 | 0.208 / -0.001 / 13.6% | -0.001 / 0.000 / 0.7% | -0.003 / 0.000 / 0.6% |
| 8 | 0.231 / -0.003 / 12.4% | 0.082 / -0.002 / 2.6% | 0.073 / 0.002 / 2.5% |
| 13 | 0.248 / -0.003 / 17.4% | 0.147 / -0.001 / 7.7% | 0.146 / 0.007 / 8.0% |
| 20 | 0.276 / -0.002 / 15.7% | 0.128 / -0.002 / 4.4% | 0.105 / 0.001 / 3.7% |
| 27 | 0.109 / -0.000 / 3.7% | -0.032 / 0.000 / 0.3% | -0.038 / -0.002 / 0.3% |

Sink first2 最明显的层：

```text
layers 4, 8, 13, 20
```

Layer 0 反而不明显。这和前面的 ablation 互补：sink anchor 不是简单“所有层都看第一个 token”，而是在中间层和中深层的特定 heads 中形成强公共方向。

## 8. 最强 sink heads

Post-RoPE sink_first2 full cosine 最高的 heads：

| layer | head | full cosine | residual cosine | K cos q_mean | common energy | logit |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 10 | 0.447 | 0.000 | 0.514 | 27.5% | 12.37 |
| 4 | 13 | 0.436 | 0.003 | 0.517 | 36.0% | 13.65 |
| 13 | 4 | 0.430 | -0.018 | 0.556 | 31.2% | 14.04 |
| 20 | 13 | 0.407 | -0.004 | 0.490 | 27.3% | 10.07 |
| 8 | 6 | 0.390 | -0.007 | 0.481 | 27.4% | 11.85 |

这些 head 的共同特征是：

```text
sink_first2 K 和 q_mean 的 cosine 可达 0.48-0.56，
K 在 q_mean 方向上的能量可达 27%-36%，
但 residual cosine 仍接近 0。
```

所以这些 head 几乎就是专门的 common-direction sink heads。

## 9. 当前解释

sink token 看起来“神奇”，不是因为它能匹配所有不同语义的 q，而是因为很多 q 共享一个很强的公共分量。

更准确地说：

```text
q = q_common + q_specific
k_sink_first2 ≈ aligned_to(q_common) + small residual
```

于是：

```text
cos(q, k_sink_first2)
≈ cos(q_common, k_sink_common)
```

这会让 sink 对很多 query 都有较高 cosine/logit。去掉 q_common 后，sink 的 residual cosine 约等于 0，说明它并没有和具体语义 residual 对齐。

## 10. 和 RoPE 的关系

这次实验说明：

1. pre-RoPE already has common-direction alignment；
2. post-RoPE 后，sink_first2 仍然保持最高 common alignment；
3. 前面 move ablation 显示，原 sink 内容移走后 mass 大幅下降，新的前排位置获得 mass。

合起来看：

```text
pre-RoPE 的 q/k 投影已经学出 common direction；
RoPE 起始位置相位决定这个 common direction 在 attention logits 中是否能稳定发挥。
```

所以不是“纯 RoPE”，也不是“纯内容”。更合理的说法是：

```text
模型学出了一个 common query direction；
前 1-2 个 token 的 K 向量承载了这个 direction；
RoPE 起始相位让这个承载关系在 causal attention 中最稳定。
```

## 11. 对方法设计的启发

1. 不应该随便 drop 前 1-2 个 token KV，因为它们是 common-direction anchor。
2. 第 3-16 个 sink token 可以大胆压缩或删除，前面 ablation 已经显示损失很小。
3. 如果做低秩投影分类器，应该显式把 common-direction/sink-anchor 方向和 evidence-specific residual 方向分开：

```text
score(q, k) = common_anchor_score + residual_evidence_score
```

否则 classifier 可能只学到 sink/common direction，而不是学到真正的 evidence selection。

