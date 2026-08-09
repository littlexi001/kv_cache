# War and Peace 4k→512 首轮结果

日期：2026-07-14

## 1. 运行设置

```text
host = df / CISL-NF5468M5
gpu = physical GPU 4, NVIDIA RTX 3090
model = /home/fdong/hrj/prove/Qwen3-0.6B
text = War and Peace
prefill_tokens = 4096
eval_tokens = 512
chunk_size = 64
dtype = float16
attention = eager
budget = ceil(ratio * historical_tokens), current self kept outside budget
```

远程输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_token_mechanism/outputs/full_war4k_20260714_162846
```

本地同步输出：

```text
ymluo/projects/qwen3_top2_token_mechanism/outputs/full_war4k_20260714_162846
```

## 2. Sanity checks

全部通过：

- Top-attention 100% 与 full 的 delta loss 为 `0.0`；
- 所有 mode 评分完全相同的 token indices；
- `sink_recent_s0` 与 pure recent 的逐 token 最大 NLL 差为 `0.0`；
- 所有等预算 selector 的实际 keep-percent spread 为 `0.0`。

## 3. Top-attention ratio 曲线

| 历史 token 比例 | 实际 keep | PPL | Delta NLL vs full |
| ---: | ---: | ---: | ---: |
| 0.1% | 0.115% | 40.6101 | +0.55025 |
| 0.5% | 0.510% | 26.5015 | +0.12344 |
| 1% | 1.011% | 24.3977 | +0.04073 |
| 2% | 2.011% | 23.3741 | -0.00213 |
| **4%** | **4.011%** | **22.7105** | **-0.03094** |
| 8% | 8.011% | 23.0062 | -0.01800 |
| 16% | 16.011% | 23.1964 | -0.00976 |
| 32% | 32.011% | 23.3425 | -0.00349 |
| 100% | 100% | 23.4240 | 0.00000 |

这一次单文本运行的最低点是 4%，不是 2%。配对 block-bootstrap：

```text
2% - full:  mean Delta NLL = -0.00213
             95% CI = [-0.04372, +0.03198]

4% - full:  mean Delta NLL = -0.03094
             95% CI = [-0.06457, -0.00132]

4% - 2%:    mean Delta NLL = -0.02880
             95% CI = [-0.03862, -0.01813]
```

因此，在这个 4k→512 continuation 条件下，2% 与 full 无法区分；4% 的改善较稳定。但这不否定此前下游任务曲线里的 2% 低点，说明最优比例可能依赖数据、评分区间和任务。

## 4. Top-2% 为什么已经接近 full

全 layer/head/query 聚合：

```text
mean entropy                         = 2.9698
mean exp(entropy) effective support = 82.42 tokens
mean 1/sum(p^2) effective support   = 20.16 tokens
normalized L2 effective support     = 0.463%
Top-2% historical attention mass    = 77.16%
Top-2% + self attention mass        = 85.34%
mean 2% cutoff score gap            = 0.01275
```

这支持“attention 高度集中”解释：2% 虽然只保留约 87 个历史位置，但已经覆盖约 77% 的历史 attention mass；按 inverse participation ratio 计算，平均有效支持甚至只有约 20 个 token。

同时，4% 比 2% 更好说明 2% 后的尾部并非全是噪声。2% 到 4% 之间仍有少量对 PPL 有益的 token；更大的 8%–32% 则逐渐回到 full。

## 5. Top-2% token 是什么

按 query-relative、互斥位置角色：

| 角色 | 选择事件 | 事件占比 |
| --- | ---: | ---: |
| sink（前 4 个位置） | 217,759 | 1.08% |
| recent（最近 256，排除 sink） | 9,968,281 | 49.65% |
| remote | 9,889,288 | 49.26% |

选择事件几乎正好一半 recent、一半 remote。sink 的事件数很少，但 mass 极端集中：token 0 `The` 单独只占 `1.02%` 的选择事件，却占所有 Top-2% selected attention mass 的 `51.57%`，选择率为 `89.4%`。

按 eligible exposure 校正后的词法统计：

| 类别 | 事件占比 | Mass 占比 | 相对 exposure 富集 |
| --- | ---: | ---: | ---: |
| word | 75.39% | 87.50% | 1.11x |
| punctuation | 19.32% | 10.15% | 2.73x |
| newline | 2.85% | 1.28% | 0.24x |
| mixed/subword | 1.79% | 0.98% | 1.60x |
| number | 0.47% | 0.07% | 0.19x |
| whitespace | 0.18% | 0.02% | 0.019x |

标点位置被明显过选，但绝大多数 selected mass 仍在 word token；newline/whitespace 并不是主导类别。

## 6. 等预算 sink + recent 是否能替代 Top-2%

| Selector | PPL | PPL / Top-2% | Delta NLL vs Top-2% | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| Top-2% oracle | 23.3741 | 1.000 | 0.000 | [0, 0] |
| sink=8 + recent | **35.1149** | **1.502** | **+0.4070** | **[+0.2944, +0.5394]** |
| sink=2 + recent | 35.6930 | 1.527 | +0.4233 | [+0.2981, +0.5709] |
| sink=1 + recent | 35.7059 | 1.527 | +0.4237 | [+0.2988, +0.5696] |
| sink=4 + recent | 35.7926 | 1.531 | +0.4261 | [+0.3048, +0.5687] |
| sink=16 + recent | 37.4381 | 1.602 | +0.4711 | [+0.3379, +0.6158] |
| recent only | 1917.8586 | 82.05 | +4.4073 | [+3.9427, +4.9112] |

所有 sink+recent 方案都未通过 `±0.01 nat/token` 等价检验。最佳分配是 8 个 sink token，其 PPL 仍比 Top-2% 高约 50.2%。

机制诊断出现一个重要反差：

| Sink allocation | Oracle position recall | Oracle mass recall | Pruned-distribution cosine |
| ---: | ---: | ---: | ---: |
| 0 | 33.01% | 31.49% | 0.3959 |
| 1 | **33.87%** | **83.02%** | **0.9626** |
| 2 | 33.71% | 82.97% | 0.9625 |
| 4 | 33.41% | 82.87% | 0.9622 |
| 8 | 32.83% | 82.71% | 0.9619 |
| 16 | 31.24% | 82.26% | 0.9607 |
| 32 | 27.60% | 81.18% | 0.9578 |

只加入 token 0，就把 oracle mass recall 从 31.5% 提高到 83.0%，分布 cosine 提高到 0.963；但真实 PPL 仍明显差于 oracle Top-2%。这说明：

```text
高 attention mass / 高分布 cosine 并不足以保证层层传播后的语言模型质量。
低 mass、query-conditioned remote token 仍然具有不成比例的功能影响。
```

Top-2% 中只保留 sink+recent 部分的 drop-remote 模式实际剩约 0.987% 历史 token，PPL 为 31.2899；相对 Top-2% 的 Delta NLL 为 `+0.2917`，95% CI `[+0.1182, +0.4842]`。这进一步支持 remote selected tokens 有实际贡献，但该 drop 消融同时降低了预算，不能当作严格等预算对照。

## 7. 当前结论

1. Top-attention 稀疏化的收益不是单纯来自 sink+recent 位置规则。
2. token 0 是极强 sink，解释了超过一半的 selected attention mass，但它不等于完整功能记忆。
3. Top-2% 选择事件约一半 recent、一半 remote；远程的 query-conditioned token 对 PPL 必要。
4. 这次 continuation PPL 的最优点是 4%，说明“2% 最优”不是跨条件常数。
5. 下一步应在原先观察到 2% 最优的下游任务上复用同一套等预算和逐 token/逐样本配对协议，判断任务差异还是文本差异导致最优比例移动。

