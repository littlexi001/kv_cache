# Section 15. 基于 K-L2 图的 KV 检索候选集合能量覆盖实验

## 1. 实验目的

前面的实验观察到两类结构：

1. K/K L2 最近邻中，部分 head 会呈现固定相对距离的横带；
2. Q/K attention top-k 中，部分 head 会呈现固定 token 的斜带或固定 lag 的横带。

这些现象说明 K cache 的几何结构并不是随机的，而是包含可被利用的检索结构。本节实验尝试设计一种不直接使用当前 query 排序所有 K 的候选集合方法，并评估该候选集合能覆盖多少真实 attention energy。

本节只做阶段 A：评估 attention energy 覆盖率，不修改模型 forward，也不计算 loss/PPL。

## 2. 候选集合设计

对于当前 query token `t`，当前可见上下文长度为：

```text
visible = t + 1
```

候选集合 `S_t` 由三部分组成：

```text
S_t = prefix_1% ∪ recent_1% ∪ expanded_middle_seeds
```

其中：

- `prefix_1%`：当前可见上下文最前面的 1% token；
- `recent_1%`：当前 query 最近的 1% token；
- `middle`：去掉 `prefix_1%` 和 `recent_1%` 后的中间区域；
- `seeds`：同一层、同一 attention head 的上一个 query token `t-1` 在 `middle` 区域中 attention 分数最高的 1% token；
- `expanded_middle_seeds`：每个 seed 自身，加上它在同一层、同一 KV head 中 previous-only K-L2 最近的 top-20 邻居。

所有候选 token 去重，并限制在 causal 可见范围内：

```text
j <= t
```

由于 Qwen3-0.6B 使用 GQA，attention head 和 KV head 数量不同：

```text
num_attention_heads = 16
num_key_value_heads = 8
kv_head = attention_head // 2
```

因此实验中采用 KV-head 粒度的候选集合：同一个 KV head 对应的两个 attention heads 共享同一个候选集合。

## 3. 评价指标

对于真实 attention 分布：

```text
a_t(j) = softmax(q_t · k_j)
```

方法能量定义为候选集合覆盖的 attention mass：

```text
method_energy = sum_{j in S_t} a_t(j)
```

同时记录两个 baseline。

Oracle baseline：

```text
oracle_energy = 当前 query 中 attention 分数最高的 |S_t| 个 token 的 attention 之和
```

这个 baseline 使用了当前 query 的真实 attention 排序，因此不是在线可用方法，而是同等候选预算下的上界。

Prefix/recent baseline：

```text
prefix_recent_energy = prefix_1% ∪ recent_1% 覆盖的 attention 之和
```

它衡量只保留开头和最近 token 的固定策略能覆盖多少能量。

候选集合大小用百分比表示：

```text
candidate_fraction = |S_t| / visible
```

## 4. 图的含义

典型输出图：

```text
energy_and_candidate_fraction_by_token.png
```

横轴：

```text
Query token index
```

左纵轴：

```text
Attention energy
```

右纵轴：

```text
Candidate set size (%)
```

图中四条线分别表示：

- 蓝线 `method energy`：本文候选集合方法覆盖的真实 attention energy；
- 橙线 `oracle top-s energy`：同等候选数量下，直接选当前 attention top-|S| 的 oracle 上界；
- 绿线 `prefix+recent energy`：只选前 1% 和最近 1% token 的 baseline；
- 灰线 `candidate %`：本文方法实际选出的候选集合大小占当前可见上下文的比例。

如果蓝线接近橙线，说明检索候选集合在同样预算下接近 oracle；如果蓝线显著高于绿线，说明中间区域 seeds + K-L2 扩展确实带来了额外有效覆盖。

## 5. 实验设置

本次用户提供的图来自 5k token 实验：

```text
tokens: 5000
boundary_fraction: 0.01
seed_fraction: 0.01
neighbor_count: 20
candidate granularity: KV head
```

对应项目：

```text
ymluo/projects/qwen3_kv_retrieval_energy_analysis
```

运行方式：

```bash
bash ymluo/projects/qwen3_kv_retrieval_energy_analysis/scripts/run_analysis.sh
```

## 6. L15 KV2 Attention Head 4 结果

![Retrieval energy L15 KV2 AttnH4](../projects/qwen3_kv_retrieval_energy_analysis/outputs/retrieval_energy/plots/layer_15/kv_head_02/attention_head_04/energy_and_candidate_fraction_by_token.png)

图 15-1：`L15 KV2 AttnH4` 的候选集合能量覆盖结果。

该 head 的主要现象：

1. `method energy` 大部分时间保持在较高水平，通常接近 `0.9-1.0`。
2. `oracle top-s energy` 基本接近 `1.0`，说明在相同候选预算下，如果直接按当前真实 attention 排序，几乎可以覆盖全部 attention energy。
3. `prefix+recent energy` 波动较大，但整体明显低于 method energy，说明仅靠前 1% 和最近 1% 并不足以稳定覆盖该 head 的 attention mass。
4. `candidate %` 随 token 增加逐渐下降，大约从早期较高比例下降到后期约十几个百分点附近。这是因为候选集合中固定 prefix/recent 和 seed 扩展数量增长相对慢于可见上下文长度。

初步解释：对于 `L15 KV2 AttnH4`，K-L2 图扩展的 middle candidates 非常有效。虽然候选集合只占可见上下文的一小部分，但能覆盖大部分真实 attention energy，说明这个 head 的 attention 目标和 K-L2 邻域结构有较强一致性。

## 7. L15 KV4 Attention Head 8 结果

![Retrieval energy L15 KV4 AttnH8](../projects/qwen3_kv_retrieval_energy_analysis/outputs/retrieval_energy/plots/layer_15/kv_head_04/attention_head_08/energy_and_candidate_fraction_by_token.png)

图 15-2：`L15 KV4 AttnH8` 的候选集合能量覆盖结果。

该 head 的主要现象：

1. `method energy` 明显低于 `L15 KV2 AttnH4`，在较长区间内大约处于 `0.5-0.8`，波动更大。
2. `oracle top-s energy` 仍然较高，通常在 `0.85-0.95` 附近，说明同等候选预算本身足够覆盖较多能量，但本文检索方法没有总是选中最关键 token。
3. `prefix+recent energy` 在不同 token 区间中剧烈变化，有些片段接近或超过 method energy，有些片段非常低。这说明该 head 的 attention 目标在不同文本区间可能发生明显切换。
4. `candidate %` 大致在十几到二十多个百分点之间，候选预算并不小，但 method energy 仍与 oracle 有较大 gap。

初步解释：`L15 KV4 AttnH8` 的 attention 更难被“上一个 query 的 middle top1% seeds + K-L2 previous neighbors”稳定捕获。可能原因包括：

- 该 head 的 attention 更依赖当前 query 内容，而不是上一 token 的 attention seeds；
- attention 目标切换较快，`t-1` 的 seeds 对 `t` 的预测性较弱；
- K-L2 邻域和真实 attention 目标之间的一致性较弱；
- 该 head 可能更偏 anchor/sink 或语义跳转，导致 K-L2 扩展不能稳定覆盖关键 token。

## 8. 两个 head 的对比

两张图显示同一层不同 KV head/attention head 的行为差异很大：

```text
L15 KV2 AttnH4: method energy 高且稳定，接近 oracle
L15 KV4 AttnH8: method energy 中等且波动大，和 oracle gap 明显
```

这说明基于 K-L2 图的检索候选方法并不是对所有 head 都同样有效。它更适合那些 attention 目标和 K-cache 几何邻域一致的 head；对于目标变化快、anchor 切换明显或更强 query-dependent 的 head，上一 token seeds 加 K-L2 扩展可能不够。

从候选集合大小看，两个实验中 `candidate %` 后期都不是很大，说明方法具备一定压缩潜力。但是否能用于真实 loss/PPL 实验，需要进一步确认 method energy 在所有层/head 上的最低覆盖水平。

## 9. 当前结论

1. 阶段 A 实验证明，K-L2 图扩展的候选集合在某些 head 上可以用较小候选比例覆盖很高的 attention energy。
2. Oracle energy 高说明同等预算本身足够强，但 method 和 oracle 的 gap 衡量了检索策略本身的不足。
3. Prefix/recent baseline 不稳定，说明仅依赖开头 token 和局部窗口不足以解释这些 head 的 attention 质量。
4. 不同 head 差异明显，因此后续阶段 B 的 loss 实验可能需要 head-wise 或 layer-wise 自适应策略，而不是所有 head 使用相同检索规则。
5. 当前方法最关键的假设是：上一个 query 的中间高 attention token 可以作为当前 query 的入口 seeds。这个假设对部分 head 有效，但对另一些 head 可能不足。

## 10. 后续建议

建议继续做以下分析：

1. 汇总所有 `(layer, KV head, attention head)` 的 `method_energy_mean`、`energy_gap_to_oracle_mean` 和 `candidate_fraction_mean`，找出适合该检索策略的 head。
2. 按层画 heatmap，观察浅层/深层的 energy 覆盖差异。
3. 比较不同 `neighbor_count`，例如 10、20、40，对 energy 和 candidate fraction 的影响。
4. 比较 seed 来源：`t-1`、`t-2`、前几个 query 的 union，或者滑动平均 attention seeds。
5. 对 energy gap 大的 head 单独分析 attention top-k 图，判断失败原因是 sink token、anchor 切换，还是 K-L2 邻域不匹配。
6. 在阶段 A 覆盖率稳定后，再进入阶段 B：真正 mask 非候选 KV token，计算 loss/PPL。
