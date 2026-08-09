# Locality-Preserving RoPE Repair：冻结实验协议

## 1. 方法定义

原生 RoPE 中，第 $i$ 个二维频率对的 QK 贡献为

$$
s_i(\Delta)=A_i\cos(\Delta\omega_i)+B_i\sin(\Delta\omega_i),
$$

其中 $\Delta$ 是 Query–Key 相对距离。我们只在 $\Delta$ 超过阈值 $s$ 后压缩相位：

$$
\widetilde\Delta=
\begin{cases}
\Delta,&\Delta\le s,\\
s+\alpha(\Delta-s),&\Delta>s.
\end{cases}
$$

原生与修复分数通过连续门控混合：

$$
\widetilde s_i(\Delta)
=(1-\beta)s_i(\Delta)+\beta s_i(\widetilde\Delta).
$$

因此，任意相对距离不超过 $s$ 的 token 对都严格复用原生 attention 路径；方法不会压缩整个序列的绝对位置。

## 2. 候选位置

### F47 消融

- layers 18–23；
- KV head-group 4（Query heads 16–19）；
- frequency pair 47；
- $s\in\{8192,16384\}$；
- $\alpha\in\{0,0.25,0.5\}$，$\beta=1$。

该组候选在 validation 上未改善 Gold NLL，已淘汰。

### F46 主候选

- layer 25；
- KV head-group 3（Query heads 12–15）；
- frequency pair 46；
- $s=8192$，$\alpha=0.25$，$\beta=1$。

## 3. 数据划分

- 参数搜索：RULER-32K seeds 43–44，共 26 条；
- 冻结测试：RULER-32K seeds 54–56，共 78 条；
- 长度迁移：RULER-64K seeds 57–59，共 39 条；
- 跨任务保真：LongBench HotpotQA 18 条冻结样本、PG19 4K/32K 各 8 本书。

冻结测试的参数和候选位置不得根据 seeds 54–59 的结果重新选择。

## 4. 候选筛选规则

候选必须同时满足：

1. 两个 validation seed 的平均官方分数均不下降；
2. validation 不出现逐样本官方分数退化；
3. 平均 Gold NLL 改善为正。

满足条件的候选按 Gold NLL 改善排序，只选择一个进入冻结测试。冻结测试只有在官方分数不下降且 Gold NLL 改善为正时，才进入跨任务保真。

## 5. 实现校验

- 本地 intervention 单元测试全部通过；
- 阈值以内直接复用原生 attention 输出，避免“数学等价但 BF16 舍入不同”；
- 64K 使用模型运行时的 YaRN `inv_freq` 与 `attention_scaling`，不从原始 `rope_theta` 重算；
- 相对距离修复直接重构选定频率对的 QK 分数，未修改 V 或模型权重；
- 64K BF16 使用两张 24GB GPU 的 `balanced` 模型并行，仅改变设备放置，不改变数值精度或方法定义。
