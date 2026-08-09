# 确定性采样 QKSieve：阶段结论

## 结论

当前应冻结为下一轮论文实验基线的是 **QKSieve c64 deterministic ValueSketch**：

`qksieve_qmse_oas_requestlocal_valuesketch16_sorted_c64_k1280`

它不是 router，也不依赖任务标签或 Full Attention 回退。其决策只由当前请求的 Q/K 二阶统计、序列长度和固定数学规则产生。相较此前固定 1,024 个分位样本的 sampled QKSieve，它主要增加了两项：

1. 采样数按目标上尾概率缩放，使样本上尾始终约有 64 个锚点。
2. CUDA 候选压缩和 Value 尾项使用 bitmask、前缀扫描和固定树归约，消除 atomic 顺序造成的不确定性。

## 条件矩路线的结论

MomentSieve/条件 Wiener 路线有明确的局部正信号，但没有证明能够全方面超过 QKSieve。在 Qwen sports 96K 的真实 Q/K/V 局部审计中，query-crossfit 的 16 维条件模型在 192 个 case 上把 held-out Value 残差预测误差平均降低 `46.27%`，说明 Key 坐标确实包含未被 rank-16 ValueSketch 吸收的 Value 信息。然而，同一审计中的最终 attention output 相对 L2 误差仍约为 `1.25%`，没有随残差回归指标同比下降。

闭环 4-token probe 更清楚地暴露了问题。在五个 32K case、相同 `k=1280` 下：

| 方法 | 平均质量保持 | 最差质量保持 | 稳态延迟 |
|---|---:|---:|---:|
| rank-16 ValueSketch | 100.590% | 98.262% | 124.6 ms |
| 8 维条件残差 | 100.756% | 98.640% | 463.2 ms |
| query-crossfit 条件残差 | 100.647% | 98.424% | 1098.2 ms |
| safe-query 条件残差 | 100.588% | 98.456% | 1072.2 ms |

条件残差的平均质量只提高 `0.166` 个百分点，而且这是极小样本；延迟却是 ValueSketch 的 `3.72x`。query-crossfit 更慢约 `8.81x`，也没有进一步改善闭环质量。Wiener shrinkage 在 sports/medicine 32K 的两个 case 上同样没有稳定收益：ValueSketch、普通条件残差、Wiener 条件残差的平均质量分别为 `100.596%/100.261%/100.176%`，延迟为 `128.4/467.2/479.7 ms`。在 Llama religion 4K 的 query-calibrated probe 中还出现过 top-1 一致率降到 `75%`。

原因不是“条件相关性不存在”，而是训练目标与 decode 目标不一致。当前闭式回归拟合的是均匀历史 token 上的

`R_i = V_i - (mu_v + U z_i) ≈ b + A(x_i - mu_x)`，

但 decode 真正需要的是随当前 Query 改变的 softmax 加权量 `sum_i exp(q k_i) R_i`。全局残差 MSE 降低，不能保证当前上尾、未选集合中的加权方向误差降低。query gain 只能缩放修正量，无法修复 selected-tail 分布偏移和方向偏差。当前 prototype 还会显式构造全历史坐标、tail weight 并执行逐 token einsum，因此系统成本过高。

所以结论是：**条件矩具有研究价值，但当前实现不应作为论文主方法。** 若继续研究，门槛不是再调 ridge、维数或 gain，而是推导无需逐 token 展开的 query-weighted block sufficient statistics，并证明它在 selected-tail 条件分布下有效。

该路线中真正有效的发现被保留下来：**不能只找 top token，还要补偿未选 token 的总 Value 贡献。** 当前版本不用更强的高斯条件假设，而是用 rank-16 INT4 ValueSketch 直接累计 softmax 尾部分母和低秩分子。它是条件矩研究的稳健化结果，而不是简单回到只做 QK top-k 的旧 QKSieve。

## 完整执行流程

1. Prefill 后从当前请求的 Query/Key 二阶矩估计 request-local OAS QK-balanced 变换。
2. 将每个 128 维 Key 分成 8 个 16 维 band，在总成本 15 下，用 qMSE 为各 head/band 分配 0/1/2/4/8 bit；启用 band 的 scale 计入成本。
3. 按长度设置精确候选目标：`k(N)=min(N,max(256,min(ceil(0.06N),1280)))`。
4. 令 `p=k/N`，按 `m=align256(ceil(64/p))` 取分层、head-specific 相位样本，且把 `m` 限制在 `[256,8192]`。
5. 对样本代理 QK 分数求目标 order statistic，得到每个 Query head 的阈值。
6. 用低比特 Key 索引一次扫描全部历史；满足阈值的 token 通过 bitmask 和固定前缀扫描写入候选数组。
7. 对候选从 GPU 常驻 FP16 K/V 计算精确 attention。未选中 token 的 Value 贡献由 rank-16、block-256、INT4 ValueSketch 近似，并与候选 softmax 分子/分母合并。
8. 每步只为新 token 增量追加 Key 编码和 ValueSketch 元数据。

## 为什么选择 c64

c32 的 128K decode 是 `3.129x`，c64 是 `3.089x`，速度差只有 `1.3%`。但 c64 把 128K 每 head 的候选范围从 `[654,2291]` 收紧到 `[728,1825]`，并在 Qwen medicine probe 中把 top-1 从 `96.875%` 恢复到 `100%`。论文主方法应优先选择稳定性更强且规则仍然简单统一的 c64；c32 可作为 fast preset 消融。

## 当前最佳结果

在 Llama-3.1-8B-Instruct、held-out mixed-domain 32-token probe 上：

| 长度 | 质量保持 | Top-1 | GPU 实际 token/head | 稳态 decode |
|---:|---:|---:|---:|---:|
| 32K | 100.812% | 100% | 1,276.17 | 1.921x |
| 64K | 101.358% | 100% | 1,276.58 | 2.409x |
| 128K | 100.526% | 100% | 1,278.86 | 3.089x |

辅助索引约占完整 FP16 KV 的 `7.4%`；原始 K/V 仍在 GPU。一次性索引成本约为 `3.2--3.6 s`，32K/64K/128K 分别在约 `76/38/19` 个生成 token 后回本。因此该实现最适合多轮问答、agent 和 prefix/KV 复用后的长 decode，不应声称优化了单次短输出的 prefill 端到端延迟。

## 与上一版本相比

| 项目 | 固定 s1024 sampled 版 | 当前 c64 deterministic 版 |
|---|---|---|
| 分位样本 | 所有长度固定 1,024 | 随 `1/p` 缩放到 1,792/3,328/6,656 |
| 长上下文阈值方差 | 随上尾概率降低而增大 | 目标上尾锚点数保持约 64 |
| 候选压缩 | 全局 atomicAdd | bitmask + 固定前缀扫描 |
| Value 尾项 | atomic 浮点归约 | block partial + 固定树归约 |
| 同输入重复 | 候选顺序/尾项可抖动 | kernel 输出逐 bit 一致 |
| 128K 扫描性能 | v38 deterministic 0.635 ms/层 | v40 0.467 ms/层 |

## 冻结判断

**可以冻结算法原型和 CUDA 数据流，但还不能冻结论文主结果。** 当前方法已经形成简单、可解释、无训练 router 的统一机制，且 32K--128K 的小规模质量与速度信号都通过。下一阶段不应继续围绕 `c=48/56/72` 做参数搜索，而应验证外部有效性：

1. 在不改参数的前提下完成 LongBench、RULER 和长文本 PPL 的独立测试。
2. 至少增加一个不同架构模型和三个 seed，报告均值、方差与最坏任务。
3. 与 FIER、SparQ、BinaryPC 等在相同 GPU、相同 active-token 预算、相同 KV 驻留条件下直接测速。
4. 独立测量 dense attention、代理检索、候选精确 attention、Value 尾项及整个 decode；profile 模式只用于归因，不用于最终加速数字。
5. 单列索引建立成本和 break-even 生成长度，明确 cached-KV/agent 场景是主要部署目标。

只有这些独立实验通过后，才能把“32K--128K 通用”和“优于公开方法”写成论文结论。

## 复现入口

- 方法与公式：`docs/deterministic_sampled_valuesketch_20260804/design.md`
- 实验协议：`docs/deterministic_sampled_valuesketch_20260804/experiment_design.md`
- 数值结果：`docs/deterministic_sampled_valuesketch_20260804/visualization_results.md`
- CUDA：`src/mixedblock_spectral_cuda_20260729.py`
- 运行时：`src/run_head_top2_targeted_ppl_20260714.py`
- 长度 probe：`src/run_qksieve_coldskip_longcontext_quality_20260730.py`
- 正确性脚本：`src/validate_qksieve_valuesketch_deterministic_20260804.py`
- 最终 launcher：`scripts/launch_qksieve_v40_c64_final_profile_5gpu_20260804.sh`
