# Pre-RoPE 召回后把证据移到 Query 附近：RULER-32K 实验

## 结论

朴素的虚拟位置修复没有提升，完整近移反而明显退化。最佳保守档 $\alpha=0.05$ 在未参与调参的 24 条样本上，相对原 pre-RoPE 召回方法的样本平均 RULER 分数从 85.00% 降至 81.25%；证据召回几乎不变，但证据 attention mass 下降，而且位置变换平均提高 non-gold QK、降低 gold QK。

因此，问题不是“证据已经召回，只要把它搬近就一定更容易消费”。更准确的结论是：RoPE 近移同样作用于大量错误候选，且各频率相位随位移振荡；统一搬移不能保证真实证据获得相对优势。

## 方法

先用 pre-RoPE 语义分数，以每层每 head 2% budget 召回远程 token；sink 16 与最近 128 token 保持原位。远程候选按原位置排序并压到 Query 的局部窗口之前，得到虚拟位置 $\tilde p$。只改变 K 的 RoPE 相位，不改变 V：

$$
K_{p\rightarrow\tilde p}^{\mathrm{repair}}
=R(\tilde p)R(p)^{-1}K_p^{\mathrm{post}}.
$$

连续插值为：

$$
p_{\mathrm{eff}}=p+\alpha(\tilde p-p),
$$

$$
s_\alpha(t,p)
=\frac{q_t^\top R(p_{\mathrm{eff}}-t)k_p}{\sqrt d}.
$$

$\alpha=0$ 等于原 `local_global_postscore`；$\alpha=1$ 表示完整近移。每个解码步动态重设虚拟位置。

## 实验协议

- 模型：Qwen3-8B，NF4 权重、BF16 计算。
- 数据：13 个 RULER 官方任务，每任务 2 条，目标长度 32K；实际 prompt 为 28,768--32,656 token。
- 对比：Full、exact post-RoPE Top-2%、pre-RoPE 召回 + native post-RoPE 消费、pre-RoPE 召回 + 虚拟位置修复。
- 冒烟调参：1 条多值样例和 1 条 UUID 风险样例；扫描 $\alpha\in\{0,.02,.05,.10,.15,.25,.50,.75,1\}$。
- 冻结 $\alpha=0.05$ 后运行全部 26 条；主分析排除两条调参样例，剩余 24 条。
- 输出以完整 RULER 官方评分为主，不用首 token 正确代替完整数字或 UUID 正确。
- 26 条位置修复结果均无 2% budget 违规和重复位置违规。

## 主要结果

### Held-out 24 条

| 方法 | 样本平均分 | 13 任务宏平均 | 首答案 NLL | NIAH 证据 recall | NIAH 证据 mass | Query 时间 |
|---|---:|---:|---:|---:|---:|---:|
| Full | 85.00% | 86.15% | 2.983 | -- | -- | 0.105 s |
| exact Top-2% | 83.19% | 84.49% | 2.946 | 23.804% | 1.266% | 0.269 s |
| pre-RoPE recall, native consume | 85.00% | 86.15% | 2.796 | 23.991% | 1.332% | 0.298 s |
| + 5% virtual-position repair | 81.25% | 82.69% | 2.850 | 23.990% | 1.283% | 0.517 s |

5% 修复相对原 pre-RoPE 方法的配对分数差为 $-3.75$ 点，bootstrap 95% CI 为 $[-12.50,+1.25]$。区间包含 0，所以小样本下不能声称显著下降；但它显然没有提供提升证据。相对 exact Top-2% 的差为 $-1.94$ 点，95% CI 为 $[-11.67,+5.00]$。

首答案 NLL 从 2.796 升到 2.850，对应几何 PPL 从 16.38 升到 17.28。Query 时间增加约 73%；生成时间还受输出长度变化影响，不作为纯算子开销。

### 真正移到 Query 附近时

$\alpha=1$ 在两条冒烟样例中把远程候选的平均有效距离压到约 382 token，但：

| 样例 | $\alpha=0$ | $\alpha=1$ | gold mass 变化 | gold / non-gold QK 增量 |
|---|---:|---:|---:|---:|
| UUID multikey | 1.00 | 0.00 | 0.836% → 0.167% | $-1.758 / +3.604$ |
| multivalue | 1.00 | 0.75 | 1.925% → 1.821% | $+0.783 / +3.641$ |

即使 multivalue 的 gold QK 上升，non-gold 上升得更多，真实证据的相对优势仍下降。

### 5% 修复的内部机制

在 held-out NIAH 上：

- gold QK 平均变化：$-0.039$；
- non-gold QK 平均变化：$+0.100$；
- gold-minus-non-gold：$-0.139$；
- 14 条中 9 条更偏向 non-gold，8 条的 gold QK 直接下降；
- attention mass 从 1.332% 降至 1.283%，而 recall 保持约 23.99%。

这条链条最符合数据：

$$
\text{位置插值}
\rightarrow
\text{多频率相位非单调变化}
\rightarrow
\Delta s_{\mathrm{non\text{-}gold}} > \Delta s_{\mathrm{gold}}
\rightarrow
\text{gold 相对 softmax 权重下降}
\rightarrow
\text{精确多 token 输出可能失败}.
$$

代表性反例是 `niah_multikey_2_32768_1`：Full、exact Top-2% 和原 pre-RoPE 方法都输出正确 `6999379`，5% 修复输出 `6707911`。其首 token NLL 仍只有 0.002，因为两个答案共享开头 token，说明首 token 指标不能替代完整答案。

## 为什么 5% 仍不是“小扰动”

在 held-out 样本中，5% 插值平均移动 731 个位置，修复后的平均距离仍为 14.27K。Qwen3-8B 的最高频 RoPE 对约以 1 rad/token 旋转，数百 token 的位移已经跨过很多周期。因此全频率共用一个 $\alpha$ 会产生强烈振荡，不能把 $\alpha=0.05$ 理解为相位只变化 5%。

## 下一步

不应继续扫描单一全局 $\alpha$。更合理的下一版是：

1. 以连续证据块为单位平移并保留块内相对位置，而不是逐 token 压紧；
2. 对每个 RoPE 频率限制最大相位改变量，高频只允许很小修复；
3. 只对语义置信度高且预测能提高 gold-vs-background margin 的 head/候选开启门控；
4. 保持远程候选的总 log-sum-exp，避免位置修复整体放大 softmax 分母；
5. 若仍不稳定，转向训练期适配，因为原模型未学习在推理时消费被重新定位的 K。

## 产物

- 数值汇总：`../outputs/analysis/summary.json`
- 样本级数据：`../outputs/analysis/all_rows.csv`
- 任务分数：`../outputs/analysis/heldout_task_scores.csv`
- 图：`../outputs/analysis/smoke_alpha_sweep.png`、`smoke_mass_qk.png`、`heldout_task_scores.png`、`heldout_niah_qk_delta.png`
