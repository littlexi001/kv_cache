# RULER-32K 虚拟 RoPE 位置修复

## 可证伪问题

在保持 pre-RoPE selector、2% budget 和初始输入相同的情况下，把被召回的远程 Key 从原始 RoPE 位置重新旋转到 Query 附近，是否能提高证据 attention mass、完整答案质量和 RULER 官方分数？

这是一个端到端方法实验，不是假设所有后续层的支持集被强制冻结：第一层位置修复改变 residual 后，后续层 Query 与实际召回位置可以随之变化。这种反馈属于方法本身的效果；因此累计 selection hash 只用于识别轨迹，不要求不同 alpha 完全一致，也不把结果解释为“固定支持集下的单层纯中介效应”。

若位置修复只提高正确答案第一个 token 的信心，却损害 UUID 等多 token 实体的后续生成，则该操作不成立。

## 物理先验与数学模型

原始 Key 和 Query 为：

$$
K_p^{\mathrm{post}}=R(p)k_p,
\qquad
Q_t^{\mathrm{post}}=R(t)q_t.
$$

原生分数使用远程相对位置：

$$
s_{\mathrm{native}}(t,p)
=
\frac{q_t^\top R(p-t)k_p}{\sqrt d}.
$$

为远程 token 指定 Query 附近的虚拟位置 $\tilde p$，不移动 Value，只重新旋转 Key：

$$
K_{p\rightarrow\tilde p}^{\mathrm{repair}}
=
R(\tilde p)R(p)^{-1}K_p^{\mathrm{post}}.
$$

连续修复强度为：

$$
p_{\mathrm{eff}}
=
p+\alpha(\tilde p-p),
\qquad \alpha\in[0,1].
$$

因此消费分数为：

$$
s_\alpha(t,p)
=
\frac{q_t^\top R(p_{\mathrm{eff}}-t)k_p}{\sqrt d}.
$$

$\alpha=0$ 必须等价于当前 `local_global_postscore`；$\alpha=1$ 表示完全移动到虚拟位置。

## 精确算法

### 输入

- Qwen3-8B 的一条 RULER-32K prompt。
- 每层缓存中的 post-RoPE K 和原始 V。
- 当前 token 的 pre/post-RoPE Query。
- 固定 2% per-head budget。

### 参数

| 参数 | 值 | 含义 | 太小/太大的影响 |
|---|---:|---|---|
| `ratio` | 0.02 | 每 head 支持集比例 | 太小丢证据；太大失去 2% 研究问题 |
| `sink_tokens` | 16 | 保留原位置的开头 token | 太小可能丢起始锚点；太大挤占远程预算 |
| `local_window` | 128 | 保留原位置的最近 token | 太小破坏格式；太大挤占远程预算 |
| `alpha` | 0, .25, .5, .75, 1 | 位置移动比例 | 0 不修复；1 最强也最可能越出训练分布 |

### 中间变量

- `selected[h]`：第 h 个 head 由同一个 pre-RoPE selector 选出的支持位置。
- `remote_mask[h]`：支持集中哪些位置属于远程语义召回。
- `virtual[h]`：远程 token 的近程虚拟位置。
- `effective[h]`：按 alpha 插值得到的有效位置。
- `score_delta`：修复分数减原生 post-RoPE 分数。

### 步骤

1. 每个 head 强制保留 16 个 sink、最近 128 个 token 和当前 token。
2. 剩余预算按 pre-RoPE QK 选择远程 token。
3. 对每个 head，将远程 token 按原始位置升序排列。
4. 将这些远程 token 连续映射到最近窗口之前，即虚拟相对距离约为 `129 .. 128 + remote_count`；sink、local、current 的虚拟位置等于原位置。
5. 计算 `effective = original + alpha * (virtual-original)`。
6. 对选中 Key 执行 `R(effective)R(original)^(-1)K_post`。
7. 使用原生 post-RoPE Query、修复后的 Key 和原始 V，在相同支持集上计算 sparse softmax。
8. 每个生成步都相对当前 Query 重新锚定虚拟位置；这是本轮明确测试的动态位置修复。

### 不进入算法的信息

- gold answer 和 gold evidence 不参与召回、虚拟位置或 alpha 决策。
- 不修改 Value。
- 不重写前缀 hidden state，也不重新 prefill。
- 不扩展证据块；这样保证各 alpha 的支持集完全相同。

## 通过、失败与调试产物

通过条件：

- alpha=0 数值测试与原生 post-RoPE 分数一致。
- 所有 alpha 的首 Query 支持集 hash 完全一致。
- 2% budget 和位置去重错误率均为 0。
- 相比 alpha=0，完整 RULER score 提升且多 token 风险样例不退化。

失败条件：

- `support_changed`：不同 alpha 的支持集 hash 不同。
- `rotation_reconstruction_error`：delta 旋转与直接在新位置旋转不一致。
- `output_regression`：任务分数下降，尤其 UUID 后缀损坏。
- `mass_without_quality`：证据 mass 上升但完整答案分数不升。
- `insufficient_evidence`：点估计变化但区间跨 0。

逐样本原始预测、support hash、位置移动距离、gold/non-gold QK 变化、证据 mass、官方 score 和耗时均保存到 `outputs/`。
