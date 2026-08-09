# 实验协议

## 数据与模型

- 模型：Qwen3-8B，BF16，固定 YaRN RoPE factor 4.0。
- 当前失败长度：143,488 tokens。
- 当前失败位置之前建立历史表征库。候选偏移为 $\{1,2,4,8,16,32,48,64,96,128,192,256,384,512\}$；它是覆盖 RoPE 几何频率尺度的对数网格，并在已知 64-token 转折附近加密，不把任意一个偏移预设成最优。
- 证据、问题、filler 和 tokenizer 与既有单跳 `nine` 边界实验完全一致。
- 服务器资源：仅使用 GPU 6–7。

## 干预位置

第一阶段扫描层：L20、L21、L22、L23、L24。残差融合另保留 L16 作为此前替换实验的早期正对照。

融合比例：

$$
\alpha\in\{0.25,0.50,0.75,1.00\}.
$$

其中 $\alpha=0$ 由未干预失败 baseline 给出；$\alpha=1$ 是历史表征端点。

## 历史锚点选择

对层 $l$、历史偏移 $\delta$，先计算所有 Query head 的原生 post-RoPE 表征：

$$
u_{l,\delta}=R(t-\delta)q^{\mathrm{pre}}_{l,t-\delta}.
$$

以当前 $u_{l,0}$ 为起点，用 cosine distance 做最远点采样：第一个锚点离当前最远；之后每个锚点最大化它到当前与已选锚点集合的最小距离。分别得到 $K=1,2,4$ 的历史相位覆盖集。该选择不读取正确答案或 gold Key。

同时保留固定偏移 64 作为已知正确历史点的对照。于是可以回答两个问题：固定历史点是否有效，以及“覆盖不同相位”是否比拍脑袋选间隔更有效。

## 三个主实验

### A. 残差流融合

只修改最终 Query token 在第 $l$ 层输入处的 hidden state：

$$
\widetilde h_l=(1-\alpha)h_l^{\mathrm{now}}+\alpha h_l^{\mathrm{old}}.
$$

随后从第 $l$ 层继续正常前向。该实验回答历史中间状态能否恢复输出，但不能单独归因于 RoPE 相位。

### B. pre-RoPE Query 融合并使用当前位置相位

$$
\widetilde q_l^{\mathrm{pre}}
=\operatorname{NormMatch}\!\left(
(1-\alpha)q_l^{\mathrm{pre,now}}+
\alpha q_l^{\mathrm{pre,old}}
\right),
$$

$$
\widetilde q_l^{\mathrm{post}}=R(t_{\mathrm{now}})\widetilde q_l^{\mathrm{pre}}.
$$

这保留历史 Query 的内容方向，但不给它保留历史相位。

### C. 保留历史原生相位的 post-RoPE Query 融合

$$
q_l^{\mathrm{post,now}}=R(t_{\mathrm{now}})q_l^{\mathrm{pre,now}},
$$

$$
q_{l,j}^{\mathrm{post,old}}=R(t_j)q_{l,j}^{\mathrm{pre,old}},
$$

$$
\widetilde q_l^{\mathrm{post}}
=\operatorname{NormMatch}\!\left(
(1-\alpha)q_l^{\mathrm{post,now}}+
\alpha\sum_{j\in\mathcal S_K}\beta_j q_{l,j}^{\mathrm{post,old}}
\right).
$$

实现时先计算 $R(t_{\mathrm{now}})^{-1}\widetilde q_l^{\mathrm{post}}$，在 `q_norm` 后注入，再由模型正常施加当前位置 RoPE。这样实际进入 QK 内积的正是上式的混合 post-RoPE Query。

第一阶段令 $\beta_j=1/K$。`NormMatch` 逐 head 把混合向量范数缩放到当前 Query 的范数，避免“向量相消导致范数变小”混入相位结论。

对 A、B、C 都比较 $K=1,2,4$，因此锚点数量由实测决定，而不是预设“融合越多越好”。过多相位可能互相抵消，这是必须允许实验否定的情况。

## 第二阶段控制

仅在第一阶段最佳层、锚点数和比例附近运行：

- 历史 post-RoPE Query 的二维频率对施加随机相位，但保持每对振幅不变；
- 残差差向量做等范数维度置换；
- 用错误历史长度（若扫描中存在）替代正确历史长度。
- 把两个最佳单层按网络顺序同时干预，与最佳单层比较；只有该串联实验继续提升，才采用多层融合。

## 每次前向保存的指标

- $P(\texttt{nine})$；
- $z_{\texttt{nine}}-z_{\texttt{newline}}$；
- 首 token 及最强竞争 token；
- 29 个既定关键 head 的加权 evidence QK；
- 29 个关键 head 的加权 evidence attention mass；
- 相对当前 baseline 和历史端点的恢复比例；
- 运行时间、模型与 RoPE 配置。

## 机制判定

设同层同 $\alpha$ 下，C 与 B 的输出 margin 差为

$$
\Delta_{\mathrm{phase}}
=\operatorname{margin}_{C}-\operatorname{margin}_{B}.
$$

只有当 C 在多个相邻 $\alpha$ 上同时提高 margin、evidence QK 和 attention mass，且相位打乱控制不能复现，才把结果记为“支持相位互补”。
