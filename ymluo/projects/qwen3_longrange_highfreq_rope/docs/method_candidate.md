# 方法候选：Layer–Spectral RoPE Gating

## 当前最小有效版本

对第 $l$ 层、第 $i$ 个 RoPE 二维频率对定义相位速度 $\lambda_{l,i}$：

$$
R_{l,i}(p)=R\!\left(\lambda_{l,i}p\omega_i\right).
$$

原生 RoPE 为 $\lambda_{l,i}=1$，NoPE 为 $\lambda_{l,i}=0$。当前 Qwen3-8B、RULER-32K 的两个 Pareto 候选是：

- 准确率优先：L30–L35 的 F0–F15 设为 0；其余保持 1。
- Gold NLL 优先：L24–L35 的 F0–F11 设为 0；其余保持 1。

这两个版本不增加参数，也不增加推理 FLOPs；只改变指定维度的旋转角。

## 为什么固定 mask 还不够形成 ICLR 方法

部分层 NoPE 已有 RNoPE，部分维度 RoPE/NoPE 已有 Partial RoPE 和 RoPE-ID，按频率学习也已有 LeRoPE。一个手工的“晚层 × 高频”mask 是有用发现，但容易被评价为已有两类方法的直接组合。

## 建议的论文方法

在公开 checkpoint 上学习层–频率门控，而不是从头训练统一频率：

$$
\lambda_{l,i}=1-m_{l,i},\qquad 0\le m_{l,i}\le1.
$$

$m_{l,i}$ 表示该层该频率对的去位置化程度。训练目标为：

$$
\mathcal L
=
\mathcal L_{\text{long}}
+\beta\,\mathrm{KL}\!\left(
P_{\text{native}}^{\text{short}}
\,\|\,
P_{m}^{\text{short}}
\right)
+\gamma\,\Omega(m).
$$

- $\mathcal L_{\text{long}}$：普通长文本 next-token loss，可混入少量合成长程检索数据。
- KL 项：让改造模型在短上下文上逼近原生 checkpoint，避免破坏局部顺序、知识和推理。
- $\Omega(m)$：结构化正则，鼓励门控随层深增大、随频率降低而减小，并形成连续频带。

可使用 hard-concrete/straight-through gate，使训练结束后的 $m_{l,i}$ 二值化。这样最终模型仍是普通 RoPE/NoPE 旋转，不增加解码计算。

## 与最接近工作的差异

| 方法 | 层维度 | 频率维度 | checkpoint 后训练 | 短程保持约束 |
|---|---|---|---|---|
| RNoPE | 选择整层 | 全维统一 | 可 | 无显式约束 |
| Partial RoPE / RoPE-ID | 全层统一 | 选择部分维度 | 主要从头训练 | 无 |
| LeRoPE | 全层共享 | 每频率可学习 | 从头训练验证 | 无 |
| 本候选 | 每层可不同 | 连续频带/结构门控 | 是 | 显式蒸馏 |

真正的贡献应是“机制诊断 → 结构化门控 → checkpoint retrofit → 长短能力共同保持”的完整闭环，而不是 mask 本身。

## 进入正式 mid-training 前的门槛

1. 在更多 RULER seeds、8K/16K/32K/64K 上复现层–频带安全边界。
2. 在至少另一个 4B–8B checkpoint 上复现。
3. 完成 LongBench、扩大 WikiText/PG19 PPL，并补 MMLU/GSM8K。
4. 与 RNoPE、Partial RoPE、RoPE-ID、LeRoPE 风格频率缩放做同预算比较。
5. gate-only 训练先用 1M–10M token 验证梯度、吞吐和门控是否从原生位置形成连续晚层频带；通过后再做 100M–500M token 方法筛选。

在这些门槛完成前，当前状态是“有强实证信号和明确方法方向”，还不是可直接投稿的最终算法。
