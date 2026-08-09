# 关键位置动态预算：Evidence-Sufficiency Budgeting

日期：2026-07-15

## 1. 研究问题

新的核心假设是：不同生成位置需要的历史证据量不同。预测高频、局部可确定的 token 时可以使用很低预算；预测依赖远程事实、数字或专有实体的 token 时应提高预算。

直接按目标 token 的词频或类型分配预算并不可靠。此前 1536 个目标 token 的逐 token 探针显示：未在历史中出现的 token 通常更难，但数字 token 并没有稳定表现出更高的预算收益；`and` 的不同上下文也同时存在“几乎不需要历史”和“明显受益于高预算”两类情况。因此，预算应由上下文中的证据充分度决定，而不是由词本身决定。

## 2. 已排除的路径

### 2.1 标量风险路由

使用低预算 logits、置信度、margin、entropy、token 频率、token 类型、topic 和 attention 统计训练 ExtraTrees 路由。训练使用 window 0/1，window 2 完全独立测试。

20% 高预算校准点在独立测试上的结果为：

| 方法 | committed links | executed links | PPL |
|---|---:|---:|---:|
| full attention | 100% | 100% | 14.001 |
| fixed 2% | 2.000% | 2.000% | 14.400 |
| scalar two-stage router | 1.721% | 1.961% | 约 14.50 |

该路由能降低平均预算，但没有稳定支配 fixed 2%。两阶段重算还增加了 executed links。

### 2.2 隐藏状态路由

从低 1% 前向收集 4096 维 hidden state，测试 Ridge、PCA + ExtraTrees、PCA + GBDT。独立 window-2 上最好的 AUC 仍约为 0.66，与标量路由相当，PPL/预算前沿没有实质改善。

结论：继续增加 token-level router 特征并不是当前最有效的方向。

## 3. 新方法：逐 head 证据充分度预算

工作名：**Evidence-Sufficiency Budgeting（ESB）**。

对位置 \(t\)、层 \(l\)、query head \(h\)，记历史 key 的 attention score 为

\[
s_i = \frac{q_{t,l,h}^{\mathsf T} k_{i,l,h}}{\sqrt d},\qquad i < t.
\]

给定预算阶梯

\[
\mathcal B=\{0.25\%, 0.5\%, 1\%, 2\%, 4\%\},
\]

对每个预算 \(b\)，取 score 最大的 \(\lceil b(t-1)\rceil\) 个历史位置组成 \(S_b\)，并始终保留当前 token。定义保留的 full-attention mass：

\[
M_b =
\frac{\sum_{i\in S_b\cup\{t\}}\exp(s_i)}
{\sum_{i\le t}\exp(s_i)}.
\]

选择满足阈值的最小预算：

\[
b^*_{t,l,h}=\min\{b\in\mathcal B:M_b\ge\tau\}.
\]

如果 4% 仍不满足阈值，则使用 4%。不同位置、层和 head 可以选择不同预算。与“预测下一个 token 是否困难”的标量路由相比，该规则直接回答当前 head 是否已经收集到足够证据。

当前实现只在一次模型前向内完成稀疏 softmax/V 聚合，没有 low/high 两次模型重算。但它为了精确计算 \(M_b\) 仍执行 full QK，因此是机制验证版本，而不是最终部署版本。

## 4. 32K 体育/医学定向 PPL 结果

模型：Llama-3.1-8B-Instruct。数据为 sports/medicine 各 3 个不重叠 32K 窗口，每个窗口评估 256 个严格因果 continuation token，共 1536 token。

### 4.1 全部六个窗口

| 方法 | Attention links | PPL | PPL / full | Attention-V 上界 |
|---|---:|---:|---:|---:|
| full attention | 100.000% | 8.394 | 1.0000 | 1.00x |
| fixed 1% | 1.000% | 8.750 | 1.0424 | 100.00x |
| fixed 2% | 2.000% | 8.605 | 1.0252 | 50.00x |
| fixed 4% | 4.000% | 8.520 | 1.0151 | 25.00x |
| **ESB, tau=0.75（压缩点）** | **0.828%** | **8.468** | **1.0088** | **120.77x** |
| ESB, tau=0.80 | 1.012% | 8.449 | 1.0066 | 98.86x |
| **ESB, tau=0.85（质量点）** | **1.277%** | **8.433** | **1.0047** | **78.33x** |
| ESB, tau=0.875 | 1.457% | 8.436 | 1.0050 | 68.62x |
| **ESB, tau=0.90** | **1.689%** | **8.439** | **1.0054** | **59.22x** |
| ESB, tau=0.95 | 2.443% | 8.460 | 1.0079 | 40.94x |
| ESB, tau=0.97 | 2.981% | 8.485 | 1.0108 | 33.54x |
| ESB, tau=0.98 | 3.349% | 8.499 | 1.0125 | 29.86x |
| ESB, tau=0.99 | 3.758% | 8.517 | 1.0146 | 26.61x |

`tau=0.75` 在低于固定 1% 的平均 links 下，把 PPL 损失从 fixed 1% 的 4.24% 降到 0.88%。`tau=0.85` 则以 1.277% links 将 PPL 损失进一步降到 0.47%。`tau=0.90` 的预算分配为：42.73% 的 layer/head/query 使用 0.25%，6.96% 使用 0.5%，8.78% 使用 1%，10.19% 使用 2%，31.35% 使用 4%。这说明统一预算浪费严重，少数困难 head 需要高预算，大量 head 可以激进压缩。

### 4.2 独立 window-2

阈值扫描之前没有用 window-2 训练任何 router。独立窗口结果为：

| 方法 | Attention links | PPL | PPL / full |
|---|---:|---:|---:|
| full attention | 100.000% | 14.001 | 1.0000 |
| fixed 1% | 1.000% | 14.689 | 1.0492 |
| fixed 2% | 2.000% | 14.400 | 1.0285 |
| fixed 4% | 4.000% | 14.252 | 1.0179 |
| **ESB, tau=0.75** | **0.800%** | **14.100** | **1.0071** |
| ESB, tau=0.80 | 0.980% | 14.049 | 1.0035 |
| **ESB, tau=0.85** | **1.241%** | **14.031** | **1.0021** |
| **ESB, tau=0.90** | **1.649%** | **14.062** | **1.0044** |

因此，该现象并非只存在于用于观察的 window 0/1。

### 4.3 去除 full partition：tail-corrected sampling 诊断

精确 ESB 需要 full softmax denominator。为去掉该依赖，测试“高分候选精确计算 + 未检索部分均匀抽样估计”。当前诊断仍用 exact top-k 作为候选，仅验证 denominator estimator；每层每 head 额外打分 0.25% 固定历史样本。

| 估计阈值 | 候选 links | 候选+抽样打分 | 平均真实 mass | P10 真实 mass | 覆盖率 |
|---|---:|---:|---:|---:|---:|
| 0.75 | 0.55% | 0.80% | 0.886 | 0.783 | 90.85% |
| 0.80 | 0.66% | 0.91% | 0.893 | 0.800 | 87.52% |
| 0.85 | 0.83% | 1.08% | 0.903 | 0.823 | 81.93% |
| 0.90 | 1.17% | 1.42% | 0.916 | 0.850 | 74.17% |

对应的 total log-partition MAE 仅为 0.030--0.045，明显优于 block centroid/variance 估计。

完整逐 token PPL 结果如下。这里的候选仍由 exact top-k 产生，但预算判定只看候选分数和 0.25% 固定残差样本，不再使用 full denominator。

| 方法 | Candidate links | 额外残差 sample | PPL | PPL / full |
|---|---:|---:|---:|---:|
| sampled-tail, tau=0.75 | 0.832% | 0.25% | 8.450 | 1.0067 |
| sampled-tail, tau=0.80 | 1.008% | 0.25% | 8.438 | 1.0053 |
| sampled-tail, tau=0.85 | 1.259% | 0.25% | 8.436 | 1.0051 |
| sampled-tail, tau=0.90 | 1.650% | 0.25% | 8.429 | 1.0042 |

独立 window-2 上，`tau=0.75` 为 0.800% candidate links、PPL/full=1.0092；`tau=0.85` 为 1.215% links、PPL/full=1.0036。说明 sampled-tail estimator 的误差没有在 256-step 自回归轨迹上造成明显累积。

### 4.4 无 full-QK 候选路径的短测

进一步接入 QAbs partial-QK 候选器，在 sports window-0 的前 64 个 target token 上测试。相同 token 的 full PPL 为 8.628，exact ESB `tau=0.75` 为 8.910。

| 候选器 | Candidate links | PPL | PPL / full | 近似 QK 打分比例 |
|---|---:|---:|---:|---:|
| QAbs8, candidate 7% | 1.225% | 9.437 | 1.0938 | 约 13.5% |
| QAbs16, candidate 7% | 1.089% | 9.062 | 1.0503 | 约 19.8% |
| QAbs8, candidate 15% | 1.102% | 9.121 | 1.0571 | 约 21.5% |

QAbs16/candidate-7% 已接近 95% 质量，但只验证了单个 64-token 短样本；QAbs8 明显不够。增加 query 通道比单纯扩大候选集合更有效，说明候选误差主要来自 partial-score 排序，而不是 rerank pool 太小。

随后完成 QAbs16/candidate-7% 的全部六窗口、每窗口 256-token 评估：

| Split | Final attention links | PPL | PPL / full | 等价 QK 点积比例 |
|---|---:|---:|---:|---:|
| calibration window-0/1 | 0.881% | 6.553 | 1.0083 | 约 19.75% |
| independent window-2 | 0.833% | 14.166 | 1.0118 | 约 19.75% |
| **all** | **0.865%** | **8.474** | **1.0095** | **约 19.75%** |

等价 QK 比例由 16/128 维全历史 partial scan、7% candidate full rerank 和 0.25% residual sample 相加得到，对应约 5.06x 的理想 QK 乘加减少。该结果已经不计算 full QK，但仍保存完整 K/V。当前纯 PyTorch 原型平均 online 时间约 92 秒/样本，full attention 约 34 秒/样本，因此还没有 wall-clock 加速证据；主要开销是通道 gather、多个 top-k、候选索引构造和非融合 sparse attention。

## 5. 新发现

1. **关键性是 layer/head/query 级属性。** 仅使用一个全局 position budget 会把真正的差异平均掉。
2. **词频只是弱先验。** 罕见词通常更难，但同一个高频词在不同上下文中的预算需求也会显著变化。
3. **预算与端到端 PPL不严格单调。** 将阈值从 0.90 提高到 0.99，links 增加但 PPL 略差。局部 attention 更接近 full 并不保证多层自回归轨迹上的 token NLL 单调改善。
4. **两阶段 token router 不是必需的。** 直接以证据充分度控制每个 head 的预算，当前质量/预算前沿明显更好。
5. **高分尾部必须单独处理。** block centroid 即使加入投影方差，在 block=32、估计阈值 0.90 时也只保留约 0.80 真实 mass；高分候选精确计算后再抽样残差，partition 估计才足够准确。
6. **预算估计问题已基本解耦，候选召回成为主瓶颈。** sampled-tail 可以替代 full denominator；QAbs8 失败而 QAbs16 成功，说明候选索引必须显式满足高召回约束。
7. **首个无 full-QK 质量点已经成立。** QAbs16 sampled-tail 在 0.865% final links 下把六窗口 PPL 控制在 full 的 1.0095 倍，但 kernel 与物理 KV 管理尚未兑现。

## 6. 不能过度宣称的部分

- 1.689% 是平均 attention link ratio，不是已经兑现的物理 KV 存储比例。
- 精确 ESB 实验保留完整 K/V，并计算 full QK 后才知道精确 attention mass。
- sampled-tail 完整 PPL 虽不使用 full denominator，但仍用 full-QK exact top-k 生成候选；它只解决预算判定，不等于完整部署系统。
- QAbs16 sampled-tail 已去掉 full QK，但仍保存 full K/V，并且当前 Python 实现慢于 full attention。
- 当前 full-QK 诊断实现平均 online 时间约 62 秒/样本，full attention 约 34 秒/样本；它目前没有实际 wall-clock 加速。
- 59.22x 只是稀疏 softmax/V 聚合的理论 links 上界，不能写成端到端加速。
- 目前只有一个模型、两个主题和 1536 个评估 token，尚不足以支撑通用性结论。

## 7. 下一步：把 full-QK 证据充分度变成可部署证书

应与已有 block retrieval 合并，而不是再训练一个黑盒 token router。

对每个 block \(b\) 保存 key centroid \(\mu_b\)、半径 \(r_b\) 和 token 数 \(n_b\)。利用 Cauchy-Schwarz：

\[
q^\mathsf{T}\mu_b-\lVert q\rVert r_b
\le q^\mathsf{T}k_i
\le q^\mathsf{T}\mu_b+\lVert q\rVert r_b,\qquad i\in b.
\]

可以构造每个 block 对 softmax partition 的上下界：

\[
L_b=n_b\exp(q^\mathsf{T}\mu_b-\lVert q\rVert r_b),\qquad
U_b=n_b\exp(q^\mathsf{T}\mu_b+\lVert q\rVert r_b).
\]

按 block 上界/中心分数逐步展开候选，并计算保守的已选 mass 下界：

\[
\underline M(S)=
\frac{\sum_{b\in S}L_b}
{\sum_{b\in S}L_b+\sum_{b\notin S}U_b}.
\]

当 \(\underline M(S)\ge\tau\) 时停止检索，否则继续展开 block。这样可把当前精确但昂贵的规则改造成分层、可提前停止、带安全证书的动态预算系统，同时自然利用现有 block-size-aware retrieval。

## 8. 代码与结果

- 固定预算/attention mass 实现：`src/run_head_top2_targeted_ppl_20260714.py`
- 多预算逐 token 探针：`src/run_critical_position_budget_probe_20260715.py`
- 标量动态路由：`src/run_dynamic_critical_position_ppl_20260715.py`
- ESB 评估：`src/run_adaptive_mass_budget_ppl_20260715.py`
- block partition / sampled-tail 诊断：`src/run_block_partition_estimator_probe_20260715.py`
- ESB 启动脚本：`scripts/launch_adaptive_mass_budget_ppl_20260715.sh`
- 统一汇总：`scripts/summarize_adaptive_mass_frontier_20260715.py`
- 结果：`results/20260715_adaptive_mass_frontier/frontier.csv`
