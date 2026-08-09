# KVQ-R 最小边诊断协议（Qwen3-8B）

**状态：** runner、CPU tests 和 GPU6/7 launcher 已实现；本阶段不接 sparse consumer，也不把 AUROC 改善写成端到端收益。

## 1. 唯一研究问题

在冻结 Qwen3-8B 原生 KV cache 的条件下，把候选源 block $B$ 的原始 Value 按 final-query 的 pre-RoPE 相关性聚合，并经对应 Query heads 的 $W_O$ 写入残差方向后，下一层的 pre-RoPE Query 是否会朝候选终点 block $C$ 的 Key 方向移动？

换言之，我们只检验：

> 不使用答案、gold span 或梯度的 $K\rightarrow V\rightarrow Q\rightarrow K$ relay score，能否区分真实两跳边和结构/难度匹配的干扰边？

本实验不修改 RoPE、attention logits、K/V 或模型输出，也不报告 sparse PPL/accuracy。

## 2. 与 sequential-query 既有工作的边界

[Guo et al., *How Do LLMs Perform Two-Hop Reasoning in Context?*, arXiv:2502.13913](https://arxiv.org/abs/2502.13913) 已在合成两跳任务中分析 sequential-query mechanism：早期检索前项和桥接概念，随后利用它们推断终点。

因此，本实验**不主张首次发现 sequential-query mechanism**。可主张的范围仅限于：冻结的预训练 Qwen3-8B、真实原生 KV cache、无 answer direction/gradient 的 layer-boundary relay edge，以及该 edge 是否可作为后续 sparse support closure 的候选信号。

即使 AUROC 很高，也只能说明这个 frozen-KV edge diagnostic 有预测性；它不能单独证明自然前向严格按该边执行两跳推理。

## 3. 数据、候选与标签隔离

- 复用 `run_length_causal_mechanism_20260717.py` 的受控两跳数据；默认 `mixed`、`prefix`、`english_single_token` 和 8K/16K/32K。
- block 只由可见换行 token 与 `max_block_tokens=64` 划分，不读取 `RuleEvent`。
- final-query pre-RoPE QK 只在固定四个 source layers 上聚合并选 Top-64；gold block 不强制加入。
- proposal、全部主分数、controls 和跨层聚合冻结之后，`events` 才进入 `_edge_rows_for_case`：positive 为 relevant step-0 $\rightarrow$ relevant step-1；negative 优先使用完整 conflict/competitor chain，再以 pre-score 和距离匹配补足。
- `score_case_label_free` 的函数签名不接受 case、events、gold、answer 或 loss/answer gradient。

因此，gold 两跳候选覆盖率必须与 edge AUROC 分开汇报；gold 未召回的 case 只影响 coverage，不得以“只有 negatives”的形式污染 edge AUROC。

## 4. 计算定义

固定四个 source layers：

$$
\mathcal L=\left\{\left\lfloor\frac L4\right\rfloor,
\left\lfloor\frac L2\right\rfloor,
\left\lfloor\frac{3L}4\right\rfloor,L-2\right\}.
$$

对每个 $l\in\mathcal L$，捕获 $l$ 与 $l+1$ 的 final-query pre-RoPE Q、prefix pre-RoPE K；原始 V 从未修改的 native cache 读取。$g(h)$ 表示 Query head $h$ 对应的 GQA KV head。

### 4.1 Label-free block proposal

$$
s_{li}^{h}=\frac{(q_l^h)^\top k_{li}^{g(h)}}{\sqrt d},
$$

$$
r_l^h(B)=\tau_b\log\sum_{i\in B}\exp\left(\frac{s_{li}^h}{\tau_b}\right)-\tau_b\log|B|.
$$

每层先跨 head 求均值，再用 median/MAD 在 blocks 间标准化，最后只跨四个 source layers 平均并取 Top-64。额外捕获的 $l+1$ 只用于 destination scoring/control，不参与 proposal。

### 4.2 从原始 V 得到源 block write

$$
\pi_{li}^{h}(B)=\frac{\exp(s_{li}^h/\tau_v)}{\sum_{j\in B}\exp(s_{lj}^h/\tau_v)},
$$

$$
\bar v_{lB}^{h}=\sum_{i\in B}\pi_{li}^{h}(B)v_{li}^{g(h)},
\qquad
\delta h_{lB}=W_O^l\operatorname{Concat}_{h=1}^{H_q}\bar v_{lB}^{h}.
$$

实现用 `o_proj(x)-o_proj(0)` 消除 bias。

### 4.3 下一层局部有限差分

定义：

$$
F_{l+1}^{Q}(h)=q\_norm_{l+1}\!\left(W_{Q,l+1}\,RMSNorm_{l+1}(h)\right).
$$

在捕获的原生下一层输入 $h_{l+1}$ 上做 batched central finite difference：

$$
\Delta q_{l\rightarrow l+1}(B)=
\frac{F_{l+1}^{Q}(h_{l+1}+\epsilon\delta h_{lB})-
F_{l+1}^{Q}(h_{l+1}-\epsilon\delta h_{lB})}{2\epsilon}.
$$

默认 $\epsilon=0.05$。这不是完整 block JVP：它把 $\delta h_{lB}$ 当作直接到达 $l+1$ 输入的边界扰动，没有建模第 $l$ 层 attention write 经该层 residual/MLP 后的变化。故它只检验**局部 layer-boundary relay compatibility**，不是自然前向的完整因果效应。

### 4.4 Directed edge score

$$
e_l(B\rightarrow C)=\frac1{H_q}\sum_{h'=1}^{H_q}
\left[\tau_e\log\sum_{j\in C}\exp\left(
\frac{(\Delta q_{l\rightarrow l+1}^{h'}(B))^\top k_{l+1,j}^{g(h')}}
{\tau_e\sqrt d}\right)-\tau_e\log|C|\right].
$$

每层 edge matrix 在非对角元素上做 median/MAD 标准化，再对四层求均值，得到最终 `kvq_relay`。

## 5. Controls

1. **shuffled-V relay：** source blocks 间使用无固定点置换。
2. **norm-matched random-V relay：** 逐 block/head 生成随机 V 方向；经相同 $W_O$ 后，再严格匹配原生 $\|\delta h_{lB}\|_2$，避免各向异性 $W_O$ 造成扰动尺度不等。
3. **reverse edge：** 对 $B\rightarrow C$ 使用原生矩阵中的 $e_l(C\rightarrow B)$。
4. **K-K similarity：** query-weighted source/destination pre-RoPE K 代表的逐 head cosine。
5. **pre-score pair：** $r_l(B)+r_{l+1}(C)$。

只有 `kvq_relay` 显著优于这些 controls，才说明信号不只是 QK 相关性、Value/write norm 或无方向的 block relevance。

## 6. 两个硬审计

### 6.1 Prefix immutable

在 query capture 前、query cache 截回 prefix 后、离线 edge scoring 后，对所有读取层的原生 prefix K/V 逐字节、分块计算完整 SHA-256，并同时记录 shape、dtype 和数值矩。三次必须完全相同；任何差异立即使 case 无效。

### 6.2 全候选 finite-difference audit

对每层**全部候选 blocks（最多 64）**同时计算 $\epsilon$ 与 $\epsilon/2$：

$$
E_{FD}(B)=\frac{\|\Delta q_{\epsilon}(B)-\Delta q_{\epsilon/2}(B)\|_2}
{\|\Delta q_{\epsilon/2}(B)\|_2+10^{-12}}.
$$

还需重构 $F_{l+1}^{Q}(h_{l+1})$ 并与 hook 捕获的 baseline pre-Q 比较。逐 source 保存误差；默认整 case 有效门槛：

- `fd_halving_relative_error_max <= 0.35`
- `fd_halving_cosine_min >= 0.90`
- `baseline_q_reconstruction_max_abs <= 1e-4`

失败 case 可保存排错，但不得进入方法结论。

## 7. 汇总、断点续跑与 stop rules

- AUROC 必须先在每个可解析且通过 FD 审计的 case 内计算，再做 case macro mean；95% CI 按 seed cluster bootstrap。禁止跨 case 直接池化 raw scores。
- 无 positive 或 negative 时 AUROC 写为 JSON `null`，并记录 invalid reason；禁止非标准 `NaN`。
- 每个 case 先原子写入 raw tensor、audit、edge rows 和 case row，最后写 `done.json` commit marker；resume 只认可该 marker。
- 输出目录使用不可变 `full_config_hash`；merge 要求 shard `done.txt`、一致的 `method_config_hash`、完整 seed 覆盖和无重复 case key，并写 merged manifest 与 raw artifact 引用。

首轮满足任一项即停止扩展 KVQ-R：

1. 两条 gold records 同时进入 label-free Top-64 的比例低于 80%；
2. 有效 cases 的 `kvq_relay` AUROC 均值低于 0.65，或 seed-bootstrap 95% CI 下界不高于 0.5（不能排除 chance）；
3. shuffled-V 保留原生 relay 超过 80% 的 above-chance AUROC；
4. random-V、reverse、K-K 或 pre-score pair 达到同等/更好 AUROC；
5. 超过 5% cases 未通过 FD audit；
6. 任一 prefix immutable audit 失败。

通过本轮只允许进入下一阶段的严格 2% sparse-consumer 实验，不能直接写成端到端方法成立。

## 8. 产物

- runner：`src/run_kvq_relay_edge_probe_8b.py`
- tests：`tests/test_kvq_relay_edge_probe.py`
- launcher：`scripts/run_kvq_relay_edge_probe_gpu67_20260801.sh`
- 每 case：`cases/<case>/raw.pt`、`audit.json`、`case_row.json`、`edge_rows.json`、`done.json`
- 汇总：`case_rows.{jsonl,csv}`、`case_summary.{json,csv}`、`edge_rows.{jsonl,csv}`、`summary.{json,csv}`

launcher 只允许物理 GPU6/7；本阶段只创建文件，不启动服务器任务。
