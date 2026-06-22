# Experiment Design

## 0. 当前诊断：Top-2% Attention Token 的 Feature 从哪里来？

### 0.1 可证伪问题

已知每层每个 head 的 top-2% attention token 足以支撑推理。当前要回答：

> 这些 top-2% token 共享的 feature 到底在哪里？它是在输入 hidden state 中已经存在，还是由 `W_Q/W_K` 抽取出来，还是由完整 `W_Q/W_K` SVD 后按 head 切输出空间得到的少数 query/key feature pairs 定义？

这个问题的输出不是一个新模型，而是一个设计判断：inverse KV 的 routing 应该依赖 hidden、Q/K，还是显式学习 Q/K-like feature projection。

### 0.2 输入

1. 一个训练好的 Qwen / inverse-KV 相关 checkpoint；
2. 一批真实文本 token sequences；
3. 指定 layer set 和 head set；
4. 每个 head 的 `W_Q`、`W_K`；
5. 每个 token 在每层的 pre-attention hidden state `x_i`。

### 0.3 正样本、负样本与距离控制

对每个 layer/head/query token $i$：

- 正样本 key $j$：$j<i$，且 $j$ 在该 layer/head/query 的 strict-history top-2% QK score key 集合中。
- 负样本 key $j'$：$j'<i$，且 $j'$ 不在 top-2% 集合中。

这里不能只做一种负样本。距离本身可能就是 feature 的一部分，因为 RoPE 会把相对位置直接注入 Q/K。若最后发现 top-2% 主要由距离解释，这也是一个有效结论：模型在该 head 上主要使用 position/RoPE feature 做 retrieval。

因此需要同时报告两组负样本：

1. **随机负样本**：从所有非 top-2% 历史 token 中采样。它回答：top-2% token 相比普通历史 token 是否有可分辨 feature。
2. **距离匹配负样本**：按 `i-j` 的距离桶匹配正样本。它回答：在相同距离条件下，top-2% token 是否仍有额外内容 feature、Q/K feature 或 SVD feature。

两组结果的解释不同：

1. 如果随机负样本可分、距离匹配后不可分，说明 top-2% 主要由距离/RoPE feature 解释；
2. 如果距离匹配后仍可分，说明除距离外还有内容或参数矩阵抽取出的 feature；
3. 如果两组都不可分，说明当前指标没有解释 top-2% selection，需要检查 norm、sink token、后续层补偿或指标定义。

### 0.4 Head-level 分析是基本单位

Attention score 是逐 layer、逐 head 计算的：

$$
\mathrm{score}(l,h,i,j)=q_{l,h,i}^{\top}k_{l,h,j}
$$

因此所有诊断都必须以 `(layer, head)` 为基本单位。不能先把 head 平均后再分析 feature 来源，因为平均会把不同 head 的 feature 混在一起。

对原始 hidden state `x` 的分析也必须绑定到具体 head：

1. `cos(x_i, x_j)`：测试不经过该 head 的 `W_Q/W_K` 时，原始 residual hidden 是否已经足以解释该 head 的 top-2%；
2. `cos(q_{h,i}, k_{h,j})` 或 `q_{h,i}^T k_{h,j}`：测试经过该 head 的 `W_Q/W_K` 后，feature 是否被增强；
3. 完整 `W_Q/W_K` 的 SVD 后按 head 切输出空间：测试全局 singular feature 如何写入不同 head，以及某个 head 的 high-QK score 来自哪些 query/key singular feature pair。

跨 head 的统计只作为二级汇总：

1. 每个 head 各自的可解释性；
2. 不同 head 的 top-2% key 集合是否重叠；
3. 同一个 key token 被多少个 head 选中；
4. 所有 head 的 top-2% union 覆盖了多少历史 token。

### 0.5 测量分数

对每个 pair `(i,j)` 计算：

| 分数 | 定义 | 测试什么 |
|---|---|---|
| hidden cosine | $\cos(x_i,x_j)$ | feature 是否已在 residual hidden space 中天然存在 |
| K-space cosine | $\cos(k_{h,i},k_{h,j})$ | 该 head 的 key/address space 是否已有聚类 |
| QK score | $q_{h,i}^{\top}k_{h,j}$ | 真实 attention retrieval score |
| centered K cosine | $\cos(k_{h,i}-c_h,k_{h,j}-c_h)$ | common center 是否解释了虚假相似 |
| full-WQ/WK SVD head mass | $U_Q,U_K$ 的奇异向量在第 $h$ 个 head 输出坐标上的 mass | 全局 singular feature 是否集中写入特定 head |
| SVD pair contribution concentration | $c^{(h)}_{r,s}$ 的 top mass | high score 是否由少数 $W_Q/W_K$ singular feature pair 主导 |
| SVD pair overlap | top-contribution feature pairs 在 top pairs 之间的重合度 | 同一 head 是否反复使用稳定的 query/key feature pair |

其中：

$$
W_Q=U_Q\Sigma_QV_Q^{\top},\quad
W_K=U_K\Sigma_KV_K^{\top}
$$

$$
q_{h,i}=x_iV_Q\Sigma_QU_{Q,h}^{\top},\quad
k_{h,j}=x_jV_K\Sigma_KU_{K,h}^{\top}
$$

这里的 head-level SVD 不是对 $W_{Q,h}$ 或 $W_{K,h}$ 单独做 SVD，而是先对完整 $W_Q/W_K$ 做 SVD，再把输出空间奇异向量 $U_Q/U_K$ 按 head 坐标切片。这样所有 head 共享同一套输入 singular feature basis，才能比较一个 feature 被写入哪些 head。

### 0.6 完整 WQ/WK SVD 如何解释某个 head 的 QK score

对某一层某个 head：

$$
W_Q=U_Q\Sigma_QV_Q^{\top}
$$

$$
W_K=U_K\Sigma_KV_K^{\top}
$$

其中 $U_Q,U_K$ 位于 concat 后的 Q/K 输出空间，可以按 head 切分。记第 $h$ 个 head 的输出坐标切片为 $U_{Q,h}$ 和 $U_{K,h}$。

于是：

$$
q_{h,i}=x_iV_Q\Sigma_QU_{Q,h}^{\top}
$$

$$
k_{h,j}=x_jV_K\Sigma_KU_{K,h}^{\top}
$$

该 head 的 QK score 为：

$$
s_{h,i,j}
=x_iV_Q\Sigma_QU_{Q,h}^{\top}U_{K,h}\Sigma_KV_K^{\top}x_j^{\top}
$$

第 $r$ 个 $W_Q$ singular feature 和第 $s$ 个 $W_K$ singular feature 对该 head score 的贡献为：

$$
c^{(h)}_{i,j,r,s}
=(x_iV_Q)_r
\sigma_{Q,r}
\left(U_{Q,h}^{\top}U_{K,h}\right)_{r,s}
\sigma_{K,s}
(x_jV_K)_s
$$

这里要区分三个问题：

1. **全局 singular feature 写入哪个 head**：看 $U_Q[:,r]$ 或 $U_K[:,s]$ 在各 head 输出坐标上的 mass；
2. **某个 head 的 Q/K 输出方向是否对齐**：看 $U_{Q,h}^{\top}U_{K,h}$ 是否集中或接近对角；
3. **当前 query/key pair 实际用了哪些 feature pair**：看 $|c^{(h)}_{i,j,r,s}|$ 或正贡献 $\max(c^{(h)}_{i,j,r,s},0)$ 的 top mass；
4. **同一个 head 是否反复使用稳定 feature pair**：看不同 top-2% pair 的 top-contribution $(r,s)$ overlap。

奇异值大不等于该 feature pair 一定解释当前 pair。因为贡献还乘了 token 在 $V_Q,V_K$ 输入奇异方向上的投影，以及 $U_{Q,h}^{\top}U_{K,h}$ 的输出对齐项。因此需要报告：

1. $\Sigma_Q,\Sigma_K$ 本身的 sharpness；
2. 每个 singular feature 写入不同 head 的 mass 分布；
3. 当前 pair 的 contribution matrix $c^{(h)}_{i,j,r,s}$ 是否集中；
4. 正样本 top-2% pair 与负样本 pair 的 contribution concentration 是否不同；
5. 高贡献 $(r,s)$ feature pair 在不同 query、不同文本中的稳定性。

如果 top-2% pair 的 QK score 主要由少数稳定 $(r,s)$ feature pair 贡献，而距离匹配负样本没有这种现象，就说明该 head 的 top attention 可以被完整 $W_Q/W_K$ 奇异空间中的 feature pair 解释。

补充 sanity check 可以对组合后的矩阵 $M_h=W_{Q,h}^{\top}W_{K,h}$ 做 SVD。它回答的是“最终双线性匹配方向是否集中”，但它已经把 $W_Q$ 和 $W_K$ 的原始奇异空间合并，因此不能作为主分析。

### 0.7 指标

1. `AUROC`：某个分数能否区分正样本 pair 和负样本 pair；
2. `precision@top2%`：只用某个分数排序时，能否恢复真实 attention top-2% key；
3. `positive_negative_gap`：正样本分数均值减去负样本分数均值；
4. `svd_pair_top_mass`：QK score 中由 top contribution feature pairs 解释的比例；
5. `head_feature_mass_entropy`：一个 singular feature 被集中写入少数 head，还是分散写入多个 head；
6. `head_pair_direction_entropy`：一个 head 使用少数稳定 $W_Q/W_K$ feature pairs，还是使用很多分散 feature pairs；
7. `head_pair_jaccard`：不同 head 的 top-2% key 集合重合度；
8. `selected_head_count_per_key`：同一个 key token 被多少个 head 强烈选中；
9. `layer_head_union_coverage`：一个 layer 或多层多 head 的 top-2% key union 覆盖了多少历史 token；
10. `distance_only_baseline`：只用相对距离预测 top-2% key 的效果，用来判断 position/RoPE feature 的解释力；
11. `norm_only_baseline`：只用 key norm 或 q/k norm 预测 top-2% key 的效果，用来排除 norm artifact。

### 0.8 结果解释

| 观察结果 | 解释 |
|---|---|
| hidden cosine 已能预测 top-2% | top token feature 在 residual hidden space 中已经存在 |
| QK / K-space 明显强于 hidden cosine | `W_Q/W_K` 主动抽取了 head-specific feature |
| full-WQ/WK SVD contribution pair 在每个 head 内稀疏且稳定 | high-score token relation 可以被完整 $W_Q/W_K$ 奇异空间中的 feature pair 解释 |
| 不同 head 的 top-key overlap 低 | 支持 head-level specialization |
| 大多数 key 被很多 head 同时选中 | head-level specialization 较弱，top attention 可能由 sink、position 或全局公共 feature 驱动 |
| centered K 提高区分度 | common center 掩盖了有用 feature geometry |
| 随机负样本可分、距离匹配后不可分 | top-2% 主要由距离/RoPE feature 解释 |
| 距离匹配后仍可分 | 除距离外，还有内容 feature、Q/K feature 或 SVD feature |
| 没有任何分数能区分正负样本 | top-2% 有效性可能来自后续层聚合、norm artifact、sink token 或当前指标定义不对 |

### 0.9 通过、失败与证据不足

支持 feature-source conjecture 的条件：

1. 正样本 pair 与距离匹配负样本 pair 在至少一个 feature-space 指标上有清晰差异；
2. QK/full-WQ-WK-SVD 指标比 position-only baseline 更能解释 top-2%；
3. 不同 head 存在可测的 head-specific 差异；
4. 代表性样本显示可解释的方向集中，而不只是 aggregate number 上有微弱差异。

削弱该 conjecture 的条件：

1. top-2% pair 与匹配负样本无法区分；
2. 所有 head 几乎选择相同 key；
3. high score 主要由 token norm、sink token 或距离解释；
4. SVD feature pairs 在不同样本中分散且不稳定。

证据不足的条件：

1. 只测试 synthetic data；
2. 只测试一个 layer/head；
3. 没有同时报告随机负样本和距离匹配负样本；
4. 只报告 aggregate scores，没有代表性 pair 的 stage-level evidence。

### 0.10 下一步结构判断

该诊断直接决定 router 设计：

1. hidden-native feature 强：优先测试 `layer_input` routing；
2. K/Q extracted feature 强：优先测试 `k`、`q` 或 Q/K-like learned projection；
3. full-WQ/WK SVD feature pair 强：设计 router 去近似 dominant query/key singular feature pairs；
4. head-specific feature 强：保留 head-level bucket/expert；
5. head overlap 高：token-level 或 shared global routing 可能已经足够；
6. distance/RoPE feature 强：router 需要显式接收或学习相对位置结构，而不能只看 token content。

### 0.11 已完成结果与下一步统一归因

当前实验已经确认：

1. centered hidden space 已有 top-key feature；
2. K projection 进一步放大该 feature；
3. K 前 256 个奇异方向贡献约 `86.5%` 的净正向 top-vs-random margin；
4. RoPE 显著改变 retrieval membership，并相对随机历史 token 提供约 `53.1%` 的条件 margin 增量；
5. 控制 token distance 后，RoPE 的条件增量约为 `24.1%`。

由于最终 score 是 input、learned Q/K 和 RoPE 的乘性交互，下一步若要给出统一贡献比例，必须在同一 score metric 上完成八组 factorial ablation：

1. input relationship：原始 pair relationship / 距离桶内打乱 key hidden；
2. Q/K transformation：训练后的 `W_Q/W_K` / 保持奇异值谱但随机化奇异向量方向的 matched transformation；
3. position：RoPE / identity rotation。

统一输出使用固定完整模型 top-2% label 的 standardized top-vs-control score margin，并同时报告 top-set recall。最后使用 Shapley value 分配三项主效应及其交互贡献。该设计避免把 hidden cosine、K cosine 与 QK score 上的百分比直接相加。

## 1. Falsifiable Question

在真实 DCLM 数据上，head-level shared KV/expert bucket 能否仅靠 NTP：

1. 保持有限且持续下降的 training loss；
2. 避免所有 token 坍缩到同一 expert；
3. 把每层 attention candidate ratio 降到 full causal attention 的明显子集；
4. 让 router 从 NTP 获得非零梯度。

本轮只检验训练可行性，不声称已经节省物理 KV cache。

## 2. Fixed Setup

| Parameter | Value | Reason |
|---|---:|---|
| base config | Qwen3-0.6B | fits below the 2B limit and matches existing training stack |
| experts/buckets per head | 4 | first low-risk compression setting |
| router top-k | 1 | bucket id and expert id remain identical |
| ordinary expert intermediate size | 3072 | ordinary expert is `1024 -> 3072 -> 1024` |
| derived head expert intermediate size | 512 | shared expert is `64 -> 512 -> 1024`, derived by equal-parameter matching |
| sequence length | 1024 | existing DCLM training setting |
| loss | NTP only | isolates whether the structure learns without auxiliary objectives |

## 3. Minimal Runs

### Run 0: Equal-budget ordinary MoE

```text
architecture = ordinary_moe
attention = full causal attention
experts per layer = 4
expert shape = 1024 -> 3072 -> 1024
router input = post-attention residual
```

The full model has about `1.388888B` parameters. The shared-bucket model has about `1.389003B`, so this is the primary size-matched baseline.

### Run A: Main structure

```text
router_input = k
center_router_input = true
router_normalization = l2
local_window = 32
sink_tokens = 4
```

### Run B: Centering ablation

Same as A, except:

```text
center_router_input = false
```

### Run C: Router representation ablation

Run `layer_input`, `q`, and `v` after A passes the basic training checks.

## 4. Recorded Evidence

Each step is written to `run_dir/train_metrics.jsonl`:

| Metric | Meaning |
|---|---|
| `loss`, `perplexity` | NTP quality |
| `next_token_accuracy` | teacher-forced token accuracy |
| `gradient_norm` | overall optimization health |
| `candidate_ratio` | allowed attention pairs / full causal pairs |
| `router_max_probability` | routing confidence |
| `router_margin` | top-1 minus top-2 router probability |
| `router_token_entropy` | mean per-token routing entropy, normalized |
| `router_load_entropy` | aggregate expert-load entropy, normalized |
| `effective_experts` | `exp(load entropy)` |
| `max/min_expert_load` | collapse and dead-expert evidence |

## 5. Pass, Fail, and Insufficient Evidence

Pass for the implementation stage:

1. smoke test shows nonzero router gradient；
2. no NaN/Inf in loss or routing metrics；
3. all four experts remain active in most layers during early training；
4. candidate ratio is materially below 1；
5. loss decreases over a meaningful training interval。

Fail:

1. router gradient remains zero；
2. attention rows become fully masked；
3. one expert load approaches 1 and remains there；
4. loss does not improve relative to initialization；
5. model exceeds the 2B hard parameter limit。

Insufficient evidence:

1. only a few debug steps are available；
2. training loss decreases but no full-attention/dense control exists；
3. logical candidate ratio decreases but physical KV bytes and latency are not measured。

## 6. Required Next Controls

After the main run is stable, compare under the same data, token budget, and optimizer:

1. standard Qwen3 dense attention + dense FFN；
2. bucket attention + dense FFN；
3. full attention + head experts；
4. shared bucket attention + head experts；
5. separate attention bucket and expert routing IDs。
