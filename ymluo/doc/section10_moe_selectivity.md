# Section 10 — MoE Selectivity 合成数据实验

> 新增日期：2026-05-19；最近同步日期：2026-05-23

## 实验背景

指标说明：

- `same_higher_by_layer`：同一个 higher-level unit 的 token 对被分发到同一个 expert 的比例（越高越好的 selectivity）。
- `same_higher_occurrence_by_layer`：用纯位置 signal（occurrence ID）做同样计算的交叉验证指标。
- `higher_mass_by_layer`：attention 落在同一 higher-level history token 上的质量。
- `expert_load_by_layer`：每层不同 expert 的平均路由负载分布。

所有实验默认使用：3 层，hidden=128，4 attention heads，2 KV heads，head_dim=32，4 experts，`num_experts_per_tok=1`，`moe_head_level=false`（每层只有 1 个 gate）。

---

## Exp1：对其它 expert / gate 列向量施加 negative gradient

实验目的：让 expert / gate 对非同类的 token 产生抑制信号，观察是否能提升 expert selectivity。

| step | loss | acc | same_higher_by_layer | higher_mass_by_layer | expert_load |
| --- | ---: | ---: | --- | --- | --- |
| 100 | 1.4925 | 0.7676 | L0:0.4316 L1:0.5558 L2:0.6765 | L0:0.2997 L1:0.2578 L2:0.3173 | [0.1736, 0.2397, 0.2717, 0.3149] |
| 1100 | 0.2673 | 0.9128 | L0:0.4673 L1:0.9781 L2:1.0000 | L0:0.3636 L1:0.4257 L2:0.4200 | [0.1393, 0.3073, 0.2292, 0.3240] |
| 10000 | 0.2607 | 0.9139 | L0:0.4568 L1:0.9826 L2:1.0000 | L0:0.3808 L1:0.4341 L2:0.4273 | [0.1402, 0.3076, 0.2288, 0.3232] |

结论：反向负梯度效果非常好，L2 达到完美的 1.0000 selectivity。但 expert 负载存在一定不均衡（min 14%，max 32.3%）。

---

## Exp2：小 batch / token 更新

实验目的：每次只更新一个 token，验证类似人脑 inhibition 的逐条数据学习机制是否能提升 gating selectivity。

| step | loss | acc | same_higher_by_layer | higher_mass_by_layer | expert_load |
| --- | ---: | ---: | --- | --- | --- |
| 10000 | 4.5524 | 0.0556 | L0:0.7572 L1:0.5807 L2:0.7301 | L0:0.2470 L1:0.2680 L2:0.2648 | [0.2160, 0.3237, 0.1273, 0.3328] |
| 100000 | 1.0697 | 0.7883 | L0:0.2849 L1:0.5347 L2:0.5712 | L0:0.2746 L1:0.2415 L2:0.2627 | [0.2137, 0.2238, 0.2935, 0.2687] |
| 300000 | 0.9383 | 0.8090 | L0:0.3002 L1:0.6541 L2:0.6161 | L0:0.2947 L1:0.2485 L2:0.2763 | [0.2323, 0.2549, 0.2259, 0.2866] |

结论：逐 token 学习确实能学到一定 selectivity，但训练极慢（30w step 的 loss 仍远高于其他实验 1w step 的水平）。

---

## Exp3：expert 初始化调整

实验目的：验证 expert 初始化或 gate 初始化是否显著影响最终训练出的 selectivity。

### Exp3.1：expert 两两 Frobenius 内积正交初始化

| step | loss | acc | same_higher_by_layer | higher_mass_by_layer | expert_load |
| --- | ---: | ---: | --- | --- | --- |
| 100 | 1.4884 | 0.7679 | L0:0.3209 L1:0.3625 L2:0.3840 | L0:0.2994 L1:0.2572 L2:0.3140 | [0.2174, 0.2864, 0.2649, 0.2311] |
| 1000 | 0.2687 | 0.9126 | L0:0.3033 L1:0.3477 L2:0.3419 | L0:0.3483 L1:0.3862 L2:0.3762 | [0.2429, 0.2413, 0.2692, 0.2463] |
| 10000 | 0.2608 | 0.9125 | L0:0.3027 L1:0.3207 L2:0.3410 | L0:0.3602 L1:0.4174 L2:0.3955 | [0.2430, 0.2486, 0.2365, 0.2717] |

结论：Frobenius 正交初始化**无法**有效让相同 higher-level unit 选择相同 expert。selectivity 几乎就是随机的 0.25~0.34。

### Exp3.2：前 1000 步强制同一 higher-level unit 选择相同 expert

| step | loss | acc | same_higher_by_layer | higher_mass_by_layer | expert_load |
| --- | ---: | ---: | --- | --- | --- |
| 100 | 1.5110 | 0.7650 | L0:0.3451 L1:0.3422 L2:0.3560 | L0:0.2996 L1:0.2564 L2:0.3118 | [0.3052, 0.2031, 0.2548, 0.2367] |
| 1000 | 0.4252 | 0.8808 | L0:0.4413 L1:0.4842 L2:0.5174 | L0:0.3351 L1:0.2831 L2:0.2990 | [0.5068, 0.0635, 0.0426, 0.3869] |
| 10000 | 0.2604 | 0.9136 | L0:0.3796 L1:0.3106 L2:0.3158 | L0:0.3851 L1:0.4466 L2:0.4259 | [0.2868, 0.2260, 0.1511, 0.3360] |

结论：强制 warmup 在 warmup 期间（step 1000）确实提升了 selectivity，但去除 force 后（step 10000）selectivity 迅速退化到接近正常水平。模型会逐渐"遗忘"被强制的路由模式。

**Exp3 总结**：初始化参数对模型后续训练影响较小，模型会逐渐将参数向 baseline 靠近。

---

## Exp4.1：Attention Cluster — 基础版（uniform 数据 + 纯 attention 信号）

> 新增日期：2026-05-22

实验目的：用 attention 权重作为监督信号，让彼此关注度高的 token 被路由到同一个 expert。

**配置**：`attention_cluster_weight=0.05`、`attention_cluster_topk=4`、`attention_cluster_negative_weight=0.01`、`synthetic_sampling_distribution=uniform`、无 pre_router、无 load_balance。

| step | loss | acc | same_higher_by_layer | higher_mass_by_layer | expert_load |
| --- | ---: | ---: | --- | --- | --- |
| 10000 | 0.2015 | 0.9422 | L0:0.7944 L1:0.9988 L2:0.9997 | L0:0.3256 L1:0.3194 L2:0.4631 | [0.283, 0.285, 0.242, 0.190] |

**详细指标**（step 10000）：

- `same_higher_same_expert`：整体 0.9310，L2 近乎完美 (0.9997)。
- `same_higher_occurrence_same_expert`：L0:0.8322, L1:0.9981, L2:0.9995。
- `expert_load_by_layer`：L0:[0.234,0.307,0.257,0.203] L1:[0.257,0.305,0.242,0.195] L2:[0.359,0.242,0.227,0.172]。

**结论**：纯 attention cluster 效果优秀，L2=0.9997 接近 Exp1 的水平，且 expert 负载均衡（最不均衡比例 2.1x）。

---

## Exp4.2：Pre-Router + Attention Cluster（zipf 数据 + load balance）

> 新增日期：2026-05-23

实验目的：在 attention cluster 基础上增加 pre-router（基于 Q 投影提前路由）和 load balance loss，同时改用 Zipf 分布数据。

**配置变更**（相比 Exp4.1）：

| 参数 | Exp4.1 | Exp4.2 | 变更说明 |
| --- | ---: | ---: | --- |
| `use_pre_router` | false | **true** | 新增 pre-router |
| `pre_router_input` | — | **q** | 从 query 投影做提前路由 |
| `attention_cluster_weight` | 0.05 | **0.01** | 降低 cluster loss 权重 |
| `attention_cluster_topk` | 4 | **8** | 增加 top-k |
| `attention_cluster_negative_weight` | 0.01 | **0** | 去掉负样本 loss |
| `moe_load_balance_loss_weight` | 0.0 | **0.01** | 新增负载均衡 loss |
| `synthetic_sampling_distribution` | uniform | **zipf** | Zipf α=1.1 |
| `moe_expert_input_attention_topk` | — | **8** | 新增参数 |

**训练过程**（末尾 step）：

```
step 9990: loss=0.1994 acc=0.9419 attn_cluster=0.4842 load_balance=1.4404
step 10000: loss=0.1977 acc=0.9432 attn_cluster=0.4955 load_balance=1.4598
```

**最终结果**（step 10000）：

| 指标 | 值 |
| --- | --- |
| eval_loss | 0.2015 |
| eval_acc | 0.9421 |
| same_higher_same_expert | **0.7935** (vs Exp4.1: 0.9310) |
| higher_mass | 0.3738 |
| expert_load (平均) | [0.128, 0.328, 0.423, 0.121] |

**逐层对比**：

| Layer | same_higher | same_higher_occurrence | higher_mass | expert_load |
| --- | ---: | ---: | ---: | --- |
| L0 | 0.6297 | 0.6007 | 0.3485 | [0.055, 0.305, **0.580**, 0.059] |
| L1 | 0.7724 | 0.7967 | 0.3178 | [0.322, 0.383, 0.281, **0.014**] |
| L2 | 0.9785 | 0.9804 | 0.4551 | [**0.008**, 0.294, 0.407, 0.290] |

**关键发现**：

1. **L0 近坍缩**：Expert 2 独占 58%，Expert 0 和 3 仅 5~6%。L0 的 selectivity 从 0.79 降到 0.63。
2. **L1 有死 expert**：Expert 3 仅 1.4%，接近完全不被使用。
3. **L2 仍表现良好**：selectivity=0.9785，与 Exp4.1 的 0.9997 接近，但 Expert 0 在这层几乎完全被跳过（0.8%）。
4. **load_balance loss 未起效**：虽然加了 load_balance（权重 0.01，loss 约 1.44-1.46），expert 分布反而比 Exp4.1 更不均衡。
5. **pre_router 干扰了 cluster 信号**：pre_router 提前做了路由决策，可能和 attention cluster loss 形成冲突——pre_router 只看到 Q 投影信息，而 attention cluster 信号来自完整的 attention 计算，两者不完全一致。

## 全部实验对比

| 实验 | L2 same_higher | same_higher (avg) | expert_load 范围 | 备注 |
| --- | ---: | ---: | --- | --- |
| Exp1 negative gradient | **1.0000** | — | [0.140, 0.323] | 效果最好但负载不均 |
| Exp2 single token (30w) | 0.6161 | — | [0.226, 0.287] | 训练太慢 |
| Exp3.1 orthogonal init | 0.3410 | — | [0.237, 0.272] | 无效 |
| Exp3.2 forced warmup | 0.3158 | — | [0.151, 0.336] | 去掉后退化 |
| **Exp4.1 attn cluster (基础版)** | **0.9997** | **0.9310** | [0.172, 0.359] | **效果接近 Exp1 且负载更均衡** |
| Exp4.2 pre-router + attn cluster | 0.9785 | 0.7935 | [0.008, 0.580] | selectivity 下降，负载严重失衡 |

## 结论

- **Exp4.1（纯 attention cluster）是最佳方案**：selectivity 接近完美，expert 负载健康，且是自监督信号。
- **Exp4.2 加入 pre-router 反而变差**：pre-router 的提前路由与 attention cluster 信号形成冲突，导致 expert 坍缩。
- **推荐的实验配置**是 Exp4.1 的参数组合。

运行命令：

```bash
# Exp4.1 基础版（推荐）
ATTENTION_CLUSTER_WEIGHT=0.05 \
ATTENTION_CLUSTER_TOPK=4 \
ATTENTION_CLUSTER_TEMPERATURE=1.0 \
ATTENTION_CLUSTER_NEGATIVE_WEIGHT=0.01 \
MOE_LOAD_BALANCE_LOSS_WEIGHT=0.0 \
USE_PRE_ROUTER=false \
bash ymluo/projects/qwen3_moe_attention_cluster/scripts/run_train.sh
```

主要输出：

```text
ymluo/projects/qwen3_moe_attention_cluster/outputs/train/moe-attention-cluster/metrics.jsonl
ymluo/projects/qwen3_moe_attention_cluster/outputs/train/moe-attention-cluster/checkpoints/<step>.pth
ymluo/projects/qwen3_moe_attention_cluster/outputs/train/moe-attention-cluster/checkpoints/runtime_config.json
```

---

## Exp5 — Attention Cluster 真实数据实验（Qwen1.5-MoE + DCLM）

> 新增日期：2026-05-25

### 实验目的

将 Exp4 的 attention cluster 核心思想从合成数据迁移到真实 MoE 模型和真实文本，验证该方法在真实场景下的可行性。

### 与合成数据版的核心差异

| 维度 | 合成数据版 (Exp4) | 真实数据版 (Exp5) |
| --- | --- | --- |
| 模型 | 自定义 `MyQwen3ForCausalLM`（3 层，hidden=128） | 真实 **Qwen1.5-MoE-A2.7B** 缩至 **0.6B**（12 层，hidden=768，12 experts） |
| 数据集 | fdong hierarchical synthetic data | **DCLM** 真实文本（随机采样行 → tokenize → 固定长度 block） |
| Top-k 方式 | 绝对数量 top-k（如 top-4 个 token） | **比例 top-ratio**（默认前 10% 历史 token） |
| Router 输入 | 可配置（`layer_input` / `q` / `k` / `v`） | **固定用 Q 投影**（`q_proj` + `q_norm` 后的状态） |
| Expert 输入 | 标准 post-attention MLP 输入 | **替换为稀疏 attention 输出**（top-10% V 加权 + `o_proj`） |
| 负样本对 loss | 有（基于 hierarchical metadata 特征 ID） | **无**（真实文本无层级标注） |
| Head-level 路由 | 支持 `moe_head_level` | **不支持**，仅 per-layer 路由 |
| 训练框架 | 自定义训练循环 | **HuggingFace Trainer + DeepSpeed ZeRO-3** + 8 GPU |

### 辅助损失函数

合成版和真实版的核心 loss 公式一致：

```
same_expert_prob(i, j) = dot(router_prob_i, router_prob_j)
L_attn_cluster = -attention(i, j) * log(same_expert_prob(i, j))
```

真实版在实现上支持两个额外选项：
- `attention_cluster_detach_attention`（默认 true）：detach attention 权重，使其不通过 cluster loss 回传梯度到 attention 参数。
- `attention_cluster_detach_key_router`（默认 false）：detach key 侧的 router 概率，只让 query 侧的 router 被优化。

总训练损失：

```
L = L_lm + attention_cluster_weight * L_attn_cluster + load_balance_loss_weight * L_load_balance
```



前向的时候分发到k个expert上面，k << num_expert , 然后反向更新参数的时候，只修改1个expert，然后剩下的k-1个expert负向更新梯度。



### Patch 机制

真实版通过 PyTorch forward hooks（而非修改模型代码）注入 attention cluster 逻辑：

1. **Layer pre-hook**：保存残差连接的输入（用于后续还原）。
2. **Attention pre-hook**：保存 attention 输入 hidden states。
3. **Attention hook**：从 attention 输出中提取 attention weights；用保存的 hidden states 计算 Q 投影作为 router 输入；用 sparse top-ratio attention 计算新的 expert 输入（稀疏 V 加权 + `o_proj`），替代标准 post-attention 输出进入 MLP。
4. **MLP pre-hook**：将 expert 输入替换为上一步计算的稀疏 attention 输出。
5. **Gate pre-hook**：将 gate 输入替换为 Q 投影状态（即 router 看到的是 pre-attention 的 Q 信息）。
6. **Gate hook**：收集每层的 router logits 和 attention weights 用于计算辅助 loss。

### Baseline 模式

设 `EXPERIMENT_MODE=baseline` 可跳过所有 patch，训练标准 MoE LM，用于对照实验。

### 运行命令

```bash
# attention_cluster 模式（默认 0.6B preset）
bash ymluo/projects/qwen15_moe_real_attention_cluster/scripts/nohup_train.sh

# baseline 对照
EXPERIMENT_MODE=baseline \
bash ymluo/projects/qwen15_moe_real_attention_cluster/scripts/nohup_train.sh

# 自定义参数
ATTENTION_CLUSTER_WEIGHT=0.1 \
ATTENTION_TOP_RATIO=0.15 \
MAX_STEPS=20000 \
bash ymluo/projects/qwen15_moe_real_attention_cluster/scripts/nohup_train.sh
```

### 主要默认参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `ATTENTION_TOP_RATIO` | 0.10 | attention cluster loss 使用的 top 历史 token 比例 |
| `EXPERT_INPUT_TOP_RATIO` | 0.10 | expert 输入使用的稀疏 attention 比例 |
| `ATTENTION_CLUSTER_WEIGHT` | 0.01 | cluster loss 权重 |
| `LOAD_BALANCE_LOSS_WEIGHT` | 0.01 | load balance loss 权重 |
| `SEQ_LENGTH` | 1024 | 训练序列长度 |
| `MODEL_SIZE_PRESET` | moe_0_6b | 模型规模 preset（12 层，hidden=768，12 experts） |
| `GRADIENT_ACCUMULATION_STEPS` | 4 | 梯度累积步数 |

### 当前状态

项目代码已完成，待实际训练运行和结果记录。
