# 第五轮：在 Synthetic 数据上系统尝试 MoE Specialization 结构

## 总判断

Round5 表明，NTP 上 full/head MoE 与 dense 差异不大，spectral 当前失败；但从 specialization 看，`k/head` 明显更接近 feature bucket，尤其配合 common expert 后更强。不过自然训练得到的 bucket 仍不够硬，若要服务 reverse KV，需要进一步用 attention-derived signal 显式约束 gate。

## 这一轮要回答的问题

Round 5 对应 specialization overview 中的第 3 部分：**什么结构能做到 specialization**。

前几轮实验已经说明：

1. baseline MoE 不会自然形成 ground-truth slot specialization；
2. attention 中存在可用的 hierarchy feature signal；
3. ground-truth routing 本身有收益；
4. load-balance loss 只能改变 expert usage shape，不能直接产生 feature-level specialization；
5. 为了服务 KV cache reverse indexing，最终可部署 routing signal 最好在 attention 前产生。

因此，这一轮不再只诊断现有 MoE 为什么失败，而是开始在 synthetic 数据上系统尝试不同 MoE 结构，目标是找到能稳定形成 feature-level expert bucket 的结构。

## 目标

这一轮的目标不是单纯提高 next-token prediction accuracy，而是同时验证三件事：

1. **Predictive utility：** 模型仍然能完成 synthetic next-token prediction；
2. **Feature selectivity：** expert assignment 与 ground-truth local / higher-level feature 对齐；
3. **Deployability：** routing signal 尽可能能在 attention 前产生，从而有潜力服务 KV cache reverse indexing。

## 最终参考设定

Round 5 后续实验统一使用下面的 synthetic 数据设定，除非某个实验明确声明自己在做 ablation。

### 数据超参

```text
dataset_type = hierarchical_pattern
seq_len = 128
synthetic_block_size = 4
synthetic_num_hierarchy_layers = 2
synthetic_content_token_count = 512
synthetic_num_units_per_layer = 512
synthetic_seed = 0
synthetic_sampling_distribution = zipf
synthetic_zipf_alpha = 1.0
synthetic_zipf_shuffle_ranks = true
debug_vocab_size = 513
```

选择这个设定的原因是：

1. `inside-local` 仍然可以被 dense 参考模型基本学会，说明数据没有难到基础 pattern 都无法拟合；
2. `local-boundary-not-high` 仍有明显上升空间，适合检验模型是否更好地学到 high slot 内部的 local slot 组合；
3. `high-boundary` 本身是 top-level unit 采样边界，不应作为主要优化目标，只作为随机边界 sanity check。

在 `dense 2-layer h64`、训练 2000 step 的参考结果为：

```text
overall accuracy:                 92.69%
inside-local accuracy:            98.98%
local-boundary-not-high accuracy: 93.33%
high-boundary accuracy:           11.99%
```

因此，1000-step dense baseline 低估了 dense 小模型在该数据设定上的能力。后续结构比较必须使用 2000-step dense 作为参考，核心指标是：

```text
inside-local accuracy
local-boundary-not-high accuracy
expert routing selectivity
attention / routing bucket alignment
```

### 模型超参

当前 Round 5 结构实验统一使用小模型：

```text
num_hidden_layers = 2
hidden_size = 64
intermediate_size = 128
num_attention_heads = 4
num_key_value_heads = 2
head_dim = 16
max_position_embeddings = 256
```

MoE 默认超参为：

```text
use_moe = true
moe_num_unique_experts = 4
moe_num_experts_per_tok = 1
moe_intermediate_size = 128
moe_use_common_expert = false
moe_router_type = linear
```

其中 `dense 2-layer h64` 作为预测能力参考；MoE 结构的主要评价重点是是否在预测能力接近 dense 的同时形成更强 feature specialization。

## 2000-step 实验结果汇总

已完成的 2000-step 实验包括：

1. `dense 2-layer h64` baseline；
2. 固定 expert 输入为 `attention_output_residual/full`，扫描 router input 位置与形态；
3. 针对最有代表性的 router 结构，扫描 expert input 位置。

MoE router sweep 固定 expert 输入为 full hidden，也就是：

```text
expert_input_pos = attention_output_residual
expert_input_shape = full
```

这意味着 expert 始终读取完整的 `attention output + residual`，该组实验只测试 router 的输入位置和输入形态是否影响 specialization。

Router input 位置包括：

```text
attn_output, q, k, v, layer_input, hidden
```

这里 `attn_output` 对应 attention output without residual。若后续需要严格测试“attention input”，它在当前实现中等价于 `layer_input`。

Router input 形态包括：

```text
full, head, spectral
```

Targeted expert-input sweep 选择代表性 router，并测试不同 expert input 位置。

| ID | Run | Type | Router | Expert | Train loss | Eval loss | NTP acc | Inside-local | Local-boundary-not-high | High-boundary | Local spec | High spec | Eff. experts | Attn-local | Attn-high | Attn-expert |
|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `round5-dense-b4-u512-h64-s2000` | dense | `-` | `dense FFN` | - | 0.3792 | 92.69% | 98.98% | 93.33% | 11.99% | - | - | - | - | - | - |
| 2 | `round5-full-attn-output-exp-resid` | router-sweep | `attention_output/full` | `attention_output_residual/full` | 0.4250 | 0.4139 | 91.80% | 98.27% | 91.35% | 12.05% | 34.39% | 28.08% | 3.9462 | 50.73% | 76.34% | 51.33% |
| 3 | `round5-full-q-exp-resid` | router-sweep | `q/full` | `attention_output_residual/full` | 0.3680 | 0.3637 | 92.72% | 98.89% | 93.68% | 12.33% | 38.54% | 30.98% | 3.9298 | 50.08% | 78.16% | 51.61% |
| 4 | `round5-full-k-exp-resid` | router-sweep | `k/full` | `attention_output_residual/full` | 0.3620 | 0.3648 | 92.72% | 98.85% | 93.83% | 12.43% | 43.50% | 35.42% | 3.6992 | 48.48% | 77.34% | 52.87% |
| 5 | `round5-full-v-exp-resid` | router-sweep | `v/full` | `attention_output_residual/full` | 0.3690 | 0.3675 | 92.63% | 98.78% | 93.60% | 12.48% | 33.74% | 25.94% | 3.9678 | 46.23% | 75.47% | 44.12% |
| 6 | `round5-full-layer-input-exp-resid` | router-sweep | `layer_input/full` | `attention_output_residual/full` | 0.3530 | 0.3565 | 92.77% | 98.88% | 94.01% | 12.40% | 34.86% | 28.35% | 3.9706 | 47.39% | 75.18% | 45.76% |
| 7 | `round5-full-hidden-exp-resid` | router-sweep | `hidden/full` | `attention_output_residual/full` | 0.3730 | 0.3765 | 92.43% | 98.65% | 93.12% | 12.38% | 36.87% | 28.40% | 3.9806 | 48.02% | 74.70% | 48.00% |
| 8 | `round5-head-attn-output-exp-resid` | router-sweep | `attention_output/head` | `attention_output_residual/full` | 0.4200 | 0.4099 | 91.79% | 98.21% | 91.56% | 12.10% | 37.03% | 30.73% | 3.9581 | 53.37% | 78.01% | 55.21% |
| 9 | `round5-head-q-exp-resid` | router-sweep | `q/head` | `attention_output_residual/full` | 0.3740 | 0.3747 | 92.54% | 98.79% | 93.21% | 12.13% | 41.27% | 33.19% | 3.9840 | 49.58% | 77.97% | 56.82% |
| 10 | `round5-head-k-exp-resid` | router-sweep | `k/head` | `attention_output_residual/full` | 0.3550 | 0.3639 | 92.75% | 98.93% | 93.72% | 12.30% | 50.34% | 43.16% | 3.5054 | 53.34% | 83.52% | 63.21% |
| 11 | `round5-head-v-exp-resid` | router-sweep | `v/head` | `attention_output_residual/full` | 0.3450 | 0.3528 | 92.95% | 99.01% | 94.50% | 12.18% | 37.28% | 29.27% | 3.9612 | 52.32% | 83.41% | 47.14% |
| 12 | `round5-head-layer-input-exp-resid` | router-sweep | `layer_input/head` | `attention_output_residual/full` | 0.3370 | 0.3462 | 92.97% | 98.98% | 94.63% | 12.53% | 36.04% | 29.38% | 3.9822 | 48.31% | 80.64% | 46.75% |
| 13 | `round5-head-hidden-exp-resid` | router-sweep | `hidden/head` | `attention_output_residual/full` | 0.3570 | 0.3599 | 92.81% | 98.90% | 94.11% | 12.43% | 34.38% | 27.99% | 3.9939 | 50.53% | 81.15% | 47.11% |
| 14 | `round5-spectral-attn-output-exp-resid` | router-sweep | `attention_output/spectral` | `attention_output_residual/full` | 0.4820 | 0.4874 | 90.55% | 97.72% | 87.08% | 11.62% | 35.19% | 29.35% | 13.7743 | 55.08% | 79.59% | 41.97% |
| 15 | `round5-spectral-q-exp-resid` | router-sweep | `q/spectral` | `attention_output_residual/full` | 0.5050 | 0.5113 | 90.20% | 97.46% | 86.24% | 11.72% | 41.68% | 31.96% | 12.7323 | 53.34% | 79.78% | 41.51% |
| 16 | `round5-spectral-k-exp-resid` | router-sweep | `k/spectral` | `attention_output_residual/full` | 0.4690 | 0.4743 | 90.93% | 98.05% | 87.83% | 11.64% | 38.51% | 30.15% | 12.1473 | 56.91% | 85.26% | 42.13% |
| 17 | `round5-spectral-v-exp-resid` | router-sweep | `v/spectral` | `attention_output_residual/full` | 0.4990 | 0.5023 | 90.24% | 97.47% | 86.51% | 11.49% | 36.52% | 27.47% | 12.8272 | 54.78% | 82.98% | 37.32% |
| 18 | `round5-spectral-layer-input-exp-resid` | router-sweep | `layer_input/spectral` | `attention_output_residual/full` | 0.5250 | 0.5221 | 90.07% | 97.48% | 85.60% | 11.31% | 35.96% | 27.56% | 15.0174 | 55.98% | 84.19% | 36.92% |
| 19 | `round5-spectral-hidden-exp-resid` | router-sweep | `hidden/spectral` | `attention_output_residual/full` | 0.5030 | 0.5192 | 90.07% | 97.44% | 85.85% | 11.09% | 37.82% | 28.09% | 14.9591 | 52.48% | 77.50% | 37.46% |
| 20 | `round5-expertpos-targeted-rhead-k-eresid` | expert-input | `k/head` | `attention_output_residual/full` | 0.3500 | 0.3630 | 92.80% | 98.97% | 93.77% | 12.46% | 51.98% | 44.26% | 3.5483 | 52.89% | 83.16% | 63.53% |
| 21 | `round5-expertpos-targeted-rhead-k-eattn` | expert-input | `k/head` | `attention_output/full` | 0.3620 | 0.3678 | 92.81% | 98.98% | 93.87% | 12.18% | 48.37% | 39.68% | 3.6857 | 63.40% | 87.21% | 68.19% |
| 22 | `round5-expertpos-targeted-rhead-k-elayerin` | expert-input | `k/head` | `layer_input/full` | 0.3940 | 0.3957 | 92.26% | 98.64% | 92.63% | 11.26% | 42.04% | 35.26% | 3.9258 | 47.63% | 75.56% | 55.55% |
| 23 | `round5-expertpos-targeted-rhead-k-eq` | expert-input | `k/head` | `q/full` | 0.4390 | 0.4424 | 91.37% | 98.15% | 89.96% | 10.83% | 41.86% | 34.39% | 3.8701 | 36.53% | 64.39% | 48.12% |
| 24 | `round5-expertpos-targeted-rhead-k-ek` | expert-input | `k/head` | `k/full` | 0.4570 | 0.4594 | 90.89% | 97.72% | 89.26% | 10.58% | 37.92% | 30.67% | 3.9346 | 40.23% | 67.37% | 47.47% |
| 25 | `round5-expertpos-targeted-rhead-k-ev` | expert-input | `k/head` | `v/full` | 0.4470 | 0.4418 | 91.57% | 98.34% | 90.34% | 10.76% | 46.43% | 39.12% | 3.8655 | 57.22% | 84.56% | 62.75% |
| 26 | `round5-expertpos-targeted-rhead-k-ehidden` | expert-input | `k/head` | `hidden/full` | 0.3600 | 0.3633 | 92.81% | 98.91% | 94.03% | 12.46% | 52.88% | 46.03% | 3.4853 | 52.96% | 83.92% | 64.02% |
| 27 | `round5-expertpos-targeted-rhead-layerin-eresid` | expert-input | `layer_input/head` | `attention_output_residual/full` | 0.3390 | 0.3459 | 93.01% | 99.01% | 94.66% | 12.58% | 35.97% | 29.25% | 3.9744 | 48.10% | 80.53% | 46.22% |
| 28 | `round5-expertpos-targeted-rhead-layerin-eattn` | expert-input | `layer_input/head` | `attention_output/full` | 0.3550 | 0.3556 | 92.99% | 99.07% | 94.43% | 12.23% | 39.74% | 31.67% | 3.9551 | 62.22% | 86.78% | 59.93% |
| 29 | `round5-expertpos-targeted-rhead-layerin-elayerin` | expert-input | `layer_input/head` | `layer_input/full` | 0.3870 | 0.3938 | 92.19% | 98.56% | 92.54% | 11.21% | 38.27% | 29.27% | 3.9863 | 46.67% | 74.35% | 46.81% |
| 30 | `round5-expertpos-targeted-rhead-layerin-eq` | expert-input | `layer_input/head` | `q/full` | 0.4080 | 0.4106 | 92.02% | 98.54% | 91.71% | 11.24% | 37.11% | 29.76% | 3.9389 | 40.15% | 71.75% | 41.96% |
| 31 | `round5-expertpos-targeted-rhead-layerin-ek` | expert-input | `layer_input/head` | `k/full` | 0.4290 | 0.4418 | 91.36% | 98.12% | 90.11% | 10.73% | 36.22% | 29.46% | 3.9461 | 39.37% | 69.91% | 42.08% |
| 32 | `round5-expertpos-targeted-rhead-layerin-ev` | expert-input | `layer_input/head` | `v/full` | 0.4190 | 0.4202 | 91.95% | 98.57% | 91.36% | 10.88% | 35.84% | 28.76% | 3.9276 | 56.70% | 84.92% | 51.51% |
| 33 | `round5-expertpos-targeted-rhead-layerin-ehidden` | expert-input | `layer_input/head` | `hidden/full` | 0.3400 | 0.3458 | 93.00% | 99.05% | 94.50% | 12.46% | 36.05% | 30.06% | 3.9740 | 48.15% | 80.49% | 46.49% |
| 34 | `round5-expertpos-targeted-rhead-q-eattn` | expert-input | `q/head` | `attention_output/full` | 0.3660 | 0.3724 | 92.82% | 99.05% | 93.74% | 11.90% | 50.62% | 42.28% | 3.7744 | 65.82% | 88.50% | 72.58% |
| 35 | `round5-expertpos-targeted-rhead-q-ek` | expert-input | `q/head` | `k/full` | 0.4640 | 0.4784 | 90.57% | 97.54% | 88.21% | 10.65% | 40.02% | 29.94% | 3.9608 | 37.58% | 62.74% | 48.59% |
| 36 | `round5-expertpos-targeted-rfull-k-eattn` | expert-input | `k/full` | `attention_output/full` | 0.3740 | 0.3749 | 92.70% | 98.94% | 93.42% | 12.15% | 50.08% | 44.32% | 2.9656 | 65.55% | 88.55% | 70.99% |
| 37 | `round5-expertpos-targeted-rfull-layerin-eattn` | expert-input | `layer_input/full` | `attention_output/full` | 0.3520 | 0.3603 | 92.83% | 98.97% | 94.09% | 12.05% | 37.70% | 30.26% | 3.8144 | 62.31% | 86.65% | 58.02% |
| 38 | `round5-headhead-rattn-eattn` | overnight | `attention_output/head` | `attention_output/head` | 0.4750 | 0.4877 | 90.97% | 98.13% | 87.84% | 11.16% | 34.48% | 28.60% | 3.9707 | 66.79% | 88.88% | 62.81% |
| 39 | `round5-headhead-rattn-ehidden` | overnight | `attention_output/head` | `hidden/head` | 0.4490 | 0.4717 | 91.14% | 98.15% | 88.58% | 11.44% | 33.22% | 27.79% | 3.9524 | 66.07% | 88.86% | 60.35% |
| 40 | `round5-headhead-rattn-ek` | overnight | `attention_output/head` | `k/head` | 0.5260 | 0.5447 | 89.76% | 97.21% | 85.47% | 9.97% | 34.57% | 29.23% | 3.9629 | 58.60% | 82.87% | 57.38% |
| 41 | `round5-headhead-rattn-elayerin` | overnight | `attention_output/head` | `layer_input/head` | 0.4740 | 0.4924 | 90.82% | 97.95% | 88.03% | 10.40% | 32.96% | 28.02% | 3.9546 | 63.15% | 87.94% | 58.41% |
| 42 | `round5-headhead-rattn-eq` | overnight | `attention_output/head` | `q/head` | 0.5450 | 0.5413 | 89.87% | 97.25% | 85.96% | 9.87% | 34.65% | 29.77% | 3.9265 | 53.45% | 79.93% | 53.06% |
| 43 | `round5-headhead-rattn-eresid` | overnight | `attention_output/head` | `attention_output_residual/head` | 0.4540 | 0.4710 | 91.13% | 98.18% | 88.36% | 11.57% | 32.89% | 27.66% | 3.9618 | 65.58% | 88.69% | 59.63% |
| 44 | `round5-headhead-rattn-ev` | overnight | `attention_output/head` | `v/head` | 0.4970 | 0.5216 | 90.47% | 97.76% | 87.05% | 9.99% | 35.04% | 29.90% | 3.9378 | 63.24% | 87.92% | 59.24% |
| 45 | `round5-headhead-rhidden-eattn` | overnight | `hidden/head` | `attention_output/head` | 0.4690 | 0.4793 | 91.09% | 98.20% | 88.22% | 11.11% | 39.90% | 29.54% | 3.9354 | 66.58% | 89.26% | 61.99% |
| 46 | `round5-headhead-rhidden-ehidden` | overnight | `hidden/head` | `hidden/head` | 0.4440 | 0.4510 | 91.49% | 98.43% | 89.38% | 11.19% | 39.74% | 29.28% | 3.9704 | 65.13% | 88.44% | 59.27% |
| 47 | `round5-headhead-rhidden-ek` | overnight | `hidden/head` | `k/head` | 0.5020 | 0.5328 | 90.04% | 97.49% | 85.94% | 9.72% | 38.10% | 28.72% | 3.9863 | 55.63% | 81.29% | 53.26% |
| 48 | `round5-headhead-rhidden-elayerin` | overnight | `hidden/head` | `layer_input/head` | 0.4660 | 0.4821 | 91.09% | 98.10% | 88.86% | 10.40% | 38.58% | 28.39% | 3.9830 | 61.60% | 87.65% | 57.01% |
| 49 | `round5-headhead-rhidden-eq` | overnight | `hidden/head` | `q/head` | 0.5330 | 0.5420 | 89.68% | 97.20% | 85.13% | 9.87% | 37.76% | 28.97% | 3.9543 | 51.24% | 78.98% | 50.42% |
| 50 | `round5-headhead-rhidden-eresid` | overnight | `hidden/head` | `attention_output_residual/head` | 0.4390 | 0.4548 | 91.41% | 98.35% | 89.28% | 11.19% | 39.67% | 29.42% | 3.9697 | 64.84% | 88.51% | 59.07% |
| 51 | `round5-headhead-rhidden-ev` | overnight | `hidden/head` | `v/head` | 0.5020 | 0.5156 | 90.46% | 97.80% | 86.84% | 10.05% | 39.41% | 29.68% | 3.9513 | 64.14% | 88.51% | 59.20% |
| 52 | `round5-headhead-rk-eattn` | overnight | `k/head` | `attention_output/head` | 0.4530 | 0.4707 | 91.23% | 98.33% | 88.51% | 10.83% | 54.88% | 46.34% | 3.6093 | 68.25% | 90.15% | 73.58% |
| 53 | `round5-headhead-rk-ehidden` | overnight | `k/head` | `hidden/head` | 0.4290 | 0.4436 | 91.71% | 98.52% | 90.19% | 11.26% | 51.29% | 43.46% | 3.6817 | 66.00% | 89.95% | 70.29% |
| 54 | `round5-headhead-rk-ek` | overnight | `k/head` | `k/head` | 0.5130 | 0.5134 | 90.41% | 97.69% | 86.80% | 10.63% | 41.60% | 33.89% | 3.9188 | 58.00% | 83.14% | 60.58% |
| 55 | `round5-headhead-rk-elayerin` | overnight | `k/head` | `layer_input/head` | 0.4590 | 0.4734 | 91.23% | 98.33% | 88.68% | 10.35% | 49.02% | 41.10% | 3.6694 | 63.41% | 89.14% | 66.15% |
| 56 | `round5-headhead-rk-eq` | overnight | `k/head` | `q/head` | 0.5120 | 0.5155 | 90.41% | 97.74% | 86.81% | 9.99% | 51.84% | 44.89% | 3.4521 | 54.29% | 81.57% | 64.27% |
| 57 | `round5-headhead-rk-eresid` | overnight | `k/head` | `attention_output_residual/head` | 0.4390 | 0.4424 | 91.78% | 98.58% | 90.29% | 11.31% | 51.77% | 43.11% | 3.7192 | 65.43% | 89.72% | 69.96% |
| 58 | `round5-headhead-rk-ev` | overnight | `k/head` | `v/head` | 0.5010 | 0.5133 | 90.57% | 97.85% | 87.22% | 10.05% | 50.82% | 41.93% | 3.6890 | 62.60% | 88.38% | 66.81% |
| 59 | `round5-headhead-rlayerin-eattn` | overnight | `layer_input/head` | `attention_output/head` | 0.4370 | 0.4497 | 91.68% | 98.46% | 90.36% | 10.98% | 39.62% | 29.99% | 3.9728 | 68.50% | 90.86% | 62.73% |
| 60 | `round5-headhead-rlayerin-ehidden` | overnight | `layer_input/head` | `hidden/head` | 0.4410 | 0.4345 | 91.82% | 98.59% | 90.31% | 11.85% | 39.14% | 29.36% | 3.9765 | 65.81% | 89.47% | 59.42% |
| 61 | `round5-headhead-rlayerin-ek` | overnight | `layer_input/head` | `k/head` | 0.4890 | 0.5007 | 90.62% | 97.90% | 87.16% | 10.48% | 40.57% | 31.71% | 3.9311 | 56.75% | 82.54% | 54.06% |
| 62 | `round5-headhead-rlayerin-elayerin` | overnight | `layer_input/head` | `layer_input/head` | 0.4360 | 0.4578 | 91.53% | 98.43% | 89.82% | 10.58% | 38.96% | 29.42% | 3.9704 | 60.97% | 87.35% | 55.66% |
| 63 | `round5-headhead-rlayerin-eq` | overnight | `layer_input/head` | `q/head` | 0.4950 | 0.5108 | 90.51% | 97.81% | 87.05% | 10.05% | 37.84% | 29.64% | 3.9785 | 51.60% | 78.63% | 49.95% |
| 64 | `round5-headhead-rlayerin-eresid` | overnight | `layer_input/head` | `attention_output_residual/head` | 0.4380 | 0.4326 | 91.87% | 98.60% | 90.60% | 11.64% | 39.48% | 29.37% | 3.9798 | 65.37% | 89.47% | 59.12% |
| 65 | `round5-headhead-rlayerin-ev` | overnight | `layer_input/head` | `v/head` | 0.4870 | 0.5032 | 90.70% | 97.86% | 87.79% | 10.20% | 39.07% | 29.31% | 3.9857 | 62.81% | 88.12% | 57.07% |
| 66 | `round5-headhead-rq-eattn` | overnight | `q/head` | `attention_output/head` | 0.4660 | 0.4705 | 91.31% | 98.31% | 88.97% | 11.04% | 57.44% | 49.52% | 3.7571 | 68.63% | 90.39% | 76.72% |
| 67 | `round5-headhead-rq-ehidden` | overnight | `q/head` | `hidden/head` | 0.4410 | 0.4488 | 91.70% | 98.51% | 90.11% | 11.39% | 59.65% | 51.24% | 3.6352 | 66.87% | 90.10% | 76.72% |
| 68 | `round5-headhead-rq-ek` | overnight | `q/head` | `k/head` | 0.5200 | 0.5198 | 90.32% | 97.74% | 86.27% | 10.25% | 60.30% | 52.39% | 3.4832 | 57.27% | 82.83% | 72.66% |
| 69 | `round5-headhead-rq-elayerin` | overnight | `q/head` | `layer_input/head` | 0.4680 | 0.4739 | 91.20% | 98.27% | 88.75% | 10.50% | 58.34% | 51.98% | 3.5930 | 62.90% | 88.70% | 74.50% |
| 70 | `round5-headhead-rq-eq` | overnight | `q/head` | `q/head` | 0.5210 | 0.5152 | 90.35% | 97.80% | 86.26% | 10.05% | 50.27% | 42.71% | 3.6387 | 52.26% | 79.39% | 61.85% |
| 71 | `round5-headhead-rq-eresid` | overnight | `q/head` | `attention_output_residual/head` | 0.4400 | 0.4450 | 91.79% | 98.63% | 90.12% | 11.42% | 59.19% | 51.67% | 3.6149 | 67.03% | 90.37% | 77.25% |
| 72 | `round5-headhead-rq-ev` | overnight | `q/head` | `v/head` | 0.5030 | 0.5143 | 90.64% | 97.94% | 87.09% | 10.40% | 60.74% | 53.98% | 3.4029 | 64.49% | 88.97% | 76.56% |
| 73 | `round5-headhead-rv-eattn` | overnight | `v/head` | `attention_output/head` | 0.4480 | 0.4584 | 91.50% | 98.48% | 89.24% | 11.24% | 35.20% | 26.96% | 3.9680 | 68.34% | 90.06% | 61.17% |
| 74 | `round5-headhead-rv-ehidden` | overnight | `v/head` | `hidden/head` | 0.4280 | 0.4392 | 91.76% | 98.55% | 90.23% | 11.49% | 37.35% | 27.47% | 3.9835 | 66.16% | 89.73% | 59.02% |
| 75 | `round5-headhead-rv-ek` | overnight | `v/head` | `k/head` | 0.5000 | 0.5137 | 90.35% | 97.69% | 86.73% | 9.89% | 36.50% | 27.52% | 3.9703 | 57.87% | 83.59% | 53.17% |
| 76 | `round5-headhead-rv-elayerin` | overnight | `v/head` | `layer_input/head` | 0.4520 | 0.4685 | 91.28% | 98.35% | 88.90% | 10.38% | 35.17% | 27.76% | 3.9830 | 61.81% | 88.16% | 54.70% |
| 77 | `round5-headhead-rv-eq` | overnight | `v/head` | `q/head` | 0.5010 | 0.5118 | 90.57% | 97.85% | 87.05% | 10.48% | 34.82% | 27.66% | 3.9697 | 52.50% | 79.82% | 49.13% |
| 78 | `round5-headhead-rv-eresid` | overnight | `v/head` | `attention_output_residual/head` | 0.4370 | 0.4414 | 91.74% | 98.51% | 90.23% | 11.64% | 34.43% | 27.03% | 3.9800 | 65.43% | 89.70% | 58.26% |
| 79 | `round5-headhead-rv-ev` | overnight | `v/head` | `v/head` | 0.4970 | 0.5061 | 90.73% | 97.95% | 87.67% | 9.94% | 34.04% | 27.07% | 3.9826 | 62.40% | 88.10% | 55.45% |
| 80 | `round5-cap-rhead-k-ne16-topk1-c0` | overnight | `k/head` | `attention_output_residual/full` | 0.3430 | 0.3498 | 92.85% | 98.87% | 94.45% | 12.46% | 30.50% | 23.53% | 12.6973 | 47.23% | 76.53% | 45.98% |
| 81 | `round5-cap-rhead-k-ne16-topk1-c1` | overnight | `k/head` | `attention_output_residual/full` | 0.3250 | 0.3288 | 93.21% | 99.13% | 95.19% | 12.68% | 35.02% | 28.93% | 10.9758 | 48.39% | 81.65% | 45.70% |
| 82 | `round5-cap-rhead-k-ne16-topk2-c0` | overnight | `k/head` | `attention_output_residual/full` | 0.3250 | 0.3269 | 93.25% | 99.15% | 95.38% | 12.63% | 21.61% | 13.99% | 14.3079 | 45.62% | 78.63% | 34.36% |
| 83 | `round5-cap-rhead-k-ne16-topk2-c1` | overnight | `k/head` | `attention_output_residual/full` | 0.3160 | 0.3193 | 93.32% | 99.22% | 95.48% | 12.66% | 23.72% | 14.90% | 13.7301 | 46.68% | 80.94% | 34.34% |
| 84 | `round5-cap-rhead-k-ne4-topk1-c0` | overnight | `k/head` | `attention_output_residual/full` | 0.3620 | 0.3677 | 92.70% | 98.87% | 93.70% | 12.25% | 49.53% | 41.63% | 3.6429 | 51.37% | 79.91% | 62.10% |
| 85 | `round5-cap-rhead-k-ne4-topk1-c1` | overnight | `k/head` | `attention_output_residual/full` | 0.3310 | 0.3365 | 93.13% | 99.11% | 94.97% | 12.43% | 60.52% | 54.36% | 3.7117 | 49.54% | 81.14% | 67.18% |
| 86 | `round5-cap-rhead-k-ne4-topk2-c0` | overnight | `k/head` | `attention_output_residual/full` | 0.3340 | 0.3383 | 93.15% | 99.10% | 95.05% | 12.66% | 38.44% | 28.13% | 3.9827 | 51.14% | 83.12% | 49.87% |
| 87 | `round5-cap-rhead-k-ne4-topk2-c1` | overnight | `k/head` | `attention_output_residual/full` | 0.3230 | 0.3299 | 93.26% | 99.17% | 95.35% | 12.68% | 36.92% | 29.63% | 3.9566 | 46.72% | 78.60% | 46.11% |
| 88 | `round5-cap-rhead-k-ne8-topk1-c0` | overnight | `k/head` | `attention_output_residual/full` | 0.3540 | 0.3630 | 92.65% | 98.83% | 93.70% | 12.00% | 34.76% | 26.04% | 7.5119 | 47.24% | 75.24% | 48.83% |
| 89 | `round5-cap-rhead-k-ne8-topk1-c1` | overnight | `k/head` | `attention_output_residual/full` | 0.3310 | 0.3329 | 93.17% | 99.12% | 95.11% | 12.51% | 50.03% | 42.31% | 6.9308 | 50.82% | 83.02% | 57.56% |
| 90 | `round5-cap-rhead-k-ne8-topk2-c0` | overnight | `k/head` | `attention_output_residual/full` | 0.3220 | 0.3290 | 93.23% | 99.15% | 95.28% | 12.56% | 28.70% | 19.47% | 7.8453 | 49.52% | 81.19% | 40.78% |
| 91 | `round5-cap-rhead-k-ne8-topk2-c1` | overnight | `k/head` | `attention_output_residual/full` | 0.3210 | 0.3238 | 93.30% | 99.21% | 95.36% | 12.68% | 29.57% | 20.20% | 7.8101 | 48.10% | 80.54% | 40.04% |
| 92 | `round5-cap-rhead-layerin-ne16-topk1-c0` | overnight | `layer_input/head` | `attention_output_residual/full` | 0.3270 | 0.3323 | 93.13% | 99.06% | 95.02% | 12.81% | 19.82% | 12.18% | 14.9590 | 42.13% | 74.46% | 28.24% |
| 93 | `round5-cap-rhead-layerin-ne16-topk1-c1` | overnight | `layer_input/head` | `attention_output_residual/full` | 0.3140 | 0.3192 | 93.30% | 99.18% | 95.48% | 12.73% | 20.58% | 12.57% | 14.9511 | 46.11% | 80.37% | 29.98% |
| 94 | `round5-cap-rhead-layerin-ne16-topk2-c0` | overnight | `layer_input/head` | `attention_output_residual/full` | 0.3200 | 0.3210 | 93.29% | 99.21% | 95.35% | 12.68% | 17.97% | 9.81% | 15.6894 | 44.89% | 77.89% | 28.83% |
| 95 | `round5-cap-rhead-layerin-ne16-topk2-c1` | overnight | `layer_input/head` | `attention_output_residual/full` | 0.3120 | 0.3177 | 93.31% | 99.21% | 95.48% | 12.63% | 17.96% | 9.87% | 15.4651 | 44.62% | 79.53% | 27.27% |
| 96 | `round5-cap-rhead-layerin-ne4-topk1-c0` | overnight | `layer_input/head` | `attention_output_residual/full` | 0.3470 | 0.3498 | 92.94% | 98.92% | 94.69% | 12.38% | 35.44% | 29.37% | 3.9678 | 48.43% | 80.68% | 46.45% |
| 97 | `round5-cap-rhead-layerin-ne4-topk1-c1` | overnight | `layer_input/head` | `attention_output_residual/full` | 0.3270 | 0.3324 | 93.21% | 99.14% | 95.20% | 12.68% | 39.02% | 30.50% | 3.8991 | 48.83% | 80.00% | 46.05% |
| 98 | `round5-cap-rhead-layerin-ne4-topk2-c0` | overnight | `layer_input/head` | `attention_output_residual/full` | 0.3320 | 0.3349 | 93.22% | 99.17% | 95.17% | 12.48% | 36.53% | 27.09% | 3.9927 | 50.80% | 83.15% | 46.59% |
| 99 | `round5-cap-rhead-layerin-ne4-topk2-c1` | overnight | `layer_input/head` | `attention_output_residual/full` | 0.3200 | 0.3312 | 93.23% | 99.18% | 95.14% | 12.66% | 38.25% | 28.90% | 3.9740 | 46.90% | 78.75% | 44.25% |
| 100 | `round5-cap-rhead-layerin-ne8-topk1-c0` | overnight | `layer_input/head` | `attention_output_residual/full` | 0.3330 | 0.3339 | 93.16% | 99.12% | 95.02% | 12.61% | 24.97% | 17.40% | 7.7696 | 50.88% | 82.41% | 37.75% |
| 101 | `round5-cap-rhead-layerin-ne8-topk1-c1` | overnight | `layer_input/head` | `attention_output_residual/full` | 0.3270 | 0.3312 | 93.16% | 99.11% | 95.07% | 12.63% | 22.65% | 15.38% | 7.9048 | 44.45% | 74.39% | 33.71% |
| 102 | `round5-cap-rhead-layerin-ne8-topk2-c0` | overnight | `layer_input/head` | `attention_output_residual/full` | 0.3180 | 0.3258 | 93.23% | 99.16% | 95.30% | 12.43% | 24.90% | 16.09% | 7.8040 | 49.47% | 81.71% | 35.49% |
| 103 | `round5-cap-rhead-layerin-ne8-topk2-c1` | overnight | `layer_input/head` | `attention_output_residual/full` | 0.3230 | 0.3225 | 93.32% | 99.23% | 95.47% | 12.46% | 23.76% | 16.50% | 7.8771 | 47.94% | 81.10% | 34.91% |
表中指标含义：

1. `Local spec` / `High spec`：同一 local slot / high slot 内 token 被分到同一 expert bucket 的比例；
2. `Attn-local` / `Attn-high`：attention mass 落在同 local slot / high slot 的比例，包含 self attention；
3. `Attn-expert`：attention mass 落在同 expert bucket token 上的比例，衡量 routing bucket 与 attention retrieval bucket 的重合度。

## 2000-step 实验结论

1. **dense h64 训练到 2000 step 后已经很强，不能再把 1000-step dense 当作可靠 baseline。** 2000-step dense 达到 `NTP acc=92.69%`、`local-boundary-not-high=93.33%`，已经接近普通 MoE baseline。因此 Round5 当前数据设定下，MoE 的主要价值不是显著提高 NTP，而是观察是否能在预测能力相近时形成更好的 feature specialization。
2. **预测性能最好的结构是 `layer_input/head + attention_output_residual expert`。** 它达到 `NTP acc=93.01%`、`local-boundary-not-high=94.66%`，比 dense 2000 分别高约 `0.32%` 和 `1.33%`。提升存在，但幅度不大。
3. **specialization 最强的结构是 `k/head + full residual expert`。** 代表性结果是 `local spec=51.98%~52.88%`、`high spec=44.26%~46.03%`、`attention-expert mass=63.53%~64.02%`。这说明 `k` 空间和 head-level routing 更容易形成接近 attention retrieval bucket 的 expert bucket。
4. **expert 输入最好仍然保持完整 residual stream，即 `attention_output_residual/full` 或等价的 `hidden/full`。** 对 `k/head` 和 `layer_input/head` 来说，expert 输入改成 `q/k/v/layer_input` 会明显损害 NTP；`attention_output` 能提高 attention-expert mass，但通常不比 residual expert 更稳。当前结论是：router 可以使用更 feature-selective 的表征，expert 仍应处理完整 token state。
5. **`attention_output/full` expert 能提高 routing bucket 与 attention bucket 的重合度，但不是最稳的预测选择。** 例如 `q/head + attention_output expert` 的 `Attn-expert=72.58%`，`k/full + attention_output expert` 的 `Attn-expert=70.99%`，但 NTP 没有超过最好的 residual expert 结构。这说明 alignment 和 prediction utility 仍有 tradeoff。
6. **spectral router 当前不是有效方案。** 它的 effective experts 很高，但 NTP 与 local-boundary 明显低于 full/head router，说明当前 periodic SVD + band routing 主要增加了 bucket 数量，没有稳定产生 task-useful specialization。可能原因是 PCA/SVD 捕捉的是 batch variance 方向，而不是 next-token prediction 所需的 feature 方向。
7. **性能和 specialization 不是同一个目标。** `layer_input/head` 预测最好但 specialization 一般；`k/head` specialization 最强但 NTP 略低。后续如果目标是 reverse indexing，应优先沿 `k/head` 路线继续；如果目标是单纯 NTP，应优先沿 `layer_input/head` 路线继续。

## Overnight Sweep 结果补充

在上述实验之后，又补充了两组完整实验：

1. `head/head` sweep：`router_shape=head` 且 `expert_shape=head`，共 42 个配置；
2. capacity ablation：在 `k/head` 与 `layer_input/head` 上扫描 `num_experts={4,8,16}`、`topk={1,2}`、`common_expert={false,true}`，共 24 个配置。

Spectral sample-size / stability ablation 原计划运行 24 个配置，但在 MPS 上因 `torch.linalg.qr` 不支持而中断，只完成了第一个配置，因此这里不使用该组结果更新 spectral 结论。

### Head/Head Sweep

`head/head` 结构把 expert input 也限制在 per-head 子空间，而不是让每个 head-router expert 处理完整 hidden。该结构的平均结果是：

```text
NTP acc:                  90.97%
local-boundary-not-high:  88.19%
local specialization:     42.63%
high specialization:      34.61%
attention-expert mass:    61.62%
```

最强 specialization 配置包括：

```text
round5-headhead-rq-ev:
  NTP acc:              90.64%
  local specialization: 60.74%
  high specialization:  53.98%
  attention-expert:     76.56%

round5-headhead-rq-eresid:
  NTP acc:              91.79%
  local specialization: 59.19%
  high specialization:  51.67%
  attention-expert:     77.25%
```

结论是：**head/head 更容易形成 feature bucket 与 attention retrieval bucket 的对齐，但明显损害 NTP。** 这说明把 expert 输入也切成 head 子空间是一种更强 specialization inductive bias，但表达能力不足；当前更合理的结构仍然是 head-router + full expert input。

### Capacity Ablation

Capacity ablation 的平均趋势是：

```text
topk=1:
  NTP acc:              93.05%
  local specialization: 35.24%
  high specialization:  27.85%
  attention-expert:     45.79%

topk=2:
  NTP acc:              93.26%
  local specialization: 28.20%
  high specialization:  19.55%
  attention-expert:     38.57%

common=false:
  NTP acc:              93.07%
  local specialization: 30.26%
  high specialization:  22.06%

common=true:
  NTP acc:              93.24%
  local specialization: 33.17%
  high specialization:  25.34%
```

最好的 NTP 配置是：

```text
round5-cap-rhead-k-ne16-topk2-c1:
  NTP acc:                 93.32%
  local-boundary-not-high: 95.48%
  local specialization:    23.72%
  high specialization:     14.90%
```

最好的 balanced specialization 配置是：

```text
round5-cap-rhead-k-ne4-topk1-c1:
  NTP acc:                 93.13%
  local-boundary-not-high: 94.97%
  local specialization:    60.52%
  high specialization:     54.36%
  attention-expert:        67.18%
```

Capacity ablation 的结论是：

1. **为了 NTP，增加 capacity 有效，但这不再是同等 active budget 对比。** `topk=2` 与 `common_expert=true` 都会增加 active compute，因此这些实验说明“容量更大可以提升 NTP”，不说明结构本身在公平预算下更优。
2. **为了 specialization，`topk=1` 比 `topk=2` 更干净。** `topk=2` 让 token 同时进入多个 expert，预测更好，但 hard specialization 指标下降。
3. **expert 数量更多不等于 specialization 更强。** `num_experts=4` 的 local/high specialization 明显强于 `num_experts=8/16`；更多 experts 会把同一个 ground-truth feature 拆散到更多 bucket。
4. **`common_expert=true` 对 `k/head` 的 specialization 有帮助。** 当前最有价值的 specialization candidate 是 `k/head + topk=1 + common_expert=true`。

### Round5 当前总判断

从 specialization -> reverse KV 的角度，当前结论可以压缩为：

1. **为了 NTP，除 spectral 外，大多数 full/head MoE 结构差异不大。** 在当前 synthetic setting 下，dense h64 已经达到 `92.69%`，普通 full/head MoE 多数在 `92%~93%` 区间；capacity 增大后可以到 `93.3%` 左右，但这主要是容量收益。
2. **为了 specialization，`k/head` 是目前最有价值的 router 设计。** 这里的 `k/head` 指用 attention key 表征作为 router input，并按 attention head 切分后分别 routing。它比 residual/hidden/layer_input 更容易形成和 local/high slot 以及 attention retrieval bucket 对齐的 expert bucket。
3. **当前 specialization 仍然不够强，不能直接支撑稳定 reverse KV。** 即使最好的自然训练配置也只是 `local/high specialization` 约 `60%/54%`，还不是接近 deterministic 的 feature bucket。因此如果目标是让 expert routing 作为 KV reverse index，需要额外约束 routing 与 attention retrieval 对齐。
4. **下一步应保留 attention-derived routing objective 的方向。** 如果把 attention retrieval 结果显式约束到 gate 上能进一步提高 specialization，那么它正好补上当前自然训练 MoE 的缺口：`k/head` 提供较好的 pre-attention / attention-adjacent 表征，attention-derived loss 提供更硬的 feature bucket supervision。


当前实现注意事项：

1. `full` router + `full` expert 已支持；
2. `spectral` router + `full` expert 已支持；
3. `head` router + `full` expert 已支持：每个 head 单独 routing，但 expert 输入/输出仍是 full hidden；每个 head 的 expert intermediate 会按 `num_heads` 缩小，使总激活 intermediate budget 与 baseline MoE 对齐。

SD-MoE 第一版固定为 full-output expert：

```text
router input:
  spectral band

expert input:
  full attention_output + residual

expert output:
  full hidden

final output:
  common_output + routed_band_outputs
```

对应 hidden size 64 的第一版 spectral 设置为：

```text
moe_spectral_band_dims = 8,32,64
moe_spectral_num_experts_per_band = 0,4,4
moe_spectral_topk_per_band = 1,1,1
moe_spectral_intermediate_sizes = 32,48,48
```

其中第一段是 common band，不做 sparse routing；后两段分别做 top-1 routing。激活 intermediate budget 约为 `32 + 48 + 48 = 128`，与 baseline MoE 的单 expert intermediate size 对齐。


## 候选结构：Router Input 与 Expert Input 的组合

### Router input 位置

候选项包括：

1. hidden / residual stream；
2. attention output without residual；
3. layer input；
4. q；
5. k；
6. v；
7. SD / PCA 后的谱空间坐标。

### Router input 形态

候选项包括：

1. **完整表征：** 普通 MoE，对整个 hidden 做一次 routing；
2. **head-level 表征：** 每个 attention head 单独 routing；
3. **spectral-band 表征：** 对不同谱坐标段分别 routing。

### SD-MoE on Forward Representation

SD-MoE 的目标是把 token representation 从原始 hidden 坐标系转换到谱空间，再按谱空间中的不同 coordinate band 做不同层次的 routing / expert ownership。

核心假设是：

```text
hidden state 不是单一 feature，
而是 token identity / position / local context / higher-level context / common feature / rare feature 等多种 feature 的混合。

谱空间中的不同方向可能对应不同强度、不同频率、不同抽象层次的 feature。
```

因此，一个最小实现版本是：

```text
H_sample = sampled hidden states
H_sample = H_sample - mean(H_sample)
H_sample = U S V^T
z = h V

top spectral band      -> common expert
middle spectral band   -> routed expert group
tail spectral band     -> routed expert group
```

这里要特别明确：**SVD / PCA basis `V` 不参与反向传播，也不产生额外 SVD loss。** 它不是一个通过梯度学习的参数，而是一个周期性更新的统计量 / buffer，作用类似 running statistics 或外部 codebook。

具体训练方式建议如下：

```text
1. warmup 若干 step，让模型初步形成 representation；
2. 每隔 K step 采样一批 hidden states；
3. 对采样表征做 PCA / SVD，得到新的 basis V；
4. 将 V 写入模型 buffer；
5. forward 时使用 z = h @ V 做 spectral-band routing；
6. backward 时梯度只回传到 h、router、expert，不回传到 V，也不回传到 SVD 过程。
```

这个设计修正了一个重要问题：不能假设训练前就有一个已知的最终表征空间，因此不能直接离线采样“训练后模型”的 hidden 来得到 basis。更合理的是：

1. 训练早期 warmup 后估计一次 basis；
2. 或者训练过程中 periodic SVD update；
3. 或者未来测试 learnable orthogonal basis 作为对照。

Round 5 的优先实现应是第二种：**periodic SVD update + stop-gradient + `V` as buffer**。这样既不需要预先知道最终表征空间，也能让训练和推理使用同一套显式 basis。

### Spectral Band 与 Expert Group

SD-MoE 不应只把谱空间切成若干段，还应允许不同 spectral band 使用不同 expert 结构。

例如 hidden size 为 100 时，可以设计：

```text
dim 0-10:
  common band
  不做 sparse routing，进入 common expert

dim 10-20:
  middle/common-specific band
  4 experts, top-1 routing

dim 20-50:
  mid-level feature band
  8 experts, top-1 routing

dim 50-100:
  tail / rare feature band
  8 或更多 experts, top-k routing
```

这里“前 10 维不分发”的含义不是这些维度不经过 expert，而是这些 common feature 进入 common expert。后续 spectral bands 再进入各自的 routed expert group。

不同 band 使用不同 expert 数量是合理的，因为不同谱段可能对应不同粒度和不同频率的 feature：

1. **top band：** 更可能是 common / high-frequency / global feature，适合 common expert；
2. **middle band：** 可能对应稳定的 compositional feature，适合中等数量 routed experts；
3. **tail band：** 可能包含 rare / specific feature，也可能包含噪声，需要实验判断是增加 experts 还是压低容量。

第一版不建议把配置做得太复杂。可以从一个最小结构开始：

```text
top band:    common expert
middle band: 4 routed experts
tail band:   4 routed experts
```

如果这个版本能提升 feature selectivity，再扩展到：

```text
top band:    common expert
middle band: 4 routed experts
tail band:   8 routed experts
```

这样可以判断 tail band 是否真的需要更多 expert capacity。

### Expert input 位置

候选项包括：

1. 与 router input 相同；
2. attention output；
3. attention output + residual。

已完成实验优先固定 expert input 为 `attention output + residual`，只改变 router input。这样可以避免 expert 缺失预测所需信息，使实验更集中地回答 routing 是否形成 specialization。

### Expert input 形态

候选项包括：

1. **完整表征输入 expert：** gate 根据某个 feature 判断 expert，但 expert 处理完整 token state。这更稳，也更接近分类后由专家处理完整输入的结构；
2. **切分后的子空间输入 expert：** router 和 expert 都按相同子空间切分，expert 只处理对应 feature band。这更接近真正的 feature-subspace ownership，但实现更复杂，也更容易损失 NTP 性能。

Round 5 第一版采用方案 A：**full-output expert**。

具体定义如下：

```text
router input:
  spectral band / head / selected representation

expert input:
  full standard hidden, usually attention_output + residual

expert output:
  full hidden

final output:
  common_output + routed_band_outputs
```

也就是说，spectral band 只用于决定 routing，不限制 expert 只能读取或写回该 band。这样可以把实验问题集中在“spectral routing 是否产生更好的 specialization”，而不是同时引入“expert 只能处理子空间是否损害表达能力”的额外变量。

另一种方案是 band-output expert：

```text
expert_i: full hidden or band hidden -> intermediate -> band_dim
z_out = concat(band outputs)
h_out = z_out @ V^T
```

但在有线性 projection / mixing 的情况下，`concat + projection` 和 `projected sum` 本质上是两种等价的线性组织方式。band-output expert 的主要区别不是表达能力，而是更强的 inductive bias：它限制某个 expert 只能写回对应 spectral band。这个限制可能有研究价值，但第一版不使用，因为它会让 NTP 下降时难以判断原因。

### 参数预算与公平对比

SD-MoE 必须控制激活参数量，否则实验无法判断收益来自 specialization 还是来自更多 activated parameters。

设 hidden size 为 `d`，baseline MoE 的单个激活 expert 是：

```text
baseline expert: d -> B -> d
```

其中 `B` 是 baseline active intermediate budget。

SD-MoE v1 中，每个 token 会同时经过 common expert 和若干 routed band expert。为了公平，应满足：

```text
B_sd = r_common + k_middle * r_middle + k_tail * r_tail ~= B
```

其中：

1. `r_common` 是 common expert 的 intermediate size；
2. `r_middle` 是 middle band expert 的 intermediate size；
3. `r_tail` 是 tail band expert 的 intermediate size；
4. `k_middle / k_tail` 是对应 band 的 top-k 激活数量。

例如 baseline 是：

```text
d = 100
B = 200
baseline active expert: 100 -> 200 -> 100
```

则 SD-MoE v1 可以设置：

```text
common expert: 100 -> 40 -> 100
middle expert: 100 -> 60 -> 100, top-1
tail expert:   100 -> 100 -> 100, top-1

active budget = 40 + 60 + 100 = 200
```

这样每个 token 激活的 intermediate budget 与 baseline 对齐。第一版实现应优先保证这一点。

建议实验顺序是：

```text
阶段 1：router 按子空间 / head / spectral band 分发，expert 仍处理完整 hidden；
阶段 2：如果阶段 1 成立，再测试 expert 是否也按子空间切分。
```

## 评价指标

每个结构至少需要报告：

1. NTP loss / accuracy；
2. expert load / entropy / effective expert count；
3. local slot 与 expert assignment 的 MI / NMI；
4. higher-level slot 与 expert assignment 的 MI / NMI；
5. feature-to-expert purity；
6. same-feature same-expert rate；
7. attention retrieval bucket 与 expert bucket 的重合度；
8. 如果结构支持 pre-attention routing，还需要报告 KV 保留比例与 reverse-indexing NTP accuracy。

## 预期输出

Round 5 最终应回答：

```text
哪些 MoE 结构能在 synthetic 数据上形成更强 specialization；
这种 specialization 是 local-level、higher-level，还是 token-level shortcut；
这种结构是否比 baseline MoE 更适合作为 KV reverse index；
哪些结构值得进一步迁移到真实语料实验。
```
