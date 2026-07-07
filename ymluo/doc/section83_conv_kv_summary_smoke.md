# Section 83: Conv1D KV Summary Smoke

日期：2026-07-06

## 目标

验证一个新想法：

```text
能不能借鉴 CNN 的局部特征提取能力，
用 Conv1D 对连续 K/V block 做 summary，
把 old KV 压成少量 synthetic KV slots？
```

本节先不跑端到端生成，而是做 attention-output reconstruction：

```text
full output = Attn(Q, K_full, V_full)
compressed output = Attn(Q, K_summary, V_summary)

目标：compressed output 尽量接近 full output。
```

如果 reconstruction 都不行，端到端 PPL / generation 基本不会稳。

## 新增脚本

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_conv_kv_summary_reconstruction.py
```

脚本比较：

```text
mean_pool:
  每个 block 或 sub-block 做 K/V 平均。

last_token:
  每个 block 或 sub-block 取最后一个 token 的 K/V。

trained_conv:
  训练一个 Conv1D compressor:
  (K_block, V_block) -> (K_summary, V_summary)
  loss = MSE(Attn(Q, K_summary, V_summary), Attn(Q, K_full, V_full))
```

Conv 初始化为和 mean_pool 等价的 sub-block mean，因此如果训练后变好，说明学习到的局部权重确实有用。

## 数据模式

### smooth_local

局部平滑/短语型信息。

```text
K/V 有局部连续结构；
query 主要读取同一局部 block 的后半段信息。
```

这是 CNN 应该擅长的场景。

### needle_exact

单点精确查找。

```text
一个随机位置包含 needle key/value；
query 直接匹配这个 needle key。
```

这是 KV compression 最容易出问题的场景，理论上需要 raw fallback。

## 结果

### 6.25% active KV

命令：

```bash
python ymluo/projects/learned_hierarchical_summary_memory/src/run_conv_kv_summary_reconstruction.py \
  --output_dir ymluo/projects/learned_hierarchical_summary_memory/outputs/conv_kv_summary_smoke_b16_20260706 \
  --device cpu \
  --epochs 8 \
  --train_samples 256 \
  --test_samples 64 \
  --seq_len 256 \
  --dim 32 \
  --query_count 16 \
  --block_size 16 \
  --slots_per_block 1
```

| mode | mean rel MSE | conv rel MSE | conv / mean MSE | mean cosine | conv cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| smooth_local | 0.6283 | 0.4677 | 0.7443 | 0.6630 | 0.7426 |
| needle_exact | 0.9807 | 0.9403 | 0.9588 | 0.6972 | 0.2599 |

结论：

```text
高压缩下，Conv 对局部平滑信息有明显收益；
但 needle exact 仍然很差，不能替代 raw fallback。
```

### 12.5% active KV

命令：

```bash
python ymluo/projects/learned_hierarchical_summary_memory/src/run_conv_kv_summary_reconstruction.py \
  --output_dir ymluo/projects/learned_hierarchical_summary_memory/outputs/conv_kv_summary_smoke_20260706 \
  --device cpu \
  --epochs 6 \
  --train_samples 192 \
  --test_samples 48 \
  --seq_len 256 \
  --dim 32 \
  --query_count 16 \
  --block_size 8 \
  --slots_per_block 1
```

| mode | mean rel MSE | conv rel MSE | conv / mean MSE | mean cosine | conv cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| smooth_local | 0.3353 | 0.1552 | 0.4630 | 0.8626 | 0.9252 |
| needle_exact | 0.9516 | 0.8324 | 0.8747 | 0.8912 | 0.4552 |

结论：

```text
12.5% active KV 下，Conv 对 smooth_local 的 reconstruction error 降低约 54%。
needle_exact 的 MSE 略有下降，但绝对误差仍然很高。
```

### 25% active KV

命令：

```bash
python ymluo/projects/learned_hierarchical_summary_memory/src/run_conv_kv_summary_reconstruction.py \
  --output_dir ymluo/projects/learned_hierarchical_summary_memory/outputs/conv_kv_summary_smoke_slots2_20260706 \
  --device cpu \
  --epochs 6 \
  --train_samples 192 \
  --test_samples 48 \
  --seq_len 256 \
  --dim 32 \
  --query_count 16 \
  --block_size 8 \
  --slots_per_block 2
```

| mode | mean rel MSE | conv rel MSE | conv / mean MSE | mean cosine | conv cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| smooth_local | 0.0778 | 0.0788 | 1.0126 | 0.9729 | 0.9645 |
| needle_exact | 0.7792 | 0.6500 | 0.8342 | 0.9652 | 0.6756 |

结论：

```text
25% active KV 下，sub-block mean 已经很强；
Conv 对 smooth_local 没有额外收益。
needle_exact 仍然不可靠。
```

## 当前判断

这次 smoke 支持下面判断：

1. **Conv KV summary 是可行 backend，但适用范围有限。**

   它适合局部平滑/短语型/可聚合信息，尤其是在压缩率很高时比 mean pooling 更好。

2. **Conv KV summary 不应该处理 exact retrieval。**

   needle/exact lookup 仍需要 raw span / prefix span / retrieval fallback。

3. **Conv 的价值区间可能是 6%-12.5% active KV。**

   到 25% active KV 时，简单 sub-block mean 已经足够强，Conv 的额外收益变小。

4. **下一步应该接真实 Qwen KV/Q 做 reconstruction。**

   当前结果是 synthetic attention manifold，不是端到端模型结果。
   下一步需要从真实 forward 中采集：

   ```text
   Q_test from decode/query positions
   K_old, V_old from historical cache
   full attention output per layer/head
   ```

   然后按 layer/head 评估：

   ```text
   mean_pool
   trained_conv
   typed_summary
   raw_span fallback
   ```

## 对论文主线的影响

比较稳的定位是：

```text
typed summary KV 作为主记忆路径；
Conv KV summary 作为局部可压缩 old spans 的 synthetic KV backend；
exact / needle / code / multi-value 查询走 raw span fallback；
risk-aware router 决定是否使用 Conv KV。
```

不建议把它写成：

```text
Conv KV 可以通用替代 raw KV。
```

目前证据不支持这个 claim。
