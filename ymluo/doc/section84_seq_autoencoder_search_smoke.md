# Section 84: Seq Autoencoder Compression Search Smoke

日期：2026-07-06

## 目标

验证一个新的压缩搜索思路：

```text
先把 K/V 序列沿 seq 维降维成 latent sequence；
需要时再升维重建 K/V；
或者直接在低维 latent sequence 上做 block search。
```

这个实验专门检查一个风险：

```text
重建 MSE 小，不一定代表搜索 top-k / attention target 保序。
```

因此本节同时报告：

```text
1. K/V reconstruction relative MSE
2. reconstructed K/V 的 attention-output relative MSE
3. reconstructed K 的 token top-k recall
4. mean block / reconstructed block / latent block 的 block search recall
```

## 新增脚本

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_seq_autoencoder_search_smoke.py
```

模型：

```text
encoder:
  Conv1D stride=block_size
  (K,V) seq -> latent seq

decoder:
  ConvTranspose1D stride=block_size
  latent seq -> reconstructed (K,V) seq

latent searcher:
  q -> q_latent
  score(block) = q_latent · latent_block
```

可选：

```text
--joint_search_weight
```

用于在 autoencoder 训练时加入 block-search CE loss，让 latent 不只为 MSE 服务。

## 数据模式

### smooth_local

局部平滑/短语型信息。适合压缩。

### needle_exact

一个稀有 needle key/value 随机插在序列中，query 精确匹配 needle key。

这个模式用于测试：

```text
整体重建 MSE 看起来可接受时，
是否仍然会漏掉搜索上最重要的稀有 token。
```

## 结果一：3.125% latent storage

命令：

```bash
python ymluo/projects/learned_hierarchical_summary_memory/src/run_seq_autoencoder_search_smoke.py \
  --output_dir ymluo/projects/learned_hierarchical_summary_memory/outputs/seq_autoencoder_search_smoke_20260706 \
  --device cpu \
  --ae_epochs 10 \
  --search_epochs 6 \
  --train_samples 256 \
  --test_samples 64 \
  --seq_len 256 \
  --dim 32 \
  --block_size 8 \
  --latent_dim 16 \
  --query_count 16
```

latent storage ratio：

```text
blocks * latent_dim / (seq_len * 2 * dim) = 3.125%
```

| mode | K rel MSE | V rel MSE | attention rel MSE | recon token top1 | mean block top1 | recon block top1 | latent block top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smooth_local | 0.7258 | 0.5011 | 0.6260 | 0.1719 | 0.9258 | 0.6865 | 0.7207 |
| needle_exact | 0.7581 | 0.5353 | 1.0009 | 0.0244 | 0.3535 | 0.0928 | 0.0586 |

结论：

```text
3.125% 太激进。
smooth_local 还能做一些 block search；
needle_exact 基本失败。
```

## 结果二：12.5% latent storage

命令：

```bash
python ymluo/projects/learned_hierarchical_summary_memory/src/run_seq_autoencoder_search_smoke.py \
  --output_dir ymluo/projects/learned_hierarchical_summary_memory/outputs/seq_autoencoder_search_smoke_lat64_20260706 \
  --device cpu \
  --ae_epochs 12 \
  --search_epochs 8 \
  --train_samples 256 \
  --test_samples 64 \
  --seq_len 256 \
  --dim 32 \
  --block_size 8 \
  --latent_dim 64 \
  --query_count 16
```

| mode | K rel MSE | V rel MSE | attention rel MSE | recon token top1 | mean block top1 | recon block top1 | latent block top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smooth_local | 0.2253 | 0.1325 | 0.1808 | 0.6270 | 0.9258 | 0.9688 | 0.9922 |
| needle_exact | 0.2458 | 0.1667 | 1.0004 | 0.3330 | 0.3574 | 0.5791 | 0.0928 |

结论：

```text
smooth_local 明显可行：
  latent block top1 = 99.22%
  recon block top1 = 96.88%

needle_exact 仍然不安全：
  K/V MSE 已经比 3.125% 好很多，
  但 attention rel MSE 仍约 1.0，
  latent block top1 只有 9.28%。
```

这说明：

```text
低 MSE 不等于可搜索。
```

## 结果三：12.5% latent storage + joint search loss

命令：

```bash
python ymluo/projects/learned_hierarchical_summary_memory/src/run_seq_autoencoder_search_smoke.py \
  --output_dir ymluo/projects/learned_hierarchical_summary_memory/outputs/seq_autoencoder_search_smoke_lat64_joint_20260706 \
  --device cpu \
  --ae_epochs 12 \
  --search_epochs 8 \
  --joint_search_weight 0.2 \
  --train_samples 256 \
  --test_samples 64 \
  --seq_len 256 \
  --dim 32 \
  --block_size 8 \
  --latent_dim 64 \
  --query_count 16
```

| mode | K rel MSE | V rel MSE | attention rel MSE | recon token top1 | mean block top1 | recon block top1 | latent block top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smooth_local | 0.1637 | 0.1361 | 0.1062 | 0.5723 | 0.9258 | 0.9854 | 0.9971 |
| needle_exact | 0.2571 | 0.2105 | 1.0077 | 0.2988 | 0.3574 | 0.5195 | 0.1914 |

结论：

```text
joint search loss 对 smooth_local 有帮助；
needle_exact 的 latent block top1 从 9.28% 提到 19.14%，
但仍然低于 mean block baseline 的 35.74%，不能作为 exact search 主路径。
```

额外测试 `joint_search_weight=1.0` 没有救 needle，反而使 reconstruction 变差，因此不是简单加大 search loss 就能解决。

## 结果四：增强 loss 消融

新增参数已经接入脚本：

```text
--attention_loss_weight
--block_search_weight
--topk_score_weight
--score_topk
--score_temperature
--rare_recon_weight
--rare_token_fraction
```

对应训练目标：

```text
loss = reconstruction_loss
     + attention_loss_weight * MSE(Attn(Q, K_recon, V_recon), Attn(Q, K, V))
     + block_search_weight * CE(latent_block_scores, oracle_block)
     + topk_score_weight * top-k block score distillation
     + rare_recon_weight * high-norm token reconstruction
```

### 4.1 Balanced multi-loss

命令：

```bash
python ymluo/projects/learned_hierarchical_summary_memory/src/run_seq_autoencoder_search_smoke.py \
  --output_dir ymluo/projects/learned_hierarchical_summary_memory/outputs/seq_autoencoder_search_smoke_lat64_multiloss_20260706 \
  --device cpu \
  --ae_epochs 12 \
  --search_epochs 8 \
  --train_samples 256 \
  --test_samples 64 \
  --seq_len 256 \
  --dim 32 \
  --block_size 8 \
  --latent_dim 64 \
  --query_count 16 \
  --attention_loss_weight 2.0 \
  --block_search_weight 0.2 \
  --topk_score_weight 0.2 \
  --rare_recon_weight 1.0 \
  --rare_token_fraction 0.02
```

| mode | K rel MSE | V rel MSE | attention rel MSE | recon block top1 | latent block top1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| smooth_local | 0.2325 | 0.1849 | 0.1656 | 0.9795 | 0.9951 |
| needle_exact | 0.5906 | 0.8694 | 0.9903 | 0.3135 | 0.1621 |

结论：

```text
balanced multi-loss 对 smooth_local 仍然很好；
但 needle_exact 没有改善，rare/attention 权重过大还会破坏 K/V reconstruction。
```

### 4.2 Block + top-k score distillation

命令：

```bash
python ymluo/projects/learned_hierarchical_summary_memory/src/run_seq_autoencoder_search_smoke.py \
  --output_dir ymluo/projects/learned_hierarchical_summary_memory/outputs/seq_autoencoder_search_smoke_lat64_block_topk_20260706 \
  --device cpu \
  --ae_epochs 12 \
  --search_epochs 8 \
  --train_samples 256 \
  --test_samples 64 \
  --seq_len 256 \
  --dim 32 \
  --block_size 8 \
  --latent_dim 64 \
  --query_count 16 \
  --block_search_weight 0.2 \
  --topk_score_weight 1.0
```

| mode | K rel MSE | V rel MSE | attention rel MSE | recon block top1 | latent block top1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| smooth_local | 0.2012 | 0.1840 | 0.1740 | 0.9746 | 0.9883 |
| needle_exact | 0.4551 | 0.3928 | 1.0037 | 0.3447 | 0.0752 |

结论：

```text
top-k score distillation 没有救 exact needle。
当前简单 latent dot-product scorer 不是 exact retrieval 的合适读出方式。
```

### 4.3 Rare-token preservation

低权重 rare preservation：

```bash
python ymluo/projects/learned_hierarchical_summary_memory/src/run_seq_autoencoder_search_smoke.py \
  --output_dir ymluo/projects/learned_hierarchical_summary_memory/outputs/seq_autoencoder_search_smoke_lat64_rare_low_20260706 \
  --device cpu \
  --ae_epochs 12 \
  --search_epochs 8 \
  --train_samples 256 \
  --test_samples 64 \
  --seq_len 256 \
  --dim 32 \
  --block_size 8 \
  --latent_dim 64 \
  --query_count 16 \
  --rare_recon_weight 0.2 \
  --rare_token_fraction 0.01
```

| mode | K rel MSE | V rel MSE | attention rel MSE | recon token top1 | recon block top1 | latent block top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smooth_local | 0.2328 | 0.1293 | 0.1636 | 0.5986 | 0.9668 | 0.9834 |
| needle_exact | 0.3016 | 0.2404 | 1.0225 | 0.4619 | 0.6543 | 0.0791 |

中等权重 rare preservation：

```bash
python ymluo/projects/learned_hierarchical_summary_memory/src/run_seq_autoencoder_search_smoke.py \
  --output_dir ymluo/projects/learned_hierarchical_summary_memory/outputs/seq_autoencoder_search_smoke_lat64_rare_mid_20260706 \
  --device cpu \
  --ae_epochs 12 \
  --search_epochs 8 \
  --train_samples 256 \
  --test_samples 64 \
  --seq_len 256 \
  --dim 32 \
  --block_size 8 \
  --latent_dim 64 \
  --query_count 16 \
  --rare_recon_weight 0.5 \
  --rare_token_fraction 0.01
```

| mode | K rel MSE | V rel MSE | attention rel MSE | recon token top1 | recon block top1 | latent block top1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| smooth_local | 0.2738 | 0.1606 | 0.2057 | 0.5537 | 0.9463 | 0.9824 |
| needle_exact | 0.3888 | 0.4232 | 1.0044 | 0.4922 | 0.5908 | 0.0928 |

结论：

```text
rare_recon_weight=0.2 是当前最有用的增强项。
它把 needle_exact 的 reconstructed-K block top1 从 baseline 57.91% 提高到 65.43%，
同时 smooth_local 仍保持 96.68% reconstructed-K block top1。

但是 latent block top1 仍然很低。
这说明当前可行路径更像：

  compressed latent storage -> query-time decode reconstructed K -> search/gather raw span

而不是：

  direct q · latent block search
```

## 当前判断

这个思路不应该废弃，但必须重新定义目标：

### 可行部分

```text
seq autoencoder latent 可以作为局部/平滑/summary/generation 类 old memory 的 compressed search index。
```

在 smooth_local 上：

```text
12.5% latent storage:
  latent block top1 = 99.22%

12.5% + joint search:
  latent block top1 = 99.71%
```

### 不可行部分

```text
不能用 reconstruction MSE 证明 exact retrieval 安全。
```

needle_exact 上：

```text
12.5% latent storage:
  K rel MSE = 0.2458
  V rel MSE = 0.1667
  attention rel MSE = 1.0004
  latent block top1 = 9.28%
```

这就是低 MSE 误导搜索的典型例子。

## 下一步建议

如果继续做 compression search，应该换成下面的训练目标：

```text
loss = reconstruction_loss
     + λ1 * attention_output_distillation
     + λ2 * block_search_cross_entropy
     + λ3 * topk_score_distillation
     + λ4 * rare-token / high-norm-token preservation
```

从本节消融看，当前推荐先采用更保守的版本：

```text
reconstruction loss
+ low-weight rare-token preservation

用途：
  用 latent 压缩存储 K-search index；
  query 时升维重建 K；
  在 reconstructed K 上做 block/page search；
  命中后回退 gather raw K/V span。
```

暂时不建议把 direct latent scorer 作为 exact retrieval 主路径。

并且 action space 必须保留：

```text
compressed latent search:
  用于 summary/generation/local smooth old memory。

raw span / prefix span fallback:
  用于 exact / needle / code / multi-value / high-risk query。
```

## 对当前方法主线的影响

更稳的论文叙事是：

```text
typed summary KV 是主记忆路径；
seq-autoencoder latent 是可选 compressed search index；
Conv KV / reconstructed KV 是 synthetic backend；
exact retrieval 由 risk-aware router 切到 raw span fallback。
```

不建议写成：

```text
只要 seq autoencoder MSE 小，就可以用低维 latent 替代 KV search。
```

本节实验已经显示这个说法不成立。
