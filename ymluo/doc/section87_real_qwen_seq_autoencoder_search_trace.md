# Section 87: Real Qwen Seq-AE Search Trace Smoke

日期：2026-07-07

## 目标

把 Section 84 的 seq autoencoder compression search 从 synthetic attention manifold 推进到真实 Qwen trace：

```text
full Qwen forward -> hidden_states
每层重算 RoPE 后 Q/K/V
构造真实 (K_old, V_old, Q_query) 样本
训练 seq autoencoder:
  (K_old, V_old) -> latent seq -> reconstructed (K, V)
评估:
  mean block search
  reconstructed-K search
  direct latent block search
```

这一步只验证 search/reconstruction，不做端到端 decode。

## 新增脚本

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_real_qwen_seq_ae_search_trace.py
```

脚本特性：

```text
1. 使用 AutoModelForCausalLM 跑 Qwen full forward。
2. 开 output_hidden_states=True。
3. 对每层 hidden_states[i] 重新经过 input_layernorm + q_proj/k_proj/v_proj。
4. 使用 model.model.rotary_emb + apply_rotary_pos_emb 得到 RoPE 后 Q/K。
5. 按 KV head 分组 query heads。
6. 每个 case/layer/kv_head 形成一个训练样本。
```

主要指标：

```text
recon_k_relative_mse
recon_v_relative_mse
recon_attention_relative_mse
mean_block_top1/top3
recon_block_top1/top3
latent_block_top1/top3
```

## Tiny smoke

命令：

```bash
python ymluo/projects/learned_hierarchical_summary_memory/src/run_real_qwen_seq_ae_search_trace.py \
  --output_dir ymluo/projects/learned_hierarchical_summary_memory/outputs/real_qwen_seq_ae_search_trace_tiny_20260707 \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --local_files_only true \
  --device_map none \
  --dtype float32 \
  --attn_implementation eager \
  --prompt_tokens 128 \
  --page_tokens 32 \
  --cases old_single,two_old \
  --layers 0-1 \
  --kv_heads 0-1 \
  --max_query_tokens 4 \
  --block_size 8 \
  --latent_dim 32 \
  --ae_epochs 2 \
  --search_epochs 1 \
  --batch_size 4 \
  --rare_recon_weight 0.2 \
  --rare_token_fraction 0.01
```

这个只用于验证脚本链路，样本太少且 latent storage 只有 1.56%，不作为结论。

## Small smoke

命令：

```bash
python ymluo/projects/learned_hierarchical_summary_memory/src/run_real_qwen_seq_ae_search_trace.py \
  --output_dir ymluo/projects/learned_hierarchical_summary_memory/outputs/real_qwen_seq_ae_search_trace_small_20260707 \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --local_files_only true \
  --device_map none \
  --dtype float32 \
  --attn_implementation eager \
  --prompt_tokens 128 \
  --page_tokens 32 \
  --cases old_single,two_old,decoy_exact \
  --layers 0-5 \
  --kv_heads 0-3 \
  --max_query_tokens 4 \
  --block_size 8 \
  --latent_dim 128 \
  --ae_epochs 6 \
  --search_epochs 3 \
  --batch_size 8 \
  --rare_recon_weight 0.2 \
  --rare_token_fraction 0.01
```

设置：

```text
model = Qwen/Qwen3-0.6B
context = 128 tokens
layers = 0..5
kv heads = 0..3
cases = old_single, two_old, decoy_exact
trace samples = 72
train/test = 54/18
latent storage ratio vs K/V = 6.25%
```

### 结果：rare preservation

| split | group | samples | mean b1 | recon-K b1 | latent b1 | mean b3 | recon-K b3 | latent b3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | overall | 54 | 31.48% | 12.73% | 67.36% | 61.57% | 21.76% | 83.80% |
| test | overall | 18 | 24.31% | 13.19% | 65.28% | 46.53% | 20.14% | 75.69% |
| test | decoy_exact | 6 | 6.25% | 25.00% | 93.75% | 27.08% | 29.17% | 93.75% |
| test | old_single | 5 | 27.50% | 12.50% | 62.50% | 47.50% | 25.00% | 72.50% |
| test | two_old | 7 | 37.50% | 3.57% | 42.86% | 62.50% | 8.93% | 62.50% |

观察：

```text
1. reconstructed-K search 在真实 Qwen trace 上很弱。
2. direct latent scorer 在这个小设置里反而明显强于 mean block。
3. rare preservation 对 K/V reconstruction 有帮助，但没有让 reconstructed-K search 成为主路径。
```

注意：

```text
recon_v_relative_mse 和 recon_attention_relative_mse 很大。
这说明当前 AE decoder 还不能可靠重建 V，也不能替代 attention output。
因此这里的有效信号主要是 search recall，不是 synthetic KV replacement。
```

## Ablation: no rare preservation

命令同上，但：

```bash
--rare_recon_weight 0.0
```

| split | group | samples | mean b1 | recon-K b1 | latent b1 | mean b3 | recon-K b3 | latent b3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | overall | 54 | 31.48% | 11.11% | 68.52% | 61.57% | 24.31% | 83.56% |
| test | overall | 18 | 24.31% | 8.33% | 64.58% | 46.53% | 21.53% | 76.39% |
| test | decoy_exact | 6 | 6.25% | 16.67% | 93.75% | 27.08% | 25.00% | 93.75% |
| test | old_single | 5 | 27.50% | 0.00% | 55.00% | 47.50% | 17.50% | 67.50% |
| test | two_old | 7 | 37.50% | 7.14% | 46.43% | 62.50% | 21.43% | 67.86% |

对比：

```text
rare preservation:
  test recon-K top1: 8.33% -> 13.19%
  test latent top1: 64.58% -> 65.28%
  test K MSE:       0.3636 -> 0.3174
  test V MSE:       81.86 -> 38.39
```

结论：

```text
rare preservation 有正向作用，但主要是改善 reconstruction；
当前真正有用的是 latent scorer，而不是 reconstructed-K search。
```

## 当前判断

这一步改变了 Section 84 的倾向：

```text
synthetic 上：
  reconstructed-K search 更像可行路径；
  direct latent search 很弱。

真实 Qwen 小 trace 上：
  reconstructed-K search 很弱；
  trainable latent scorer 明显强于 mean block。
```

所以不能过早锁定一种后端。下一步应该扩大真实 trace，并分别评估：

```text
A. latent scorer as compressed search index
B. latent -> reconstructed K -> search
C. mean block baseline
D. raw span oracle
```

## 下一步建议

P0:

```text
在 Qwen3-0.6B 上扩大到：
  context = 1024 或 2048
  layers = all 或中后层子集
  kv_heads = all
  cases >= 8
```

P1:

```text
把 latent scorer 的训练从后置小 Linear 改为 pairwise/ranking loss：
  positive = oracle top block
  negatives = hard blocks from mean/recon search
```

P2:

```text
如果 latent top3 仍然稳定高于 mean baseline，
就接入 raw span gather smoke：

latent scorer -> top-k page/span -> gather raw K/V -> answer NLL/exact
```

暂时不建议做：

```text
latent -> reconstructed K/V -> synthetic attention replacement
```

因为当前 V reconstruction 和 attention output reconstruction 明显不够。
