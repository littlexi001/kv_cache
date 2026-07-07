# Section 88: Latent Scorer Raw Span Smoke

日期：2026-07-07

## 目标

把 Section 87 的 real-Qwen latent scorer 接到端到端检索链路：

```text
real Qwen Q/K/V trace
-> train latent scorer
-> query-time rank old remote pages
-> protect sink/recent separately
-> gather raw span or rebuild selected text prompt
-> answer NLL / exact
```

这一步不是验证 reconstructed K/V，而是验证“压缩 latent index 只负责搜索，命中后回取原始 span”。

## 新增脚本

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_latent_scorer_raw_span_smoke.py
```

脚本要点：

```text
1. 复用 run_real_qwen_seq_ae_search_trace.py 收集真实 Qwen trace 并训练 AE + latent scorer。
2. page ranking 默认过滤 page 0 sink 和 recent 覆盖页，因为这些应由固定策略保护。
3. 支持 --page_halo_pages，命中中心页后回取相邻页，避免事实 record 被 page 边界切断。
4. 同时评估 KV-native sparse gather 和 prompt rebuild selected text。
5. 输出 center page recall 和 span page recall，避免 halo 后误读命中率。
```

## 关键命令

最终主结果使用 256-token / 8-page 设置：

```bash
python ymluo/projects/learned_hierarchical_summary_memory/src/run_latent_scorer_raw_span_smoke.py \
  --output_dir ymluo/projects/learned_hierarchical_summary_memory/outputs/latent_scorer_raw_span_smoke_256_halo1_span_decode16_20260707 \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --local_files_only true \
  --device_map none \
  --dtype float32 \
  --attn_implementation eager \
  --prompt_tokens 256 \
  --page_tokens 32 \
  --recent_tokens 32 \
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
  --rare_token_fraction 0.01 \
  --top_pages 1,2,3 \
  --page_halo_pages 1 \
  --exclude_sink_pages 1 \
  --exclude_recent_from_latent true \
  --decode_steps 16 \
  --include_prompt_rebuild true
```

设置：

```text
model = Qwen/Qwen3-0.6B
context = 256 tokens
page = 32 tokens
pages = 8
recent = 32 tokens
trace samples = 72
latent storage ratio vs K/V = 6.25%
```

## 主结果

| method | active tokens | center recall | span recall | NLL | exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_kv_cache | 256.0 | 1.000 | 1.000 | 0.645 | 1.000 |
| kv_native_recent_only | 32.0 | 0.000 | 0.000 | 8.654 | 0.000 |
| kv_native_oracle_pages_plus_recent_absolute | 149.3 | 1.000 | 1.000 | 4.688 | 0.000 |
| prompt_rebuild_oracle_pages_text | 170.3 | 1.000 | 1.000 | 0.845 | 0.667 |
| kv_native_latent_remote_top1_plus_recent | 128.0 | 0.333 | 0.500 | 4.582 | 0.000 |
| kv_native_latent_remote_top2_plus_recent | 170.7 | 0.333 | 0.667 | 1.871 | 0.333 |
| kv_native_latent_remote_top3_plus_recent | 224.0 | 0.333 | 1.000 | 0.669 | 1.000 |
| prompt_rebuild_latent_remote_top3_text | 242.7 | 0.333 | 1.000 | 0.911 | 0.667 |

输出目录：

```text
ymluo/projects/learned_hierarchical_summary_memory/outputs/latent_scorer_raw_span_smoke_256_halo1_span_decode16_20260707
```

## 观察

1. latent scorer 有真实信号，但当前还不够准。
   - old_single 能把证据 page 5 排到 remote top1。
   - two_old 会偏向 page 1/5/6，中心页没命中 page 2/4，但 halo 后覆盖到了证据。
   - decoy_exact 会先选旧 decoy page 2，current page 5 需要更大 top-k 或 halo 才被覆盖。

2. halo 很重要。
   - 事实 record 是从 page 中间插入的，32-token page 会把 record 切到相邻页。
   - 不加 halo 时 oracle text 在 256-token 设置上也会失败。
   - 加 `page_halo_pages=1` 后，oracle text NLL 降到 0.845。

3. 非连续 raw KV gather 还不能作为主路线。
   - 即使 oracle span recall = 1.0，`kv_native_oracle_pages_plus_recent_absolute` 仍然 NLL = 4.688、exact = 0。
   - latent top3 + halo 能恢复到 NLL = 0.669、exact = 1.0，但 active KV 已经到 224/256，压缩率太低。
   - 这说明当前 naive sparse KV path 对位置、上下文连续性、层间状态都很敏感。

4. prompt rebuild 是更稳定的验证通道。
   - selected text 重新 prefill 成 compact KV，比直接抽取非连续 past K/V 更稳。
   - 这不等于最终高效方案，但适合先验证搜索索引是否找到了有用 span。

## 当前判断

这个方向不应该废弃，但不能按“AE 重建 K/V 或 naive sparse KV gather”继续主攻。

更合理的路线是：

```text
latent/compressed index -> page/span ranking -> retrieve raw text/span -> re-prefill 或 RoPE-aware contiguous repack
```

现在的问题不在于“压缩搜索完全没信号”，而在于：

```text
1. scorer 排序还不够强，尤其是 multi-hop 和 decoy/current 冲突。
2. 命中后需要 span halo，而不是只取中心 page。
3. 非连续 raw KV gather 本身还没被验证成可靠执行路径。
```

## 下一步

P0：把 latent scorer 从纯 attention-block 监督改成 page/span ranking 监督：

```text
positive = gold evidence span/page
hard negatives = sink page, recent page, obsolete decoy page, key-only page
loss = pairwise ranking / contrastive CE / multi-positive CE
```

P1：先用 `retrieve span -> selected text re-prefill` 做质量闭环，把 top-k 降到可用预算：

```text
目标：256 context 下 top1/top2 + halo 达到 span recall >= 0.9，active text <= 50%。
```

P2：如果还要走 raw KV cache，需要单独做 KV 执行层验证：

```text
contiguous prefix fallback
RoPE-aware repack
position remap
layer/head selective gather
```

短期不要把 reconstructed K/V replacement 当主线；它在真实 Qwen trace 上还不够可靠。
