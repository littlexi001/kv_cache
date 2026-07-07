# Section 89: Supervised Latent Page Ranker Smoke

日期：2026-07-07

## 目标

Section 88 证明了 latent scorer 有信号，但 attention-block 监督不够稳定。这里改成直接做 page/span ranking 监督：

```text
positive = gold evidence page
hard negatives = sink page, recent page, obsolete decoy page
loss = multi-positive page CE + hard-negative margin
```

同时把 `max_query_tokens` 从之前常用的 4 提到 16。原因是 4 个 query token 只覆盖 `Question: What is...`，基本看不到 key/current/bridge 等检索信息。

## 新增脚本

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_supervised_latent_page_ranker_smoke.py
```

脚本流程：

```text
1. 收集 real Qwen Q/K/V trace。
2. 训练 Seq-AE，得到 latent block sequence。
3. 保留 attention-searcher baseline。
4. 从 attention-searcher 初始化 supervised page ranker。
5. 用 evidence pages 做 supervised ranking。
6. 比较 attention vs supervised 的 page/span recall。
7. 用 selected text re-prefill 做端到端 NLL/exact。
```

## 关键命令

```bash
python ymluo/projects/learned_hierarchical_summary_memory/src/run_supervised_latent_page_ranker_smoke.py \
  --output_dir ymluo/projects/learned_hierarchical_summary_memory/outputs/supervised_latent_page_ranker_256_prompt_20260707 \
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
  --max_query_tokens 16 \
  --block_size 8 \
  --latent_dim 128 \
  --ae_epochs 6 \
  --search_epochs 3 \
  --batch_size 8 \
  --rare_recon_weight 0.2 \
  --rare_token_fraction 0.01 \
  --supervised_epochs 80 \
  --supervised_lr 0.001 \
  --hard_negative_weight 0.5 \
  --hard_negative_margin 1.0 \
  --top_pages 1,2 \
  --page_halo_pages 1 \
  --exclude_sink_pages 1 \
  --exclude_recent_from_latent true \
  --decode_steps 16 \
  --include_prompt_rebuild true \
  --eval_rankers attention,supervised
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

输出目录：

```text
ymluo/projects/learned_hierarchical_summary_memory/outputs/supervised_latent_page_ranker_256_prompt_20260707
```

## Page Ranking 结果

| ranker | top pages | center page recall | span page recall |
| --- | ---: | ---: | ---: |
| attention baseline | 1 | 0.667 | 0.833 |
| attention baseline | 2 | 0.667 | 0.833 |
| supervised | 1 | 0.833 | 0.833 |
| supervised | 2 | 1.000 | 1.000 |

关键变化：

```text
two_old:
  attention top2 -> [5, 6], 漏掉 evidence [2, 4]
  supervised top2 -> [2, 4], 正好命中两跳证据

decoy_exact:
  supervised top1 -> [5], 命中 current evidence page
  obsolete decoy page 2 被压下去
```

## Prompt Rebuild 端到端结果

| method | active tokens | span recall | NLL | exact |
| --- | ---: | ---: | ---: | ---: |
| full_kv_cache | 256.0 | 1.000 | 0.645 | 1.000 |
| prompt_rebuild_oracle_pages_text | 170.3 | 1.000 | 0.845 | 0.667 |
| prompt_rebuild_attention_top1_pages_text | 149.7 | 0.833 | 0.898 | 0.333 |
| prompt_rebuild_attention_top2_pages_text | 149.7 | 0.833 | 0.898 | 0.333 |
| prompt_rebuild_supervised_top1_pages_text | 149.0 | 0.833 | 2.173 | 0.333 |
| prompt_rebuild_supervised_top2_pages_text | 180.7 | 1.000 | 0.837 | 0.667 |

解释：

```text
1. supervised top2 + halo 已经贴近 oracle text：
   NLL 0.837 vs oracle 0.845

2. top1 还不够：
   two_old 需要两个 evidence pages，top1 只取 page 2，会漏 final-answer page 4。

3. exact 对 decoy_exact 偏严格：
   oracle/supervised 的 NLL 很低，但生成有时只输出 "EMBER-447..." 而不是完整 "VERIFIED-EMBER-447"。
   所以这里 NLL 比 exact 更能反映是否找到了正确 span。
```

## 当前判断

这个实验把方向从“可能有信号”推进到了“可训练出更准的压缩 page index”。

可以继续做，但主线应改为：

```text
compressed latent index
-> supervised page/span ranker
-> top-k pages + halo
-> selected text re-prefill 或后续 RoPE-aware contiguous repack
```

暂时不要把 naive non-contiguous KV gather 当主执行路径；它在 Section 88 里 oracle 都不稳。

## 下一步

P0：扩大监督数据和 case 类型。

```text
当前只有 3 个 synthetic cases，监督很容易过拟合。
下一步至少加：
  single old fact
  two-hop bridge
  decoy old/current
  multiple same-key records
  numeric/date conflict
  near-recent conflict
  no-answer distractor
```

P1：把预算压下来。

```text
当前 supervised top2 + halo active tokens = 180.7 / 256 = 70.6%。
目标应是 <= 50%，否则压缩意义不足。
可尝试：
  smaller page size
  token/span-level second-stage rerank
  adaptive halo
  page top2 -> span top-k
```

P2：训练/评估切分要更严格。

```text
现在是同一批 synthetic case 的 layer/head split smoke。
下一步要按 case/template/key split，避免只学到模板位置。
```

P3：如果质量稳定，再回到 KV 执行层。

```text
先验证 selected text re-prefill。
再做 RoPE-aware contiguous repack。
最后再考虑 layer/head selective raw KV reuse。
```
