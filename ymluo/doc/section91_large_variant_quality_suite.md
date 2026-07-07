# Section 91: Large Variant Quality Suite

日期：2026-07-07

## 目标

把 Section 90 的 3-case/9-case smoke 推进到更接近论文实验的 quality suite：

```text
train variants: a-f
eval variants:  g-l
families:
  old_single
  two_old
  decoy_exact
```

也就是：

```text
train cases = 18
eval cases  = 18
trace samples = 864
train samples = 432
eval samples = 432
```

核心问题：

```text
1. supervised latent page ranker 是否能在更多 held-out keys 上稳定超过 attention-search baseline？
2. page16/recent16/halo2 的低预算配置是否仍可行？
3. ranking recall 和端到端 NLL/exact 之间还有哪些缺口？
```

## 代码变化

修改：

```text
ymluo/projects/learned_hierarchical_summary_memory/src/run_real_qwen_seq_ae_search_trace.py
ymluo/projects/learned_hierarchical_summary_memory/src/run_supervised_latent_page_ranker_smoke.py
```

新增能力：

```text
1. case variants 扩到 12 组答案和位置计划。
2. ranker 脚本支持：
   --case_families
   --train_variant_suffixes
   --eval_variant_suffixes
3. decoy hard-negative page 与 make_case 共享 DECOY_PAGE_PLAN，避免训练标签漂移。
```

## 实验一：36-case Rank-only Suite

命令要点：

```bash
python ymluo/projects/learned_hierarchical_summary_memory/src/run_supervised_latent_page_ranker_smoke.py \
  --output_dir ymluo/projects/learned_hierarchical_summary_memory/outputs/supervised_latent_page_ranker_keyheldout_36case_page16_rankonly_20260707 \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --local_files_only true \
  --device_map none \
  --dtype float32 \
  --attn_implementation eager \
  --case_families old_single,two_old,decoy_exact \
  --train_variant_suffixes a,b,c,d,e,f \
  --eval_variant_suffixes g,h,i,j,k,l \
  --prompt_tokens 256 \
  --page_tokens 16 \
  --recent_tokens 16 \
  --layers 0-5 \
  --kv_heads 0-3 \
  --max_query_tokens 16 \
  --block_size 8 \
  --latent_dim 128 \
  --ae_epochs 6 \
  --search_epochs 3 \
  --supervised_epochs 80 \
  --top_pages 1,2,3 \
  --page_halo_pages 2 \
  --include_prompt_rebuild false
```

输出目录：

```text
ymluo/projects/learned_hierarchical_summary_memory/outputs/supervised_latent_page_ranker_keyheldout_36case_page16_rankonly_20260707
```

结果：

| ranker | top pages | center recall | span recall |
| --- | ---: | ---: | ---: |
| attention | 1 | 0.000 | 0.194 |
| attention | 2 | 0.167 | 0.333 |
| attention | 3 | 0.250 | 0.611 |
| supervised | 1 | 0.722 | 0.806 |
| supervised | 2 | 0.917 | 1.000 |
| supervised | 3 | 0.972 | 1.000 |

结论：

```text
supervised latent page ranker 在 18 个 held-out variants 上明显优于 attention-search baseline。
top2 + halo2 达到 span recall 1.0。
这支撑“压缩 latent index 可训练成 query-conditioned page retriever”的核心主张。
```

## 实验二：6-case Prompt Rebuild Subset

为了验证 ranking recall 是否转化为端到端质量，抽 eval variants g/h 做 prompt rebuild。

输出目录：

```text
ymluo/projects/learned_hierarchical_summary_memory/outputs/supervised_latent_page_ranker_36case_page16_prompt_gh_20260707
```

结果：

| method | active tokens | span recall | NLL | exact |
| --- | ---: | ---: | ---: | ---: |
| full_kv_cache | 256.0 | 1.000 | 0.562 | 0.500 |
| oracle text | 137.3 | 1.000 | 0.742 | 0.833 |
| attention top2 | 151.5 | 0.250 | 3.438 | 0.167 |
| supervised top1 | 118.8 | 0.833 | 1.920 | 0.500 |
| supervised top2 | 150.2 | 1.000 | 1.137 | 0.667 |
| supervised adaptive_case_top | 134.7 | 1.000 | 1.125 | 0.667 |

观察：

```text
1. supervised top2/adaptive 的 span recall 已经是 1.0。
2. NLL 仍弱于 oracle text：
   supervised adaptive NLL 1.125 vs oracle text 0.742。
3. 差距主要来自 multi-hop ordering/context：
   two_old_g 的 supervised top2 span recall = 1.0，但 NLL = 3.456；
   oracle text 同 case NLL = 1.162。
```

## 对 ICML 目标的判断

当前已有的强信号：

```text
1. 方法不是简单 mean/attention heuristic：
   supervised latent index 在 36-case held-out suite 上大幅超过 attention baseline。

2. 存储压缩有明确数字：
   latent storage ratio vs K/V = 6.25%。

3. 检索预算开始可控：
   page16/recent16/halo2 下，active prompt token 约 50%-59%。

4. 失败边界清楚：
   position-heldout 失败；
   multi-hop ordering 会让 NLL 低于 oracle；
   naive non-contiguous KV gather 不稳。
```

目前还不足以支撑 ICML 的部分：

```text
1. 数据规模还太小，仍是 synthetic micro-suite。
2. 还没有 LongBench/RULER/Needle 等公开任务结果。
3. 端到端质量还没有稳定超过 strong baseline。
4. 当前主执行路径是 selected text re-prefill，不是最终 KV-cache runtime。
5. 缺少速度/显存收益与质量的联合曲线。
```

## 下一步必须补的创新点

P0：Second-stage span ordering / evidence composer。

```text
问题：
  page recall = 1.0 不保证 NLL 接近 oracle。

方案：
  selected pages -> evidence snippets
  preserve chronological/causal order
  bridge pages before answer pages
  current record after obsolete record
  drop duplicate filler
```

P1：从 page-level 转为 page -> span 两阶段。

```text
stage 1:
  compressed latent page ranker
stage 2:
  cheap token/span reranker inside selected pages

目标：
  active <= 35%-45%
  NLL close to oracle text
```

P2：公开 benchmark bridge。

```text
先做 selected text re-prefill on synthetic-public tasks:
  Needle-in-a-haystack
  multi-hop synthetic QA
  decoy/current conflict

再接 LongBench/RULER 子集。
```

P3：论文表述建议。

```text
核心创新名可以暂定：
  Query-Conditioned Latent Page Indexing for KV-Cache Compression

主张不要写成：
  learned K/V reconstruction replaces KV cache

应该写成：
  a compact learned index selects raw evidence spans;
  the runtime can re-prefill selected text or later use RoPE-aware repack.
```

## 当前结论

继续做，有论文潜力；但距离“足够支撑 ICML”还缺两块硬证据：

```text
1. 更大公开/半公开 benchmark 上的质量曲线。
2. page recall -> NLL 的 second-stage evidence composer。
```

下一步优先做 P0/P1，而不是回到 reconstructed K/V 或 naive sparse KV gather。
