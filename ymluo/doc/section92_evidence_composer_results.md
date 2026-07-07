# Section 92: Evidence Composer Results

日期：2026-07-07

## 目标

Section 91 暴露了一个关键缺口：

```text
page/span recall = 1.0 仍然不保证 NLL 接近 oracle text。
```

主要原因是 page16 + halo2 会带入较多 filler，且 record 可能跨页。这里新增一个 second-stage evidence composer：

```text
selected pages
-> sentence / identifier extraction
-> bridge/current/obsolete/final-answer ordering
-> composed evidence prompt
```

composer 不使用 gold answer，只使用：

```text
query
selected text
identifier patterns
record keywords
```

## 代码变化

修改：

```text
ymluo/projects/learned_hierarchical_summary_memory/src/run_supervised_latent_page_ranker_smoke.py
```

新增参数：

```text
--include_evidence_composer
--composer_max_tokens
--composer_extra_halo_pages
```

关键实现：

```text
1. 从 query 中提取大写 identifier，例如 BRIDGE-TRACE-108。
2. 在 selected text 中保留包含 identifier 的句子。
3. 二次扩展新发现的 identifier，例如 NODE-TULIP-47。
4. 保留 verified/current/obsolete/bridge/intermediate/final-answer 等 record 句。
5. composer 额外使用 +1 page halo，避免 page16 截断跨页 answer。
```

## 命令

```bash
python ymluo/projects/learned_hierarchical_summary_memory/src/run_supervised_latent_page_ranker_smoke.py \
  --output_dir ymluo/projects/learned_hierarchical_summary_memory/outputs/supervised_latent_page_ranker_36case_page16_composer_extra_gh_20260707 \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --local_files_only true \
  --device_map none \
  --dtype float32 \
  --attn_implementation eager \
  --case_families old_single,two_old,decoy_exact \
  --train_variant_suffixes a,b,c,d,e,f \
  --eval_variant_suffixes g,h \
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
  --top_pages 2 \
  --page_halo_pages 2 \
  --include_prompt_rebuild true \
  --include_adaptive_case_top true \
  --include_evidence_composer true \
  --composer_max_tokens 96 \
  --composer_extra_halo_pages 1 \
  --eval_rankers supervised
```

输出：

```text
ymluo/projects/learned_hierarchical_summary_memory/outputs/supervised_latent_page_ranker_36case_page16_composer_extra_gh_20260707
```

## 结果

| method | active tokens | span recall | NLL | exact |
| --- | ---: | ---: | ---: | ---: |
| full_kv_cache | 256.0 | 1.000 | 0.562 | 0.500 |
| oracle text | 137.3 | 1.000 | 0.742 | 0.833 |
| oracle composed | 85.3 | 1.000 | 0.766 | 0.833 |
| supervised top2 text | 150.2 | 1.000 | 1.137 | 0.667 |
| supervised top2 composed | 88.2 | 1.000 | 0.757 | 0.833 |
| supervised adaptive text | 134.7 | 1.000 | 1.125 | 0.667 |
| supervised adaptive composed | 85.3 | 1.000 | 0.766 | 0.833 |

关键变化：

```text
supervised top2:
  active tokens: 150.2 -> 88.2
  NLL:           1.137 -> 0.757
  exact:         0.667 -> 0.833

supervised adaptive:
  active tokens: 134.7 -> 85.3
  NLL:           1.125 -> 0.766
  exact:         0.667 -> 0.833
```

这已经把 active prompt 降到：

```text
85.3 / 256 = 33.3%
```

同时 NLL 基本贴近 oracle text：

```text
supervised adaptive composed NLL = 0.766
oracle text NLL = 0.742
```

## Per-case 观察

two_old_g 是之前最明显的失败：

```text
supervised top2 text:
  NLL = 3.456
  exact = False

supervised top2 composed:
  NLL = 1.333
  exact = False

oracle text:
  NLL = 1.162
  exact = True
```

composer 显著缩小了 gap，但 two-hop 仍然是最难 case。

decoy_exact_g：

```text
supervised top2 text:
  NLL = 0.708
  exact = False

supervised top2 composed:
  NLL = 0.504
  exact = True
```

composer 对 current/obsolete conflict 有正向作用。

## 对论文主张的影响

这一步明显增强了论文故事：

```text
Stage 1:
  query-conditioned compressed latent page index

Stage 2:
  evidence composer over selected pages

Runtime:
  compact composed evidence re-prefill
```

相比只做 page retrieval，现在有更清晰的端到端质量收益：

```text
attention baseline:
  span recall low, NLL high

supervised latent retriever:
  span recall high, but raw selected pages contain filler

evidence composer:
  removes filler, completes cross-page facts, recovers oracle-like NLL
```

## 当前 ICML 风险

仍然不能宣称已经足够：

```text
1. 仍是 synthetic suite。
2. composer 是规则启发式，不是 learned module。
3. full KV exact 在部分 synthetic cases 上也不稳定，所以 exact 指标需要更规范。
4. 还没有公开 benchmark 和速度/显存曲线。
```

但现在创新结构更完整：

```text
Compressed latent page index + evidence composer
```

这比单纯“CNN/AE summary K/V”更像可以写成论文的方法。

## 下一步

P0：把 composer 从规则启发式升级为可学习/可替换模块。

```text
page-level latent retriever -> top pages
token/span scorer inside top pages -> evidence snippets
snippet ordering policy -> composed evidence
```

P1：把 public-style synthetic benchmark 接进来。

```text
Needle-in-a-haystack
multi-hop retrieval
decoy/current conflict
```

P2：补质量-预算曲线。

```text
active ratio:
  25%, 33%, 50%, 75%, 100%
metrics:
  NLL
  exact/contains
  span recall
  latency
```

P3：如果要冲 ICML，下一轮必须把 suite 从 Qwen3-0.6B synthetic smoke 扩到至少一个公开长上下文 benchmark 子集。
