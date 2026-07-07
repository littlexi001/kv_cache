# Section 90: Strict Generalization And Budget Smoke

日期：2026-07-07

## 目标

Section 89 在同一批 case 的 layer/head split 上效果很好，但这可能包含过拟合。这里做两件事：

```text
1. 扩展 synthetic case variants：
   old_single_a/b/c/...
   two_old_a/b/c/...
   decoy_exact_a/b/c/...

2. 改成 case-level split：
   train_cases 只用于 AE/searcher/ranker 训练
   eval_cases 只用于 page/span recall 和 prompt rebuild
```

同时开始压预算：

```text
page_tokens: 32 -> 16
recent_tokens: 32 -> 16
top pages: top1/top2/adaptive
halo: 1/2
```

## 代码变化

新增/修改：

```text
ymluo/projects/learned_hierarchical_summary_memory/src/run_real_qwen_seq_ae_search_trace.py
ymluo/projects/learned_hierarchical_summary_memory/src/run_supervised_latent_page_ranker_smoke.py
```

关键实现：

```text
1. make_case 支持 *_a / *_b / *_c variants。
2. record 插入 offset 从固定 16 改为 min(16, page_tokens // 2)，支持 page_tokens=16。
3. supervised ranker 支持 --train_cases / --eval_cases。
4. 增加 adaptive policy：
   adaptive_case_top:
     two_old -> top2
     old_single/decoy_exact -> top1
   adaptive_family_budget:
     two_old -> top2 + halo1
     old_single/decoy_exact -> top1 + halo2
```

## 实验一：position-heldout split

训练：

```text
old_single_a, old_single_b
two_old_a, two_old_b
decoy_exact_a, decoy_exact_b
```

评估：

```text
old_single_c, two_old_c, decoy_exact_c
```

输出：

```text
ymluo/projects/learned_hierarchical_summary_memory/outputs/supervised_latent_page_ranker_strict_256_rankonly_20260707
```

结果：

| ranker | top pages | center recall | span recall |
| --- | ---: | ---: | ---: |
| attention | 1 | 0.500 | 0.500 |
| attention | 2 | 0.833 | 1.000 |
| attention | 3 | 1.000 | 1.000 |
| supervised | 1 | 0.167 | 0.500 |
| supervised | 2 | 0.500 | 0.833 |
| supervised | 3 | 0.500 | 1.000 |

结论：

```text
这个 split 太严格地 hold out 了位置模式。
supervised ranker 没见过 two_old_c 的 page-6 second hop，泛化变差。
这说明当前 ranker 仍然有明显 position/template bias。
```

## 实验二：key-heldout split

训练覆盖主要位置模式：

```text
old_single_a/b/c
two_old_a/b/c
decoy_exact_a/b/c
```

评估换新 key/variant：

```text
old_single_g/h/i
two_old_g/h/i
decoy_exact_g/h/i
```

输出：

```text
ymluo/projects/learned_hierarchical_summary_memory/outputs/supervised_latent_page_ranker_keyheldout_256_rankonly_20260707
```

结果：

| ranker | top pages | center recall | span recall |
| --- | ---: | ---: | ---: |
| attention | 1 | 0.389 | 0.611 |
| attention | 2 | 0.667 | 1.000 |
| attention | 3 | 0.667 | 1.000 |
| supervised | 1 | 0.833 | 0.833 |
| supervised | 2 | 1.000 | 1.000 |
| supervised | 3 | 1.000 | 1.000 |

结论：

```text
当训练覆盖位置模式，supervised latent page ranker 能跨 key 泛化。
它不是只记住 key 字符串；但它还需要足够的位置/结构覆盖。
```

## 实验三：page16 预算压缩

配置：

```text
prompt_tokens = 256
page_tokens = 16
recent_tokens = 16
page_halo_pages = 2
held-out eval = old_single_g, two_old_g, decoy_exact_g
```

输出：

```text
ymluo/projects/learned_hierarchical_summary_memory/outputs/supervised_latent_page_ranker_keyheldout_256_page16_family_budget_prompt_g_20260707
```

端到端结果：

| method | active tokens | span recall | NLL | exact |
| --- | ---: | ---: | ---: | ---: |
| full_kv_cache | 256.0 | 1.000 | 0.627 | 1.000 |
| oracle text | 129.3 | 1.000 | 0.836 | 0.667 |
| supervised top1 | 119.0 | 1.000 | 0.790 | 0.333 |
| supervised top2 | 140.0 | 1.000 | 0.845 | 0.667 |
| supervised adaptive_case_top | 129.3 | 1.000 | 0.836 | 0.667 |
| supervised adaptive_family_budget | 118.7 | 1.000 | 1.679 | 0.333 |

观察：

```text
1. page16 + halo2 + recent16 能把 active tokens 压到约 50%。
2. supervised top1 最省，119/256 = 46.5%，NLL 0.790，但 exact 只有 0.333。
3. adaptive_case_top 更稳，129.3/256 = 50.5%，NLL 0.836，贴近 oracle text。
4. adaptive_family_budget 虽然更省，但 two_old 的 halo1 太窄，NLL 退化。
```

## 当前判断

方向继续成立，但结论要更精确：

```text
可行的是：
  supervised compressed page index
  + enough position/template coverage
  + page16/top1-or-top2/halo2/recent16
  + selected text re-prefill

还不成立的是：
  少量 case 训练后直接 position-heldout 泛化
  naive non-contiguous raw KV gather
  过窄 halo 的 family budget
```

## 下一步

P0：构造更大的 variant set。

```text
至少每类 12-24 个 variants：
  different keys
  different answers
  repeated position patterns
  held-out keys
  held-out templates
  held-out positions
```

P1：把 ranker 训练改成混合 split：

```text
train:
  cover all common position patterns
eval:
  key-heldout
  template-heldout
  partial position-heldout
```

P2：把预算目标固定为：

```text
page_tokens = 16
recent_tokens = 16
top policy = top1 or adaptive_case_top
halo = 2
target active <= 50%-55%
```

P3：下一轮不急着回 KV gather，先扩大 text re-prefill quality suite。
