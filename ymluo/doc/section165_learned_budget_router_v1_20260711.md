# Section 165: learned budget router v1

## 背景

v300_main 的主要弱点是任务级 policy 仍然由人工指定：不同 LongBench 任务手工给基础预算、block size、scorer 和 fallback 条件。为了把方法从“benchmark-specific tuning”推进到更适合论文叙事的自适应系统，本节训练一个离线 learned budget/action router。

目标不是让模型直接生成答案，而是让 router 预测：

- 当前样本是否可以使用更便宜的动作；
- 如果可以，选择哪个候选动作/预算；
- 如果不确定，就回退到 v300/reference。

## 训练数据

脚本：

```bash
scripts/train_learned_budget_router_20260711.py
```

输入：

- reference：`outputs/riskkv_v19_v300_v294_no_2wiki_20260711_m100_bDyn_pDyn`
- full：`outputs/riskkv_fullkv_m100_same_samples_20260710`
- candidates：v311/v324/v325/v326/v327/v328/v329/v330/v331 以及若干 M20 候选动作。

特征只使用一次检索后能得到的信息：

- `raw_prefix_tokens`, `raw_prompt_tokens`, `context_length_field`, `page_count`
- `ours_score_max`, `ours_score_mean`, `ours_score_gap2`, `ours_score_gap3`, `ours_score_entropy`
- `ours_query_coverage_terms`, `ours_query_coverage_covered`, `ours_query_coverage_recall`
- task family one-hot，不使用具体 task id 作为主结果。

标签生成：

- 对每个样本枚举候选动作。
- 质量约束：候选得分至少达到 reference 的 `quality_ratio`。
- 在满足质量约束的候选中选 KV keep 最低的动作。
- 如果模型预测置信度低于阈值，则回退到 reference/v300。

## 结果

主要工作点：`quality_ratio=0.90`, `confidence_fallback_threshold=0.35`, task encoding = family。

输出目录：

```bash
outputs/riskkv_v19_learned_budget_router_v1_family_q090_conf035_20260711
```

全量模拟结果：

| Method | Score | Score / v300 | KV keep | Online | Online speed vs v300 | Safe rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v300 reference | 0.4392 | 100.0% | 27.41% | 0.563s | 1.00x | 100.0% |
| learned router v1 | 0.4354 | 99.12% | 25.22% | 0.525s | 1.073x | 98.06% |
| oracle selector | 0.4432 | 100.91% | 24.47% | 0.539s | 1.045x | 100.0% |

Held-out fold：

| Method | Score | Score / v300 | KV keep | Online | Online speed vs v300 | Safe rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v300 reference | 0.4383 | 100.0% | 25.33% | 0.584s | 1.00x | 100.0% |
| learned router v1 | 0.4322 | 98.62% | 23.88% | 0.559s | 1.045x | 98.47% |
| oracle selector | 0.4391 | 100.19% | 23.11% | 0.559s | 1.044x | 100.0% |

## 特征重要性

Top features:

| Feature | Importance |
| --- | ---: |
| page_count | 0.115 |
| ours_score_entropy | 0.111 |
| ours_score_mean | 0.091 |
| raw_prompt_tokens | 0.075 |
| ours_score_gap2 | 0.075 |
| raw_prefix_tokens | 0.068 |
| ours_score_gap3 | 0.067 |
| context_length_field | 0.067 |
| ours_query_coverage_covered | 0.062 |
| family=single_doc_qa | 0.062 |

这个结果说明 router 不是单纯记住任务名；主要依据确实来自检索置信度、gap、entropy、coverage 和长度。

## 当前结论

learned router v1 已经证明方向可行：

- 能自动选择更便宜动作；
- 不需要手写每个任务的预算；
- 保守置信度回退后，质量仍为 v300 的 99% 左右；
- KV 从 27.41% 降到 25.22%。

但它还不是最终主方法：

- 当前结果是离线模拟，不是 runtime harness 真正在线调用模型。
- 候选动作集合仍来自已有实验，需要进一步扩展纯预算候选。
- qasper/narrativeqa 的低预算动作仍有风险，需要更强的 uncertainty/fallback。

下一步：

1. 把 learned router 导出成 runtime 可执行的 JSON/tree policy。
2. 在 harness 里加入 `action_router_mode=learned_v1` 或 `lookup_json`。
3. 用 M100 端到端跑 learned router，而不是只做离线 replay。
