# MoR-KV: Mixture-of-Retrievers for Function-Specialized KV Access

MoR-KV 研究一个比“不同 head 分配不同 KV 数量”更强的问题：

> 不同 attention heads 是否应该使用不同的 KV **检索算子**，并由当前 query 动态选择，而不是只改变统一 eviction 策略的预算？

当前项目是论文主线的第一阶段实现。它复用 Qwen3-0.6B 的真实 Q/K block rankings，在严格 train/dev/test 划分上验证 query-conditioned head portfolio、operator quota 和 hard-negative 风险。它不是最终 sparse-attention kernel，也不应被描述成已经达到 ICLR 完成度。

## 核心方法

MoR-KV 的目标形态包含四类 operator：

1. `streaming`：sink + recent，服务位置/局部 heads；
2. `lexical/structural`：token hash、标点、delimiter 和 record boundary side index；
3. `semantic QK`：低秩 Q/K block retrieval，服务 query-dependent 远程检索 heads；
4. `dense/risk fallback`：不确定时扩大 budget 或回退 full attention。

当前 v1 实验实现其中两类远程 operator：

- BM25 lexical block index；
- 真实 pre-RoPE Q/K 的 all-layer/all-query-head SVD32 rankings。

它还实现：

- 由每个 head 的 top score、top1-top4 margin 和候选 score spread 构造 query signature；
- 只用 train split 拟合 nearest-centroid operator router；
- train split 校准每个任务的 specialist heads，dev split 固定 operator/head/quota action；
- test split 一次性报告，禁止再调参；
- Qwen3 GQA 映射和同一 KV head 内的 query-head 去重；
- `weighted_rrf` 与保护少数专业 head nomination 的 `minority_max`。

## 项目结构

```text
src/run_mor_kv_offline.py       held-out operator-routing experiment
src/evaluate_mor_answer_nll.py  retrieved-context answer NLL
src/compile_nll_routed_policy.py dev-NLL policy compilation and test audit
src/analyze_real_nll_route.py   natural LongBench small-sample No-Go audit
src/analyze_natural_operator_library.py natural operator complementarity + LODO routing
src/evaluate_operator_proxy_nll.py answer-free self-verification probe
src/analyze_proxy_route.py       LODO proxy and action-regret routing
src/run_lodo_natural_specialist_retrieval.py cross-dataset specialist operator
src/apply_frozen_specialist_retrieval.py calibration-only holdout policy
src/build_operator_portfolios.py equal-budget operator portfolios
src/analyze_portfolio_selection.py out-of-fold quota audit
src/summarize_frozen_holdout.py  zero-overlap holdout comparison
src/apply_frozen_proxy_router.py frozen answer-free risk gate
src/apply_head_confidence_router.py frozen QK head-confidence gate
src/generate_head_distortion_teacher.py exact post-RoPE per-head causal labels
src/merge_head_distortion_teacher.py risk-constrained oracle compilation
src/train_head_distortion_router.py query-disjoint conformal operator router
src/evaluate_sparse_attention_reference_nll.py causal model-forward NLL intervention
src/merge_sparse_attention_reference_nll.py paired bootstrap aggregation for reference NLL
src/plot_reference_nll_frontier.py causal mean/tail NLL frontier plot
src/plot_risk_threshold_sweep.py learned-router risk/quality Pareto plot
src/plot_external_allocation_validation.py frozen external-allocation frontier
src/benchmark_gqa_grouped_sdpa.py physical-GQA-union gather+SDPA benchmark
src/analyze_layer_risk_sensitivity.py cross-layer NLL amplification audit
src/analyze_query_risk_gate.py deployable conformal query-fallback audit
src/analyze_compiled_gqa_oracle.py exact query-head actions to physical GQA union
scripts/run_synthetic_server.sh full server pipeline
tests/test_mor_kv.py            pure logic tests
doc/method.md                   algorithm and theory
doc/novelty_audit.md            latest related-work boundary
doc/initial_results.md          frozen v1 findings
doc/iclr_execution_plan.md      paper-level evidence gates
doc/paper_skeleton.md           draft paper narrative, claims, and required evidence
doc/natural_holdout_report.md   independent holdout and router failure audit
doc/causal_teacher_pilot.md     exact distortion teacher and learned-router results
```

## 本地逻辑测试

```bash
python -m unittest discover \
  -s ymluo/projects/mor_kv_operator_routing/tests \
  -p 'test_*.py'
```

## 服务器复现

依赖 Section 128 已生成的真实 Q/K rankings 和 synthetic BM25 side index：

```bash
ssh fdong@10.176.37.31
cd /home/fdong/ymluo/projects/mor_kv_operator_routing
GPU_IDS=4,5,6,7 \
OUT=/home/fdong/ymluo/projects/mor_kv_operator_routing/outputs/synthetic_v1 \
bash scripts/run_synthetic_server.sh
```

默认数据：

```text
corpus:
  /home/fdong/ymluo/projects/parallel_block_retrieval/data/
  synthetic_controlled_100k_500_v1

all-head QK rankings:
  /home/fdong/ymluo/projects/parallel_block_retrieval/outputs/
  synthetic_controlled_100k_500_allhead_consensus_v1/per_head_topk.npz

BM25 scores:
  /home/fdong/ymluo/projects/parallel_block_retrieval/outputs/
  synthetic_controlled_100k_500_bm25_v1/block_scores.npy
```

## v1 结果摘要

500 queries 按 `300 train / 100 dev / 100 test` 固定，四种任务各 125 条。router 在 test 上为 `100/100`，但这是受控数据结果，不能外推为自然分布准确率。

在 4-block test 预算：

| 方法 | Evidence fraction | Hard-negative hit | Utility |
| --- | ---: | ---: | ---: |
| BM25 | 0.775 | 0.980 | 0.530 |
| single global hybrid | 0.785 | 0.930 | 0.553 |
| wrong router | 0.605 | 0.770 | 0.413 |
| **MoR-KV** | **0.755** | **0.770** | **0.563** |

`utility = evidence_fraction - 0.25 × hard_negative_hit_rate`。MoR-KV 在所有测试预算 `1,4,8,16,39` 上均获得最高 utility；错误路由对照显著下降，说明 operator/head portfolio 匹配是有效因素。

只用 dev NLL 在 `BM25 / global hybrid / task MoR` 三个动作之间编译 task policy 后，冻结 test 结果为：

| 方法 | Test mean answer NLL |
| --- | ---: |
| BM25 | 3.765 |
| single global hybrid | 3.706 |
| raw utility-routed MoR | 3.763 |
| **dev-NLL compiled MoR** | **3.520** |

相对 BM25 的 paired mean delta 为 `-0.246`，bootstrap 95% CI `[-0.425, -0.083]`；相对 global hybrid 为 `-0.186`，CI `[-0.357, 0.002]`，后者仍是边界显著，必须增加自然数据和样本量。

完整结果见 `doc/initial_results.md` 和 `outputs/synthetic_v1/summary.json`。

## 当前结论边界

- v1 证明的是候选 block selection 和 distractor exposure，不是端到端 kernel speedup。
- BM25 在最终系统中应实现为 KV block 的轻量 token/hash side index，而不是外部 RAG 文档库。
- 100% router accuracy 来自受控任务族，必须做自然任务、跨域和跨模型测试。
- 当前只覆盖两个远程 operator；streaming/structure/dense fallback 尚需接入真实 sparse decode。
- 论文成立的最终标准是：在强 KV baselines、8B models、真实长上下文和实际 kernel latency 上同时得到质量与效率 Pareto 改进。
