# MoR-KV v1 Held-out Results

日期：2026-07-11

## 1. 实验设置

| 项目 | 设置 |
| --- | --- |
| model profiles | Qwen3-0.6B 真实 pre-RoPE Q/K |
| context scale | 4 records × 25,088 tokens，约 100K tokens corpus |
| query count | 500 |
| task families | lexical, semantic paraphrase, hard negative, multihop |
| fixed split | 300 train / 100 dev / 100 test |
| query heads | 28 layers × 16 heads = 448 |
| GQA | 2 query heads / KV head |
| QK index | per-layer/KV-head centered SVD32 |
| budgets | 1, 4, 8, 16, 39 blocks |
| utility | evidence fraction - 0.25 × hard-negative hit rate |

所有 head utility 只在 train 编译；operator/head-count/depth/quota 只在 dev 选择；test 不做调参。

## 2. Router

Score signature 只使用正式 gather 前已经得到的：

```text
per-head top1 score
per-head top1 - top4 margin
per-head candidate score standard deviation
```

| split | Accuracy |
| --- | ---: |
| train | 1.000 |
| dev | 0.990 |
| test | 1.000 |

这是受控任务族可分性证据，不代表自然任务可以达到相同准确率。错误路由消融是更关键的因果对照。

## 3. Test 主表

| Budget | Method | Evidence fraction | Hard-negative hit | Utility |
| ---: | --- | ---: | ---: | ---: |
| 1 | BM25 | 0.275 | 0.560 | 0.135 |
| 1 | single hybrid | 0.275 | 0.560 | 0.135 |
| 1 | wrong router | 0.180 | 0.140 | 0.145 |
| 1 | **MoR-KV** | 0.225 | **0.070** | **0.208** |
| 4 | BM25 | 0.775 | 0.980 | 0.530 |
| 4 | single hybrid | **0.785** | 0.930 | 0.553 |
| 4 | wrong router | 0.605 | 0.770 | 0.413 |
| 4 | **MoR-KV** | 0.755 | **0.770** | **0.563** |
| 8 | BM25 | **0.870** | 1.000 | 0.620 |
| 8 | single hybrid | 0.860 | 1.000 | 0.610 |
| 8 | wrong router | 0.770 | 0.810 | 0.568 |
| 8 | **MoR-KV** | 0.845 | **0.800** | **0.645** |
| 16 | BM25 | **0.890** | 1.000 | 0.640 |
| 16 | single hybrid | 0.880 | 1.000 | 0.630 |
| 16 | wrong router | 0.820 | 0.930 | 0.588 |
| 16 | **MoR-KV** | 0.855 | **0.790** | **0.658** |
| 39 | BM25 | **0.940** | 1.000 | 0.690 |
| 39 | single hybrid | 0.930 | 1.000 | 0.680 |
| 39 | wrong router | 0.810 | 0.980 | 0.565 |
| 39 | **MoR-KV** | 0.905 | **0.830** | **0.698** |

MoR-KV 的 test utility 在五个预设预算上均最高，但它不是 evidence-recall 单指标最优。它的主要第一轮优势是避免明显 distractor，同时保留大部分 evidence。

## 4. Budget-4 分任务

| Task | Method | Evidence fraction | Hard-negative hit | Utility |
| --- | --- | ---: | ---: | ---: |
| lexical | BM25 | 0.840 | 0.960 | 0.600 |
| lexical | single hybrid | 0.880 | 0.760 | 0.690 |
| lexical | **MoR-KV** | **0.880** | **0.440** | **0.770** |
| semantic paraphrase | BM25 | 1.000 | 1.000 | 0.750 |
| semantic paraphrase | MoR-KV | 1.000 | 1.000 | 0.750 |
| hard negative | BM25 | **0.760** | 0.960 | **0.520** |
| hard negative | MoR-KV | 0.640 | **0.640** | 0.480 |
| multihop | BM25 | 0.500 | 1.000 | 0.250 |
| multihop | MoR-KV | 0.500 | 1.000 | 0.250 |

平均增益主要来自 lexical family；hard-negative family 仍存在 evidence/distractor trade-off，multihop 没有改善。这两个失败点是下一轮必须解决的核心，而不是可以隐藏的边角。

## 5. 关键消融

| Budget 4 method | Utility |
| --- | ---: |
| all-head RRF consensus | 0.003 |
| routed QK only | 0.505 |
| BM25 only | 0.530 |
| single global hybrid | 0.553 |
| **MoR-KV** | **0.563** |
| wrong-router MoR | 0.413 |

解释：

1. all-head consensus 再次失败，支持 Section 128 的“多数票淹没专业证据”；
2. 单一 global hybrid 已经很强，MoR-KV 的绝对增益目前只有 `+0.010`；
3. wrong router 的 `-0.150` gap 显示 operator matching 有真实作用，但也暴露 router 错误风险；
4. 当前结果属于 promising mechanism evidence，尚不是 paper headline result。

## 6. Answer NLL：utility route 的失败与 NLL compilation 的修正

四个 operator policy 在 test 100 queries、相同 4-block context 上的答案 NLL：

| 方法 | Mean answer NLL | Delta vs BM25 |
| --- | ---: | ---: |
| BM25 | 3.765 | 0.000 |
| single global hybrid | 3.706 | -0.059 |
| raw utility-routed MoR | 3.763 | -0.003 |
| wrong router | 4.942 | +1.177 |

这说明 evidence/distractor utility 不能替代模型 loss：raw MoR 虽然 retrieval utility 更好，但总体 NLL 与 BM25 基本持平。

随后只用 dev NLL 在 `BM25 / single hybrid / raw MoR` 三个冻结动作之间，为每个 routed task family 选择 policy：

| Task | Dev 选择的 action |
| --- | --- |
| hard negative | single global hybrid |
| lexical | BM25 |
| multihop | raw MoR |
| semantic paraphrase | BM25 |

该 policy 没有查看 test NLL。冻结 test 结果：

| 方法 | Test mean NLL |
| --- | ---: |
| BM25 | 3.765 |
| single global hybrid | 3.706 |
| raw MoR | 3.763 |
| **dev-NLL compiled MoR** | **3.520** |

Paired bootstrap：

| Reference | Mean NLL delta | 95% CI |
| --- | ---: | ---: |
| BM25 | **-0.246** | **[-0.425, -0.083]** |
| single global hybrid | -0.186 | [-0.357, 0.002] |
| raw utility MoR | -0.243 | [-0.720, 0.193] |

相对 BM25 的区间不跨 0；相对最强 global hybrid 的上界为 `0.002`，仍属于边界结果，不能宣称稳定显著胜出。

## 7. 输出

```text
outputs/synthetic_v1/summary.json
outputs/synthetic_v1/summary.csv
outputs/synthetic_v1/query_metrics.csv
outputs/synthetic_v1/query_results.csv
outputs/synthetic_v1/router_predictions.csv
outputs/synthetic_v1/policies.json
outputs/synthetic_v1/head_portfolios.csv
outputs/synthetic_v1/plots/
outputs/synthetic_v1/answer_nll_b4/
outputs/synthetic_v1/answer_nll_b4_dev/
outputs/synthetic_v1/nll_routed_b4/
```

## 8. 尚未完成

- natural LongBench held-out operator router；
- exact sparse-attention output fidelity；
- 8B/7B cross-model validation；
- 真实 paged KV gather latency、memory 和 index overhead；
- streaming/structure/dense fallback operator；
- utility lambda sweep 与 constrained Pareto evaluation。

## 9. Natural LongBench 小样本 No-Go 审计

复用现有 64 条真实 LongBench answer-NLL rows，在每个 dataset 内按顺序交替拆成 `33 calibration / 31 held-out`。Calibration 在以下三个 39-block operator 中按 dataset 选择：

```text
BM25 record39
risk BM25 + SVD32
deep question-likelihood + record39 + SVD32
```

Held-out：

| 方法 | Mean answer NLL |
| --- | ---: |
| BM25 record39 | 3.829 |
| risk BM25 + SVD32 | 3.601 |
| **global deep-QK** | **3.304** |
| dataset-routed operator | 3.510 |

Routed policy 相对 strongest global deep-QK 退化 `+0.206`，bootstrap 95% CI `[+0.032,+0.439]`。这说明当前 synthetic task routing 没有自然泛化证据，而且小样本 dataset-level action selection会过拟合。

即使在 held-out 上事后对每条 query 从三个 actions 取 NLL 最小值，oracle mean NLL 也只有 `3.247`，相比 strongest global deep-QK `3.304` 的 headroom 仅 `0.057`。因此问题不仅是 router：当前自然 operator library 本身缺少足够互补性。

直接结论：下一轮不能继续依赖 dataset label 或 6–7 条 calibration 样本；还必须先扩展真正互补的 structure/streaming/multihop operators，再学习 query-level action regret/risk。

## 10. Group-saturating Submodular Ablation

在不改变每个 task 的 head-count/depth/BM25 quota 的条件下，将 weighted RRF 换成 group-saturating submodular greedy，并只在 dev 选择 temperature：

| Blocks | weighted MoR utility | submodular utility |
| ---: | ---: | ---: |
| 1 | 0.208 | **0.218** |
| 4 | **0.563** | 0.543 |
| 8 | **0.645** | 0.640 |
| 16 | **0.658** | 0.648 |
| 39 | 0.698 | **0.700** |

结论：submodular saturation 目前只在极小和较大预算有微弱收益，中间预算反而下降。理论动机仍成立，但当前 utility/temperature 和 group definition 尚未得到实证支持，不能作为 paper contribution 宣称完成。

## 11. Expanded natural operator library

补齐三个 semantic record-routing operators，并新增 nested-LODO GQA specialist-head operator 后，第一批 64 条自然 query 的 per-query oracle mean NLL 从 `3.247` 下降到 `2.877`；相对 strongest global deep-QK `3.278` 的 oracle headroom 从 `0.057` 增大到 `0.402`。specialist action 在 22/64 条 query 上进入 oracle，证明其具有独立互补性。

但是 query-text/retrieval-disagreement router 为 `3.498`，full-model answer-free proxy router 约为 `3.281–3.285`，均没有超过 deep-QK。

## 12. Frozen zero-overlap holdout

构建了 64 条新 query 的冻结 holdout：与 calibration `record_uid` 重叠为 0；10M-token blocks 和 records 的 SHA-256 完全相同；134GB K index 复用；所有策略和 quota 在 target NLL 前冻结。

| action | holdout mean NLL |
| --- | ---: |
| **deep-QK** | **3.147** |
| BM25 | 3.239 |
| frozen specialist | 3.313 |
| frozen deep27 + specialist12 | 3.239 |

原样本中 `deep27 + specialist12 = 3.258`、`deep = 3.278` 的探索性小幅收益没有复现，不能作为主结果。

同时，specialist 在 45.3% target queries 上胜过 deep，四 action oracle 为 `2.799`，仍有 `0.347` headroom。这说明真正瓶颈是 tail-risk routing，而不是没有互补 operator。

## 13. Frozen router audit

- answer-free question-NLL/entropy gate：切换 4/64，mean `3.166`，比 deep 退化 `+0.0195`；
- action-wise proxy regret gate：安全回退 deep 64/64；
- 61-feature head QK confidence gate：安全回退 deep 64/64。

因此下一阶段必须使用 exact per-head attention-output distortion/omitted-mass 作为 causal dense teacher，而不能继续把 retrieval recall、task label、generic entropy 或 QK margin 当作下游 regret 代理。完整报告见 `doc/natural_holdout_report.md`。

## 14. Exact causal head-distortion teacher

在 64 条自然 query、Qwen3-0.6B 全部 448 heads、4K context 上生成了 `172,032` 个 exact post-RoPE head/action labels。每个 label 直接比较 full attention output 与候选 operator output。

固定 8-block operator 的 tail distortion 很大：QK p95 relative output L2 为 `0.149`，lexical 为 `0.312`，uniform 为 `0.344`；2-block streaming 为 `0.658`。因此统一 sparse operator 无法满足 `relative L2 <= 0.05`。

Exact per-query/head oracle 在阈值 0.05 下平均只需 `8.28/15.36` logical blocks，p95 error `0.0447`。193/448 heads 在至少 80% query 上保持相同 action，说明 static prior 与 query-conditioned activation 同时存在。

## 15. Query-disjoint conformal head router

按 query 划分 `17 fit / 16 conformal / 31 test`。Router 只使用 head identity 与 QK/lexical score signatures。

| policy | mean blocks | p95 error | violation rate |
| --- | ---: | ---: | ---: |
| global conformal 95% | 14.35 | 0.0020 | 0.00% |
| head-local conformal 95% | 11.53 | 0.0208 | 0.44% |
| **head-local conformal 90%** | **10.95** | **0.0275** | **0.86%** |
| head-local conformal 80% | 10.15 | 0.0375 | 2.08% |
| static head prior | 11.07 | 0.0294 | 1.18% |
| test oracle | 8.24 | 0.0445 | 0.00% |

90% head-local conformal 是当前第一个 query-disjoint 正结果：相对 static head prior 同时降低 blocks、mean/p95 error 和 violation rate。完整结果见 `doc/causal_teacher_pilot.md`。

GQA physical union 后，90% learned router 的 mean physical blocks 为 `12.77`，full 为 `15.29`，实际物理 block 节省 `16.46%`；static prior 节省 `15.37%`，test oracle 节省 `32.12%`。因此 GQA union 会压缩逻辑收益，但不会消除它。
