# Value-aware KV 检索探索

更新时间：2026-07-17

## 1. 动机

只按 attention score 选择 token，优化的是保留 attention mass。但单个 token 对 attention 输出的影响还取决于 Value 向量大小。

对于 token `i`，一个简单的输出贡献上界为：

`contribution_i = exp(qk_i) * ||V_i||`

因此固定预算下，可以按下面的分数选择 token：

`priority_i = qk_i + log(||V_i||)`

该方法只需为每个 KV token 增加一个 Value norm 标量，不训练 router，也不使用任务标签。

## 2. 真实 Q/K/V trace

在 Llama-3.1-8B-Instruct 的 sports 和 medicine 32K trace 上，采集第 0、8、16、24、31 层，共 320 个 query-head 样本。这里使用 exact QK，直接比较相同 token 数下重建的 attention output 与 Full output。

| Token budget | QK output error | Value-aware output error | 相对降低 | QK cosine | Value-aware cosine |
|---:|---:|---:|---:|---:|---:|
| 0.5% | 0.3131 | 0.3105 | 0.8% | 0.9351 | 0.9482 |
| 1% | 0.2633 | 0.2541 | 3.5% | 0.9482 | 0.9623 |
| 2% | 0.2176 | 0.2022 | 7.1% | 0.9587 | 0.9732 |
| 4% | 0.1711 | 0.1484 | 13.3% | 0.9680 | 0.9831 |

Value-aware 排名稳定降低了平均 attention-output 误差，但会轻微降低 retained attention mass。例如 2% 预算下，平均 mass 从 0.8794 降到 0.8702。

## 3. 端到端 PPL

协议固定为 32K 历史、sports/medicine window 0、每主题 128 个 target tokens。所有稀疏方法使用同一组动态预算、PCA64 INT4、0.25% 样本校准和 2x exact overfetch。

| 方法 | Sports PPL | Medicine PPL | 几何 PPL | 相对 global | Links | 在线时间 |
|---|---:|---:|---:|---:|---:|---:|
| Full Attention | 8.0139 | 5.8937 | 6.8725 | -1.66% | 100% | 29.18s |
| Global QK | 8.2795 | 5.8987 | 6.9884 | 0.00% | 2.31% | 42.86s |
| Value candidate only | 8.2447 | 5.9100 | 6.9804 | -0.11% | 2.31% | 45.50s |
| Value exact rerank only | 8.2756 | 5.9005 | 6.9879 | -0.01% | 2.31% | 45.96s |
| Value candidate + rerank | 8.2359 | 5.9105 | 6.9769 | -0.16% | 2.31% | 46.91s |

## 4. 结论

1. `qk + log(||V||)` 是成立的单层输出误差准则，2% 到 4% 预算时子系统收益明显。
2. 该收益没有稳定转化为端到端 PPL：最佳组合只比 global 改善 0.16%，medicine 反而略差。
3. 当前实现还增加了约 9% 的原型在线时间，因此不应加入默认主方法。
4. 该方向保留为论文 ablation 或未来低预算专用模式，不继续调 Value 权重。
5. 主方法仍使用 attention-mass planner；更值得继续探索的是如何避免每步扫描完整索引。

## 5. 代码与结果

- Value norm 索引与候选实现：`src/run_head_top2_targeted_ppl_20260714.py`
- Trace 分析：`src/analyze_value_weighted_retrieval_20260717.py`
- Trace 汇总：`results/20260717_value_weighted_trace_analysis/summary.json`
- Global 对照：`results/20260717_global_twotheme_eval128_32k`
- Value candidate：`results/20260717_value_bound_alpha1_twotheme_32k`
- Value rerank：`results/20260717_contribution_rerank_twotheme_eval128_32k`
- Value candidate + rerank：`results/20260717_contribution_full_twotheme_eval128_32k`
