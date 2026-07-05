# V8: Oracle Regret Predictor 实验记录

日期：2026-07-03

## 目标

这一步从“继续调单个 expert”转向训练 regret predictor：

```text
query/page features + candidate expert proxy features -> predicted regret per expert
```

在线选择时，对每个候选 expert 预测 regret，然后选择 predicted regret 最小的 expert。候选 expert 包括：

```text
full_kv_cache
recent_kv_gather_topk
lexical_kv_gather_topk
learned_causal_kv_gather_topk
set_utility_kv_gather_v7
```

核心变化是：不再假设某个 KV expert 永远最好，而是学习“当前 query、page 结构、budget、SLA 下哪个 expert 的 regret 最低”。

## 代码

新增脚本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/train_oracle_regret_predictor_v8.py
```

输入 V7 输出目录：

```text
strategy_results.csv
oracle_regret_labels.csv
causal_page_influence_labels.csv
```

输出：

```text
predicted_regret_cases.csv
predicted_regret_summary.csv
model_summary.json
feature_names.json
```

模型是一个轻量 MLP：

```text
Linear -> GELU -> Linear
target = log1p(deployable_objective_regret)
loss = weighted smooth L1
```

脚本现在支持两个 feature policy：

| Policy | 用途 | 是否作为主结果 |
|---|---|---|
| `online_proxy` | 只使用选择前可获得的 proxy：budget、mode、page feature 聚合、keep fraction、selected pages/tokens、estimated causal recall 等 | 是 |
| `oracle_debug` | 额外使用 teacher/gold outcome 字段：correct、gold PPL、margin、true causal label/loss delta 等 | 否，只做诊断上界 |

这个区分很重要。早期 V8 proof-of-concept 的结果偏高，因为混入了部分在线选择前不可获得的 outcome 字段。现在主结论以 `online_proxy` 为准。

## 运行命令

严格 online-proxy：

```bash
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
/home/fdong/miniconda3/bin/python src/train_oracle_regret_predictor_v8.py \
  --input_dir outputs/oracle_regret_memory_planner_v7_20x_b1358_20260703_v7_expanded_5x20 \
  --output_dir outputs/oracle_regret_predictor_v8_online_proxy_from_v7_expanded_5x20 \
  --epochs 800 \
  --hidden_dim 48 \
  --feature_policy online_proxy

/home/fdong/miniconda3/bin/python src/train_oracle_regret_predictor_v8.py \
  --input_dir outputs/oracle_regret_memory_planner_v7_10x_b358_20260703_v7_hardnoise_5x10 \
  --output_dir outputs/oracle_regret_predictor_v8_online_proxy_from_v7_hardnoise_5x10 \
  --epochs 800 \
  --hidden_dim 48 \
  --feature_policy online_proxy
```

诊断上界：

```bash
/home/fdong/miniconda3/bin/python src/train_oracle_regret_predictor_v8.py \
  --input_dir outputs/oracle_regret_memory_planner_v7_20x_b1358_20260703_v7_expanded_5x20 \
  --output_dir outputs/oracle_regret_predictor_v8_oracle_debug_from_v7_expanded_5x20 \
  --epochs 800 \
  --hidden_dim 48 \
  --feature_policy oracle_debug

/home/fdong/miniconda3/bin/python src/train_oracle_regret_predictor_v8.py \
  --input_dir outputs/oracle_regret_memory_planner_v7_10x_b358_20260703_v7_hardnoise_5x10 \
  --output_dir outputs/oracle_regret_predictor_v8_oracle_debug_from_v7_hardnoise_5x10 \
  --epochs 800 \
  --hidden_dim 48 \
  --feature_policy oracle_debug
```

## 严格结果：online_proxy

### Expanded 5x20

输出目录：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/oracle_regret_predictor_v8_online_proxy_from_v7_expanded_5x20
```

| Selector | SLA | Acc | PPL | Online sec | KV kept | Oracle match |
|---|---:|---:|---:|---:|---:|---:|
| full_kv_cache | quality | 50.0% | 13.14 | 0.267 | 100.0% | 56.0% |
| set_utility_kv_gather_v7 | quality | 37.0% | 19.56 | 0.173 | 13.1% | 9.5% |
| deployable_oracle | quality | 67.0% | 9.44 | 0.226 | 62.0% | 84.0% |
| predicted_regret_v8 | quality | 47.0% | 14.91 | 0.229 | 64.8% | 46.0% |
| predicted_regret_v8 | balanced | 47.5% | 14.63 | 0.227 | 62.9% | 45.0% |
| predicted_regret_v8 | speed | 44.5% | 15.94 | 0.222 | 58.6% | 34.0% |

训练信息：

| SLA | Train rows | MAE log1p(regret) | Feature count |
|---|---:|---:|---:|
| quality | 1000 | 1.151 | 91 |
| balanced | 1000 | 1.120 | 91 |
| speed | 1000 | 1.120 | 91 |

### Hard-noise 5x10

输出目录：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/oracle_regret_predictor_v8_online_proxy_from_v7_hardnoise_5x10
```

| Selector | SLA | Acc | PPL | Online sec | KV kept | Oracle match |
|---|---:|---:|---:|---:|---:|---:|
| full_kv_cache | quality | 60.0% | 12.45 | 0.425 | 100.0% | 48.0% |
| set_utility_kv_gather_v7 | quality | 45.3% | 14.22 | 0.173 | 8.2% | 8.0% |
| deployable_oracle | quality | 74.7% | 7.63 | 0.295 | 52.1% | 88.0% |
| predicted_regret_v8 | quality | 58.7% | 13.19 | 0.354 | 74.3% | 45.3% |
| predicted_regret_v8 | balanced | 61.3% | 11.92 | 0.361 | 76.8% | 44.0% |
| predicted_regret_v8 | speed | 58.7% | 13.28 | 0.333 | 66.9% | 30.7% |

训练信息：

| SLA | Train rows | MAE log1p(regret) | Feature count |
|---|---:|---:|---:|
| quality | 375 | 0.379 | 91 |
| balanced | 375 | 0.393 | 91 |
| speed | 375 | 0.362 | 91 |

## 诊断结果：oracle_debug

`oracle_debug` 不是公平在线结果，因为它使用了在线选择前不可获得的字段，例如 true correct、gold PPL、margin、true causal label、loss delta。它的意义是判断：如果我们能训练出一个足够强的 page/query outcome estimator，regret planner 上界有多高。

| Dataset | Selector | SLA | Acc | PPL | Online sec | KV kept | Oracle match |
|---|---|---:|---:|---:|---:|---:|---:|
| expanded 5x20 | full_kv_cache | quality | 50.0% | 13.14 | 0.267 | 100.0% | 56.0% |
| expanded 5x20 | deployable_oracle | quality | 67.0% | 9.44 | 0.226 | 62.0% | 84.0% |
| expanded 5x20 | predicted_regret_v8 | quality | 65.0% | 10.79 | 0.224 | 60.3% | 62.5% |
| hard-noise 5x10 | full_kv_cache | quality | 60.0% | 12.45 | 0.425 | 100.0% | 48.0% |
| hard-noise 5x10 | deployable_oracle | quality | 74.7% | 7.63 | 0.295 | 52.1% | 88.0% |
| hard-noise 5x10 | predicted_regret_v8 | quality | 74.7% | 9.00 | 0.341 | 69.0% | 65.3% |

## 分析

V8 结果现在应分两层看：

1. `oracle_debug` 说明 regret predictor 方向是有潜力的。只要有足够强的 page/query outcome estimator，planner 可以接近 deployable oracle，并明显超过单个 expert。
2. `online_proxy` 说明当前上线前可获得的浅层特征还不够。expanded 集合上它低于 full；hard-noise 上 balanced SLA 略好于 full，但提升不稳定。

这不是否定 regret predictor，反而说明下一步应该做“teacher-distilled causal/page outcome predictor”，而不是继续调 expert heuristic。

当前瓶颈：

- `oracle_regret_labels.csv` 的 label 很强，但 online feature 还弱。
- 当前 page feature 主要是 lexical/typed/position/role heuristic，缺少真正语义 embedding 和 causal influence 预测。
- 如果不给 predictor 提供“这个 query 下哪些 page 会影响答案/logits”的可泛化信号，它只能学到粗糙的 mode/budget 偏好。

## 结论

主结论应该写成：

```text
Oracle regret labels reveal a strong planner upper bound, but shallow online proxy features are insufficient.
The next key contribution should be a teacher-distilled causal page influence / outcome predictor,
then use its predicted influence features to drive regret-based memory planning.
```

也就是说，V8 把研究方向定位清楚了：

- 不再主打单个 sparse KV expert。
- 不把 `oracle_debug` 当最终结果。
- 真正创新点应是：用 full/ablation teacher 产生 causal regret labels，再训练在线可用的 predictor，最后在 range-SDPA / sparse prefill path 里执行 planner。

## 下一步

推荐 V9：

1. 用 `causal_page_influence_labels.csv` 训练一个在线可用的 page influence predictor，输入只用 page/query features 或 embedding/reranker 分数。
2. 把 V8 的 `online_proxy` 特征增强为 `predicted_causal_positive_rate`、`predicted_loss_delta_topk`、`predicted_evidence_recall`、`predicted_full_risk`。
3. 做 cross-run 验证：expanded 训练，hard-noise / 新任务测试，避免同分布过拟合。
4. 加 confidence calibration：当 predicted regret 差距小或 full risk 高时 fallback。
5. 把 selector 接进 range-SDPA / sparse prefill execution path，报告真实 online latency 和 end-to-end latency。
