# V10：Pairwise Regret Ranker 与 RULER Scorer

日期：2026-07-04

这一轮做了两件事：

1. 新增 `pairwise/listwise regret ranker`，把 planner 学习目标从 V8 的 pointwise regret regression 改成同一 query 内 expert 排序。
2. 接入 KVCache-Factory RULER 的 `string_match_all` 公式，补 RULER smoke 分数汇总。

## 1. 新增代码

- `src/train_oracle_regret_ranker_v10.py`
  - 复用 V8 的 online / learned causal page influence 特征。
  - 训练目标：
    - pairwise：同一个 `(variant, task_id, budget, sla)` 内，低 regret expert 的 rank score 应低于高 regret expert。
    - listwise：用 `softmax(-log1p(regret))` 做组内软标签。
    - pointwise：保留一个弱的 log regret smooth-l1 正则。
  - 评测时输出：
    - `pairwise_ranker_v10`
    - `pairwise_ranker_v10_fullfb_{margin}`，当 sparse expert 相对 full 的预测优势不够时回退 full。
- `src/summarize_kvcache_factory_ruler.py`
  - 复现 KVCache-Factory `eval_ruler.py` 使用的 `string_match_all` 公式。
  - 避免因官方 `metrics.py` 顶层依赖 `jieba/rouge/fuzzywuzzy` 缺失而不能跑 scorer。

## 2. Pairwise Ranker 结果

### Expanded workload

输入：

`/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/oracle_regret_memory_planner_v7_20x_b1358_20260703_v7_expanded_5x20`

输出：

- 强模型版：`outputs/oracle_regret_ranker_v10_learned_causal_from_v7_expanded_5x20`
- 强正则版：`outputs/oracle_regret_ranker_v10_regstrong_expanded_5x20`

关键对比：

| Selector | SLA | Acc | PPL | Online Sec | KV Kept |
|---|---|---:|---:|---:|---:|
| full_kv_cache | balanced | 50.0% | 13.14 | 0.267 | 100.0% |
| V8 learned causal pointwise | balanced | 50.5% | 13.65 | 0.219 | 55.2% |
| V10 pairwise strong | balanced | 41.5% | 16.49 | 0.212 | 48.8% |
| V10 pairwise regstrong | balanced | 42.5% | 15.64 | 0.222 | 57.9% |
| deployable oracle | balanced | 67.0% | 9.44 | 0.224 | 60.7% |

结论：

- V10 在 expanded 上不成功。
- 强模型版 train pair accuracy 接近 0.99，但 test acc 明显下降，说明过拟合 query/workload。
- 强正则版降低了过拟合，但仍没有超过 V8 pointwise。
- 这说明“pairwise ranking”方向本身合理，但当前数据量和特征还不足以支撑 MLP ranker 泛化。

### Hard-noise workload

输入：

`/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/oracle_regret_memory_planner_v7_10x_b358_20260703_v7_hardnoise_5x10`

输出：

- 强模型版：`outputs/oracle_regret_ranker_v10_learned_causal_from_v7_hardnoise_5x10`
- 强正则版：`outputs/oracle_regret_ranker_v10_regstrong_hardnoise_5x10`

关键对比：

| Selector | SLA | Acc | PPL | Online Sec | KV Kept |
|---|---|---:|---:|---:|---:|
| full_kv_cache | balanced | 60.0% | 12.45 | 0.425 | 100.0% |
| V8 learned causal pointwise | balanced | 61.3% | 12.90 | 0.333 | 67.4% |
| V10 pairwise strong | balanced | 52.0% | 13.13 | 0.303 | 56.2% |
| V10 pairwise regstrong | balanced | 58.7% | 13.48 | 0.305 | 56.5% |
| V10 regstrong fullfb_0.5 | balanced | 58.7% | 12.30 | 0.322 | 62.6% |
| deployable oracle | balanced | 74.7% | 7.63 | 0.288 | 49.7% |

结论：

- hard-noise 上 V10 强正则版比 expanded 稳定，但仍没有超过 V8 balanced。
- `fullfb_0.5` 有一个有意思的点：PPL 12.30 比 full 的 12.45 略低，同时 online 从 0.425s 降到 0.322s，KV kept 62.6%；但 acc 58.7% 低于 full 的 60.0%。
- 这说明 fallback 能改善 PPL/速度折中，但还没有解决 accuracy risk。

## 3. 为什么 V10 没有超过 V8

主要原因不是 causal page predictor 失效，而是 ranker 泛化不足：

- 训练样本太少。expanded 只有 1000 train rows，hard-noise 只有 375 train rows；每个 query 只有 5 个 candidate expert。
- Pairwise pair 数虽然看起来多，但都来自同一批 query，不是真正独立样本。
- MLP 很容易记住 variant 和 candidate pattern。强模型版 train group oracle match 高，但 test 明显掉。
- 当前 feature 仍然偏 workload-specific，尤其依赖 synthetic variant、budget、candidate mode one-hot。
- full fallback 只解决“风险校准”，不能修正 ranker 对 sparse expert 的错误排序。

所以目前最好的 planner 仍是 V8 pointwise learned causal proxy，而不是 V10 pairwise。

## 4. RULER 官方公式 Scorer

RULER smoke 输出目录：

`/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/kvcache_factory_official_ruler_1shot_20260703_official_ruler_smoke_8b_b512_ctx4096_1shot`

设置：

- model：Llama-3.1-8B
- context length：4096
- budget：512
- 每个 RULER task 1 条
- scorer：KVCache-Factory `string_match_all`

汇总：

| Method | Tasks | Score | Total Eval Sec | Mean Eval Sec |
|---|---:|---:|---:|---:|
| FullKV | 11 | 100.00 | 32.11 | 2.92 |
| SnapKV | 11 | 76.06 | 33.28 | 3.03 |
| PyramidKV | 11 | 73.33 | 31.28 | 2.84 |
| H2O | 11 | 33.79 | 40.78 | 3.71 |
| StreamingLLM | 11 | 19.09 | 30.95 | 2.81 |

解释：

- 这只是 1-shot smoke，但能说明 RULER generation + scoring path 已经打通。
- FullKV 在 4096/1-shot 下是 100，不意外。
- SnapKV/PyramidKV 明显比 StreamingLLM/H2O 稳。
- 后续正式 benchmark 要跑 256/512/1024/2048 budget sweep，并扩大样本数。

## 5. 当前结论

这轮得到的最重要结论是负面的，但很有价值：

> 直接把 regret predictor 改成 pairwise MLP ranker，并不会自然变好；在当前数据规模下，它更容易过拟合。

下一步更合理的 planner 方向：

1. 先做更大的 oracle label 数据集，而不是继续调 ranker。
2. 用 cross-task / cross-template split，避免 train/test 都是同类 synthetic pattern。
3. 用更简单的模型先建立稳健 baseline：linear / ridge / GBDT ranker。
4. 把 fallback 训练成 risk classifier：预测 sparse 是否会错，而不是只看 rank score margin。
5. 正式接同一 generation path，把 ours 和 KVCache-Factory 官方方法放到同一批 sampled IDs 上比较。
