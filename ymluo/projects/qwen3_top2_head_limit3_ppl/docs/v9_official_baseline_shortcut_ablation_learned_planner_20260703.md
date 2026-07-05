# V9：官方 baseline、去捷径 ablation、planner 学习化

日期：2026-07-03

这一轮主要补三件事：

1. 接入 KVCache-Factory 官方实现，替代之前的 proxy baseline。
2. 做去捷径 ablation，检查 typed range-SDPA 是否只靠 answerline 格式取巧。
3. 把 planner 从手写 online proxy 推进一步，加入可训练的 causal page influence predictor。

## 1. 代码改动

新增/修改：

- `scripts/run_range_sdpa_shortcut_ablation_v9_server.sh`
  - 增加 `answerline_override`、`answerline_no_override`、`natural_no_override`、`raw_sparse_kv` 四种 ablation。
  - 固定层级分页参数：L1 page 256-1024 tokens，L2 section 8 pages，L3 chapter 64 pages，global index 128 pages。
- `src/train_oracle_regret_predictor_v8.py`
  - 新增 `--feature_policy learned_causal_proxy`。
  - 先用 train split 的 `causal_page_influence_labels.csv` 训练 page influence MLP。
  - 再把每个 candidate expert 选中 page 的 predicted influence recall/top-k coverage/mass miss 作为 regret predictor 特征。
  - 输出 `learned_causal_page_predictions.csv` 和 page predictor AUC。
- `src/summarize_kvcache_factory_longbench.py`
  - 修复 KVCache-Factory 输出目录匹配：`model_budget/task/method.json`。
  - 增加 `model_budget` 和 `budget` 字段。

## 2. 去捷径 ablation

输出目录：

`/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/range_sdpa_shortcut_ablation_v9_20260703_shortcut_ablation_v9_10k20k_x1`

设置：

- context：10k / 20k
- layout：`e05_d90,e20_d80,e35_d70`
- 每个长度每个 layout 1 条，所以这是小规模 sanity，不是最终统计。

| Ablation | Context | Full Acc | Typed Acc | Full PPL | Typed PPL | Query Speedup | Typed Kept | Typed Decoy Hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| answerline_override | 10k | 100.0% | 100.0% | 17.93 | 22.74 | 1.20x | 10.24% | 0% |
| answerline_override | 20k | 100.0% | 100.0% | 17.37 | 18.79 | 1.54x | 5.67% | 0% |
| answerline_no_override | 10k | 100.0% | 100.0% | 17.93 | 22.74 | 1.03x | 10.23% | 0% |
| answerline_no_override | 20k | 100.0% | 100.0% | 17.37 | 18.79 | 1.27x | 5.66% | 0% |
| natural_no_override | 10k | 100.0% | 100.0% | 18.35 | 23.38 | 1.20x | 10.24% | 0% |
| natural_no_override | 20k | 100.0% | 100.0% | 17.85 | 20.84 | 1.53x | 5.66% | 0% |
| raw_sparse_kv | 10k | 100.0% | 66.7% | 19.17 | 24.96 | 0.93x | 10.26% | 0% |
| raw_sparse_kv | 20k | 33.3% | 66.7% | 20.06 | 23.37 | 1.07x | 5.67% | 0% |

解释：

- 去掉 answerline override 后，typed accuracy 仍然是 100%，说明当前结果不是单纯靠“答案行直接覆盖”的后处理捷径。
- `natural_no_override` 仍然保持 100%，说明自然摘要格式在这批样本上也能支撑路由。
- `raw_sparse_kv` 掉到 66.7%，说明 sidecar/summary 不是可有可无，它确实承担了 page-level 语义组织和去 decoy 的作用。
- 但这个 ablation 很小，结论只能是“没有发现明显 answerline 捷径”，不能当作最终去捷径证明。最终需要更多随机样本、换模板、隐藏 answer literal、换实体名。

## 3. Learned Causal Planner

输出目录：

- expanded mixed workload：
  `/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/oracle_regret_predictor_v9_learned_causal_from_v7_expanded_5x20`
- hard-noise workload：
  `/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/oracle_regret_predictor_v9_learned_causal_from_v7_hardnoise_5x10`

### Page influence predictor

| Workload | Split | Positive Rate | AUC | Mean Pred Prob |
|---|---|---:|---:|---:|
| expanded | train | 18.90% | 0.937 | 0.315 |
| expanded | test | 19.28% | 0.904 | 0.302 |
| hard-noise | train | 8.33% | 0.954 | 0.223 |
| hard-noise | test | 9.33% | 0.836 | 0.226 |

这个结果说明 page-level causal influence 是可学习的，尤其 expanded workload 上泛化还可以。hard-noise 的 test AUC 下降到 0.836，说明噪声和干扰页会显著增加 page scorer 难度。

### Regret planner 结果

Expanded，200 test cases：

| SLA | Selector | Acc | PPL | Online Sec | KV Kept |
|---|---|---:|---:|---:|---:|
| quality | full_kv_cache | 50.0% | 13.14 | 0.267 | 100.0% |
| quality | learned causal planner | 51.0% | 14.04 | 0.221 | 57.4% |
| quality | deployable oracle | 67.0% | 9.44 | 0.226 | 62.0% |
| balanced | full_kv_cache | 50.0% | 13.14 | 0.267 | 100.0% |
| balanced | learned causal planner | 50.5% | 13.65 | 0.219 | 55.2% |
| balanced | deployable oracle | 67.0% | 9.44 | 0.224 | 60.7% |
| speed | full_kv_cache | 50.0% | 13.14 | 0.267 | 100.0% |
| speed | learned causal planner | 48.5% | 14.63 | 0.220 | 56.7% |
| speed | deployable oracle | 67.0% | 9.81 | 0.212 | 49.4% |

Hard-noise，75 test cases：

| SLA | Selector | Acc | PPL | Online Sec | KV Kept |
|---|---|---:|---:|---:|---:|
| quality | full_kv_cache | 60.0% | 12.45 | 0.425 | 100.0% |
| quality | learned causal planner | 58.7% | 13.98 | 0.326 | 64.9% |
| quality | deployable oracle | 74.7% | 7.63 | 0.295 | 52.1% |
| balanced | full_kv_cache | 60.0% | 12.45 | 0.425 | 100.0% |
| balanced | learned causal planner | 61.3% | 12.90 | 0.333 | 67.4% |
| balanced | deployable oracle | 74.7% | 7.63 | 0.288 | 49.7% |
| speed | full_kv_cache | 60.0% | 12.45 | 0.425 | 100.0% |
| speed | learned causal planner | 56.0% | 13.03 | 0.306 | 57.7% |
| speed | deployable oracle | 74.7% | 7.85 | 0.272 | 43.5% |

结论：

- page influence predictor 本身是有效的，但第一版 regret predictor 还没有稳定把这个收益转成最终策略选择收益。
- expanded 上 learned planner 用约 55%-57% KV 把 online latency 从 0.267s 降到 0.219-0.221s，acc 基本持平；但 PPL 变差。
- hard-noise balanced 下 acc 从 60.0% 到 61.3%，online latency 从 0.425s 到 0.333s，KV 从 100% 到 67.4%；但 PPL 仍比 full 差。
- deployable oracle 仍然明显更好，说明真正的空间在 planner 学习目标，而不是继续手调单个 expert。

下一版 planner 不应该继续只做 pointwise regret regression。更合适的是：

- pairwise ranking：同一个 query/budget 下，让模型直接学习哪个 expert 比哪个 expert regret 更小。
- confidence fallback：如果 learned planner 对 top-1 和 full 的 margin 不够大，回退 full 或扩大 KV。
- Pareto frontier learning：训练目标直接贴近 quality/balanced/speed 三个 SLA 的 frontier，而不是一个标量回归。

## 4. 官方 KVCache-Factory baseline

官方仓库：

`/home/fdong/ymluo/external/KVCache-Factory`

模型：

`/home/fdong/qwen/LlaMa-3.1-8B`

### LongBench mini

输出目录：

`/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/kvcache_factory_official_longbench_1shot_20260703_official_mini_8b_b512_10tasks_1shot`

设置：

- 8B model
- `attn_implementation=sdpa`
- budget = 512
- 每个 LongBench task 1 条
- 任务：`narrativeqa,qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,passage_retrieval_en,passage_count,gov_report,multi_news`
- 方法：`FullKV, StreamingLLM, H2O, SnapKV, PyramidKV, AdaKV`

所有方法 status 都是 OK。

| Method | Samples | Mean Score | Total Eval Sec | Mean Eval Sec |
|---|---:|---:|---:|---:|
| FullKV | 10 | 0.0848 | 71.31 | 7.13 |
| StreamingLLM | 10 | 0.0588 | 67.43 | 6.74 |
| H2O | 10 | 0.0412 | 89.38 | 8.94 |
| SnapKV | 10 | 0.0623 | 62.04 | 6.20 |
| PyramidKV | 10 | 0.0667 | 56.59 | 5.66 |
| AdaKV | 10 | 0.0635 | 258.75 | 25.88 |

注意：

- 这是 1-shot sanity，不能作为 paper 数字。
- 这个结果主要说明官方实现能在当前 3090 环境下跑通，H2O/AdaKV 这次没有 OOM 或 flash-attn 阻塞。
- AdaKV 在这个配置下明显慢，可能来自它的官方 runtime path 和 cache update 逻辑。
- Quest 在 KVCache-Factory 里没有直接入口；需要单独接官方 Quest 仓库或 runtime patch，不能把 proxy 数字混进官方 baseline。

### RULER smoke

输出目录：

`/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/kvcache_factory_official_ruler_1shot_20260703_official_ruler_smoke_8b_b512_ctx4096_1shot`

设置：

- 8B model
- context length = 4096
- budget = 512
- 每个 RULER task 1 条
- 方法：`FullKV, StreamingLLM, H2O, SnapKV, PyramidKV`

所有方法 status 都是 OK。

| Method | Tasks | Total Eval Sec | Mean Eval Sec |
|---|---:|---:|---:|
| FullKV | 11 | 32.11 | 2.92 |
| StreamingLLM | 11 | 30.95 | 2.81 |
| H2O | 11 | 40.78 | 3.71 |
| SnapKV | 11 | 33.28 | 3.03 |
| PyramidKV | 11 | 31.28 | 2.84 |

RULER 这次只确认了官方 generation path 和耗时。因为不同 RULER 子任务 scorer 不同，这里没有用简单字符串包含率冒充分数；下一步需要接 RULER 官方 scorer。

## 5. 当前判断

这轮之后，项目比之前更扎实：

- 官方 baseline 路径已经打通，不再只依赖 proxy baseline。
- answerline shortcut 的主要风险暂时被削弱，但还需要更大规模和更强模板扰动。
- causal page influence 可以学，page-level AUC 是正信号。
- regret planner 第一版还不够强，离 oracle gap 很大，说明创新重点应该转向 memory planning 的训练目标。

要往 ICML 方向继续推进，下一步优先级是：

1. 官方 baseline 正式 sweep：budget 256/512/1024/2048，LongBench 每任务固定 sampled IDs，RULER 接官方 scorer。
2. Ours adapter 接同一 generation path：不要再用不同 evaluation harness 比 speed/quality。
3. Pairwise regret planner：从 pointwise regret regression 升级到 query-conditioned expert ranking + confidence fallback。
4. 反捷径数据：移除显式 answer line、实体改写、模板随机化、decoy 同义扰动。
5. 真速度闭环：继续区分 online decode-side、query pipeline、end-to-end；只有 fused sparse prefill 之后再挑战 end-to-end full baseline。
