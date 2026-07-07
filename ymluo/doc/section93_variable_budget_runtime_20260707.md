# Section 93: Variable-budget planner runtime 接入（2026-07-07）

## 本轮目标

把上一节的 replay-only variable-budget planner 接入真实 runtime。

现在新增的方法不再只是离线根据结果选动作，而是在实际 benchmark 流程里：

1. full-context prefill 一次，得到完整 KV cache；
2. 对 context pages 做 lexical page scoring；
3. 对 `k1/k2/k3/k4/k6/k8` 分别构造 runtime features；
4. planner 选择 `kN_compact` 或 `full`；
5. 对选中的 KV pages 做 RoPE-aware repack；
6. 在 repacked KV cache 上 query/decode。

主方法仍然不是 RAG，也不是 prompt rebuild；它不拼接文本块重新 prefill，而是操作 full-context prefill 之后的 KV pages。

## 新增代码

文件：

- `ymluo/projects/learned_hierarchical_summary_memory/src/run_rope_aware_kv_repack_benchmark.py`

新增参数：

- `--variable_budget_planner_path`
- `--variable_budget_policy {argmax,tail_risk}`
- `--variable_budget_tail_threshold`
- `--variable_budget_source`

新增 runtime method：

- `variable_budget_kv_planner`

planner checkpoint 使用上一节训练出的：

- `/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/variable_budget_planner_qwen8b_m4_plus_longbench12_k1k2k3k4k6k8_best_20260707/variable_budget_planner.pt`

## Runtime smoke

输出目录：

- `/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/rope_repack_runtime_variable_budget_smoke_qwen8b_20260707`

设置：

- Qwen3-8B
- 2 tasks：`hotpotqa` + `niah_single_1`
- `max_examples_per_task=1`
- `max_context_tokens=4096`
- policy：tail-risk, threshold=0.35

结果：

| method | score | KV ratio | online speed |
|---|---:|---:|---:|
| full KV cache | 50.00% | 100.00% | 1.000x |
| RoPE compact k2 | 100.00% | 26.45% | 1.032x |
| variable-budget planner | 100.00% | 13.22% | 1.007x |

动作：

- `k1_compact`: 2/2

结论：runtime 路径可以跑通，planner 能真实选择比固定 k2 更低的预算。

## 13-task m=1 runtime 小评估

输出目录：

- `/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/rope_repack_runtime_variable_budget_qwen8b_13tasks_m1_20260707`

设置：

- Qwen3-8B
- LongBench 5 tasks + RULER 8 tasks
- `max_examples_per_task=1`
- `max_context_tokens=4096`
- `page_tokens=512`
- policy：tail-risk, threshold=0.35

整体结果：

| method | score | KV ratio | online speed | e2e speed | amort16 e2e |
|---|---:|---:|---:|---:|---:|
| full KV cache | 69.23% | 100.00% | 1.000x | 1.000x | 1.000x |
| prompt rebuild | 61.54% | 27.16% | 0.893x | 1.372x | 0.923x |
| RoPE compact k2 | 69.23% | 26.08% | 1.005x | 1.003x | 1.005x |
| variable-budget planner | 69.23% | 15.05% | 0.991x | 0.994x | 0.992x |

分组：

| group | full score | k2 score | variable score | variable KV |
|---|---:|---:|---:|---:|
| LongBench | 20.00% | 20.00% | 20.00% | 15.00% |
| RULER 4096 | 100.00% | 100.00% | 100.00% | 15.08% |

动作分布：

- `k1_compact`: 11/13
- `k2_compact`: 2/13

关键观察：

1. variable-budget runtime 达到 full-level 分数，同时 KV ratio 从 100% 降到约 15%。
2. 它比固定 RoPE compact k2 更省 KV：15.05% vs 26.08%。
3. 当前 online/e2e 速度没有兑现，约 0.99x，原因是 planner + RoPE repack 的 Python/HF 开销抵消了 4k 下的 attention 节省。
4. planner 概率非常尖锐，大量样本 `p(k1)` 接近 1；这说明 replay planner 可以工作，但校准还不够稳健。

## 当前论文判断

这一步非常重要：方法现在已经从 replay planner 变成了真实 runtime 方法。

可以主张：

**在真实 Qwen3-8B runtime 中，variable-budget KV planner 能在 full-context cache 上动态选择 KV 预算，以约 15% KV 达到 full-level accuracy。**

但还不能夸大：

- 4k 下端到端速度没有明显提升；
- 13-task m=1 样本仍小；
- LongBench full 自身分数低，不能只靠 m=1 说明 worst-case 稳健；
- planner 概率过度自信，tail-risk 校准在这个 checkpoint 上还没有充分发挥作用。

## 下一步

1. 训练 cost-sensitive/risk-calibrated variable-budget planner：
   - loss 直接惩罚 `score < full`；
   - 在安全前提下惩罚 KV；
   - 降低过度自信的 `k1` 预测。
2. 跑 LongBench m12 的 runtime variable-budget：
   - 使用 `--variable_budget_source group1__k1__k2__k3__k4__k6__k8`；
   - 看 worst-case 下是否需要升到 k4/k6/k8。
3. 做 8k runtime：
   - 4k 下速度很难兑现；
   - 8k/16k 才更接近论文里的系统收益场景。
4. 把 variable-budget 和 two-stage 放到同一个 runtime benchmark 表里：
   - full
   - prompt rebuild
   - fixed k2
   - two-stage calibrated
   - variable-budget argmax
   - variable-budget tail-risk
   - oracle

当前最好的方法应更新为：

**Runtime variable-budget risk-constrained KV planner + RoPE-aware KV repack**

当前最强 runtime 数字：

- score：69.23%，达到 full；
- KV ratio：15.05%；
- online speed：0.991x；
- e2e speed：0.994x。

## LongBench-only m=2 runtime

输出目录：

- `/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/rope_repack_runtime_variable_budget_qwen8b_longbench_m2_group1_20260707`

设置：

- LongBench 5 tasks
- `max_examples_per_task=2`
- 强制使用 `--variable_budget_source group1__k1__k2__k3__k4__k6__k8`
- policy：tail-risk, threshold=0.35

整体结果：

| method | score | KV ratio | online speed |
|---|---:|---:|---:|
| full KV cache | 20.00% | 100.00% | 1.000x |
| prompt rebuild | 30.00% | 26.39% | 0.894x |
| RoPE compact k2 | 10.00% | 25.00% | 1.008x |
| variable-budget planner | 20.00% | 16.25% | 0.989x |
| oracle sparse | 20.00% | 25.00% | 1.009x |
| oracle with full | 30.00% | 55.00% | 1.004x |

动作分布：

- `k1_compact`: 8/10
- `k2_compact`: 1/10
- `k3_compact`: 1/10

结论：

- 在 LongBench-only worst-case 小样本上，variable-budget runtime 达到 full-level，并且 KV 只有 16.25%。
- 但是它没有达到 prompt rebuild / oracle-with-full 的 30% 分数；这说明当前方法的“省 KV”已经成立，但 worst-case 质量还有明显空间。
- 动作仍偏低预算，且概率很尖锐；需要更强的 risk-aware/cost-sensitive 训练，而不是只靠普通 CE。

## 校准训练初试

新增训练参数：

- `--label_smoothing`
- `--confidence_penalty`

新增 runtime 参数：

- `--variable_budget_temperature`

初试 checkpoint：

- `/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/variable_budget_planner_qwen8b_m4_plus_longbench12_k1k2k3k4k6k8_best_calibrated_ls005_cp001_20260707`

参数：

- label smoothing = 0.05
- confidence penalty = 0.01

结果不作为主结果：

| policy | score | KV ratio |
|---|---:|---:|
| fixed full | 37.14% | 100.00% |
| learned argmax | 28.57% | 22.59% |
| tail-risk tau=0.05 | 37.14% | 77.69% |
| tail-risk tau=0.8 | 34.29% | 18.30% |

结论：

简单 label smoothing + entropy penalty 不够好，会牺牲 argmax 质量；tail-risk 虽然能恢复 full-level，但 KV 太高。下一步不能只做普通校准，要改成真正的 cost-sensitive objective：显式惩罚 `score < full`，并在安全前提下最小化 KV。
