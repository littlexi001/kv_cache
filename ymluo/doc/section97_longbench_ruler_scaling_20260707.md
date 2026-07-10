# Section 97: LongBench 泛化与 RULER 长上下文 scaling（2026-07-07）

## 目的

在 Section 96 的 13-task runtime 结果之后，继续验证：

1. output verifier prefix 是否能泛化到更多 LongBench 样本；
2. KV 降低是否能在更长上下文 RULER 上转化成速度收益；
3. 长上下文场景下 verifier 是否有新的安全失败模式。

## 方法配置

主方法：

**Output-verified risk-constrained KV budget planner with RoPE-aware KV repack**

统一参数：

- model: Qwen3-8B
- verifier checkpoint: `output_level_verifier_multiseed_qwen8b_m4_plus_longbench12_20260707/verifier_seed_2026070811/output_level_risk_verifier.pt`
- mode: `prefix`
- threshold: `tau=0.7`
- candidate budgets: `k1,k2,k3,k4,k6,k8`
- page size: 512

新增安全参数：

- `--output_verifier_min_budget`
- `--output_verifier_long_ruler_min_budget`
- `--output_verifier_long_ruler_context_threshold`

原因：RULER 8k 上 verifier 过度相信 k1，必须给长上下文 RULER 一个最小安全预算下界。

## LongBench runtime

### LongBench m=2

输出目录：

`/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/output_verifier_runtime_qwen8b_longbench_m2_tau07_prefix_20260707`

| 方法 | Samples | Score | KV ratio | Online speed | E2E speed |
|---|---:|---:|---:|---:|---:|
| full KV | 10 | 20.00% | 100.00% | 1.000x | 1.000x |
| output verifier prefix | 10 | 20.00% | 15.00% | 0.931x | 0.940x |
| prompt rebuild | 10 | 30.00% | 26.39% | 0.917x | 1.328x |
| RoPE compact k2 | 10 | 10.00% | 25.00% | 1.004x | 1.002x |

Action 分布：

- `k1_compact`: 9
- `k3_compact`: 1

结论：output verifier match full，并显著低 KV，但没有超过 prompt rebuild；LongBench QA 仍是弱项。

### LongBench m=4

输出目录：

`/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/output_verifier_runtime_qwen8b_longbench_m4_tau07_prefix_20260707`

| 方法 | Samples | Score | KV ratio | Online speed | E2E speed |
|---|---:|---:|---:|---:|---:|
| full KV | 20 | 20.00% | 100.00% | 1.000x | 1.000x |
| output verifier prefix | 20 | 20.00% | 18.75% | 0.892x | 0.908x |
| prompt rebuild | 20 | 20.00% | 26.46% | 0.892x | 1.394x |
| RoPE compact k2 | 20 | 10.00% | 25.00% | 1.005x | 1.003x |

Action 分布：

- `k1_compact`: 17
- `k2_compact`: 1
- `k3_compact`: 1
- `k8_compact`: 1

结论：LongBench m=4 仍能 match full，说明 verifier 没有只过拟合 13-task smoke。但速度还没有兑现，主要因为 4k 下 verifier/decode 开销占比仍高。

## RULER 长上下文 scaling

### RULER 4k

输出目录：

`/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/output_verifier_runtime_qwen8b_ruler_4k_m1_tau07_prefix_20260707`

| 方法 | Samples | Score | KV ratio | Online speed | E2E speed |
|---|---:|---:|---:|---:|---:|
| full KV | 8 | 100.00% | 100.00% | 1.000x | 1.000x |
| output verifier prefix | 8 | 100.00% | 15.08% | 0.938x | 0.950x |
| RoPE compact k2 | 8 | 100.00% | 26.81% | 1.008x | 1.005x |
| prompt rebuild | 8 | 87.50% | 27.84% | 0.896x | 1.376x |

Action 分布：

- `k1_compact`: 7
- `k2_compact`: 1

结论：4k 下 output verifier full-level，KV 更低，但速度还略低于 full。

### RULER 8k 原始 prefix

输出目录：

`/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/output_verifier_runtime_qwen8b_ruler_8k_m1_tau07_prefix_20260707`

| 方法 | Samples | Score | KV ratio | Online speed |
|---|---:|---:|---:|---:|
| full KV | 8 | 100.00% | 100.00% | 1.000x |
| output verifier prefix | 8 | 62.50% | 6.57% | 1.066x |
| RoPE compact k2 | 8 | 100.00% | 13.14% | 1.076x |

失败原因：verifier 在所有 8 个任务上都选择 `k1_compact`，其中 3 个任务 k1 错但 k2 对。这个暴露了训练分布外的长上下文风险。

### RULER 8k + long-ruler floor k2

输出目录：

`/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/output_verifier_runtime_qwen8b_ruler_8k_m1_tau07_prefix_floor2_v2_20260707`

| 方法 | Samples | Score | KV ratio | Online speed | E2E speed |
|---|---:|---:|---:|---:|---:|
| full KV | 8 | 100.00% | 100.00% | 1.000x | 1.000x |
| output verifier prefix + floor k2 | 8 | 100.00% | 13.14% | 1.080x | 1.037x |
| RoPE compact k2 | 8 | 100.00% | 13.14% | 1.085x | 1.039x |
| prompt rebuild | 8 | 100.00% | 13.65% | 0.961x | 1.976x |

Action 分布：

- `k2_compact`: 8

结论：长上下文安全 floor 修复了 RULER 8k 的质量，并且首次稳定拿到真实 online speedup：`1.08x`。

### RULER 16k 单任务 smoke

全 8 任务 Qwen8B 16k 在 24GB GPU 上第二个 case OOM，原因是 full baseline + verifier + 多方法缓存组合显存过高。记录为系统限制。

单任务 `niah_single_1` 成功：

输出目录：

`/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/output_verifier_runtime_qwen8b_ruler_16k_niah1_tau07_prefix_floor2_20260707`

| 方法 | Samples | Score | KV ratio | Online speed | E2E speed |
|---|---:|---:|---:|---:|---:|
| full KV | 1 | 100.00% | 100.00% | 1.000x | 1.000x |
| output verifier prefix + floor k2 | 1 | 100.00% | 6.54% | 1.670x | 1.187x |
| RoPE compact k2 | 1 | 100.00% | 6.54% | 1.688x | 1.191x |
| prompt rebuild | 1 | 100.00% | 6.74% | 1.499x | 3.809x |

结论：16k 单任务已经显示 KV 压缩可以转化为明显 online speedup；但全任务需要更大显存、减少 baseline 同跑、或分进程逐方法评估。

## 当前判断

积极点：

1. 13-task 4k: full-level, KV 14.05%。
2. LongBench m=2/m=4: match full，KV 15%-18.75%。
3. RULER 8k + floor: full-level，KV 13.14%，online 1.08x。
4. RULER 16k 单任务: full-level，KV 6.54%，online 1.67x。
5. 方法已经不是简单 two-stage/router，而是 output-verified runtime KV planner。

风险点：

1. LongBench 上没有超过 prompt rebuild，且 full 自身分数较低。
2. RULER 8k 暴露出 OOD 风险，需要 long-context floor 或更强训练数据。
3. 16k 全任务在 24GB 卡上 OOM，系统实验需要拆分方法或用更大显存。
4. 目前速度收益主要在长上下文出现，4k 下还不强。

## 下一步

1. 把 long-context floor 写成方法的一部分：conservative budget lower bound from task/context family。
2. 生成 RULER 8k/16k 的训练标签，重新训练 verifier，而不是依赖手工 floor。
3. 分方法运行 16k 全任务，避免 full baseline 和所有候选同时占显存。
4. 做 tau/floor ablation：无 floor、floor k2、learned floor。
5. 写论文时强调：短上下文主要是 KV reduction，长上下文才兑现 speedup。
