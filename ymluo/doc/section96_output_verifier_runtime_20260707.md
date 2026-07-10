# Section 96: Output-level verifier runtime 化结果（2026-07-07）

## 结论

今天已经把 replay 里表现最好的 **output-level risk verifier v1** 接入真实 Qwen8B runtime benchmark。

主方法现在可以表述为：

**Output-verified risk-constrained KV budget planner with RoPE-aware KV repack**

它不再是简单 router，也不是 RAG/prompt compression，而是在 full-context prefill 后，对 compact KV 候选进行输出级风险验证，并选择最小安全 KV budget。

## Runtime 实现

实现位置：

`ymluo/projects/learned_hierarchical_summary_memory/src/run_rope_aware_kv_repack_benchmark.py`

新增参数：

- `--output_verifier_path`
- `--output_verifier_threshold`
- `--output_verifier_source`
- `--output_verifier_budgets`
- `--output_verifier_mode {all,prefix}`

两个 runtime 模式：

1. `all`: 生成所有候选预算的输出，再统一验证。这是最接近 replay 的完整 verifier，但速度慢。
2. `prefix`: 按 k1 -> k2 -> k3 -> k4 -> k6 -> k8 顺序生成候选，当前候选一旦 `p_safe >= tau` 就提前停止。这是实际可用的 fast cascade。

## Qwen8B 13-task runtime 结果

设置：

- model: Qwen3-8B
- max context: 4096
- page size: 512
- max examples per task: 1
- tasks: 5 LongBench + 8 RULER
- verifier checkpoint: `output_level_verifier_multiseed_qwen8b_m4_plus_longbench12_20260707/verifier_seed_2026070811/output_level_risk_verifier.pt`
- threshold: `tau=0.7`

### 完整候选 all 模式

输出目录：

`/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/output_verifier_runtime_qwen8b_13tasks_m1_tau07_20260707`

| 方法 | Score | KV ratio | Online speed |
|---|---:|---:|---:|
| full KV cache | 69.23% | 100.00% | 1.000x |
| output verifier all | 69.23% | 14.05% | 0.167x |
| RoPE compact k2 | 69.23% | 26.08% | 1.005x |
| prompt rebuild | 61.54% | 27.16% | 0.893x |

all 模式验证了质量闭环：真实 runtime 下也能 full-level，同时 KV 降到 14.05%。但它每个样本都生成多个候选，所以速度很慢。

Action 分布：

- `k1_compact`: 12/13
- `k2_compact`: 1/13

### Fast prefix 模式

输出目录：

`/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/output_verifier_runtime_qwen8b_13tasks_m1_tau07_prefix_20260707`

| 方法 | Score | KV ratio | Online speed | E2E speed |
|---|---:|---:|---:|---:|
| full KV cache | 69.23% | 100.00% | 1.000x | 1.000x |
| output verifier prefix | 69.23% | 14.05% | 0.957x | 0.966x |
| RoPE compact k2 | 69.23% | 26.08% | 1.005x | 1.003x |
| prompt rebuild | 61.54% | 27.16% | 0.893x | 1.375x |

prefix 模式保住了 all 模式的质量和 KV ratio，同时把 online speed 从 `0.167x` 提到 `0.957x`。

这说明速度问题主要来自 all-candidate verification，而不是 KV repack 本身。实际系统应该使用 prefix/early-exit cascade。

## 当前最强方法

目前主结果应该写：

**Output verifier prefix** 在真实 Qwen8B 13-task runtime 上：

- full-level score: `69.23%`
- KV ratio: `14.05%`
- online speed: `0.957x`
- action: mostly `k1`, one case `k2`

Replay 多 seed 上：

- full/k8: `39.35%`, KV `100%`
- output verifier tau=0.7: `40.65%`, KV `20.92%`, full-level `5/5`
- oracle min-safe: `40.65%`, KV `17.13%`

组合起来看，方法已经从“一个简单 router”升级成了一个比较完整的论文方法：

1. 有明确问题定义：risk-constrained variable-budget KV planning；
2. 有 cache-native 系统实现：full prefill + RoPE-aware KV repack；
3. 有 output-level verifier：不是只看 retriever/ranker 分数；
4. 有 replay 多 seed 稳定结果；
5. 有真实 Qwen8B runtime 闭环结果。

## 仍然不足

当前还不能说“已经足够稳投 ICML”，主要缺口是：

1. runtime 规模还小：13-task m=1 不够。
2. prefix runtime 只测了一个 checkpoint 和一个 tau。
3. speedup 在 4k 长度下还没有兑现为 >1x，虽然 KV 已经降到 14.05%。
4. 需要长上下文 8k/16k/32k runtime，证明 KV 降低会转化成端到端速度收益。
5. 需要更系统的 ablation：无 output features、无 stability、all vs prefix、不同 tau。

## 下一步

优先级最高：

1. 跑 Qwen8B LongBench m=2 或 m=4 runtime prefix，确认不只是在 13-task smoke 上有效。
2. 跑 8k/16k context length 的 RULER runtime，验证 speed scaling。
3. 做 tau sweep：`0.3/0.5/0.7/0.9`，看质量-速度-KV 曲线。
4. 做 ablation，把 output-level verifier 和普通 variable-budget router、two-stage、fixed k2 放在同一表里。

如果这些结果继续稳，论文主线就比较像 ICML/ICLR 方法了。
