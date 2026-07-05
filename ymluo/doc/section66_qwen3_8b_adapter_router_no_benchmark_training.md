# Section 66: Qwen3-8B 专用 LoRA adapter 与 router 非 benchmark 训练

## 目标

本节训练 Qwen3-8B 专用的 summary-memory LoRA adapter 和 runtime router。

关键约束：

```text
不使用 LongBench / RULER / 之后正式 benchmark 的数据做 adapter 训练或 router oracle 蒸馏。
```

训练数据只使用本地普通长文本：

```text
War and Peace
Monte Cristo
```

路径：

```bash
ymluo/projects/qwen3_top2_head_limit3_ppl/data/war_and_peace_pg2600.txt
ymluo/projects/qwen3_top2_head_limit3_ppl/data/count_monte_cristo_pg1184.txt
```

## 模型

Qwen3-8B 本地 snapshot：

```bash
/home/fdong/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
```

服务器没有 `bitsandbytes`，因此本次不是 4bit QLoRA，而是 fp16 LoRA。单张 RTX 3090 24GB 可跑，但显存接近上限。

## 代码修改

为了让训练 action space 和后续 benchmark 一致，补了 ratio-summary 动作：

```text
summary1_8
summary1_4
summary1_2
```

修改文件：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_static_summary_ppl_speed.py
ymluo/projects/learned_hierarchical_summary_memory/src/run_task_adaptive_memory_policy_eval.py
ymluo/projects/learned_hierarchical_summary_memory/src/run_static_summary_lora_adaptation.py
ymluo/projects/learned_hierarchical_summary_memory/src/run_qwen8b_paper_benchmarks.py
```

主要变化：

- `context_for_method` 支持 `summary1_8/summary1_4/summary1_2`。
- LoRA adapter 训练支持 `--train_methods`，每个 step 从多个 memory format 里采样。
- task-adaptive oracle 支持 ratio-summary 方法。
- benchmark runtime 支持 router 输出 `recent_only`。

## LoRA Adapter 训练

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_lora_4k_1ksteps_no_bench_20260705
```

adapter 路径：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_lora_4k_1ksteps_no_bench_20260705/adapter
```

训练配置：

```text
train_steps = 1000
prefill_tokens = 4096
target_tokens = 64
block_tokens = 1024
recent_tokens = 512
LoRA r = 8
LoRA alpha = 16
train_methods = static_hier, summary1_8, summary1_4, recent_only
summary_backend = learned
```

训练耗时：

```text
约 1364 秒，约 22.7 分钟
```

可训练参数：

```text
trainable_params = 21,823,488
total_params = 8,212,558,848
trainable_fraction = 0.2657%
```

adapter 文件大小：

```text
adapter_model.safetensors: 约 87 MB
```

## 非 benchmark PPL 初评

评估仍然只在 War and Peace / Monte Cristo 的 held-out token range 上做，不使用正式 benchmark。

| phase | method | PPL | token ratio | speedup vs full_raw |
|---|---|---:|---:|---:|
| base | full_raw | 13.61 | 100.00% | 1.00x |
| base | recent_only | 14.70 | 13.85% | 7.12x |
| base | static_hier | 15.32 | 32.42% | 3.00x |
| base | summary1_2 | 14.83 | 57.22% | 1.84x |
| base | summary1_4 | 15.48 | 35.91% | 2.89x |
| base | summary1_8 | 15.82 | 25.24% | 3.91x |
| adapted | full_raw | 11.23 | 100.00% | 1.00x |
| adapted | recent_only | 12.16 | 13.85% | 6.84x |
| adapted | static_hier | 11.64 | 32.42% | 2.92x |
| adapted | summary1_2 | 11.36 | 57.22% | 1.77x |
| adapted | summary1_4 | 11.63 | 35.91% | 2.77x |
| adapted | summary1_8 | 11.87 | 25.24% | 3.77x |

观察：

- adapter 后 full_raw PPL 也从 13.61 降到 11.23，说明 LoRA 也适配了该 held-out 文本分布。
- 压缩格式下降更明显：
  - `summary1_8`: 15.82 -> 11.87
  - `summary1_4`: 15.48 -> 11.63
  - `static_hier`: 15.32 -> 11.64
- `summary1_8` 用约 25.2% tokens，PPL 接近 adapted full_raw。

这只是非 benchmark sanity，不作为正式 LongBench/RULER 结论。

## Router Oracle 生成

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_router_oracle_no_bench_20260705
```

oracle 使用 Qwen3-8B + 上面的 LoRA adapter 生成。

数据仍然只来自 War and Peace / Monte Cristo：

- generation cases：普通 next-token LM，选择 NLL 接近 full_raw 且 token 最少的方法。
- exact cases：在书籍文本里插入 synthetic private access code，做精确回忆；这不是 LongBench/RULER。

候选方法：

generation:

```text
full_raw
recent_only
static_hier
summary1_8
summary1_4
summary1_2
```

exact:

```text
full_raw
static_hier
summary1_8
summary1_4
summary1_2
retrieval_raw_k1
retrieval_raw_k2
```

oracle 结果：

| task family | samples | success | avg token ratio | fallback full_raw | selection |
|---|---:|---:|---:|---:|---|
| overall | 64 | 100.00% | 37.50% | 1.56% | recent/retrieval/static/summary mixed |
| exact | 32 | 100.00% | 56.96% | 0.00% | retrieval_raw_k1 90.6%, retrieval_raw_k2 9.4% |
| generation | 32 | 100.00% | 18.04% | 3.12% | recent_only 78.1%, static_hier 9.4%, summary1_8 9.4%, full_raw 3.1% |

这个 oracle 分布符合预期：

- 精确回忆任务需要 raw retrieval。
- 普通生成任务多数可以用 recent/summary memory。
- full_raw fallback 很低。

## Router 蒸馏

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_router_distill_no_bench_20260705
```

router checkpoint：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_router_distill_no_bench_20260705/router.pt
```

router label space：

```text
full_raw
recent_only
retrieval_raw_k1
retrieval_raw_k2
static_hier
summary1_8
```

蒸馏结果：

| split | family | samples | label acc | routed success | avg token ratio |
|---|---|---:|---:|---:|---:|
| train | overall | 42 | 100.00% | 100.00% | 38.21% |
| train | exact | 21 | 100.00% | 100.00% | 56.96% |
| train | generation | 21 | 100.00% | 100.00% | 19.47% |
| test | overall | 22 | 77.27% | 86.36% | 37.17% |
| test | exact | 11 | 81.82% | 90.91% | 60.79% |
| test | generation | 11 | 72.73% | 81.82% | 13.55% |

## 当前状态

已经得到两个不污染 benchmark 的 8B 专用产物：

```bash
LoRA adapter:
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_lora_4k_1ksteps_no_bench_20260705/adapter

Router:
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_router_distill_no_bench_20260705/router.pt
```

这些可以用于后续正式 LongBench/RULER/PPL 测试。

## 注意

本节训练和蒸馏没有使用正式 benchmark 数据，因此后续可以把 LongBench/RULER 作为相对干净的 test set。

但这还不是最终 ICML 证据：

- 只用了两本文本，router 数据量只有 64 cases。
- adapter 只训了 1k steps，且 context 是 4k，不是 8k/16k。
- 后续需要在不训练的情况下测试 LongBench/RULER/PPL。
- 如果 8B benchmark 上效果好，再扩大非benchmark训练数据和 steps。
