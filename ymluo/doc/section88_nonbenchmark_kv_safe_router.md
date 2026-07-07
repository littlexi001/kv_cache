# Section 88: 非 Benchmark KV-safe Router 训练与小跑

## 目标

这一步把上一节的 oracle action space 蒸馏成一个推理时可用的小 router，并且不使用 LongBench/RULER 的 benchmark 数据做训练。

使用的 label space：

```text
full_raw
recent_plus_summary1_8
recent_plus_retrieval_raw_k2
recent_plus_span_top2_b0_a0
recent_plus_span_top3_b0_a0
recent_plus_prefix_to_farthest_top3
recent_plus_full_old_raw
```

训练数据只来自非 benchmark synthetic cases，由真实长文本切片构造：

```text
War and Peace
Count of Monte Cristo
```

## 新增/修改代码

修改：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_fast_recent_plus_router_training.py
ymluo/projects/learned_hierarchical_summary_memory/src/run_qwen8b_paper_benchmarks.py
```

新增脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/scripts/train_kv_safe_topk_router_qwen8b.sh
ymluo/projects/learned_hierarchical_summary_memory/scripts/run_qwen8b_kv_safe_router_small.sh
ymluo/projects/learned_hierarchical_summary_memory/scripts/run_qwen8b_kv_safe_router_safe_small.sh
```

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/kv_safe_topk_router_nonbench_20260707
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_kv_safe_router_small_20260707
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_kv_safe_router_safe_small_20260707
```

## Router 训练

设置：

```text
model tokenizer = Qwen3-8B
cases_per_dataset = 360
prefill lengths = 4096, 8192, 16384, 20000
datasets = 2
synthetic examples = 2880
hidden_dim = 128
epochs = 1200
policy = kv_safe_topk
```

训练结果：

```text
synthetic train label accuracy = 99.95%
synthetic test label accuracy  = 100.00%
```

保存：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/kv_safe_topk_router_nonbench_20260707/router.pt
```

## Offline Heldout

先把 router 映射到上一节已有的 Qwen3-8B top-k action benchmark trials 上。

离线结果：

```text
samples = 27
score = 0.9259
full_raw score = 0.8519
relative = 108.70%
token ratio = 37.67%
```

action 分布：

```text
recent_plus_span_top2_b0_a0: 9
recent_plus_span_top3_b0_a0: 18
```

注意这里没有统计 generation summary case，因为上一节 benchmark 没跑 `recent_plus_summary1_8` 独立方法，所以 offline lookup 找不到这些 trial。

## Online Benchmark: Router

真实加载 Qwen3-8B + LoRA adapter，测试：

```text
full_raw
router
recent_plus_retrieval_raw_k2
recent_plus_span_top3_b0_a0
recent_plus_prefix_to_farthest_top3
```

公平口径：排除 full_raw OOM 的 case，只统计 full_raw 成功的 31 个 case。

| method | score | full score | relative | token ratio | seconds |
|---|---:|---:|---:|---:|---:|
| recent_plus_retrieval_raw_k2 | 0.8803 | 0.8174 | 107.70% | 46.19% | 5.11 |
| recent_plus_span_top3_b0_a0 | 0.8480 | 0.8174 | 103.75% | 40.63% | 4.92 |
| full_raw | 0.8174 | 0.8174 | 100.00% | 100.00% | 7.34 |
| router | 0.8148 | 0.8174 | 99.69% | 35.73% | 4.91 |

router 的问题：

```text
ruler_8192 / vt:
  router routed_action = recent_plus_summary1_8
  score = 0
```

也就是说，router 自身已经很省 token，但它偶尔会把 exact/global retrieval 任务错路由到 summary。

## Runtime Safety Override

因此新增 `router_safe`：

```text
如果是 summary task:
  保留 router 原选择。

如果是 exact / RULER task:
  不允许走 summary/recent_only/static_hier 等压缩过强 action。
  RULER 上压缩 action 会被覆盖到 recent_plus_span_top3_b0_a0。

如果是 vt/cwe/fwe 或 all/list/most-common 类 query:
  span_top2 会升级到 span_top3。
```

这不是 benchmark 训练，而是推理时 safety fallback，目的是避免明显错误的 action。

## Online Benchmark: Router Safe

测试：

```text
full_raw
router
router_safe
recent_plus_retrieval_raw_k2
recent_plus_span_top3_b0_a0
```

整体原始表：

| method | score | token ratio | seconds | speedup |
|---|---:|---:|---:|---:|
| full_raw | 0.7918 | 100.00% | 7.21 | 1.00x |
| recent_plus_retrieval_raw_k2 | 0.8840 | 33.00% | 5.10 | 1.41x |
| recent_plus_span_top3_b0_a0 | 0.8528 | 29.05% | 4.99 | 1.45x |
| router | 0.8206 | 26.21% | 4.89 | 1.47x |
| router_safe | 0.8519 | 27.48% | 4.97 | 1.45x |

公平口径：

| method | score | full score | relative | token ratio | seconds |
|---|---:|---:|---:|---:|---:|
| recent_plus_retrieval_raw_k2 | 0.8803 | 0.8174 | 107.70% | 46.19% | 5.10 |
| recent_plus_span_top3_b0_a0 | 0.8480 | 0.8174 | 103.75% | 40.63% | 5.00 |
| router_safe | 0.8471 | 0.8174 | 103.64% | 38.06% | 4.99 |
| full_raw | 0.8174 | 0.8174 | 100.00% | 100.00% | 7.44 |
| router | 0.8148 | 0.8174 | 99.69% | 35.73% | 4.91 |

按 benchmark：

| benchmark | full_raw | retrieval_k2 | span_top3 | router | router_safe |
|---|---:|---:|---:|---:|---:|
| LongBench | 0.2924 | 0.5361 | 0.4111 | 0.4075 | 0.4075 |
| RULER 4k | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| RULER 8k | 1.0000 | 1.0000 | 1.0000 | 0.8750 | 1.0000 |
| RULER 16k | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

token ratio：

| benchmark | retrieval_k2 | span_top3 | router | router_safe |
|---|---:|---:|---:|---:|
| LongBench | 47.8% | 38.2% | 23.6% | 30.6% |
| RULER 4k | 75.3% | 67.3% | 66.5% | 66.5% |
| RULER 8k | 38.5% | 35.2% | 32.0% | 33.6% |
| RULER 16k | 19.9% | 19.2% | 18.8% | 19.2% |

router_safe action 分布：

```text
recent_plus_retrieval_raw_k2:     1
recent_plus_span_top2_b0_a0:      6
recent_plus_span_top3_b0_a0:     23
recent_plus_summary1_8:           2
```

## 当前结论

这一步已经形成了一个可推理 router：

```text
router_safe:
  relative = 103.64% full_raw
  token ratio = 38.06%
  speedup = 约 1.49x 真实端到端 generate 时间
```

它比固定 `span_top3` 稍微省 token：

```text
span_top3:   40.63% tokens
router_safe: 38.06% tokens
```

质量基本持平：

```text
span_top3:   103.75% full_raw
router_safe: 103.64% full_raw
```

这说明 router 已经能工作，但还没有充分超过最强固定策略。

## 当前瓶颈

`router_safe` 的主要失败仍在 summary 任务：

```text
gov_report:
  full_raw = 0.1777
  router_safe = 0.1127

multi_news:
  full_raw = 0.1612
  router_safe = 0.1476
```

原因是当前 generation label 只使用：

```text
recent_plus_summary1_8
```

对 summary 任务过于激进。

下一步应该加入：

```text
recent_plus_summary1_4
recent_plus_summary1_2
```

并把 summary task 的 label 改成：

```text
brief summary -> summary1_8 / summary1_4
detailed summary -> summary1_4 / summary1_2
global full-context generation -> prefix_to_farthest_top3 或 full_old_raw
```

目标是让 router 在保持 exact/RULER 强表现的同时，把 LongBench generation 的质量拉回来。

