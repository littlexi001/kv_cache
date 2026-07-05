# 第 48 节：Summary Memory 时间测试与路由占比统计

日期：2026-07-03

## 0. 本节目标

本节补充两个实验：

```text
1. 统计 adaptive_no_raw / adaptive_with_raw 实际有多少 query 走 summary10、summary100、summary1000、full attention。
2. 用 Qwen3-0.6B 做真实 prefill wall time 测试，观察摘要记忆作为 prompt/context 时能带来多少实际时间加速。
```

注意：这里的时间测试不是自定义 KV-cache gather kernel 的极限性能测试，而是更贴近当前方法定义的版本：

```text
先根据 query 选择 memory level，再把对应 summary 或 raw text 拼成 prompt，直接喂给普通 Hugging Face Qwen3 做前向。
```

也就是说，本节测的是“摘要记忆减少模型输入长度”带来的真实 prefill 加速。

## 1. 新增与修改的脚本

新增计时脚本：

```text
ymluo/projects/learned_hierarchical_summary_memory/src/run_summary_prompt_timing.py
```

三个已有评测脚本都新增了 `route_mix.csv`：

```text
ymluo/projects/learned_hierarchical_summary_memory/src/run_learned_hier_summary_memory.py
ymluo/projects/learned_hierarchical_summary_memory/src/run_real_text_summary_memory_eval.py
ymluo/projects/learned_hierarchical_summary_memory/src/run_classic_kv_retrieval_summary_benchmark.py
```

服务器输出：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/default_20260703_route
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/real_text_warpeace_20260703_route
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/classic_kv_retrieval_20260703_route
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/summary_prompt_timing_qwen_classic_tokenpadded_20260703
```

本地已同步的计时输出：

```text
ymluo/projects/learned_hierarchical_summary_memory/outputs/summary_prompt_timing_qwen_classic_tokenpadded_20260703
```

## 2. 路由占比统计

### 2.1 Learned synthetic summary memory

文件：

```text
ymluo/projects/learned_hierarchical_summary_memory/outputs/default_20260703_route/route_mix.csv
```

| 方法 | 任务数 | summary10 | summary100 | summary1000 | full attention |
| --- | ---: | ---: | ---: | ---: | ---: |
| learned_adaptive_no_raw | 600 | 200 / 33.33% | 160 / 26.67% | 240 / 40.00% | 0 / 0.00% |
| learned_adaptive_with_raw | 600 | 200 / 33.33% | 160 / 26.67% | 160 / 26.67% | 80 / 13.33% |

解释：

```text
no_raw 把 exact_code 也压到 summary1000，所以有 40% 走 summary1000。
with_raw 遇到 exact_code 才回退 full attention，占 13.33%。
```

### 2.2 真实文本 War and Peace

文件：

```text
ymluo/projects/learned_hierarchical_summary_memory/outputs/real_text_warpeace_20260703_route/route_mix.csv
```

| 方法 | 任务数 | summary10 | summary100 | summary1000 | full attention |
| --- | ---: | ---: | ---: | ---: | ---: |
| adaptive_no_raw | 25 | 9 / 36.00% | 8 / 32.00% | 8 / 32.00% | 0 / 0.00% |
| adaptive_with_raw | 25 | 9 / 36.00% | 8 / 32.00% | 0 / 0.00% | 8 / 32.00% |

解释：

```text
exact_sentence 这类任务在 with_raw 里直接回退 raw，占 32%。
no_raw 对 exact_sentence 只能尝试 summary1000，所以准确率会掉到 68%。
```

### 2.3 Classic KV retrieval benchmark

文件：

```text
ymluo/projects/learned_hierarchical_summary_memory/outputs/classic_kv_retrieval_20260703_route/route_mix.csv
```

| 方法 | 任务数 | summary10 | summary100 | summary1000 | full attention |
| --- | ---: | ---: | ---: | ---: | ---: |
| adaptive_no_raw | 480 | 0 / 0.00% | 240 / 50.00% | 240 / 50.00% | 0 / 0.00% |
| adaptive_with_raw | 480 | 0 / 0.00% | 240 / 50.00% | 160 / 33.33% | 80 / 16.67% |

解释：

```text
passkey / needle / conflict_latest 走 summary100。
kv_lookup / multihop 走 summary1000。
exact_code 在 with_raw 里走 full attention，占 16.67%。
```

## 3. Qwen3-0.6B 真实 prefill 时间测试

### 3.1 设置

服务器：

```text
fdong@10.176.37.31
GPU: NVIDIA RTX 3090
model: /home/fdong/hrj/prove/Qwen3-0.6B
dtype: float16
attention: sdpa
```

运行设置：

```text
tasks_per_variant = 3
variants = 6
total_cases = 18
distractor_records = 256
methods = full_raw, adaptive_no_raw, adaptive_with_raw, summary1000_only
warmup = 1
repeats = 1
pad_context_to_budget = true
prompt_overhead_tokens = 64
```

这里的 `pad_context_to_budget=true` 是为了让计时更接近名义 token budget：

```text
summary10 约按 10 tokens 预算
summary100 约按 100 tokens 预算
summary1000 约按 1000 tokens 预算
raw 约按 10000 tokens 预算
```

输出：

```text
ymluo/projects/learned_hierarchical_summary_memory/outputs/summary_prompt_timing_qwen_classic_tokenpadded_20260703/timing_summary.csv
ymluo/projects/learned_hierarchical_summary_memory/outputs/summary_prompt_timing_qwen_classic_tokenpadded_20260703/timing_rows.csv
ymluo/projects/learned_hierarchical_summary_memory/outputs/summary_prompt_timing_qwen_classic_tokenpadded_20260703/route_mix.csv
```

### 3.2 结果

| 方法 | cases | 平均输入 tokens | token ratio | 平均 prefill 秒 | prefill time ratio | prefill 加速 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_raw | 18 | 10064.0 | 100.00% | 0.4786 | 100.00% | 1.00x |
| adaptive_no_raw | 18 | 894.8 | 8.89% | 0.0459 | 9.59% | 10.42x |
| adaptive_with_raw | 18 | 2394.8 | 23.80% | 0.1192 | 24.91% | 4.02x |
| summary1000_only | 18 | 1344.8 | 13.36% | 0.0503 | 10.51% | 9.51x |

如果把 prompt 构造和 tokenizer 时间也算进去：

| 方法 | 平均 pipeline 秒 | pipeline 加速 |
| --- | ---: | ---: |
| full_raw | 0.5175 | 1.00x |
| adaptive_no_raw | 0.0474 | 10.91x |
| adaptive_with_raw | 0.1236 | 4.19x |
| summary1000_only | 0.0518 | 9.99x |

### 3.3 计时实验里的路由占比

| 方法 | cases | summary10 | summary100 | summary1000 | full attention |
| --- | ---: | ---: | ---: | ---: | ---: |
| adaptive_no_raw | 18 | 0 / 0.00% | 9 / 50.00% | 9 / 50.00% | 0 / 0.00% |
| adaptive_with_raw | 18 | 0 / 0.00% | 9 / 50.00% | 6 / 33.33% | 3 / 16.67% |

## 4. 结论

这次时间测试支持以下判断：

```text
1. 如果 query 不需要 exact raw detail，直接走 summary memory 可以带来真实 wall time 加速。
2. adaptive_no_raw 的 prefill 加速约 10.42x，但会牺牲 exact_code / exact_sentence 类任务。
3. adaptive_with_raw 的准确率更高，但 16.67% 的 raw fallback 会把平均加速降到约 4.02x。
4. raw fallback 的比例虽然不高，但每次 fallback 都很贵，所以 with_raw 的总体时间主要由这些 full attention case 拉高。
```

这也说明：如果方法定义中保留 raw fallback，需要同时报告两类指标：

```text
1. route mix：多少 query 真的回退 full attention。
2. wall time：raw fallback 对平均 prefill 时间造成多大拖累。
```

## 5. 关于 gather 开销

当前 `adaptive_with_raw` 的 prompt 版实验没有做 KV-cache 内部 gather，而是直接把 raw text 拼回 prompt，所以它测到的是普通 dense attention 的真实耗时。

如果后续改成 KV-cache 级别实现，raw fallback 仍然需要访问原始 token。这里有两个工程判断：

```text
1. 不应该做任意 token 级随机 gather。
   exact_code / exact_sentence 通常需要的是某个连续 block 或少数连续 span。

2. raw fallback 应该设计成 coarse block/range fallback。
   例如：summary router 判断需要 raw 后，只打开对应 1w-token block 或更小的连续 range。
```

这样可以避免最差的随机 gather 开销，并且更容易用现有 SDPA / FlashAttention 风格 kernel 跑起来。也就是说，raw fallback 不应该成为默认路径，而应该是低频、粗粒度、连续区间的保险机制。
