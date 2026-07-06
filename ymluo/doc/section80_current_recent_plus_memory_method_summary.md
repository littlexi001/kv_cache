# Section 80: 当前 Recent-plus Hierarchical KV Memory 方法总结

## 一句话总结

当前方法可以概括为：

```text
recent raw 固定保留；
old context 不再完整 attention，而是由 router 根据任务需求选择 summary / retrieval raw / full raw；
推理时只对选中的 active KV 做 attention，从而降低 KV 长度和 attention/KV 子系统时间。
```

这和普通 RAG 的区别是：

```text
目标不是只在文本层面重组 prompt；
最终工程形态应该直接对已有 KV page 做 gather / compact / paged attention，
避免重新 prefill 原始文本。
```

当前实验里有一部分仍然用 prompt 重组做质量验证，但速度实验已经按照 KV subsystem 口径模拟了：

```text
router + page scoring + top-k + KV gather/compact + compact KV attention
```

## 方法结构

### 1. Context 切分

对长上下文切成两部分：

```text
old context:    前面的历史内容
recent context: 最近的若干 token
```

当前默认：

```text
page_size = 1024
recent_tokens = 512
```

recent raw 必选，原因是：

```text
1. 生成时最近上下文对局部连贯性最重要；
2. 很多任务答案可能就在最近内容；
3. 如果允许 router 选择 recent_only，容易学到过于激进的策略，导致真实任务漏掉历史证据。
```

### 2. Old context 的候选策略

router 只决定 old context 用什么记忆形式。

当前主要 action：

```text
recent_plus_summary1_8
recent_plus_summary1_4
recent_plus_summary1_2
recent_plus_static_hier
recent_plus_retrieval_raw_k1
recent_plus_retrieval_raw_k2
recent_plus_retrieval_raw_k3
recent_plus_retrieval_raw_k4
recent_plus_retrieval_raw_k8
full_raw
```

其中 retrieval raw 的含义是：

```text
从 old context 里按 page/block 打分；
选 top-k old raw page；
再拼上 recent raw；
后续 attention 到 compact KV。
```

例如：

```text
recent_plus_retrieval_raw_k2:
  active old KV = 2 * 1024
  recent KV = 512
  total active KV = 2560
```

### 3. Router

router 的目标不是固定 k，而是根据 query 和 context 状态自动选择策略。

输入特征包括：

```text
任务类型特征：exact / generation
query 长度、关键词、数字、all/list/count/compare 等模式
prefix token 长度、old token 长度、block 数
retriever top1/top2/top3 overlap
retriever score gap
positive block 数量
top block 位置
文本数字/大写词/unique word ratio 等统计
```

当前 best router 是小 MLP：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/recent_plus_router_best_20260706/router.pt
```

训练数据没有使用 LongBench/RULER benchmark label，而是用非 benchmark 文本构造 synthetic：

```text
War and Peace
The Count of Monte Cristo
```

synthetic 任务包括：

```text
single evidence
multi evidence
natural QA style retrieval
count/frequency style task
summary task
recent generation
full-context task
```

## 当前代码入口

### Benchmark 和方法实现

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_qwen8b_paper_benchmarks.py
```

包含：

```text
recent_plus_* action 的 memory 构造
router features
safety override
LongBench/RULER benchmark runner
```

### Router 训练

benchmark trial 蒸馏：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_qwen8b_router_distill_from_trials.py
```

非 benchmark fast synthetic router：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_fast_recent_plus_router_training.py
```

非 benchmark synthetic 数据构造：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_synthetic_router_distillation.py
```

### Offline policy evaluation

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_router_policy_offline_eval.py
```

### Attention/KV 子系统速度

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_recent_plus_attention_subsystem_timing.py
```

分析脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/scripts/analyze_recent_plus_attention_timing.py
```

## 质量结果

### Qwen3-8B recent-plus benchmark

测试目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/qwen8b_recent_plus_bench_m4_parallel_20260706/merged
```

规模：

```text
LongBench: 32 cases
RULER 4k: 32 cases
RULER 8k: 32 cases
RULER 16k: 32 cases
total: 128 cases
```

固定策略结果：

| method | relative | token ratio |
|---|---:|---:|
| recent_plus_retrieval_raw_k2 | 100.97% | 32.77% |
| recent_plus_retrieval_raw_k3 | 103.90% | 38.50% |
| recent_plus_retrieval_raw_k4 | 101.93% | 43.46% |

解释：

```text
固定 recent 后，old retrieval k2/k3/k4 已经是很强的 baseline。
```

### Oracle 上界

在同一批 benchmark 上：

```text
oracle_match_full:
  relative = 105.96% full_raw
  token ratio = 23.90%
```

这个结果说明：

```text
方法上界很好；
理论上 20%-30% active KV 是有希望接近或超过 full_raw 的。
```

### 当前 best learned router

best router：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/recent_plus_router_best_20260706/router.pt
```

heldout benchmark 结果：

| group | relative | token ratio |
|---|---:|---:|
| overall | 100.95% | 41.09% |
| exact | 101.00% | 42.00% |
| generation | 97.11% | 27.49% |
| LongBench | 88.73% | 26.93% |
| RULER 4096 | 100.00% | 73.32% |
| RULER 8192 | 100.00% | 42.45% |
| RULER 16384 | 107.14% | 21.65% |

当前结论：

```text
router 已经可用，overall 超过 full_raw；
但 token ratio 还没接近 oracle；
最大短板是 LongBench。
```

## 速度结果

最新 attention/KV subsystem 多长度测试：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/recent_plus_attention_subsystem_qwen8b_multilen_warm_20260706
```

测试的是 Qwen3-8B attention 形状：

```text
layers = 36
query heads = 32
KV heads = 8
head_dim = 128
dtype = fp16
```

计入：

```text
router
page scoring
top-k
KV gather / compact
后续多步 attention 到 compact KV
```

不计入：

```text
tokenizer
MLP
lm_head
HF generate
采样
```

### 1024 步 decode

20k 长度：

| method | active KV | speedup |
|---|---:|---:|
| full_attention | 20000 | 1.00x |
| recent_plus_k2 | 2560 | 6.95x |
| recent_plus_k3 | 3584 | 5.43x |
| recent_plus_k4 | 4608 | 4.47x |

32k 长度：

| method | active KV | speedup |
|---|---:|---:|
| full_attention | 32768 | 1.00x |
| recent_plus_k2 | 2560 | 11.21x |
| recent_plus_k3 | 3584 | 8.75x |
| recent_plus_k4 | 4608 | 7.16x |

### 新方法额外开销

page_once、1024 步下平均：

| method | router + scoring + top-k | gather + compact | total overhead |
|---|---:|---:|---:|
| recent_plus_k2 | 0.315 ms | 2.134 ms | 2.448 ms |
| recent_plus_k3 | 0.294 ms | 3.070 ms | 3.365 ms |
| recent_plus_k4 | 0.289 ms | 3.897 ms | 4.187 ms |

结论：

```text
新增开销不是瓶颈。
摊到 1024 步后，overhead share 约 0.04%。
真正决定速度的是 active KV 长度。
```

## 和 RAG 的边界

当前 prompt 版质量实验看起来像 RAG，因为它把 old memory/retrieved block 重新组织成文本再 prefill。

但论文方法应该强调最终系统形态是：

```text
历史阶段已有 KV cache；
query 到来时 router 选策略；
对 KV page 做 gather / compact / paged attention；
不重新读取完整原始文本；
不重新 prefill full raw prompt。
```

因此和 RAG 的关键区别是：

```text
RAG: 文本检索 + prompt 拼接 + 重新 prefill。
本方法: KV memory routing + KV page selection/compact + warm-cache decode。
```

## 当前短板

### 1. LongBench 还不够强

当前 best router：

```text
LongBench relative = 88.73%
```

原因：

```text
LongBench 更像真实自然问答/摘要/多文档任务；
答案不一定是显式 key-value；
证据可能跨多个 block；
当前 synthetic 仍然更接近 RULER/needle retrieval。
```

### 2. learned router 距离 oracle 还有差距

```text
oracle: 105.96% full_raw, 23.90% token
learned router: 100.95% full_raw, 41.09% token
```

说明主要瓶颈是：

```text
router 还不够接近 oracle；
不是方法上界不够。
```

### 3. 端到端 serving 还没完全工程化

attention/KV subsystem 速度很好，但完整端到端还需要：

```text
真实 KV page 管理
paged attention kernel
serving runtime 集成
避免 Python/HF generate 调度稀释收益
```

## 下一步优先级

### P0: 提升 router 泛化

继续用非 benchmark 数据增强 synthetic：

```text
自然多跳问答
多证据且答案数量不确定
跨段实体关系
全文统计 / 全文摘要
query 模糊或 retriever gap 小的 hard case
```

目标：

```text
learned router:
  95%-100%+ full_raw
  25%-35% active KV
```

### P1: LongBench 分项补测

应该分开看：

```text
exact QA
multi-doc QA
summary generation
passage retrieval/counting
```

不要只看总分。

### P2: KV-native demo

做一个小规模真实 KV gather demo：

```text
先 prefill full raw 得到 KV；
query 到来后 gather selected old KV page + recent KV；
后续 decode 使用 compact KV；
和 prompt 重组版质量做 smoke 对齐。
```

这个 demo 能更清楚地区分本方法和 RAG。

## 目前可以写进论文的核心 claim

比较稳的说法：

```text
1. Recent raw should be preserved as a mandatory local memory.
2. Old context can be routed among summary and raw evidence pages.
3. An oracle policy reaches full_raw-level quality with about 24% active KV.
4. A learned non-benchmark router currently reaches about 101% full_raw with 41% active KV.
5. Attention/KV subsystem speed scales with active KV length; at 20k context, k2/k3/k4 give about 6.95x/5.43x/4.47x speedup including router/scoring/top-k/gather overhead.
```

不应该过早声称：

```text
端到端 serving 已经有同等倍数加速；
LongBench 已全面超过 full_raw；
learned router 已达到 oracle。
```

这些还需要后续实验补齐。
