# Controlled Public KV Benchmark V1（2026-07-03）

## 目标

这组实验用于回答一个更严格的问题：

```text
在相同模型、相同样本、相同 prompt、相同生成长度、相同 KV token budget、相同计时口径下，
我们的 page-level KV gather / memory planning 方法，是否能超过最近两年论文里的 KV cache 方法。
```

当前不能只和 full baseline 或 synthetic workload 比。需要补公开 benchmark，优先：

- LongBench 原版：覆盖 multi-doc QA、single-doc QA、retrieval、counting、summarization、code 等长上下文任务。
- RULER：覆盖 NIAH、多 key、多 value、多 query、变量追踪、聚合等 synthetic long-context stress test。

## 已实现入口

新增 runner：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_controlled_public_kv_benchmark_v1.py
```

新增服务器脚本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_controlled_public_kv_benchmark_v1_server.sh
```

服务器 smoke 已跑通：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/controlled_public_kv_benchmark_v1_smoke_20260703
```

## 当前支持的数据

### LongBench 原版

数据来源：

```text
THUDM/LongBench data.zip
```

已缓存到服务器：

```text
/home/fdong/ymluo/hf_cache/datasets--THUDM--LongBench/snapshots/.../data.zip
```

当前 runner 先支持这些 task：

```text
passage_retrieval_en
hotpotqa
2wikimqa
musique
qasper
multifieldqa_en
triviaqa
passage_count
narrativeqa
```

### RULER

数据来源：

```text
/home/fdong/lm-evaluation-harness/lm_eval/tasks/ruler
```

当前 runner 先支持：

```text
niah_single_1
niah_single_2
niah_multikey_1
```

依赖已补：

```text
wonderwords
nltk
```

## 当前支持的方法

| 方法名 | 含义 | 对应论文关系 |
| --- | --- | --- |
| `full_kv` | 不压缩 KV，完整前向 + greedy decode | full baseline |
| `streamingllm_sink_recent` | 保留 sink + recent token | StreamingLLM-like |
| `h2o_observe` | 用 query suffix attention 累积重要性，选择 top KV token | H2O-like proxy |
| `snapkv_observe` | 用 observation window attention，加 pooling 后选择 top KV token | SnapKV-like proxy |
| `ours_page_gather` | 自然分页，按 query/entity/结构分数选 page，再 gather 真实 KV | 当前方法的 public benchmark adapter |

注意：`h2o_observe` / `snapkv_observe` 是同 runner 下的 controlled proxy，不等同论文官方 kernel。最终论文级结论还需要接入作者实现或 KVCache-Factory。

## 控制变量

固定项：

```text
model = /home/fdong/hrj/prove/Qwen3-0.6B
dtype = float16
attn_implementation = eager
decode = greedy
temperature = 0
same sampled IDs
same prompt template
same max_context_tokens
same max_new_tokens
same sink/recent protection
same context-token budget
same metric function
same timing split
```

计时字段：

```text
prefill_seconds:
  完整 context prefix prefill 时间。

kv_gather_seconds:
  从 full prefix cache gather selected KV 的时间。

query_seconds:
  question / suffix 继续前向时间。

decode_seconds:
  greedy decode 时间。

online_seconds:
  kv_gather_seconds + query_seconds + decode_seconds。

total_seconds:
  prefill_seconds + online_seconds。
```

对于 KV compression 方法，真正该比较的速度主要是 `online_seconds` 和 decode 侧开销；如果把完整 prefill 也算进去，当前 V5 adapter 不是端到端压缩 prefill，而是 full prefill 后再 gather。

## Smoke 结果

命令：

```bash
python src/run_controlled_public_kv_benchmark_v1.py \
  --model_name_or_path /home/fdong/hrj/prove/Qwen3-0.6B \
  --output_dir outputs/controlled_public_kv_benchmark_v1_smoke_20260703 \
  --benchmarks longbench,ruler \
  --longbench_tasks passage_retrieval_en,hotpotqa \
  --ruler_tasks niah_single_1 \
  --max_samples_per_task 1 \
  --max_context_tokens 4096 \
  --max_new_tokens_override 8 \
  --methods full_kv,streamingllm_sink_recent,h2o_observe,snapkv_observe,ours_page_gather \
  --budget_tokens 512 \
  --sink_tokens 64 \
  --recent_tokens 256 \
  --page_tokens 256 \
  --ruler_lengths 4096
```

聚合结果：

| method | score | mean online sec | mean kept prefix tokens | mean keep frac |
| --- | ---: | ---: | ---: | ---: |
| full_kv | 0.250 | 0.318 | 3965.3 | 1.000 |
| streamingllm_sink_recent | 0.250 | 0.309 | 337.0 | 0.085 |
| h2o_observe | 0.250 | 0.308 | 529.0 | 0.134 |
| snapkv_observe | 0.250 | 0.307 | 529.0 | 0.134 |
| ours_page_gather | 0.250 | 0.310 | 529.0 | 0.134 |

解释：

1. 这个 smoke 只证明 pipeline 可以跑通，不能作为质量结论。
2. RULER 用 `max_new_tokens=8` 时，Qwen3-0.6B 的 full KV 也没答出完整 needle，所以该项不适合看压缩优劣。
3. LongBench HotpotQA 单样本上所有方法都达到相同 F1；passage retrieval 单样本 full 本身也错，因此不能比较方法优劣。
4. sparse 方法在 `online_seconds` 上略低于 full，但因为这里是 full prefill 后 gather，不能声称端到端快于 full baseline。

## 下一步正式小表

服务器网络恢复后直接运行：

```bash
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
bash scripts/run_controlled_public_kv_benchmark_v1_server.sh
```

默认配置：

```text
LongBench:
  passage_retrieval_en, hotpotqa, 2wikimqa, multifieldqa_en

RULER:
  niah_single_1 at 4096

samples per task:
  3

context cap:
  8192

generation cap:
  32

budget:
  512 context tokens
```

## 正式小表结果

SSH 恢复后已完成该小表：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/controlled_public_kv_benchmark_v1_lb4_ruler_3shot_b512_20260703
```

配置：

```text
LongBench:
  passage_retrieval_en, hotpotqa, 2wikimqa, multifieldqa_en

RULER:
  niah_single_1 at 4096

samples:
  3 per task, 15 total

max_context_tokens:
  8192

max_new_tokens:
  32

budget:
  512 context tokens
```

### Overall

| method | score | online sec | kept prefix tokens | keep frac |
| --- | ---: | ---: | ---: | ---: |
| full_kv | 0.0816 | 1.0920 | 6635.6 | 1.000 |
| streamingllm_sink_recent | 0.0601 | 1.0849 | 337.4 | 0.0626 |
| h2o_observe | 0.1763 | 1.0802 | 529.4 | 0.0986 |
| snapkv_observe | 0.1133 | 1.0799 | 529.4 | 0.0986 |
| ours_page_gather | 0.1702 | 1.0804 | 529.4 | 0.0986 |

整体看，`ours_page_gather` 明显高于 full / StreamingLLM / SnapKV proxy，但略低于 H2O proxy。这个 overall 会被 RULER needle 强烈影响，不能单独作为“赢过 paper baseline”的证据。

### LongBench Only

| method | score | online sec | kept prefix tokens | keep frac |
| --- | ---: | ---: | ---: | ---: |
| full_kv | 0.1020 | 1.0935 | 7382.1 | 1.000 |
| streamingllm_sink_recent | 0.0751 | 1.0852 | 341.8 | 0.0563 |
| h2o_observe | 0.0537 | 1.0803 | 533.8 | 0.0882 |
| snapkv_observe | 0.0583 | 1.0798 | 533.8 | 0.0882 |
| ours_page_gather | 0.0461 | 1.0806 | 533.8 | 0.0882 |

LongBench 子集上，当前 `ours_page_gather` 没有赢。它低于 full、StreamingLLM、H2O proxy、SnapKV proxy。主要失败来自：

```text
passage_retrieval_en:
  all methods score = 0

2wikimqa:
  all methods score = 0

hotpotqa:
  ours_page_gather = 0.0351
  h2o_observe = 0.0952
  snapkv_observe = 0.0909
  streamingllm = 0.1000
  full = 0.1861

multifieldqa_en:
  ours_page_gather = 0.1494
  h2o_observe = 0.1194
  snapkv_observe = 0.1424
  streamingllm = 0.2005
  full = 0.2220
```

解释：当前 page scorer 还是轻量 lexical/entity/structure 打分。它在 RULER needle 这类显式 key-value 检索上有效，但在 LongBench 的 multi-hop QA、abstract-to-paragraph retrieval、开放式 QA 中，page 级相关性判断不够准，容易选到“看起来相关但不是证据”的页。

### RULER NIAH

| method | score | online sec | kept prefix tokens | keep frac |
| --- | ---: | ---: | ---: | ---: |
| full_kv | 0.0000 | 1.0858 | 3649.7 | 1.000 |
| streamingllm_sink_recent | 0.0000 | 1.0837 | 320.0 | 0.0877 |
| h2o_observe | 0.6667 | 1.0797 | 512.0 | 0.1403 |
| snapkv_observe | 0.3333 | 1.0800 | 512.0 | 0.1403 |
| ours_page_gather | 0.6667 | 1.0796 | 512.0 | 0.1403 |

RULER needle 上，`ours_page_gather` 和 H2O proxy 持平，超过 SnapKV proxy、StreamingLLM 和 full。这里 full 失败不是因为 KV 不足，而是 Qwen3-0.6B 在完整重复 haystack 下容易继续复读上下文；稀疏选择反而减少干扰，让模型更容易输出目标数字。

## V2 扩展：任务覆盖、budget sweep 与语义 page scorer

本轮已经把 runner 从 V1 的小 smoke 扩展成更接近论文比较需要的 controlled public benchmark。

新增 LongBench 覆盖：

```text
single-doc QA:
  narrativeqa, qasper, multifieldqa_en

multi-doc QA:
  hotpotqa, 2wikimqa, musique

retrieval/counting:
  passage_retrieval_en, passage_count

summarization:
  gov_report, multi_news
```

新增 RULER 覆盖：

```text
niah_single_1
niah_multikey_1
niah_multivalue
niah_multiquery
vt
cwe
fwe
qa_squad
qa_hotpot
```

其中 `qa_hotpot` 在本次服务器运行时由于 CMU HotpotQA 下载连接被 reset，被 runner 自动跳过；`qa_squad` 可正常缓存和运行。

新增脚本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_controlled_public_kv_benchmark_v2_sweep_server.sh
```

默认 sweep：

```text
budget_tokens:
  256 / 512 / 1024 / 2048

methods:
  full_kv
  streamingllm_sink_recent
  h2o_observe
  snapkv_observe
  ours_page_gather
```

### ours_page_gather 的 scorer 升级

V1 的 `ours_page_gather` 主要依赖 lexical/entity/structural heuristic。V2 做了两个关键修正：

1. 去掉答案泄漏：旧版在部分 retrieval/counting 任务里把 `answers` 拼进 page scorer query，这会让 page routing 不公平。新版只使用真实 query 或 suffix。
2. 增加语义 scorer：使用当前语言模型自己的 input embedding table，对 query/page 做轻量 embedding 打分，不额外下载 embedding 模型。

当前实现支持：

```text
lexical:
  词面/entity/结构符号打分。

semantic:
  LM input embedding mean pooling cosine。

hybrid:
  semantic + lexical + entity + structural + global coverage。

hybrid_mmr:
  hybrid 后用 MMR 抑制重复 page。

late_interaction:
  query token 到 page token 的 MaxSim 轻量 reranker。

hybrid_late_mmr:
  用 late-interaction 作为主语义分数，再叠加 lexical/entity/structural/coverage，并用 MMR 做 page 去冗余。
```

`hybrid_late_mmr` 是当前推荐默认值。它比 mean embedding 更像一个小 reranker：不是把整页压成一个平均向量，而是让 query 里的每个关键词/实体 token 在 page 内找最匹配 token，然后聚合 MaxSim 分数。

另一个修正是 budget 256 的公平性：当 `sink64 + recent256` 本身超过 256 时，旧逻辑会按 token 位置截断，偏向 sink、丢掉 recent；新版会同时保留上下文开头和结尾，避免 256 档变成不合理的 sink-only。

### V2 Qwen3-0.6B proxy sweep 结果

命令：

```bash
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
env SAMPLES=1 BUDGETS='256 512 1024 2048' MAX_CONTEXT=8192 MAX_NEW=64 STAMP=20260703_v2_expanded \
  bash scripts/run_controlled_public_kv_benchmark_v2_sweep_server.sh
```

输出目录：

```text
outputs/controlled_public_kv_benchmark_v2_expanded_1shot_b256_20260703_v2_expanded
outputs/controlled_public_kv_benchmark_v2_expanded_1shot_b512_20260703_v2_expanded
outputs/controlled_public_kv_benchmark_v2_expanded_1shot_b1024_20260703_v2_expanded
outputs/controlled_public_kv_benchmark_v2_expanded_1shot_b2048_20260703_v2_expanded
```

整体聚合结果：

| budget | method | score | online sec | kept prefix | keep frac |
| ---: | --- | ---: | ---: | ---: | ---: |
| 256 | full_kv | 0.4334 | 1.7425 | 5168.9 | 1.000 |
| 256 | streamingllm_sink_recent | 0.1710 | 1.7361 | 274.4 | 0.066 |
| 256 | h2o_observe | 0.1710 | 1.7321 | 274.4 | 0.066 |
| 256 | snapkv_observe | 0.1710 | 1.7321 | 274.4 | 0.066 |
| 256 | ours_page_gather | 0.1710 | 1.7319 | 274.4 | 0.066 |
| 512 | full_kv | 0.4418 | 1.7117 | 5165.3 | 1.000 |
| 512 | streamingllm_sink_recent | 0.1653 | 1.7029 | 338.4 | 0.082 |
| 512 | h2o_observe | 0.2582 | 1.6987 | 530.4 | 0.129 |
| 512 | snapkv_observe | 0.2256 | 1.6984 | 530.4 | 0.129 |
| 512 | ours_page_gather | 0.2953 | 1.6988 | 530.4 | 0.129 |
| 1024 | full_kv | 0.4362 | 1.7534 | 5205.4 | 1.000 |
| 1024 | streamingllm_sink_recent | 0.1699 | 1.7479 | 338.4 | 0.065 |
| 1024 | h2o_observe | 0.3233 | 1.7454 | 1042.4 | 0.251 |
| 1024 | snapkv_observe | 0.3840 | 1.7450 | 1042.4 | 0.251 |
| 1024 | ours_page_gather | 0.4006 | 1.7453 | 1042.4 | 0.251 |
| 2048 | full_kv | 0.4057 | 1.7508 | 5182.1 | 1.000 |
| 2048 | streamingllm_sink_recent | 0.2023 | 1.7432 | 338.4 | 0.065 |
| 2048 | h2o_observe | 0.4118 | 1.7503 | 2045.4 | 0.490 |
| 2048 | snapkv_observe | 0.4122 | 1.7521 | 2045.4 | 0.490 |
| 2048 | ours_page_gather | 0.4008 | 1.7504 | 2045.4 | 0.490 |

分 benchmark 看：

```text
budget 512:
  LongBench:
    full 0.0685
    streaming 0.0642
    h2o 0.0615
    snap 0.0527
    ours 0.0282

  RULER:
    full 0.9083
    streaming 0.2917
    h2o 0.5042
    snap 0.4417
    ours 0.6292

budget 1024:
  LongBench:
    full 0.0685
    streaming 0.0642
    h2o 0.0502
    snap 0.0646
    ours 0.0745

  RULER:
    full 0.8958
    streaming 0.3021
    h2o 0.6646
    snap 0.7833
    ours 0.8083

budget 2048:
  LongBench:
    full 0.0685
    streaming 0.0642
    h2o 0.0696
    snap 0.0703
    ours 0.0697

  RULER:
    full 0.8271
    streaming 0.3750
    h2o 0.8396
    snap 0.8396
    ours 0.8146
```

解释：

1. 这组仍然是 Qwen3-0.6B + proxy baselines，不是论文级最终对比。
2. `ours_page_gather` 在 512/1024 budget 的 RULER 上明显强于 H2O/SnapKV proxy，说明 typed/natural page routing 对显式长程检索有价值。
3. 在 LongBench 上，1-shot/task 太小，且 Qwen3-0.6B 本身分数很低，不能据此得出“通用长上下文已经赢”的结论。
4. 2048 budget 时 H2O/SnapKV proxy 和 ours 接近，说明当保留接近一半 prefix token 时，page routing 的优势会变小。
5. `online_seconds` 的差异很小，因为当前 adapter 仍是 full prefill 后 gather；它体现 decode-side KV 缩短后的收益，不代表端到端已经快于 full。

## 官方 KVCache-Factory 接入状态

已经在服务器准备官方实现入口：

```text
external repo:
  /home/fdong/ymluo/external/KVCache-Factory

isolated transformers 4.44 dependency target:
  /home/fdong/ymluo/pydeps/kvcache_factory_tf444

official model:
  /home/fdong/qwen/LlaMa-3.1-8B
```

复现/重建脚本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/prepare_kvcache_factory_server.sh
```

已补 transformers 4.53 与 KVCache-Factory 之间的兼容问题：

```text
StaticCache import fallback:
  transformers.generation.utils -> transformers.cache_utils

is_quanto_available fallback:
  如果 transformers.utils 中不存在该符号，则置为 False
```

官方 SnapKV LongBench smoke 已在 Llama3.1-8B 上跑通：

```bash
cd /home/fdong/ymluo/external/KVCache-Factory
source /home/fdong/miniconda3/bin/activate moe
PYTHONPATH=/home/fdong/ymluo/pydeps/kvcache_factory_tf444:$PWD:$PYTHONPATH CUDA_VISIBLE_DEVICES=0 \
  python run_longbench.py \
    --method SnapKV \
    --model_path /home/fdong/qwen/LlaMa-3.1-8B \
    --max_capacity_prompts 512 \
    --attn_implementation sdpa \
    --save_dir /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/kvcache_factory_official_smoke_20260703 \
    --use_cache True \
    --datasets hotpotqa \
    --max_num_examples 1 \
    --sample_method topk \
    --dtype float16
```

新增官方 LongBench sweep 脚本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_kvcache_factory_official_longbench_sweep_server.sh
```

默认方法：

```text
FullKV
StreamingLLM
H2O
SnapKV
PyramidKV
AdaKV
```

Quest 需要单独处理：KVCache-Factory README 的 supported methods 中提到 Quest，但当前 LongBench common arguments 和 `replace_llama` runtime patch 分支里没有 Quest。仓库里存在 `pyramidkv/quest.py` 和测试文件，但它不像 PyramidKV/SnapKV/H2O/AdaKV 那样已经接到 LongBench attention hot path。为了避免把 no-op 误标成 Quest，默认官方 sweep 暂不包含 Quest；后续需要接入 Quest 原作者实现或补齐 KVCache-Factory 的 Quest runtime patch 后再加入正式表。

新增官方 RULER sweep 脚本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_kvcache_factory_official_ruler_sweep_server.sh
```

KVCache-Factory 的 RULER runner 当前主要覆盖：

```text
FullKV
StreamingLLM
H2O
SnapKV
PyramidKV
```

官方脚本现在会记录每个 method/budget 的 `run_status.csv`。这样 Quest/AdaKV 如果因为官方 runner 或 attention backend 不兼容而失败，不会阻断其它方法。

为了让我们的 adapter 与官方方法使用同一批 LongBench sampled IDs，新增 Llama 版 ours 脚本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_ours_adapter_longbench_llama_sweep_server.sh
```

它使用同样的 LongBench task 列表、同样的 `SAMPLES`、同样的 first-N topk sampled IDs，并额外写出：

```text
sampled_ids.csv
sampled_ids.json
```

需要注意：严格论文级公平比较不仅要 same IDs，还要 same prompt template、same max context truncation、same generation cap。KVCache-Factory 的官方 LongBench runner 通常使用自己的 prompt formatting；后续需要核对并尽量对齐 prompt。

## 当前推荐下一步

服务器 SSH 恢复后，先跑最小官方矩阵，不直接启动完整 7 methods × 4 budgets × 10 tasks：

```bash
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
env SAMPLES=1 BUDGETS='512' METHODS='FullKV StreamingLLM H2O SnapKV PyramidKV AdaKV' STAMP=20260703_official_b512 \
  bash scripts/run_kvcache_factory_official_longbench_sweep_server.sh
```

同时跑 ours Llama adapter：

```bash
cd /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
env SAMPLES=1 BUDGETS='512' STAMP=20260703_ours_llama_b512 \
  bash scripts/run_ours_adapter_longbench_llama_sweep_server.sh
```

如果这两组都稳定，再启动完整预算：

```bash
env SAMPLES=1 BUDGETS='256 512 1024 2048' STAMP=20260703_official_full \
  bash scripts/run_kvcache_factory_official_longbench_sweep_server.sh

env SAMPLES=1 BUDGETS='256 512 1024 2048' STAMP=20260703_ours_llama_full \
  bash scripts/run_ours_adapter_longbench_llama_sweep_server.sh
```

速度报告必须分两个口径：

```text
online decode-side latency:
  当前 ours adapter 可以公平报告，表示 full prefill 后 gather KV，再跑 query/decode 的收益。

end-to-end latency:
  当前不能声称赢 full baseline，因为 prefill 仍然完整执行。
  只有后续实现 prefill-time compression 或 fused sparse prefill/range prefill 后，才有资格挑战 full baseline 的端到端时间。
```

## 当前结论

这组正式小表支持一个更谨慎的判断：

```text
当前方法在显式 long-range key-value / needle retrieval 上有希望；
但还没有在 LongBench 通用长上下文 QA/retrieval 上超过已有 KV 方法。
```

因此现在不能 claim：

```text
ours beats recent KV-cache papers on public benchmarks.
```

可以 claim 的是：

```text
我们已经建立了同模型、同数据、同 budget、同计时口径的 public benchmark runner；
初步结果显示 page-level memory planning 对 RULER needle 有收益；
但 LongBench 暴露出 page scorer / planner 还不够强，需要引入更可靠的 semantic evidence scorer 或 causal influence label。
```

下一步优先级应该是：

1. 接入官方 PyramidKV / Ada-KV / Quest / SnapKV 实现，替换 proxy baseline。
2. 把 `ours_page_gather` 的 page scorer 从 lexical/entity heuristic 升级成 semantic embedding 或小模型 reranker。
3. 增加 budget sweep，看 ours 是在低 budget 失败，还是 scorer 本身失败。
4. 增加更强模型，否则 Qwen3-0.6B 的 full baseline 太弱，会干扰 benchmark 解读。

## 论文级对比还缺什么

当前 V1 还只是 controlled runner，不是最终 paper-grade 对比。

必须继续补：

1. 接入 PyramidKV / Ada-KV / Quest 的官方实现或 KVCache-Factory。
2. 用同一批 sampled IDs 跑官方方法和我们的 adapter。
3. 增加 budget sweep：

```text
256 / 512 / 1024 / 2048 context tokens
```

4. 增加 LongBench task 覆盖：

```text
single-doc QA:
  narrativeqa, qasper, multifieldqa_en

multi-doc QA:
  hotpotqa, 2wikimqa, musique

retrieval/counting:
  passage_retrieval_en, passage_count

summarization:
  gov_report, multi_news
```

5. 增加 RULER task 覆盖：

```text
niah_single_1
niah_multikey_1
niah_multivalue
niah_multiquery
vt
cwe
fwe
qa_squad
qa_hotpot
```

6. 报告两个速度口径：

```text
online decode-side latency:
  对应 V5 gather 后 query/decode 的真实收益。

end-to-end latency:
  包含 prefill；只有实现 prefill-time compression / fused sparse prefill 后才能挑战 full baseline。
```

## 当前判断

从 smoke 不能得出“赢过论文方法”。它只说明现在已经有了一个可控实验框架。

更严谨的判断标准应该是：

```text
如果 ours_page_gather 在相同 budget 下：
  LongBench/RULER score >= SnapKV/H2O/PyramidKV/Ada-KV/Quest，
并且 online latency 或 memory 明显更低，
才可以说方法在某类任务上超过已有 paper baseline。

如果 score 只在 retrieval 类任务好，在 summarization / general QA 上弱，
那么 claim 应改成 task-aware memory planning，而不是通用 KV compression 全面领先。
```

## V3 官方 KVCache-Factory 对比结果（Llama3.1-8B，2026-07-03）

本轮完成了 KVCache-Factory 官方 LongBench sweep：

```text
model:
  /home/fdong/qwen/LlaMa-3.1-8B

tasks:
  narrativeqa, qasper, multifieldqa_en,
  hotpotqa, 2wikimqa, musique,
  passage_retrieval_en, passage_count,
  gov_report, multi_news

samples:
  1 per task, topk first-N sampled IDs

budgets:
  256 / 512 / 1024 / 2048
```

### 官方方法可运行性

先跑了 B512 smoke：

```text
outputs/kvcache_factory_official_longbench_1shot_20260703_b512_smoke_official
```

结果：

| method | status | 说明 |
| --- | --- | --- |
| FullKV | OK | 官方 full baseline 跑通 |
| StreamingLLM | OK | 跑通 |
| SnapKV | OK | 跑通 |
| PyramidKV | OK | 跑通 |
| H2O | FAILED | Llama3.1-8B + 7.5k context 在 3090 24GB 上 OOM，attention 权重分配约 6.7GB |
| AdaKV | FAILED | 当前环境未安装 `flash_attn`，官方 AdaKV forward 调用了 flash-attn path |
| Quest | NOT RUN | KVCache-Factory 中有 `quest.py`，但当前 LongBench `replace_llama` runtime patch 没有 Quest 分支，暂不作为已接入官方方法 |

因此正式 4-budget sweep 只跑当前可运行的官方方法：

```text
FullKV / StreamingLLM / SnapKV / PyramidKV
```

输出：

```text
outputs/kvcache_factory_official_longbench_1shot_20260703_working_fullsweep_official
```

### 官方 LongBench score

用 `src/summarize_kvcache_factory_longbench.py` 重新聚合官方 prediction JSON，得到：

| budget | FullKV | StreamingLLM | SnapKV | PyramidKV |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 0.0848 | 0.0552 | 0.0626 | 0.0897 |
| 512 | 0.0848 | 0.0588 | 0.0623 | 0.0667 |
| 1024 | 0.0848 | 0.0472 | 0.0811 | 0.0727 |
| 2048 | 0.0848 | 0.0623 | 0.0844 | 0.0847 |

这张表的解读：

1. 1-shot/task 很小，不能作为最终论文数字。
2. 当前 sweep 中，PyramidKV 在 b256 略高于 FullKV，SnapKV 在 b2048 接近 FullKV。
3. StreamingLLM 整体偏弱，符合它只保留 sink/recent、缺少语义检索的预期。
4. H2O/AdaKV 不是质量失败，而是当前硬件/依赖没有跑通。

### ours adapter：raw prompt 与 aligned prompt

第一次 ours Llama adapter 使用 raw prompt，虽然 sampled IDs 相同，但没有套 KVCache-Factory 的 Llama3 chat wrapper。因此它只能看 ours 自身趋势，不能严格和官方方法横向比：

```text
outputs/controlled_public_kv_benchmark_ours_llama_longbench_1shot_b*_20260703_working_fullsweep_ours_llama
```

| budget | full score | ours score | full online sec | ours online sec | kept frac |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.0981 | 0.0715 | 1.5185 | 1.4692 | 0.0618 |
| 512 | 0.0981 | 0.0727 | 1.5158 | 1.4302 | 0.1176 |
| 1024 | 0.0981 | 0.1844 | 1.5171 | 1.4393 | 0.2290 |
| 2048 | 0.0981 | 0.0864 | 1.5222 | 1.4654 | 0.4138 |

raw prompt 下 b1024 的 score 很高，主要来自 passage_count、2wikimqa、multi_news 这类样本被 page gather 纠偏。但这不是严格官方对比。

随后补了 `--prompt_wrapper llama3`，让 ours adapter 更接近 KVCache-Factory 的 Llama3 prompt wrapper：

```text
outputs/controlled_public_kv_benchmark_ours_llama_longbench_1shot_b*_20260703_ours_llama_aligned_ours_llama
```

| budget | full score | ours score | full online sec | ours online sec | kept frac |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.0551 | 0.0470 | 1.5950 | 1.2884 | 0.0628 |
| 512 | 0.0551 | 0.0600 | 1.6048 | 1.4578 | 0.1184 |
| 1024 | 0.0551 | 0.0360 | 1.6032 | 1.2658 | 0.2297 |
| 2048 | 0.0551 | 0.0230 | 1.6020 | 1.2081 | 0.4142 |

aligned prompt 下，ours 最好是 b512，略高于自己的 full baseline，但仍低于 KVCache-Factory 官方 FullKV/PyramidKV/SnapKV 的正式表。它的优势主要是 online decode-side latency 和 kept token fraction，而不是 LongBench 质量。

还要注意：aligned ours 的 full baseline `0.0551` 和 KVCache-Factory 官方 FullKV `0.0848` 仍不完全一致，说明除了 chat wrapper，还有 stop token、generation call、prompt truncation 或 tokenizer handling 的细节差异。严格 paper-grade 对比仍需要把 ours adapter 直接接进 KVCache-Factory runner，或者把两边 generation path 完全统一。

### 当前结论

截至这轮实验，不能 claim：

```text
ours 在官方 LongBench 上超过最近两年 KV-cache paper baseline。
```

可以 claim：

```text
1. 官方 KVCache-Factory 接入已经跑通 FullKV / StreamingLLM / SnapKV / PyramidKV。
2. 当前 3090 环境下，H2O 因 OOM，AdaKV 因缺 flash-attn，Quest 因 runtime patch 未接入，暂不能给公平数字。
3. ours_page_gather 在 decode-side latency 和 kept fraction 上有效，但 LongBench 质量不稳定。
4. raw prompt 下 b1024 出现超过 full 的样本级现象，说明 page gather 可能有纠偏能力；但官方对比必须以 aligned prompt / same generation path 为准。
5. 下一步创新重点不应是继续调 heuristic，而应把 typed page scorer 升级成可训练或 teacher-distilled 的 causal page influence predictor，并接进 KVCache-Factory 的同一 generation path。
```
