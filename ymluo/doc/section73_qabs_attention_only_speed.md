# QABS attention-only speed benchmark

本节修正之前的速度测试口径：不再用整模型 eval wall-clock 作为 QABS sparse decode 的速度证据。
整模型时间会包含 MLP、norm、lm-head、tokenizer/loop 调度等与 sparse attention 方法无关的部分，
因此不能回答“attention 模块本身是否变快、QABS 新增开销有多大”。

本次新增独立脚本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/benchmark_qabs_attention_only.py
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_qabs_attention_only_benchmark_server.sh
```

测试直接构造 decode 阶段的 Q/K/V cache 张量，只测：

1. baseline eager full attention；
2. QABS query-channel candidate generation；
3. current candidate 与 previous final / previous raw 的 union；
4. candidate 内 full-QK rerank；
5. final sparse attention。

不加载模型，不跑 MLP，不跑 lm-head。

## 1. 配置

服务器：

```text
fdong@10.176.37.31
GPU: NVIDIA GeForce RTX 3090
project: /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
```

默认 shape：

```text
batch = 1
heads = 16
head_dim = 128
dtype = bfloat16
layers projected = 28
qabs_dim_count = 8
qabs_candidate_fraction = 0.03
top_fraction = 0.02
protect_sink_tokens = 10
protect_recent_tokens = 10
partial_impl = cuda_dim_major
use_cuda_full_scores = true
use_cuda_final_attention = true
```

主输出目录：

```text
outputs/qabs_attention_only_reusefinal_20260705_v1
outputs/qabs_attention_only_reusefinal_65536_20260705_v1
outputs/qabs_attention_only_reuse3set_32768_65536_20260705_v1
```

## 2. qabs8cand3reusefinal 结果

这里的 `baseline_attention_ms` 是单层 decode attention call；
`qabs_attention_path_ms` 是单层 QABS attention path，包括新增 candidate/union/rerank 开销和最终 sparse attention。

| history tokens | baseline attention ms | QABS path ms | QABS / baseline | projected baseline 28 layers ms | projected QABS 28 layers ms |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.152 | 0.740 | 4.878x | 4.249 | 20.726 |
| 4,096 | 0.151 | 0.729 | 4.813x | 4.241 | 20.414 |
| 8,192 | 0.206 | 0.711 | 3.460x | 5.755 | 19.915 |
| 16,384 | 0.373 | 0.723 | 1.936x | 10.453 | 20.239 |
| 32,768 | 0.697 | 0.844 | 1.210x | 19.529 | 23.632 |
| 65,536 | 1.373 | 1.099 | 0.801x | 38.449 | 30.784 |

结论：在当前 3090 + eager baseline + QABS CUDA kernel 原型下，
`qabs8cand3reusefinal` 只有到 64k history 才在 attention-only 口径下超过 full attention。
32k 仍然慢约 21%。

## 3. 64k stage breakdown

`qabs8cand3reusefinal`, history=65,536：

| stage | ms / call | fraction |
|---|---:|---:|
| qdim_topk | 0.096 | 5.7% |
| partial_scores | 0.052 | 3.1% |
| candidate_select | 0.156 | 9.2% |
| candidate_union | 0.054 | 3.2% |
| candidate_full_scores | 0.148 | 8.7% |
| final_topk | 0.176 | 10.4% |
| final_mask_and_indices | 0.291 | 17.2% |
| final_sparse_attention | 0.717 | 42.4% |

新增 overhead 约 `0.973 ms/call`，最终 sparse attention 约 `0.717 ms/call`。
注意 stage breakdown 使用同步分段计时，分段和会高于无分段同步的 end-to-end QABS path；
真正对比 baseline 的值应看 `qabs_attention_path_ms`。

## 4. reusefinal vs 3-set reuse

补测 32k/64k 的三集合版本 `qabs8cand3reuse`：

| mode | history tokens | baseline attention ms | QABS path ms | QABS / baseline |
|---|---:|---:|---:|---:|
| qabs8cand3reusefinal | 32,768 | 0.697 | 0.844 | 1.210x |
| qabs8cand3reuse | 32,768 | 0.711 | 0.910 | 1.280x |
| qabs8cand3reusefinal | 65,536 | 1.373 | 1.099 | 0.801x |
| qabs8cand3reuse | 65,536 | 1.374 | 1.101 | 0.801x |

三集合复用在 64k 下速度几乎相同；在 32k 下更慢一些。
这说明当前主要瓶颈不是多 OR 一个 previous raw mask，而是 candidate select/topk、mask-to-indices 和 final sparse attention。

## 5. 解释

之前整模型 speed benchmark 不适合作为方法速度证据，因为 MLP/lm-head 等固定计算会稀释 attention 变化。
新的 attention-only benchmark 显示：

1. 短上下文下 QABS 明显慢，因为新增开销是近似固定的。
2. 32k 左右仍未真正超过 full attention。
3. 64k 开始有 attention-only 加速，但幅度还有限。
4. 下一步优化优先级应放在 `final_mask_and_indices`、`final_topk/candidate_select` 和最终 sparse attention kernel，而不是继续用整模型 wall-clock 判断。
