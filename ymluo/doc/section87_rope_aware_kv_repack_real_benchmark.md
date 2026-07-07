# Section 87: RoPE-aware KV repack 真实 benchmark 小跑

日期：2026-07-07

## 目的

把上一节 synthetic smoke 推进到真实 LongBench/RULER 样本，验证它是否真的和 RAG/prompt-rebuild 区分开。

新增脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_rope_aware_kv_repack_benchmark.py
```

方法对比：

```text
full_kv_cache
prompt_rebuild_selected_pages
naive_kv_gather_absolute_query_pos
naive_kv_gather_compact_query_pos
rope_delta_repack_compact_query_pos
rope_delta_repack_shifted_query_pos
position_mode_oracle_sparse
position_mode_oracle_with_full
```

这里的 cache-native 路径是：

```text
prefill full context once
select KV pages
optionally apply RoPE delta repack
continue decoding from selected KV
```

RAG/prompt baseline 是：

```text
select text pages
rebuild prompt
prefill selected text again
decode
```

## 运行输出

0.6B quick：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/rope_repack_benchmark_qwen06b_quick_20260707
```

8B 小跑，7 tasks：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/rope_repack_benchmark_qwen8b_small_20260707
```

8B 13 tasks，top_k=2：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/rope_repack_benchmark_qwen8b_13tasks_20260707
```

8B 13 tasks，top_k=3：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/rope_repack_benchmark_qwen8b_13tasks_k3_20260707
```

## 8B 13 tasks, top_k=2

任务：

```text
LongBench: hotpotqa, 2wikimqa, musique, passage_retrieval_en, passage_count
RULER 4096: niah_single_1, niah_single_2, niah_multikey_1, niah_multiquery,
            niah_multivalue, vt, cwe, fwe
每个任务 1 个样本
max_context_tokens = 4096
page_tokens = 512
top_k = 2
```

整体结果：

| method | exact/score | active KV ratio | NLL |
|---|---:|---:|---:|
| full_kv_cache | 61.54% | 100.00% | 5.1860 |
| prompt_rebuild_selected_pages | 61.54% | 27.16% | 4.4901 |
| naive absolute gather | 30.77% | 26.08% | 4.3456 |
| naive compact gather | 38.46% | 26.08% | 4.8500 |
| RoPE compact repack | 69.23% | 26.08% | 4.1453 |
| RoPE shifted repack | 38.46% | 26.08% | 3.8913 |
| position oracle sparse | 69.23% | 26.08% | 3.6226 |
| position oracle + full | 69.23% | 37.56% | 3.4863 |

关键结论：

```text
1. naive sparse KV gather 很弱：30.77%-38.46%。
2. RoPE compact repack 明显修复：69.23%。
3. RoPE compact / position oracle 超过了 prompt-rebuild：69.23% vs 61.54%。
4. active KV 只有 full 的约 26.08%。
```

这说明 cache-native 路线不是只在 synthetic 上有效，真实 LongBench/RULER 小样本也有信号。

## 8B 13 tasks, top_k=3

整体结果：

| method | exact/score | active KV ratio | NLL |
|---|---:|---:|---:|
| full_kv_cache | 61.54% | 100.00% | 5.1860 |
| prompt_rebuild_selected_pages | 61.54% | 40.23% | 4.6666 |
| naive absolute gather | 46.15% | 39.13% | 5.1814 |
| naive compact gather | 30.77% | 39.13% | 4.6306 |
| RoPE compact repack | 61.54% | 39.13% | 4.3476 |
| RoPE shifted repack | 46.15% | 39.13% | 5.1792 |
| position oracle sparse | 61.54% | 39.13% | 3.8816 |

结论：

```text
top_k=3 没有更好，反而比 top_k=2 弱。
```

这支持一个重要观点：

```text
更多 KV page 不是单调收益；
必须做 action/position/budget planner。
```

## 代表性 case

### LongBench hotpotqa

top_k=2：

| method | score | comment |
|---|---:|---|
| full_kv_cache | 0 | 生成 Gates v. Collier，错 |
| prompt_rebuild | 1 | 正确 |
| naive gather | 0 | 重复 Answer 或混乱 |
| RoPE compact | 1 | 正确 |
| RoPE shifted | 0 | 退化成数字重复 |

说明：RoPE compact 在真实 LongBench multi-hop 样本上修复了 naive gather。

### RULER niah_single_2

top_k=2：

| method | score |
|---|---:|
| full_kv_cache | 1 |
| prompt_rebuild | 1 |
| naive absolute | 0 |
| naive compact | 0 |
| RoPE compact | 1 |
| RoPE shifted | 0 |

说明：这是最干净的证据之一：同样 selected pages，naive KV 失败，RoPE compact 成功。

### RULER niah_multivalue

top_k=2：

| method | score |
|---|---:|
| full_kv_cache | 1 |
| prompt_rebuild | 1 |
| naive absolute | 0 |
| naive compact | 0 |
| RoPE compact | 1 |
| RoPE shifted | 1 |

说明：multi-value 任务里 RoPE repack 能修复 naive gather。

## 当前判断

情况比之前明显变好。

之前的方法容易被说成：

```text
retrieval + prompt rebuild = RAG
```

现在可以明确写成：

```text
RAG retrieves text and re-prefills it.
Our method retrieves KV pages and applies RoPE-aware position repacking without raw-text re-prefill.
```

并且真实小 benchmark 已经看到：

```text
RoPE-aware KV repack > naive KV gather
RoPE-aware KV repack >= prompt-rebuild on this small run
```

## 限制

这还不是最终主结果。

限制：

```text
1. 每个任务只有 1 个样本，样本量太小。
2. 当前 full_kv_cache 的 prompt 形式为了清晰 KV mapping 做了简化，不能直接等同正式 LongBench prompt。
3. speedup 是 HF/Python 小跑的 online decode 时间，不是最终 CUDA kernel 级 speed。
4. 还没有 learned position-mode planner，只有 oracle 和固定 mode。
```

## 下一步

下一步应该做两个版本：

```text
1. 扩大到每任务 4-8 个样本，固定 top_k=2，跑 4k 和 8k。
2. 训练 position-mode planner：compact / shifted / absolute / full fallback。
```

position planner 标签：

```text
如果 compact 成功且最低 NLL，选 compact；
如果 shifted 成功且最低 NLL，选 shifted；
如果 naive absolute 成功且更稳，选 absolute；
如果 sparse 都失败，选 full fallback。
```

论文主线建议写成：

```text
Risk-aware typed KV cache planner
= page selection + RoPE-aware position-mode planning + safety fallback
```

这个主线已经和 RAG 明确分开。
