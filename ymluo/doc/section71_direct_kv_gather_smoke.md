# Section 71: Direct KV Gather 快速验证

## 目标

本节验证一个新的实现方向：

```text
不再把选中的文本重新拼成 prompt 做 prefill，
而是先对完整上下文 prefill 一次，然后直接从已有 past_key_values 中 gather 选中的 KV page，
再用 gather 后的 compact KV 继续做 query/decode。
```

这个实验只做 smoke test，不跑大 benchmark。重点是回答两个问题：

1. 工程上能不能直接使用 gather 后的 KV cache 继续前向。
2. 非连续 page 的 KV gather 是否能像 prompt-rebuild 那样稳定回答。

## 代码

新增脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_kv_gather_smoke.py
```

脚本支持：

- Qwen3-0.6B HF `DynamicCache`。
- 从 full KV cache 中按 token index gather K/V。
- `top_k` page selection。
- `--force_pages` 强制指定 page。
- `--query_position_mode absolute|compact` 比较 query position 处理。
- synthetic exact lookup，用于快速判断答案是否可被读取。

## 测试设置

模型：

```bash
/home/fdong/hrj/prove/Qwen3-0.6B
```

上下文长度：

```text
8192 tokens
```

page 大小：

```text
1024 tokens
```

synthetic record：

```text
key = MAGIC-CODE-7319
answer = ORBITAL-COPPER-284
```

query：

```text
Question: What is the answer for key MAGIC-CODE-7319? Answer exactly.
Answer:
```

## 结果

### 1. Gather all pages: no-drop sanity check

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/kv_gather_smoke_qwen06b_exact_8k_k8_20260705
```

| method | active KV | gather | query | decode | online total | answer NLL | exact |
|---|---:|---:|---:|---:|---:|---:|---|
| full_kv_cache | 8192 | 0.0000s | 0.0801s | 0.7876s | 0.8677s | 1.0180 | true |
| kv_gather_compact | 8192 | 0.0065s | 0.0401s | 0.7596s | 0.8061s | 1.0180 | true |
| prompt_rebuild_selected_text | 7867 | 0.0000s | 0.0000s | 0.7572s | 1.1122s | 0.9081 | true |

结论：

- gather all pages 时，`kv_gather_compact` 和 `full_kv_cache` 的 answer NLL 完全一致到 4 位小数。
- 说明直接 gather KV 后继续前向这条路径是可行的，不是 API 或 cache 格式层面的错误。
- gather 开销约 6.5 ms。

### 2. Sparse page gather: top_k=2

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/kv_gather_smoke_qwen06b_exact_8k_k2_20260705
```

选中 page：

```text
[0, 5]
```

其中 page 5 包含 answer record。

| method | active KV | gather | query | decode | online total | answer NLL | exact |
|---|---:|---:|---:|---:|---:|---:|---|
| full_kv_cache | 8192 | 0.0000s | 0.0857s | 0.7771s | 0.8627s | 1.0180 | true |
| kv_gather_compact | 2048 | 0.0021s | 0.0349s | 0.7461s | 0.7830s | 8.5448 | false |
| prompt_rebuild_selected_text | 1984 | 0.0000s | 0.0000s | 0.7460s | 0.8193s | 0.9792 | true |

结论：

- sparse KV gather 的时间是正向的，online total 从 0.8627s 降到 0.7830s。
- 但是质量失败：证据页已经被选中，仍然没有答出 answer。
- prompt-rebuild 同样只使用 page 0 和 page 5，但可以答对。
- 这说明 prompt-rebuild 和 KV gather 不是等价操作。prompt-rebuild 会让选中文本在新 prompt 中重新上下文化；KV gather 使用的是原始 full-context 中已经计算好的 hidden/KV states。

### 3. Force only evidence page

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/kv_gather_smoke_qwen06b_exact_8k_forcep5_abs_20260705
```

| method | active KV | gather | query | decode | online total | answer NLL | exact |
|---|---:|---:|---:|---:|---:|---:|---|
| full_kv_cache | 8192 | 0.0000s | 0.0794s | 0.7718s | 0.8512s | 1.0180 | true |
| kv_gather_compact | 1024 | 0.0013s | 0.0344s | 0.7377s | 0.7735s | 8.7021 | false |
| prompt_rebuild_selected_text | 1005 | 0.0000s | 0.0000s | 0.7379s | 0.7754s | 0.8424 | true |

结论：

- 即使强制只取包含答案的 page 5，direct KV gather 仍然失败。
- 这进一步说明失败不是 retrieval 漏选证据页，而是非连续/局部 KV memory 的读取方式没有被原模型自然适配。

### 4. Compact position 对比

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/kv_gather_smoke_qwen06b_exact_8k_forcep5_compact_20260705
```

| method | active KV | query position | answer NLL | exact |
|---|---:|---|---:|---|
| kv_gather_compact | 1024 | compact | 5.3644 | false |

结论：

- compact position 比 absolute position 的 NLL 更低，但仍然没有答对。
- 单纯改 query/cache position 不能解决问题。

### 5. Contiguous prefix gather

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/kv_gather_smoke_qwen06b_exact_8k_prefix0_5_abs_20260705
```

选中 page：

```text
[0, 1, 2, 3, 4, 5]
```

| method | active KV | gather | query | decode | online total | answer NLL | exact |
|---|---:|---:|---:|---:|---:|---:|---|
| full_kv_cache | 8192 | 0.0000s | 0.0847s | 0.7806s | 0.8653s | 1.0180 | true |
| kv_gather_compact | 6144 | 0.0050s | 0.0360s | 0.7498s | 0.7908s | 0.6721 | true |
| prompt_rebuild_selected_text | 5906 | 0.0000s | 0.0000s | 0.7530s | 0.9984s | 1.4003 | true |

结论：

- 当 gather 的 KV 保持连续前缀结构时，direct KV gather 可以稳定答对。
- 这说明 direct KV gather 本身可用；真正困难的是任意 sparse/non-contiguous page KV 的读取。

## 当前判断

这个新方向是值得继续做的，但不能简单把 prompt-rebuild 方法替换成 arbitrary sparse KV gather。

更准确的结论是：

```text
direct KV gather 可以把方法和 RAG/prompt-rebuild 区分开，
也能避免重新 prefill selected text；
但是原始模型没有训练过读取“被任意拼接的非连续 KV memory”，
所以 sparse KV gather 需要专门的 adapter/router 训练，或者策略上保留连续 span。
```

## 对后续方法的影响

建议下一步不要直接做“大 benchmark”，而是先把方法改成下面两类之一：

1. **KV-prefix/span policy**

   router 不只选 page，还选择连续 span 或 prefix，例如：

   ```text
   recent raw + evidence span + summary memory
   ```

   这样更符合模型原本的 causal KV 使用方式。

2. **Gathered-KV adapter**

   训练 LoRA adapter，让模型适应：

   ```text
   query attends to sparse gathered KV pages
   ```

   训练数据可以用非 benchmark synthetic exact/retrieval 数据构造，目标是让模型在 page 已选中的情况下学会从 sparse KV memory 中读答案。

## 简短结论

- 工程可行：已经跑通真实 Qwen3 HF KV cache gather。
- gather 开销很小：1k-2k active KV 时约 1-2 ms，8k all gather 约 6.5 ms。
- 连续 KV 可用：prefix `[0..5]` 可以答对。
- 任意 sparse KV 不稳：page 5 包含答案时，直接 gather page 5 仍失败。
- 所以论文里如果要强调“不是 RAG，而是真 KV cache 操作”，需要补一个 `gathered-KV adapter` 或 `span-aware router`，否则质量会被 prompt-rebuild 版本压住。
