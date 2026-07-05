# Causal Sparse Prefill V6 结果（2026-07-03）

## 做了什么

在 `causal_ridge_page_gather` 基础上新增了真正的 prefill-time sparse path：

```text
causal_ridge_sparse_prefill
```

区别是：

- `causal_ridge_page_gather`：先 full prefill，再从 full KV cache 里 gather 选中的 page KV。
- `causal_ridge_sparse_prefill`：先用 causal ridge predictor 选 page，只把选中的 token 组成 compact prefix，然后只 prefill 这个短 prefix。

因此 `causal_ridge_sparse_prefill` 的 `total_seconds` 不再包含 full prefill，能测试端到端速度潜力。

新增代码：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_causal_page_influence_predictor_v6.py
ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_causal_page_sparse_prefill_ppl_v6.py
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_causal_page_sparse_prefill_downstream_v6_server.sh
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_causal_page_sparse_prefill_ppl_v6_server.sh
```

## LongBench 7-task 8B 小表

输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/causal_page_sparse_prefill_v6_1shot_b512_20260703_7task8b_sparseprefill
```

配置：

```text
model = /home/fdong/qwen/LlaMa-3.1-8B
tasks = qasper,multifieldqa_en,hotpotqa,2wikimqa,musique,passage_retrieval_en,passage_count
samples = 1 per task
max_context = 8192
budget = 512 context tokens
sink = 64
recent = 256
page = 256
max_new = 32
max_label_pages = 6
```

整体结果：

| method | score | total sec | prefill sec | online sec | kept prefix | keep frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_kv | 0.0698 | 1.7612 | 0.5088 | 1.2525 | 6479.7 | 1.000 |
| heuristic_page_gather | 0.0907 | 1.5034 | 0.5088 | 0.9946 | 551.7 | 0.121 |
| causal_ridge_page_gather | 0.0549 | 1.5032 | 0.5088 | 0.9944 | 551.7 | 0.121 |
| causal_label_oracle | 0.0820 | 1.5036 | 0.5088 | 0.9948 | 551.7 | 0.121 |
| heuristic_sparse_prefill | 0.0778 | 1.0991 | 0.0467 | 1.0524 | 551.7 | 0.121 |
| causal_ridge_sparse_prefill | 0.0324 | 1.0398 | 0.0478 | 0.9920 | 551.7 | 0.121 |

## 速度结论

prefill-time sparse path 的速度收益是明确的：

- full prefill：`0.5088s`
- causal sparse prefill：`0.0478s`

prefill 本身约快：

```text
0.5088 / 0.0478 = 10.6x
```

端到端 total：

- full KV：`1.7612s`
- causal sparse prefill：`1.0398s`

端到端约快：

```text
1.7612 / 1.0398 = 1.69x
```

这说明“只 prefill 选中的 page”确实能释放之前 full-prefill gather 没释放出来的速度空间。

## 质量结论

质量没有直接继承 `causal_ridge_page_gather` 的优势，反而下降：

```text
causal_ridge_page_gather   score = 0.0549
causal_ridge_sparse_prefill score = 0.0324
```

主要原因不是 scorer 一定坏，而是当前 sparse prefill 是 text-level compact prompt：

```text
原始 8k context -> 选中 token 按原顺序拼成 512 token compact prefix
```

这会引入两个变化：

1. RoPE/position 被压缩。原来远程 page 在 8k 位置附近，现在被移动到 512 token 的短上下文里。
2. 文档结构被破坏。page 之间的未选内容被直接删除，模型看到的 prompt 不再是原始文档流。

因此它和 full-prefill KV gather 不是同一个计算问题。KV gather 保留的是原始 prefill 后的 K/V 表示；text sparse prefill 会重新编码选中 token。

## PPL 结果

输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/causal_page_sparse_prefill_ppl_v6_p8192_e256_b512_20260703_7task8b_sparseprefill
```

配置：

```text
texts = War and Peace, Monte Cristo
prefill = 8192
eval = 256
budget = 512
```

整体：

| method | mean PPL | total sec | prefix tokens | keep frac |
| --- | ---: | ---: | ---: | ---: |
| full_prefill | 4.447 | 2.551 | 8192 | 1.000 |
| recent_sparse_prefill | 6.129 | 0.232 | 320 | 0.039 |
| heuristic_sparse_prefill | 5.753 | 0.433 | 512 | 0.062 |
| causal_ridge_sparse_prefill | 5.861 | 0.434 | 512 | 0.062 |

分文本：

| text | full PPL | recent PPL | heuristic PPL | causal PPL |
| --- | ---: | ---: | ---: | ---: |
| War and Peace | 3.929 | 6.567 | 5.956 | 6.112 |
| Monte Cristo | 4.965 | 5.691 | 5.550 | 5.610 |

PPL 结论：

- sparse prefill 非常快，512-token path 约 `0.43s`，full 8192-token path 约 `2.55s`。
- 但 PPL 明显升高，说明普通续写需要更连续的局部上下文，而 QA/retrieval 训练出来的 causal page predictor 不一定适合 LM continuation。
- 在纯 PPL 上，heuristic 略好于 causal ridge；这说明当前 causal labels 偏向问答任务，不是通用 LM memory labels。

## Range-position sparse prefill 原型

在 compact sparse prefill 的基础上，又补了一版：

```text
heuristic_rangepos_sparse_prefill
causal_ridge_rangepos_sparse_prefill
```

它和普通 sparse prefill 的区别是：

- K/V 仍然只 prefill 选中的 token；
- 但是 RoPE 的 `position_ids` 使用 token 在原始长上下文里的位置；
- `cache_position` 仍然用 compact 位置，避免 HuggingFace `DynamicCache` 索引出错。

也就是说，这不是最终的 fused `range_sdpa`，而是一个“原始位置保留”的中间原型，用来验证：单独保留 original position 能不能修复 compact sparse prefill 的质量损失。

### LongBench 7-task 结果

输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/causal_page_sparse_prefill_v6_1shot_b512_20260703_7task8b_rangepos
```

| method | score | total sec | prefill sec | online sec | kept prefix | keep frac |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_kv | 0.0698 | 1.7602 | 0.5096 | 1.2507 | 6479.7 | 1.000 |
| heuristic_page_gather | 0.0907 | 1.4893 | 0.5096 | 0.9798 | 551.7 | 0.121 |
| causal_ridge_page_gather | 0.0549 | 1.4905 | 0.5096 | 0.9810 | 551.7 | 0.121 |
| causal_label_oracle | 0.0820 | 1.4898 | 0.5096 | 0.9802 | 551.7 | 0.121 |
| heuristic_sparse_prefill | 0.0778 | 1.0843 | 0.0465 | 1.0378 | 551.7 | 0.121 |
| causal_ridge_sparse_prefill | 0.0324 | 1.0255 | 0.0480 | 0.9775 | 551.7 | 0.121 |
| heuristic_rangepos_sparse_prefill | 0.0327 | 1.0854 | 0.0470 | 1.0384 | 551.7 | 0.121 |
| causal_ridge_rangepos_sparse_prefill | 0.0188 | 1.0252 | 0.0482 | 0.9770 | 551.7 | 0.121 |

### PPL 结果

输出：

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/causal_page_sparse_prefill_ppl_v6_p8192_e256_b512_20260703_7task8b_rangepos
```

| method | mean PPL | total sec | prefix tokens | keep frac |
| --- | ---: | ---: | ---: | ---: |
| full_prefill | 4.447 | 2.584 | 8192 | 1.000 |
| recent_sparse_prefill | 6.129 | 0.234 | 320 | 0.039 |
| heuristic_sparse_prefill | 5.753 | 0.437 | 512 | 0.062 |
| causal_ridge_sparse_prefill | 5.861 | 0.435 | 512 | 0.062 |
| heuristic_rangepos_sparse_prefill | 5.813 | 0.435 | 512 | 0.062 |
| causal_ridge_rangepos_sparse_prefill | 5.985 | 0.435 | 512 | 0.062 |

range-position 的结论很明确：

- 速度基本没有损失，仍然是 512-token sparse prefill 的速度。
- 但是质量没有改善；在 LongBench 小表上甚至比普通 compact sparse prefill 更差。
- PPL 上也没有超过普通 sparse prefill，`heuristic_rangepos` 和 `causal_ridge_rangepos` 都没有接近 full_prefill。

这说明 sparse prefill 的主要问题不只是 RoPE 绝对位置被压缩。更核心的问题是：选中的 token 被重新组成了一段新的短序列，模型在每一层里看到的上下文依赖关系已经不是原始长文档里的依赖关系。full-prefill KV gather 复用的是“原始长上下文里已经算好的 K/V 表示”，而 compact/rangepos sparse prefill 都是在“删掉中间 token 后重新编码选中 token”。

## 当前判断

现在可以明确分开两个结论：

1. **速度方向成立。**
   prefill-time sparse path 能把 prefill 从约 `0.51s` 降到 `0.05s`，端到端 LongBench 小表约 `1.69x`。

2. **只改 original position 不够。**
   range-position sparse prefill 保留了 RoPE 位置，但仍然改变了层内上下文结构，所以没有恢复 full-prefill KV gather 的质量。

3. **质量要靠“原始结构保真”的 sparse prefill。**
   如果目标是通用 long-context memory，不能只把选中 page 拼成短 prompt；需要让选中 page 的 K/V 计算尽可能接近它们在原始长上下文中的表示。

所以下一步不应该只调 page scorer，而应该改 sparse prefill 的形式：

- 不要把选中 page 重新拼成短 prompt；
- 应该做真正的 page-level sparse prefill / range-SDPA，不只保留 position ids，还要保留 page/range 的原始 attention 结构；
- 或者在 compact prompt 中显式保留 page id / original offset / section boundary，让模型知道这些 page 来自原文哪个位置。

## 下一步建议

1. 实现真正的 `range_sdpa_sparse_prefill`：不要把 page 拼成一个普通短序列，而是把选中 page 作为原始长序列里的 ranges 来计算，至少保留 page 内连续结构和原始 offset。
2. causal labels 分任务训练：
   - QA/retrieval 用 answer-token delta NLL。
   - LM/PPL 用 continuation-token delta NLL。
3. PPL 专用 predictor 不应该复用 QA predictor，应在 War/Monte 等长文本上用 continuation delta NLL 蒸馏。
4. 报告时分清三种速度口径：
   - full-prefill gather online speed；
   - text sparse prefill end-to-end speed；
   - range-position sparse prefill end-to-end speed；
   - future fused range-SDPA sparse prefill end-to-end speed。
