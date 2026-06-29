# qabs8cand3reusefinal 方法说明与实验记录

本文整理 `qabs8cand3reusefinal` 的方法定义、实验设置、主要对照实验和当前结论。实验主要在服务器
`fdong@10.176.37.31` 的项目目录 `/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl`
下完成。

## 1. 方法目标

`qabs8cand3reusefinal` 的目标是在 decode 阶段减少 full QK 计算：

1. 不对完整历史 KV 做全维度 QK。
2. 先用 query 中少量高幅值维度做 partial-QK，生成候选 token。
3. 在候选 token 上再做 full-QK rerank。
4. 最终只对少量 selected token 做 attention。

它不是 oracle top2。`top2` 需要先计算完整 full-QK，再选每个 head 的 top 2% 历史 token；而
`qabs8cand3reusefinal` 试图用便宜的 partial-QK 近似找出这些重要 token。

## 2. 方法定义

名称拆解：

```text
qabs8cand3reusefinal
```

- `qabs8`：对每个 query/head，选择 query 向量中绝对值最大的 8 个维度。
- `cand3`：用这 8 个维度和所有历史 key 做 partial-QK，选 partial score 最高的 top 3% 历史 token，得到当前 raw candidate。
- `reusefinal`：复用上一 decode step 最终进入 attention 的 final top2 token。
- 最终候选集合：

```text
candidate_union = 当前 raw candidate ∪ 上一步 final top2
```

然后在 `candidate_union` 内做 full-QK rerank，选出最终用于 sparse attention 的 token。

### 与 qabs8cand3reuse 的区别

`qabs8cand3reuse` 的候选集合更大：

```text
qabs8cand3reuse = 当前 raw candidate ∪ 上一步 raw candidate ∪ 上一步 final top2
```

`qabs8cand3reusefinal` 不复用上一步 raw candidate，只复用上一步 final top2：

```text
qabs8cand3reusefinal = 当前 raw candidate ∪ 上一步 final top2
```

因此 `reusefinal` 更激进，理论上 full-QK rerank 的候选更少，更适合未来 fused kernel；但它也可能漏掉上一 token raw candidate 中有潜力、但上一 token 没进入 final top2 的历史 token。

## 3. 实验设置

主要公共设置：

- 模型：`Qwen/Qwen3-0.6B`
- attention implementation：`eager`
- dtype：`bfloat16`
- eval chunk：`eval_chunk_size=1`
- prefill chunk：`chunk_size=8`
- `top_fraction=0.02`
- `protect_sink_tokens=10`
- `protect_recent_tokens=10`
- `always_keep_self=true`
- `qabs_fast_path=true`
- `qabs_cuda_final_kernel=true`
- `qabs_cuda_candidate_kernel=true`
- `qabs_cuda_reuse_select_kernel=false`
- `disable_sparse_stats=true`

主要数据：

- War and Peace：`data/war_and_peace_pg2600.txt`
- Monte Cristo：`data/count_monte_cristo_pg1184.txt`

计时说明：

- 表中的 `seconds` 是 decode/eval 阶段时间。
- 使用 shared prefill 时，prefill 时间单独记录为 `shared_prefill_seconds`，不计入各 mode 的 `seconds`。
- 80k 长上下文下，multi-mode shared KV + clone KV 会导致 24GB GPU OOM，因此 80k 使用 single-mode shared prefill、不 clone KV 的方式补跑；该方式下 `seconds` 仍是 eval-only。

## 4. 短上下文基准：baseline / top2 / qabs8cand3reuse

设置：`prefill=10000, eval=128`。

| 数据 | 方法 | PPL | 相对 baseline | eval 时间 | 时间相对 baseline |
|---|---:|---:|---:|---:|---:|
| War and Peace | baseline | 15.7399 | 1.000x | 4.536s | 1.00x |
| War and Peace | top2 | 15.7207 | 0.999x | 6.790s | 1.50x |
| War and Peace | qabs8cand3reuse | 15.2279 | 0.967x | 82.315s | 18.15x |
| Monte Cristo | baseline | 48.3293 | 1.000x | 4.607s | 1.00x |
| Monte Cristo | top2 | 47.0691 | 0.974x | 6.720s | 1.46x |
| Monte Cristo | qabs8cand3reuse | 45.3431 | 0.938x | 79.207s | 17.19x |

观察：

- `qabs8cand3reuse` 的 PPL 很好，两个数据上都优于 baseline。
- 但 wall-clock 非常慢，说明当前原型实现瓶颈不只是 full-QK 数量，而是 candidate select、topk、gather、状态维护、小 kernel 调度等固定开销。

## 5. 更激进参数搜索

设置：War and Peace，`prefill=10000, eval=64`。

| mode | PPL | ΔPPL vs baseline | eval 时间 | 时间相对 baseline |
|---|---:|---:|---:|---:|
| baseline | 15.5743 | 0.0000 | 2.338s | 1.00x |
| top2 | 15.6426 | +0.0683 | 3.471s | 1.48x |
| qabs4cand2reuse | 15.4658 | -0.1085 | 3.909s | 1.67x |
| qabs8cand1reuse | 15.5973 | +0.0230 | 3.922s | 1.68x |
| qabs4cand1reuse | 14.8412 | -0.7331 | 3.924s | 1.68x |
| qabs8cand2reuse | 15.2824 | -0.2919 | 3.929s | 1.68x |
| qabs8cand2reusefinal | 15.2848 | -0.2895 | 3.949s | 1.69x |
| qabs4cand1reusefinal | 14.7209 | -0.8534 | 3.980s | 1.70x |
| qabs2cand1reuse | 16.3021 | +0.7278 | 3.982s | 1.70x |

观察：

- 更激进参数可以把之前 `qabs8cand3reuse` 的巨大时间开销压到约 `1.7x baseline`。
- `qabs4cand1reusefinal` 在该短评测上 PPL 最好，但仍然没有快过 baseline。
- 当前实现中，降低候选数并不能线性降低时间，说明固定开销占比很高。

## 6. 单通道、双通道、三通道实验

设置：`prefill=10000, eval=128`。

### 6.1 单通道 qabs1

| 数据 | 方法 | PPL | eval 时间 |
|---|---:|---:|---:|
| War | baseline | 15.7399 | 5.00s |
| War | top2 | 15.7207 | 6.72s |
| War | qabs1cand1reuse | 23.1370 | 7.73s |
| War | qabs1cand2reuse | 22.1570 | 7.74s |
| War | qabs1cand3reuse | 20.5280 | 7.72s |
| War | qabs1cand5reuse | 19.6234 | 8.03s |
| Monte | baseline | 48.3293 | 4.70s |
| Monte | top2 | 47.0691 | 6.68s |
| Monte | qabs1cand1reuse | 52.3198 | 7.68s |
| Monte | qabs1cand2reuse | 50.9733 | 7.70s |
| Monte | qabs1cand3reuse | 51.7002 | 7.68s |
| Monte | qabs1cand5reuse | 49.5283 | 7.93s |

结论：单通道质量明显不够，时间也没有收益。

### 6.2 双通道 qabs2cand1reusefinal

| 数据 | mode | PPL | ΔPPL vs baseline | eval 时间 | 时间 vs baseline |
|---|---:|---:|---:|---:|---:|
| War | baseline | 15.7399 | 0 | 4.536s | 1.00x |
| War | qabs2cand1reusefinal | 19.9752 | +4.2354 | 7.682s | 1.69x |
| Monte | baseline | 48.3293 | 0 | 4.607s | 1.00x |
| Monte | qabs2cand1reusefinal | 49.1500 | +0.8206 | 7.676s | 1.67x |

结论：双通道在 Monte 上勉强接近 baseline，但 War 上明显退化，且时间没有收益。

### 6.3 三通道 qabs3cand1reusefinal

| 数据 | mode | PPL | ΔPPL vs baseline | eval 时间 | 时间 vs baseline |
|---|---:|---:|---:|---:|---:|
| War | baseline | 15.7399 | 0 | 4.536s | 1.00x |
| War | qabs3cand1reusefinal | 17.7530 | +2.0132 | 7.708s | 1.70x |
| Monte | baseline | 48.3293 | 0 | 4.607s | 1.00x |
| Monte | qabs3cand1reusefinal | 49.9882 | +1.6588 | 7.820s | 1.70x |

结论：三通道比双通道在 War 上改善，但仍不够稳，也没有时间收益。

## 7. qabs4cand1reusefinal 长上下文实验

设置：War and Peace，`eval=128`，测试 `prefill=10k/20k/40k/80k`。

| prefill | baseline PPL | qabs4 PPL | ΔPPL | baseline eval 秒 | qabs4 eval 秒 | qabs4 / baseline |
|---:|---:|---:|---:|---:|---:|---:|
| 10k | 15.7399 | 16.2803 | +0.5405 | 4.397s | 8.149s | 1.85x |
| 20k | 13.3355 | 18.3651 | +5.0297 | 4.752s | 9.988s | 2.10x |
| 40k | 38.1721 | 44.1622 | +5.9901 | 8.044s | 12.826s | 1.59x |
| 80k | 22.7025 | 26.4360 | +3.7335 | 19.693s | 23.224s | 1.18x |

观察：

- 上下文越长，qabs4 与 baseline 的时间差距越小。
- 80k 时已经从 10k 的 `1.85x` 慢缩小到 `1.18x` 慢。
- 但 qabs4 的 PPL 退化明显，尤其 20k/40k/80k。

因此 qabs4 太激进，不适合作为长上下文质量主候选。

## 8. qabs8cand3reusefinal 长上下文实验

设置：War and Peace，`eval=128`，测试 `prefill=10k/20k/40k/80k`。

| prefill | baseline PPL | qabs8 PPL | ΔPPL | baseline eval 秒 | qabs8 eval 秒 | qabs8 / baseline |
|---:|---:|---:|---:|---:|---:|---:|
| 10k | 15.7399 | 15.2920 | -0.4479 | 4.640s | 8.438s | 1.82x |
| 20k | 13.3355 | 13.1964 | -0.1391 | 5.268s | 10.961s | 2.08x |
| 40k | 38.1721 | 37.9722 | -0.1999 | 9.869s | 14.520s | 1.47x |
| 80k | 22.7025 | 23.8874 | +1.1849 | 19.693s | 23.052s | 1.17x |

观察：

- 相比 qabs4，`qabs8cand3reusefinal` 的 PPL 稳定很多。
- 10k/20k/40k 上 PPL 都略优于 baseline。
- 80k 上 PPL 比 baseline 差 `+1.1849`，但仍远好于 qabs4 的 `+3.7335`。
- 时间趋势与 qabs4 类似：上下文越长，慢的比例越小；80k 时仍慢约 `17%`。

## 9. 下游任务性能测试

这一节记录同类 QABS query-channel sparse retrieval 方法在下游任务上的表现。注意：现有下游任务主测点是 `qabs8cand5reuse` 和 `qabs8cand8`，不是 `qabs8cand3reusefinal` 的完全同参数复现；但它们使用相同的核心机制：

- query 绝对值最大通道做 partial-QK candidate generation；
- candidate union/reuse；
- 在候选集合内 exact rerank；
- decode 阶段 sparse attention。

因此这些结果可以作为 `qabs8cand3reusefinal` 的重要风险证据：PPL 接近 baseline 并不自动代表 exact key-value retrieval 下游任务稳定。

### 9.1 Broad PPL 与 KV retrieval 初测

方法：`qabs8cand5reuse`，`top_fraction=0.05`，三集合复用：

```text
current candidate ∪ previous candidate ∪ previous final
```

PPL 设置：`prefill_tokens=2048`，`eval_tokens=256`，`dtype=float16`。

| Dataset | Baseline PPL | qabs8cand5reuse PPL | Ratio |
|---|---:|---:|---:|
| hard_topic_eval_v2 | 4.6147 | 4.7197 | 1.0228 |
| hard_topic_eval_v3 | 4.4129 | 4.5117 | 1.0224 |
| hard_topic_eval_v4 | 4.1257 | 4.2665 | 1.0341 |
| topic_stress_eval | 2.5155 | 2.6497 | 1.0534 |
| War and Peace | 34.2606 | 34.0770 | 0.9946 |
| Count of Monte Cristo | 32.1917 | 33.2323 | 1.0323 |

六个数据集平均 PPL ratio 约 `1.0266`，即相对 PPL 上升约 `2.7%`。这说明同类 qabs 方法在 broad PPL 上是可行的。

但在 synthetic key-value retrieval 下游任务上，初测出现明显下降：

| Run | top_fraction | Approx retained KV | Baseline accuracy | qabs accuracy | Delta |
|---|---:|---:|---:|---:|---:|
| `downstream_kv_retrieval_qabs8cand5_tf5_v4` | 0.05 | about 6% | 17/32 = 53.1% | 14/32 = 43.8% | -9.4 pts |
| `downstream_kv_retrieval_qabs8cand8_tf8_v5` | 0.08 | about 9% | 17/32 = 53.1% | 14/32 = 43.8% | -9.4 pts |

结论：简单把 `top_fraction` 从 5% 提高到 8% 没有恢复下游 retrieval accuracy。

### 9.2 Multi-task downstream suite

为了确认下游掉点是否只是单一 prompt 格式问题，又测试了 6 种 A/B/C/D label scoring 任务格式：

- `structured_noisy`
- `compact_kv`
- `natural_kv`
- `json_kv`
- `needle_sentence`
- `topic_table`

#### 长一点上下文：64 records/task

设置：每种格式 16 个 task，每个 task 64 条 records。

| Variant | Baseline | qabs5 | Delta |
|---|---:|---:|---:|
| structured_noisy | 6/16 = 37.5% | 1/16 = 6.3% | -31.3 pts |
| compact_kv | 5/16 = 31.3% | 6/16 = 37.5% | +6.3 pts |
| natural_kv | 6/16 = 37.5% | 5/16 = 31.3% | -6.3 pts |
| json_kv | 5/16 = 31.3% | 5/16 = 31.3% | 0.0 pts |
| needle_sentence | 8/16 = 50.0% | 8/16 = 50.0% | 0.0 pts |
| topic_table | 11/16 = 68.8% | 8/16 = 50.0% | -18.8 pts |

这组里 baseline 本身经常接近随机，适合作为 stress test，但不是很干净的质量 benchmark。

#### 短上下文：16 records/task

设置：每种格式 32 个 task，每个 task 16 条 records。这里 baseline 更强，更能暴露压缩造成的 retrieval 损失。

| Variant | Baseline | qabs5 | Delta qabs5 | qabs8 | Delta qabs8 |
|---|---:|---:|---:|---:|---:|
| structured_noisy | 19/32 = 59.4% | 15/32 = 46.9% | -12.5 pts | 12/32 = 37.5% | -21.9 pts |
| compact_kv | 29/32 = 90.6% | 18/32 = 56.3% | -34.4 pts | 21/32 = 65.6% | -25.0 pts |
| natural_kv | 18/32 = 56.3% | 13/32 = 40.6% | -15.6 pts | 15/32 = 46.9% | -9.4 pts |
| json_kv | 27/32 = 84.4% | 19/32 = 59.4% | -25.0 pts | 19/32 = 59.4% | -25.0 pts |
| needle_sentence | 18/32 = 56.3% | 17/32 = 53.1% | -3.1 pts | 17/32 = 53.1% | -3.1 pts |
| topic_table | 22/32 = 68.8% | 22/32 = 68.8% | 0.0 pts | 18/32 = 56.3% | -12.5 pts |

观察：

- `compact_kv` 和 `json_kv` 是最清楚的失败格式：dense baseline 强，但 qabs 掉 8-11 个 task。
- `needle_sentence` 相对稳定，只掉 1/32。
- `topic_table` 在 qabs5 下不掉，但 qabs8 反而掉 4/32，说明 uniform budget 增加不一定改善下游任务。
- 下游任务格式影响很大；exact key-value binding 对 candidate selection 更敏感。

### 9.3 Evidence span coverage 诊断

为了验证 retrieval 失败是否来自 qabs 没保留目标证据 span，增加了 evidence span coverage 诊断。每个 task 定位三类 span：

- `key`：目标查询 key；
- `label`：目标答案 label token；
- `record`：完整目标 record 行。

然后记录 qabs mask 是否覆盖这些 span：

- `current`：当前 query-channel raw candidate；
- `union`：candidate union；
- `final`：exact rerank 后最终保留 token。

设置：`qabs8cand5reuse`，`top_fraction=0.05`，每种格式 16 个 task，每个 task 16 条 records。

Accuracy：

| Variant | Baseline | qabs5 |
|---|---:|---:|
| compact_kv | 15/16 = 93.8% | 11/16 = 68.8% |
| json_kv | 9/16 = 56.3% | 10/16 = 62.5% |
| needle_sentence | 9/16 = 56.3% | 7/16 = 43.8% |
| topic_table | 12/16 = 75.0% | 8/16 = 50.0% |

Overall coverage：

| Variant | Final key any | Final key all | Final label any | Final record any | Union key any | Union label any | Union record any |
|---|---:|---:|---:|---:|---:|---:|---:|
| compact_kv | 23.5% | 0.1% | 10.5% | 31.4% | 40.5% | 18.3% | 50.9% |
| json_kv | 18.4% | 0.0% | 9.1% | 47.4% | 34.0% | 15.8% | 70.6% |
| needle_sentence | 19.2% | 0.0% | 14.5% | 63.7% | 32.0% | 21.0% | 79.4% |
| topic_table | 20.4% | 0.0% | 7.8% | 53.4% | 33.6% | 13.2% | 72.3% |

结论：

- final retained set 对 target key 的 `any` coverage 只有约 18-24%。
- full key-span coverage 基本为 0。
- label coverage 只有约 8-15%。
- union coverage 明显高于 final coverage，说明 exact rerank/final top selection 会继续丢掉一部分 evidence。

这支持一个重要判断：同类 qabs 方法的下游 retrieval loss 不只是 prompt artifact，而是 candidate/final retained tokens 经常没有保留目标证据 span。

### 9.4 Retrieval-preserving hybrid 试验

尝试过一些 naive retrieval-preserving hybrid 变体：

| Variant | Intended retained KV | Accuracy | Result |
|---|---:|---:|---|
| `qabs8cand5reuse`, `top_fraction=0.05` | about 6% | 14/32 = 43.8% | current reference |
| `qabs8cand5reuse`, `top_fraction=0.08` | about 9% | 14/32 = 43.8% | uniform larger budget did not help |
| `qabs8cand5reuseblk8`, `top_fraction=0.01` | about 8% rough target | 13/32 = 40.6% | block expansion with too few seed tokens hurt |
| `qabs8cand5reuseblk4`, `top_fraction=0.02` | about 8% rough target | 13/32 = 40.6% | smaller block still hurt |
| default qabs5 + layer 13 full | about 9-10% | 12/32 = 37.5% | hand-picked middle full layer hurt |
| default qabs5 + layer 0 full | about 9-10% | 14/32 = 43.8% | no gain over qabs |
| default qabs5 + layers 0/11/14/15 headmix4 | about 9% rough target | 14/32 = 43.8% | no gain over qabs |

这些负结果说明：不能只靠 uniform budget、固定 block expansion 或手工选择 full layer 来修复 retrieval。更合理的方向是 evidence-aware calibration 或 oracle span retention，先确认保留 target key/value span 是否能恢复 compact/json/topic_table 的准确率。

### 9.5 Oracle span-retention 诊断

在 9.3 的 evidence span coverage 之后，进一步做了 oracle span-retention 诊断。该实验不是可部署方法，而是为了回答一个问题：

> 如果最终 retained KV 中强制保留目标 key/value evidence span，下游 retrieval accuracy 是否能恢复？

实验设置：

- 模型：`/home/fdong/hrj/prove/Qwen3-0.6B`
- 输出目录：`/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/oracle_span_retention_qabs5_shortctx_v1`
- 任务格式：`compact_kv`、`json_kv`、`topic_table`、`needle_sentence`
- 每种格式 16 个 task，每个 task 16 条 records
- 基础 sparse mode：`qabs8cand5reuse`
- `top_fraction=0.05`
- `protect_sink_tokens=10`
- `protect_recent_tokens=10`

对比模式：

- `baseline`：dense attention
- `qabs8cand5reuse`：普通 QABS
- `oracle_key_label`：普通 QABS + 在 final mask 中强制保留 target key span 和 target label span
- `oracle_record`：普通 QABS + 在 final mask 中强制保留完整 target record 行

结果：

| Variant | Baseline | qabs5 | Oracle key+label | Oracle record |
|---|---:|---:|---:|---:|
| compact_kv | 15/16 = 93.8% | 11/16 = 68.8% | 14/16 = 87.5% | 14/16 = 87.5% |
| json_kv | 9/16 = 56.3% | 10/16 = 62.5% | 11/16 = 68.8% | 10/16 = 62.5% |
| topic_table | 12/16 = 75.0% | 8/16 = 50.0% | 12/16 = 75.0% | 11/16 = 68.8% |
| needle_sentence | 9/16 = 56.3% | 7/16 = 43.8% | 12/16 = 75.0% | 11/16 = 68.8% |

关键观察：

- `compact_kv` 中，普通 qabs 相比 baseline 掉 4 个 task；强制保留 key+label 后恢复 3 个。
- `topic_table` 中，普通 qabs 掉 4 个 task；强制保留 key+label 后完全恢复到 baseline。
- `needle_sentence` 中，普通 qabs 掉 2 个 task；强制保留 key+label 后在该小样本上超过 baseline。
- `oracle_key_label` 通常不弱于 `oracle_record`，说明需要 rescue 的主要是紧凑 key/value binding span，而不一定是整条 record。

该结果基本确认：exact retrieval 掉点的主要原因不是模型完全不会做任务，也不是单纯 prompt artifact，而是 QABS final retained set 没有稳定保留目标 evidence span。

因此更合理的下一步不是继续 uniform 提高 `top_fraction`，而是做 evidence-gated span rescue：默认仍使用 QABS，但在检测到 lookup-like query 时，用很小的额外预算保留候选 key/value span。

## 10. 当前结论

### 10.1 质量结论

`qabs8cand3reusefinal` 是目前更合理的质量候选：

- 比单通道、双通道、三通道稳定得多。
- 比 `qabs4cand1reusefinal` 在长上下文上明显更稳。
- 10k/20k/40k War 长上下文 PPL 均略优于 baseline。
- 80k 仍有轻微 PPL 退化，但退化幅度可控。
- 但同类 QABS 方法在 exact key-value retrieval 下游任务上会明显掉点，尤其 `compact_kv` 和 `json_kv`。因此不能只凭 PPL 判断方法已经可用。
- evidence span coverage 和 oracle span-retention 已经验证：retrieval 掉点很大程度来自 final retained KV 没有保住目标 key/value span；强制保留 key+label 可以显著恢复 `compact_kv`、`topic_table`、`needle_sentence` 的准确率。

### 10.2 时间结论

当前实现还没有 wall-clock 速度收益：

- 10k：约 `1.82x` baseline 时间。
- 20k：约 `2.08x` baseline 时间。
- 40k：约 `1.47x` baseline 时间。
- 80k：约 `1.17x` baseline 时间。

虽然 80k 已经接近 baseline，但仍未快过 baseline。

这说明当前瓶颈主要不是 full-QK 数量，而是原型实现中的固定开销：

- 每 token、每层的 topk。
- dense bool mask 构造。
- candidate union。
- gather candidate K/V。
- 多个小 CUDA kernel launch。
- Python 状态维护。
- qabs CUDA candidate kernel 对长上下文还会额外使用 key-dim-major cache，显存压力较高。

### 10.3 方法定位

`qabs8cand3reusefinal` 当前更适合作为质量稳定的 sparse retrieval 原型，而不是已经能直接加速的最终实现。

如果目标是实际速度收益，需要继续做 GPU-friendly/fused 实现：

1. candidate selection 直接输出 compact indices，而不是 dense bool mask。
2. current raw candidate 与 previous final top2 的 union 用 compact sorted merge 或 bitset kernel 完成。
3. full-QK rerank 与 final sparse attention 尽量融合。
4. 避免每层、每 token 频繁分配临时 tensor。
5. 避免为 qabs candidate kernel 长期缓存大尺寸 key-dim-major layout，或改成分块/streaming partial score。
6. 对 exact retrieval 任务加入 evidence-aware rescue：当 query 触发 lookup 行为时，额外保留 target-like key/value span 或高风险 layer/head 的证据 token。

## 11. 建议下一步

短期实验建议：

1. 在 Monte Cristo 上补跑 `qabs8cand3reusefinal` 的 10k/20k/40k/80k，验证质量是否和 War 一样稳定。
2. 测 `qabs8cand2reusefinal` 和 `qabs8cand4reusefinal` 的长上下文，确认 `cand3` 是否是最佳质量/时间折中。
3. 打开 `qabs_profile=true` 对 40k 或 80k 单点做 stage profile，定量确认 topk/gather/union/final attention 各自占比。
4. 对 `qabs8cand3reusefinal` 补跑 downstream task suite，确认它相对 `qabs8cand5reuse` 是否改善 exact KV retrieval。
5. 基于 oracle span-retention 的正结果，做非 oracle 的 evidence-gated span rescue：在线检测 lookup-like query，从 key-like token、分隔符、JSON/table 字段或高 saliency token 中生成候选 key/value span，并只用很小预算强制保留少量 span。

工程建议：

1. 优先优化 `reusefinal` 路线，而不是 `reuse` 路线，因为 `reusefinal` 的候选集合更小，更适合 fused kernel。
2. 优先做 compact candidate index pipeline，而不是继续调小通道数。单/双/三通道实验已经显示，降低通道数会明显伤质量，但不一定带来 wall-clock 收益。
3. 如果无法短期写 fused kernel，可以考虑先做 layer-wise hybrid：只在 full attention 成本最高、且 qabs 质量稳定的层启用 qabs8cand3reusefinal，其余层保持 baseline。
4. 对 exact retrieval 场景，优先实现 evidence-gated span rescue，而不是继续盲目加大 uniform token budget；oracle 结果显示 key+label span 是更有效的 rescue 单元。
