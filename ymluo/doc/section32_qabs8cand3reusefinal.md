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

## 9. 当前结论

### 9.1 质量结论

`qabs8cand3reusefinal` 是目前更合理的质量候选：

- 比单通道、双通道、三通道稳定得多。
- 比 `qabs4cand1reusefinal` 在长上下文上明显更稳。
- 10k/20k/40k War 长上下文 PPL 均略优于 baseline。
- 80k 仍有轻微 PPL 退化，但退化幅度可控。

### 9.2 时间结论

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

### 9.3 方法定位

`qabs8cand3reusefinal` 当前更适合作为质量稳定的 sparse retrieval 原型，而不是已经能直接加速的最终实现。

如果目标是实际速度收益，需要继续做 GPU-friendly/fused 实现：

1. candidate selection 直接输出 compact indices，而不是 dense bool mask。
2. current raw candidate 与 previous final top2 的 union 用 compact sorted merge 或 bitset kernel 完成。
3. full-QK rerank 与 final sparse attention 尽量融合。
4. 避免每层、每 token 频繁分配临时 tensor。
5. 避免为 qabs candidate kernel 长期缓存大尺寸 key-dim-major layout，或改成分块/streaming partial score。

## 10. 建议下一步

短期实验建议：

1. 在 Monte Cristo 上补跑 `qabs8cand3reusefinal` 的 10k/20k/40k/80k，验证质量是否和 War 一样稳定。
2. 测 `qabs8cand2reusefinal` 和 `qabs8cand4reusefinal` 的长上下文，确认 `cand3` 是否是最佳质量/时间折中。
3. 打开 `qabs_profile=true` 对 40k 或 80k 单点做 stage profile，定量确认 topk/gather/union/final attention 各自占比。

工程建议：

1. 优先优化 `reusefinal` 路线，而不是 `reuse` 路线，因为 `reusefinal` 的候选集合更小，更适合 fused kernel。
2. 优先做 compact candidate index pipeline，而不是继续调小通道数。单/双/三通道实验已经显示，降低通道数会明显伤质量，但不一定带来 wall-clock 收益。
3. 如果无法短期写 fused kernel，可以考虑先做 layer-wise hybrid：只在 full attention 成本最高、且 qabs 质量稳定的层启用 qabs8cand3reusefinal，其余层保持 baseline。
