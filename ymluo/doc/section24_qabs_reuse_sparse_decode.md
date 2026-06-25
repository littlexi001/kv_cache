# Section 24: QABS reuse sparse decode experiments

本节整理 `qwen3_top2_head_limit3_ppl` 项目中关于 `qabs reuse` 方法的实验设计、实现路径、PPL 结果和速度结果。

项目路径：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl
```

主要脚本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/evaluate_qwen3_top2_head_limit3_ppl.py
ymluo/projects/qwen3_top2_head_limit3_ppl/src/qabs_cuda_kernels.py
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_qabs_fast_speed_server.sh
```

## 1. 问题背景

前面的实验发现，直接保留 full attention 中 attention score 最高的 top2% token 时，PPL 可以接近甚至略好于 baseline。

核心困难是：decode 时不能先算完整 attention 再选 top2%，否则计算量没有减少。必须找到一个比 full-QK 更便宜的方法，提前预测哪些 historical token 会进入真实 top2%。

本轮实验的目标是：

1. 不做 full-history full-QK。
2. 用 query 向量中幅值最大的少量维度做 partial-QK，产生候选 token。
3. 复用相邻 decode step 的候选和 top2 token，提高 recall。
4. 在候选集合内做 full-QK rerank，最终只保留 top2% token。
5. 观察 PPL 是否接近 baseline，以及速度是否有希望超过 baseline。

## 2. 方法定义

### 2.1 qabs candidate

对每个 decode step、每层、每个 attention head：

1. 取当前 query 向量绝对值最大的 `Dq` 个通道。
2. 只在这些通道上计算 query 和历史 key 的 partial dot product。
3. 根据 partial score 选出 candidate fraction 的历史 token。

记法：

```text
qabs16cand3
```

含义是：

```text
qabs_dim_count = 16
qabs_candidate_fraction = 0.03
```

也就是每个 head 使用 query 幅值最大的 16 个通道，先取历史 token 中 partial-QK 最高的 3% 作为候选。

### 2.2 reuse

`reuse` 版本不仅使用当前 query 的 qabs candidate，还复用相邻 decode step 的选择结果。

当前实现的 candidate union 是：

```text
candidate =
  current qabs partial-QK candidate
  union previous-step qabs candidate
  union previous-step final top2 token
```

然后在这个 candidate 集合内部做 full-QK rerank，选出最终 top2% historical token。

实验名：

```text
qabs16cand3reuse
```

表示：

```text
qabs_dim_count = 16
qabs_candidate_fraction = 0.03
use previous candidate/top2 reuse
final rerank target = top2%
```

### 2.3 sink/recent protection

本轮主要速度实验使用：

```text
protected_sink_tokens = 10
protected_recent_tokens = 10
always_keep_self = true
```

注意：早期实验曾经使用 `sink=1000, recent=1000`，在 1k eval/prefill 较短场景会覆盖太多 token，不利于判断 sparse 方法本身。因此后续改成 `10 + 10`。

## 3. 实现路径

### 3.1 PyTorch fast path

第一版 `qabs_fast_path` 的目标是去掉最开始的 full-history full-QK。

流程：

1. 根据 query abs 选通道。
2. partial-QK 产生 qabs candidate。
3. 和上一步 candidate/top2 做 union。
4. gather candidate K。
5. 在 candidate 内 full-QK rerank。
6. 加上 sink/recent/self。
7. gather final K/V。
8. 对 final token 做 attention。

这个版本没有 full-history full-QK，但仍然大量依赖 PyTorch 小算子：

```text
topk
bool mask
mask union
dense mask -> padded indices
gather
small matmul
softmax
```

因此它是 correctness/speed-trend prototype，不是真正的 serving kernel。

### 3.2 CUDA final attention kernel

后来新增了：

```text
src/qabs_cuda_kernels.py
--qabs_cuda_final_kernel
```

这个 CUDA extension 只融合最后一步：

```text
selected QK -> softmax -> weighted V reduction
```

它不融合 candidate 生成、topk、mask union、indices compaction 或 rerank。

因此它是 kernel 级优化的第一步，但不是完整 fused sparse attention。

### 3.3 共享 prefill cache

早期每个 mode 都重新跑一次 prefill，这会浪费大量时间，也会让速度结果难以比较。

后来新增：

```text
--reuse_prefill_cache true
--baseline_last true
```

当前默认行为：

1. dense prefill 只跑一次。
2. 每个 mode clone 同一份 prefill KV cache。
3. 每个 mode 从同一个 prefill logits 开始 eval/decode。
4. baseline 默认最后跑。

CSV 中新增：

```text
reuse_prefill_cache
shared_prefill_seconds
```

开启共享 prefill 后，`seconds` 表示：

```text
per-mode cache clone + eval/decode time
```

不再包含重复 prefill。

## 4. 实验结果

### 4.1 无 kernel 版本

设置：

```text
eval_tokens = 1000
protected_sink_tokens = 10
protected_recent_tokens = 10
qabs_fast_path = true
qabs_cuda_final_kernel = false
```

结果：

| mode | PPL | seconds | 相对 baseline PPL | 相对 baseline time |
|---|---:|---:|---:|---:|
| baseline | 26.3142 | 102.31 | 1.0000 | 1.00x |
| qabs8cand3reuse | 25.8683 | 151.99 | 0.9831 | 1.49x |
| qabs8cand7reuse | 25.8773 | 151.74 | 0.9834 | 1.48x |
| qabs16cand3reuse | 25.5103 | 153.85 | 0.9695 | 1.50x |
| qabs16cand7reuse | 25.7309 | 153.79 | 0.9778 | 1.50x |

结论：

- PPL 全部接近或好于 baseline。
- `qabs16cand3reuse` 的 PPL 最好，约比 baseline 低 `3.05%`。
- 速度仍然慢于 baseline，说明 PyTorch 小 kernel、mask compaction、gather 和动态分配开销很重。
- `cand3` 和 `cand7` 时间几乎一样，说明瓶颈不是 final selected token 数量，而是固定调度/索引开销。

### 4.2 共享 prefill + CUDA final kernel 版本

设置：

```text
eval_tokens = 1000
protected_sink_tokens = 10
protected_recent_tokens = 10
qabs_fast_path = true
qabs_cuda_final_kernel = true
reuse_prefill_cache = true
baseline_last = true
shared_prefill_seconds = 33.7023
```

结果：

| mode | PPL | seconds | qabs dim | candidate fraction | 相对 baseline PPL | 相对 baseline eval time |
|---|---:|---:|---:|---:|---:|---:|
| qabs8cand3reuse | 17.5000 | 119.20 | 8 | 0.03 | 0.9999 | 5.16x |
| qabs8cand7reuse | 17.5284 | 72.64 | 8 | 0.07 | 1.0015 | 3.15x |
| qabs16cand3reuse | 17.1648 | 73.09 | 16 | 0.03 | 0.9808 | 3.17x |
| qabs16cand7reuse | 17.4685 | 73.45 | 16 | 0.07 | 0.9981 | 3.18x |
| baseline | 17.5015 | 23.08 |  |  | 1.0000 | 1.00x |

结论：

- CUDA kernel 版本已经可以完整跑完。
- 质量最好的是 `qabs16cand3reuse`，PPL `17.1648`，比 baseline `17.5015` 低约 `1.92%`。
- `qabs8cand3reuse` 和 baseline 几乎持平。
- qabs 仍然比 baseline 慢约 `3.15x` 到 `5.16x`。
- `qabs8cand3reuse` 反而最慢，说明当前瓶颈不是最终 attention 的 selected token 数量，而是 candidate 生成、mask/indices 处理或首次 warmup/编译影响。

## 5. 当前判断

### 5.1 质量判断

从 PPL 看，qabs reuse 方向是有效的。

尤其是：

```text
qabs16cand3reuse
```

在两组结果中都显著好于 baseline，是目前最值得继续扩大样本验证的配置。

### 5.2 速度判断

当前实现还没有达到生产级 sparse attention 加速。

主要原因不是 final attention 本身，而是：

1. 每层每 token 都要做 qabs topk。
2. partial-QK 仍然扫完整 history。
3. candidate union 使用 dense bool mask。
4. `_indices_from_keep_mask` 会从 `[batch, heads, history]` dense mask 转 padded indices。
5. rerank 和 final gather 都是很多小 PyTorch kernel。
6. 动态 allocation 和 kernel launch overhead 很重。

因此，即使最终只保留 top2% token，当前 wall time 仍然慢于 dense baseline。

## 6. 下一步建议

短期建议先做 profiling，而不是继续盲目优化 final attention kernel。

需要给 qabs fast path 分段计时：

```text
qabs dim topk
partial-QK
candidate union
dense mask -> indices compaction
candidate rerank
final attention kernel
```

如果 profiling 证实 `_indices_from_keep_mask` 和 union 是主要瓶颈，下一步应该把 candidate 表示从 dense bool mask 改为 compact index list 或 bitset。

更合理的 kernel 化方向：

1. 当前 qabs candidate 直接输出 compact indices。
2. previous candidate/top2 使用 compact merge 或 bitset OR。
3. candidate 内 rerank 和 final attention 尽量融合。
4. 避免每层每 token 构造 `[heads, history]` dense bool mask。

## 7. 保留配置

后续建议重点保留两个配置：

```text
qabs16cand3reuse
qabs8cand7reuse
```

其中：

- `qabs16cand3reuse`：质量最好。
- `qabs8cand7reuse`：速度和质量折中较好，适合作为轻量候选。

同时保留 baseline 最后跑：

```text
MODES=qabs8cand7reuse,qabs16cand3reuse,baseline
```

并继续使用共享 prefill：

```text
REUSE_PREFILL_CACHE=true
BASELINE_LAST=true
```

## 8. Reuse union overlap 实验

为了判断 `candidate_union` 是否必须由三个集合组成，新增 overlap 统计：

```text
--qabs_overlap_stats true
```

当前三集合是：

```text
A = current raw qabs candidate
B = previous-step raw qabs candidate
C = previous-step final top2 token
```

输出文件：

```text
{mode}_reuse_overlap_summary.csv
{mode}_reuse_overlap_by_head.csv
```

重点看 summary 里的这些列：

```text
union_ab_fraction_of_union_all
previous_final_unique_fraction_of_union_all
previous_final_covered_by_current_previous_raw
```

含义：

- `union_ab_fraction_of_union_all`：如果只用 `A union B`，可以覆盖三集合 union 的多少比例。
- `previous_final_unique_fraction_of_union_all`：`C` 中有多少 token 是 `A union B` 没有覆盖的独有 token。
- `previous_final_covered_by_current_previous_raw`：`previous final top2` 中有多少比例已经被 `A union B` 覆盖。

如果：

```text
previous_final_unique_fraction_of_union_all 很低
previous_final_covered_by_current_previous_raw 接近 1
```

说明 `previous final top2` 这一项可能可以去掉，union 简化为：

```text
current raw qabs candidate
union previous-step raw qabs candidate
```

如果相反，说明 `previous final top2` 确实在传播经过 full-QK rerank 验证的重要 token，不能直接删除。

服务器运行脚本：

```bash
bash ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_qabs_overlap_server.sh
```

默认配置：

```text
PREFILL_TOKENS=10000
EVAL_TOKENS=1000
MODES=qabs8cand3reuse,qabs8cand7reuse,qabs16cand3reuse,qabs16cand7reuse,baseline
QABS_CUDA_FINAL_KERNEL=false
```
