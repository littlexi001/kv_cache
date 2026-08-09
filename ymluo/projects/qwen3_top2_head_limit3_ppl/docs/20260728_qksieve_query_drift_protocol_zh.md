# QKSieve Query 校准与长生成漂移实验协议

## 1. 要回答的问题

冻结方法只使用 prompt 最后 8 个 Query 位置构造 Query 二阶矩。需要回答：

1. 1/4/8/16/32 个校准位置对变换和 mixed-bit allocation 的影响；
2. prompt-tail Query 与后续 decode Query 的协方差是否持续接近；
3. 固定变换和 bit allocation 后，score、top-k 与 attention mass 是否随生成位置退化；
4. 8 个 Query 是稳定默认值，还是只在当前短输出任务上有效。

## 2. 冻结路径与记录路径解耦

生产方法保持：

- `qksieve_fullprompt_auto_plain_fulltopk`；
- 最后 8 个 Query 用于真实 QK-balanced 校准；
- shrinkage 为 0.75；
- 240-bit 物理索引槽；
- `B(N)=min(N,1280,max(256,ceil(0.06N)))`；
- 无 rerank、router、recent/sink 和 Full fallback。

trace 可以额外记录最后 32 个 prompt Query，但只有最后 8 个被送入生产方法。
因此记录行为不会把冻结方法偷偷改成 32-Query 方法。

对应参数：

```text
--official_query_tail_tokens 8
--qk_trace_prefill_query_tail_tokens 32
```

## 3. 两种互补轨迹

### 3.1 自然生成轨迹

脚本：

```text
scripts/launch_qksieve_free_generation_drift_6gpu_20260728.sh
```

它在 LongBench 上按模型自然 EOS 生成，记录真实 QKSieve 执行路径中的
decode Query。它能回答真实答案生成期间是否发生 drift，但不能保证覆盖
1K–4K 位置。`summary.json` 只在对应 step 实际出现时才把 coverage
标成 `true`。

### 3.2 Teacher-forced 长续写轨迹

脚本：

```text
scripts/launch_qksieve_teacher_forced_drift_6gpu_20260728.sh
```

它在 32K 历史后继续处理 4096 个真实语料 token，覆盖六个主题：

- computer；
- sports；
- medicine；
- space；
- politics；
- religion。

该实验保证测到 step 1023/2047/4095，但它是 corpus continuation，
不是自然生成。论文必须将两类证据分开报告。

两个脚本都只允许物理 GPU 0–5，遇到 6/7 会直接退出。

## 4. 分析器

分析器：

```text
src/analyze_qksieve_query_drift_20260728.py
```

主要输出：

| 文件 | 内容 |
|---|---|
| `per_query.csv` | 每个 layer/head/Query/step 的 score、top-k 和 mass 指标 |
| `per_head_bucket.csv` | 按生成位置聚合的 covariance drift 与 allocation regret |
| `allocations.csv` | 1/4/8/16/32 样本下的 bit allocation、奇异值 gap 和子空间角 |
| `summary.json` | 协议、coverage、聚合结果和限制 |

位置桶固定为：

```text
0-63, 64-255, 256-1023, 1024-4095, 4096+
```

## 5. 测量指标

### 5.1 变换估计稳定性

- Query moment 到 32-sample 经验参考的 operator-norm error；
- 16/32/.../112 维 band 边界的 singular gap；
- 左右奇异子空间的 `sin(Theta)`；
- 理论扰动界是否满足 `epsilon_M < gap/2`；
- `max|AD^T-I|`。

32-sample 只是经验参考，不是真实总体协方差。

### 5.2 Allocation 漂移

- frozen allocation 与当前位置 held-out oracle allocation；
- band agreement；
- sampled diagonal qMSE regret；
- 完整 prompt Key 上的 diagonal/full qMSE；
- raw 与 QK-balanced 空间的 covariance drift。

held-out oracle 只用于分析，不能进入生成路径。

### 5.3 检索与输出代理

- centered score RMSE 和归一化 RMSE；
- score error range；
- active boundary gap；
- 真实 top-2% token recall；
- selected exact attention mass；
- oracle exact top-B attention mass；
- proxy/oracle omitted-mass ratio。

选择后仍使用原始精确 K/V；Query INT8 误差包含在 selection 指标中。

## 6. 固定 Key 的限制

step 0 的 Key state 删除当前 decode token 后，得到固定 prompt Key。
所有后续 Query 都在这份固定 Key 上分析，用来隔离 Query 分布与 allocation
漂移。它不包含后来生成的 Key，因此不能替代完整生成质量实验。

最终论文需要同时报告：

1. 固定 prompt Key 的机制分析；
2. 同一路径自然生成的 LongBench/RULER 质量；
3. 1K–4K teacher-forced drift；
4. 自然 EOS 覆盖范围。

## 7. 当前状态

代码、严格 schema 校验、CPU 合成 trace 和启动脚本已经完成。
120 项相关 CPU 测试通过；尚未运行任何正式 GPU drift 实验。
