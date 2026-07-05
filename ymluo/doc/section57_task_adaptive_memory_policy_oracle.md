# Section 57: 按任务难度选择 summary level / raw retrieval 的 oracle 实验

## 背景

前面的 `static_hier` 只是一个静态 baseline：根据 block 距离选择 `sum10/sum100/sum1000`，而不是根据当前任务难度选择。这个设计不够理想。

真正想要的方法应该是：

```text
简单任务 -> sum10
稍难任务 -> sum100
更细节任务 -> sum1000
精确证据任务 -> raw retrieval
高风险任务 -> full raw fallback
```

因此这里做了一个 oracle-policy 实验。它不是训练出来的 router，而是先跑所有候选策略，然后选择“满足质量要求的最低成本策略”。这个实验的作用是验证：如果有一个足够好的策略路由器，理论上能不能比固定策略更省 token，同时保持质量。

## 实验脚本

新增脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_task_adaptive_memory_policy_eval.py
```

实验使用之前效果较好的 LoRA adapter：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/static_summary_lora_learned_longbooks_s1000_20260703/adapter
```

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/task_adaptive_policy_lora_4texts_s4_retained_20260704
```

数据：

- `moby_dick`
- `pride_prejudice`
- `origin_species`
- `republic`

每本书 4 个普通生成样例 + 4 个 exact evidence 样例，总计 32 个样例。

## 候选策略

普通生成任务候选：

```text
summary10
summary100
summary1000
static_hier
full_raw
```

exact evidence 任务候选：

```text
summary10
summary100
summary1000
static_hier
retrieval_raw_k1
retrieval_raw_k2
full_raw
```

## 成功标准

普通生成：

```text
nll <= full_raw_nll + 0.10
```

也就是压缩策略的 loss 不能比 full raw 差太多。

exact evidence：

```text
choice_correct = 1 且 answer_retained = 1
```

这里必须要求答案字符串确实还在 prompt 里。否则 summary-only 偶尔四选一猜对，不能算真正成功。

## 固定策略结果

### 普通生成

| 方法 | success | tokens vs full | mean PPL | forward time |
|---|---:|---:|---:|---:|
| full_raw | 100.00% | 100.00% | 22.52 | 0.507s |
| summary10 | 68.75% | 6.98% | 24.94 | 0.090s |
| summary100 | 81.25% | 12.35% | 24.38 | 0.070s |
| static_hier | 93.75% | 23.17% | 23.46 | 0.111s |
| summary1000 | 93.75% | 60.96% | 22.84 | 0.283s |

这个结果说明，普通生成里并不是每个样例都需要 `sum1000` 或 `static_hier`。很多样例 `summary10` 已经足够，部分样例需要升级到 `summary100/static_hier/full_raw`。

### exact evidence

| 方法 | success | choice acc | answer retained | tokens vs full | forward time |
|---|---:|---:|---:|---:|---:|
| full_raw | 100.00% | 100.00% | 100.00% | 100.00% | 2.137s |
| summary10 | 0.00% | 18.75% | 0.00% | 7.40% | 0.264s |
| summary100 | 0.00% | 37.50% | 0.00% | 12.76% | 0.277s |
| static_hier | 0.00% | 31.25% | 0.00% | 23.45% | 0.452s |
| summary1000 | 25.00% | 56.25% | 25.00% | 60.91% | 1.189s |
| retrieval_raw_k1 | 100.00% | 100.00% | 100.00% | 48.35% | 0.943s |
| retrieval_raw_k2 | 100.00% | 100.00% | 100.00% | 68.54% | 1.362s |

这里再次说明：exact evidence 任务不能依赖 summary-only。即使 `summary100` 的 choice accuracy 有 37.5%，answer retained 仍然是 0，说明它只是猜中了部分候选，不是可靠 recall。

## Oracle policy 结果

Oracle policy 每个样例选择最便宜的成功策略。

| 任务 | success | avg tokens vs full | avg forward time | full fallback |
|---|---:|---:|---:|---:|
| generation | 100.00% | 15.54% | 0.123s | 6.25% |
| exact | 100.00% | 48.35% | 0.943s | 0.00% |
| overall | 100.00% | 31.95% | 0.533s | 3.13% |

策略选择分布：

| 任务 | 选择分布 |
|---|---|
| generation | `summary10`: 68.75%, `summary100`: 12.50%, `static_hier`: 12.50%, `full_raw`: 6.25% |
| exact | `retrieval_raw_k1`: 93.75%, `retrieval_raw_k2`: 6.25% |
| overall | `summary10`: 34.38%, `summary100`: 6.25%, `static_hier`: 6.25%, `retrieval_raw_k1`: 46.88%, `retrieval_raw_k2`: 3.13%, `full_raw`: 3.13% |

这个结果和预期一致：

- 普通生成大多数时候只需要 `summary10`，少数样例升级到更细 summary 或 full raw。
- exact evidence 几乎都需要 raw retrieval。
- `k` 不应该固定，至少有一个样例需要从 `k1` 升级到 `k2`。
- 整体上，oracle 只用约 31.95% full raw tokens，就达到了 100% success。

## 对方法设计的影响

这个实验支持把方法从静态 memory 设计改成：

```text
Task-Adaptive Memory Policy
```

策略空间不应该是固定拼接：

```text
sum10 + sum100 + sum1000 + recent raw
```

而应该是动态选择：

```text
sum10
sum100
sum1000
static / mixed summary
raw retrieval with adaptive k
full raw fallback
```

最终 router 的目标函数应该是：

```text
在质量达标的前提下，选择最低成本策略。
```

训练 router 可以使用本实验生成的 oracle label：

```text
如果 summary10 达标 -> label = summary10
否则如果 summary100 达标 -> label = summary100
否则如果 summary1000/static_hier 达标 -> label = 对应 summary 策略
否则如果 raw_k1 达标 -> label = raw_k1
否则如果 raw_k2 达标 -> label = raw_k2
否则 -> full_raw
```

## 目前结论

这个实验验证了用户的核心想法：对于 10w / 100w 长上下文，不能静态拼接多级 summary，也不能固定取 `k1/k2` raw block。更合理的方案是让策略路由器根据任务难度自动选择 summary level、raw retrieval 数量和是否 fallback。

当前 oracle 结果说明这个方向有明显空间：在混合任务上，100% success 的情况下，平均输入只需要 full raw 的约 31.95%。下一步应该把 oracle policy 蒸馏成一个可推理时使用的小 router。
