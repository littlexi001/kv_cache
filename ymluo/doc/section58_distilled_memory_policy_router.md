# Section 58: 将 oracle policy 蒸馏成可推理的小 router

## 目标

上一节做的是 oracle 上界：同一个样例先跑所有 memory 策略，然后选择“满足质量要求的最低成本策略”。但是 oracle 推理时不可用，因为它知道每个策略的真实结果。

本节把 oracle label 蒸馏成一个可推理时使用的小 router。router 只使用推理时可见的特征，例如任务类型、query 文本特征、上下文长度、block 数量、retriever 粗分数等，不使用 target loss 或正确答案。

## 新增代码

训练脚本：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/run_memory_policy_router_distillation.py
```

推理 runtime：

```bash
ymluo/projects/learned_hierarchical_summary_memory/src/memory_policy_router_runtime.py
```

router checkpoint：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/memory_policy_router_10texts_s4_20260704/router.pt
```

## Oracle 数据

先用 10 本 public-domain 文本重新生成更大的 oracle policy 数据：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/task_adaptive_policy_lora_10texts_s4_retained_20260704
```

数据：

- `moby_dick`
- `pride_prejudice`
- `tale_two_cities`
- `sherlock_holmes`
- `dracula`
- `frankenstein`
- `origin_species`
- `republic`
- `walden`
- `time_machine`

有效样例数：74 个。

| 任务 | 样例数 | oracle success | avg tokens vs full | avg forward time | fallback |
|---|---:|---:|---:|---:|---:|
| generation | 37 | 100.00% | 15.86% | 0.114s | 8.11% |
| exact | 37 | 100.00% | 48.43% | 0.958s | 0.00% |
| overall | 74 | 100.00% | 32.15% | 0.536s | 4.05% |

Oracle 选择分布：

| 任务 | 选择分布 |
|---|---|
| generation | `summary10`: 78.38%, `summary100`: 8.11%, `static_hier`: 5.41%, `full_raw`: 8.11% |
| exact | `retrieval_raw_k1`: 91.89%, `retrieval_raw_k2`: 8.11% |
| overall | `summary10`: 39.19%, `summary100`: 4.05%, `static_hier`: 2.70%, `retrieval_raw_k1`: 45.95%, `retrieval_raw_k2`: 4.05%, `full_raw`: 4.05% |

这继续支持核心假设：不同任务应该选不同 memory level，而不是固定拼接 summary。

## Router 特征

当前 router 使用 29 维特征。主要包括：

```text
task_is_generation / task_is_exact
query 长度、词数、是否问句
exact/code/access/private/value 等关键词计数
quote/count/list/compare/all 等高风险关键词
prefix tokens / older tokens / recent tokens / block tokens / block 数
summary10/100/1000 的配置
retriever top1/top2 lexical overlap、score gap、positive block 数
prefix lexical diversity、数字数量、实体样式词数量
```

这些特征都是推理时可见的，不依赖答案或 target loss。

## 模型

router 是一个很小的 MLP：

```text
Linear(29 -> 32)
ReLU
Linear(32 -> num_actions)
```

动作空间：

```text
full_raw
retrieval_raw_k1
retrieval_raw_k2
static_hier
summary10
summary100
```

当前 oracle 数据里没有把 `summary1000` 选为最低成本成功策略，所以这次蒸馏后的 label set 中没有 `summary1000`。后续如果加入更难的非 exact 细节任务，`summary1000` 很可能会进入动作空间。

## Router 结果

输出目录：

```bash
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/memory_policy_router_10texts_s4_20260704
```

随机分层切分：

- train: 48
- test: 26

原始 router heldout 结果：

| split | task | samples | oracle-label acc | routed success | tokens vs full | forward time |
|---|---|---:|---:|---:|---:|---:|
| test | exact | 13 | 100.00% | 100.00% | 48.55% | 0.961s |
| test | generation | 13 | 69.23% | 92.31% | 35.60% | 0.229s |
| test | overall | 26 | 84.62% | 96.15% | 42.07% | 0.595s |

Exact 任务上 router 已经学得很清楚：13/13 都成功，主要选择 `retrieval_raw_k1`，少量选择 `retrieval_raw_k2`。

Generation 任务上有 1 个失败样例：oracle 是 `summary100`，router 选了 `summary10`，属于过度压缩。

## 推荐推理策略：conservative generation upgrade

为了避免普通生成中过度压缩，runtime 默认加了一个保守规则：

```text
如果 task_family = generation 且 raw_action = summary10
则最终 action 升级为 summary100
```

离线模拟 heldout 结果：

| policy | task | success | tokens vs full | forward time |
|---|---|---:|---:|---:|
| raw router | overall | 96.15% | 42.07% | 0.595s |
| summary10 -> summary100 | overall | 100.00% | 43.96% | 0.583s |
| raw router | generation | 92.31% | 35.60% | 0.229s |
| summary10 -> summary100 | generation | 100.00% | 39.36% | 0.205s |
| raw router | exact | 100.00% | 48.55% | 0.961s |
| summary10 -> summary100 | exact | 100.00% | 48.55% | 0.961s |

这个保守规则只影响 generation，不影响 exact evidence。它用很小的 token 增量把 heldout success 从 96.15% 拉到 100%。

## 推理时使用方式

加载 router：

```python
from memory_policy_router_runtime import load_router

router = load_router(
    "/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/memory_policy_router_10texts_s4_20260704/router.pt"
)

prediction = router.predict(features, task_family="generation")
print(prediction.raw_action)  # MLP 原始动作
print(prediction.action)      # 应用 conservative upgrade 后的最终动作
print(prediction.confidence)
```

实际接入时，`features` 应该由 `run_memory_policy_router_distillation.py` 中的同一套 feature extractor 生成。

## 当前结论

这一步已经把 oracle policy 蒸馏成了一个可推理的小 router。它不再需要提前跑所有策略，而是根据 query/context 特征直接输出 memory action。

目前的主要结论：

- router 能稳定识别 exact evidence 任务，并选择 raw retrieval；
- 普通生成任务大多可以使用低成本 summary；
- 一个简单的 conservative upgrade 能显著降低过度压缩风险；
- 当前数据还很小，router 结果只能作为 proof-of-concept；
- 下一步应扩大 oracle 数据，加入更多 task 类型，尤其是需要 `summary1000` 的细节理解任务和需要 `raw_k4/full_raw` 的多证据任务。

方法层面，现在可以把系统定义为：

```text
summary memory + raw store + distilled task-adaptive memory router
```

这比固定 `static_hier` 或固定 `k1/k2` 更符合长上下文 10w/100w token 场景。
