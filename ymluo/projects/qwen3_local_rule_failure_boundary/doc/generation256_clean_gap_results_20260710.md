# Clean gap generation@256 结果

日期：2026-07-10

## 1. 实验设置

这轮实验检验真实自由生成任务，而不是只看 candidate scoring。

设置：

```text
model = Qwen3-0.6B
lengths = 8k, 32k
depth = 50%
seeds = 0..4
distractor_count = 0
competitor_count = 0
chain_length = 4
rule_gap = 0, 512, 2048, 4096, 8192
max_new_tokens = 256
decode = greedy
attention = off
```

服务器输出：

```text
/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/clean_gap_generation256_qwen06_20260710
```

本地结果：

```text
ymluo/projects/qwen3_local_rule_failure_boundary/outputs/clean_gap_generation256_qwen06_20260710
```

## 2. 指标定义

这次同时保留原来的 candidate 指标，并新增生成诊断：

```text
candidate acc:
gold candidate 的 mean NLL 是否是所有候选中最低。

strict generation acc:
原始严格生成指标，只看 normalize 后第一个答案是否等于 gold。

contains_gold@256:
256 tokens 生成文本中是否出现过 gold answer。

final answer acc:
我按生成文本判断模型最后确定的答案是否是 gold。
判定规则是：优先取最后一个显式 answer/final answer；如果没有显式答案，再取最后一个已知 code mention。

wrong contamination:
生成文本中是否出现错误显式答案，或出现 conflict / competitor / distractor / random 候选 code。
```

## 3. 主结果

| length | gap | cases | cand acc | strict gen acc | contains gold@256 | first answer acc | final answer acc | wrong contam | mean margin |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8k | 0 | 5 | 1.00 | 0.00 | 0.60 | 0.00 | 0.20 | 0.80 | 2.917 |
| 8k | 512 | 5 | 1.00 | 0.00 | 0.40 | 0.00 | 0.00 | 0.60 | 2.775 |
| 8k | 2048 | 5 | 1.00 | 0.00 | 0.60 | 0.00 | 0.00 | 0.80 | 2.767 |
| 8k | 4096 | 5 | 1.00 | 0.00 | 0.80 | 0.00 | 0.20 | 0.80 | 3.061 |
| 8k | 8192 | 5 | 1.00 | 0.20 | 0.80 | 0.20 | 0.20 | 0.80 | 2.883 |
| 32k | 0 | 5 | 1.00 | 0.00 | 0.80 | 0.00 | 0.00 | 0.80 | 2.729 |
| 32k | 512 | 5 | 1.00 | 0.00 | 0.20 | 0.00 | 0.00 | 1.00 | 2.734 |
| 32k | 2048 | 5 | 1.00 | 0.00 | 0.60 | 0.00 | 0.00 | 0.80 | 2.622 |
| 32k | 4096 | 5 | 1.00 | 0.00 | 0.60 | 0.00 | 0.00 | 1.00 | 2.810 |
| 32k | 8192 | 5 | 1.00 | 0.00 | 0.60 | 0.00 | 0.20 | 0.80 | 2.703 |

按 length 汇总：

| length | cases | cand acc | strict gen acc | contains gold@256 | final answer acc | wrong contam |
|---:|---:|---:|---:|---:|---:|---:|
| 8k | 25 | 1.00 | 0.04 | 0.64 | 0.12 | 0.76 |
| 32k | 25 | 1.00 | 0.00 | 0.56 | 0.04 | 0.88 |

最终答案类别：

| length | gold | wrong | relevant intermediate | miss |
|---:|---:|---:|---:|---:|
| 8k | 3 | 19 | 2 | 1 |
| 32k | 1 | 22 | 2 | 0 |

## 4. 典型样本

### contains gold，但 final answer 错

```text
case = len8192_d50_seed0_dist0_low_gap0_chain4_comp0
gold = EE35-985
contains_gold@256 = 1
final_answer = 45-762
final_class = wrong
```

模型开头和显式答案都说 `45-762`，但后面复述规则链时出现了真正 gold `EE35-985`。所以 contains_gold 算对，final answer 算错。

### final answer 正确

```text
case = len8192_d50_seed1_dist0_low_gap0_chain4_comp0
gold = FE24-522
contains_gold@256 = 1
final_answer = FE24-522
final_class = gold
```

这个样本没有显式 `Answer:`，但最后一个已知 code mention 是 gold，因此 final answer 算对。

### 完全没有包含 gold

```text
case = len8192_d50_seed0_dist0_low_gap512_chain4_comp0
gold = EE89-932
contains_gold@256 = 0
final_answer = 446
final_class = wrong
```

模型反复输出 `Answer: 446`，没有在 256 tokens 中生成 gold。

## 5. 结论

这轮实验说明：

```text
clean gap 条件下，candidate acc 仍然是 100%，所以相关 rule 距离从 0 到 8192 tokens 没有造成候选排序失败。
但是自由生成任务非常不稳定：模型经常能在 256 tokens 中提到 gold，却不能把 gold 作为最后确定答案输出。
```

因此现在要区分两个结论：

```text
candidate scoring 结论：
无干扰、无竞争链时，gap 本身没有造成模型偏好错误答案。

真实生成结论：
即使没有干扰和竞争链，Qwen3-0.6B 也经常不能稳定按格式输出最终 code；
失败更多表现为输出控制、答案抽取、链式复述过程中的中间 code/局部数字污染，而不是 candidate preference 失败。
```

所以后续写法应避免把 candidate acc 直接等同于真实生成任务准确率。
