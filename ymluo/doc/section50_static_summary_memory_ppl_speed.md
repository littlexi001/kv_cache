# 第 50 节：Static Summary Memory 用于普通文本生成的 PPL 与速度

日期：2026-07-03

## 0. 目标

本节把“第一种方法”放到普通语言模型续写/PPL 场景里测试：

```text
raw 历史上下文 -> 离线 static summary memory
普通 next-token prediction 不使用 query
用 summary memory + recent raw text 预测后续 tokens
比较 PPL 和 forward wall time
```

这里的 summary 是普通静态摘要，不根据后续 eval tokens 或特殊问题生成。

## 1. 新增脚本

```text
ymluo/projects/learned_hierarchical_summary_memory/src/run_static_summary_ppl_speed.py
```

服务器输出：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/static_summary_ppl_speed_8k_16x2_20260703
```

本地输出：

```text
ymluo/projects/learned_hierarchical_summary_memory/outputs/static_summary_ppl_speed_8k_16x2_20260703
```

## 2. 实验设置

模型：

```text
Qwen3-0.6B
model = /home/fdong/hrj/prove/Qwen3-0.6B
dtype = float16
attention = sdpa
GPU = RTX 3090
```

文本：

```text
War and Peace
Count of Monte Cristo
```

PPL 设置：

```text
prefill_tokens = 8192
eval_tokens = 128
samples_per_dataset = 16
datasets = 2
total evaluated samples = 32
block_tokens = 2048
recent_tokens = 512
```

PPL 计算方式：

```text
输入 = compressed_context + 原始后续 eval tokens
loss 只计算 eval tokens
```

## 3. 方法

| 方法 | 含义 |
| --- | --- |
| full_raw | 完整 8192 raw prefix |
| recent_only | 只保留最近 512 raw tokens |
| static_sum10 | older blocks 全部用 summary10，加 recent raw |
| static_sum100 | older blocks 全部用 summary100，加 recent raw |
| static_sum1000 | older blocks 全部用 summary1000，加 recent raw |
| static_hier | 远处 block 用 summary10，中间用 summary100，最近的 older block 用 summary1000，再加 recent raw |

summary 生成方式：

```text
summary10:
  block 的 top keywords

summary100:
  keywords + named entities + high-scoring sentences

summary1000:
  high-scoring sentences up to 900 words
```

这是 extractive static summary，不是 query-conditioned summary，也不是训练好的 learned summarizer。

## 4. 总体结果

两个文本合并后：

| 方法 | PPL | avg input tokens | token ratio | avg forward sec | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_raw | 23.8765 | 8320.0 | 100.00% | 0.3927 | 1.00x |
| recent_only | 26.8514 | 640.0 | 7.69% | 0.0360 | 10.91x |
| static_sum10 | 27.9869 | 711.2 | 8.55% | 0.0353 | 11.13x |
| static_sum100 | 27.1094 | 1207.9 | 14.52% | 0.0434 | 9.06x |
| static_hier | 26.3521 | 2160.3 | 25.97% | 0.0808 | 4.86x |
| static_sum1000 | 25.9121 | 5524.2 | 66.40% | 0.2305 | 1.70x |

## 5. 分文本结果

### War and Peace

| 方法 | PPL | avg input tokens | speedup |
| --- | ---: | ---: | ---: |
| full_raw | 23.6519 | 8320.0 | 1.00x |
| recent_only | 27.3430 | 640.0 | 11.17x |
| static_sum10 | 28.5293 | 710.0 | 11.37x |
| static_sum100 | 27.9003 | 1183.4 | 9.44x |
| static_hier | 25.9837 | 2116.6 | 5.09x |
| static_sum1000 | 25.6505 | 5394.2 | 1.78x |

### Count of Monte Cristo

| 方法 | PPL | avg input tokens | speedup |
| --- | ---: | ---: | ---: |
| full_raw | 24.1032 | 8320.0 | 1.00x |
| recent_only | 26.3686 | 640.0 | 10.65x |
| static_sum10 | 27.4548 | 712.3 | 10.88x |
| static_sum100 | 26.3410 | 1232.4 | 8.68x |
| static_hier | 26.7257 | 2204.1 | 4.64x |
| static_sum1000 | 26.1764 | 5654.1 | 1.64x |

## 6. 结论

普通文本生成 PPL 和 QA/retrieval 不一样：

```text
1. full_raw 仍然是 PPL 最好的方法。
   这符合预期，因为 next-token prediction 对原文局部细节高度敏感。

2. static summary memory 可以换速度，但会损失 PPL。
   静态摘要保留主题/实体/重要句子，但不能完整保留逐词风格和局部句法。

3. static_sum1000 最接近 full_raw：
   PPL 25.91 vs 23.88，速度 1.70x。

4. static_hier 是更有意思的折中：
   PPL 26.35，速度 4.86x，输入 token 约 26% full。

5. recent_only 已经很强。
   普通短程续写主要依赖最近上下文，所以只保留 recent raw 能拿到不错 PPL 和最高速度。
```

这说明 static summary memory 更适合：

```text
长程语义一致性
实体/主题/事实 recall
长上下文 QA / retrieval / summarization
```

它不天然适合直接替代 raw context 做严格 next-token PPL，因为 PPL 会惩罚所有被摘要丢掉的细粒度 token 信息。

## 7. 下一步

如果目标是普通生成任务中的 PPL 和速度都更好，需要进一步做：

```text
1. recent raw 必须保留。
2. static summary 只补充远程语义，而不是替代所有历史 raw。
3. summary token 应该用训练过的 summarizer 生成，而不是当前的 extractive heuristic。
4. 可以训练模型适应这种输入格式：
   [memory summaries] + [recent raw] -> next-token prediction
```

当前实验已经说明：不训练模型直接换上下文格式，summary memory 能加速，但 PPL 会有可见损失。

