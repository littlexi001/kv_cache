# 第 55 节：压缩时间、压缩后推理时间与 full attention 推理时间

日期：2026-07-03

## 0. 目标

前面的实验主要报告：

```text
compressed method forward time
full_raw forward time
speedup
```

但如果压缩是在推理时在线做的，压缩本身也要花时间。因此本节把时间拆开：

```text
compression time:
  把 older raw tokens 压成 summary memory 的时间。

prompt tokenization time:
  把 summary memory + recent raw 文本重新 tokenize 的时间。

forward time:
  Qwen3-0.6B 实际 forward 时间。

online total time:
  compression + tokenization + forward。

cached total time:
  tokenization + forward。
  这个对应 summary memory 已经离线预计算/缓存好的部署场景。
```

注意：

```text
full_raw 使用已有 token ids 直接 forward，不额外统计 prompt tokenize。
teacher cold compression 不包含 teacher model load 时间，只统计已加载 teacher 后生成 block summaries 的时间。
```

## 1. 新增脚本

```text
ymluo/projects/learned_hierarchical_summary_memory/src/run_compression_inference_timing.py
```

输出字段：

```text
avg_compression_seconds
avg_prompt_tokenize_seconds
avg_forward_seconds
avg_total_online_seconds
avg_total_cached_seconds
forward_speedup_vs_full_forward
online_speedup_vs_full_forward
cached_speedup_vs_full_forward
```

## 2. Learned extractive static_hier timing

### 2.1 设置

数据：

```text
10 个 public-domain texts
samples = 28
```

模型：

```text
Qwen3-0.6B
adapter = static_summary_lora_learned_longbooks_s1000_20260703
method = learned_static_hier
```

输出目录：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/compression_timing_learned_10texts_20260703
```

本地摘要：

```text
ymluo/projects/learned_hierarchical_summary_memory/outputs/compression_timing_learned_10texts_20260703
```

### 2.2 结果

| method | samples | PPL | input tokens | compression | tokenize | forward | online total | cached total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_raw | 28 | 24.8648 | 8320.0 | 0.0000s | 0.0000s | 0.5242s | 0.5242s | 0.5242s |
| learned_static_hier | 28 | 25.8648 | 2046.9 | 0.0201s | 0.0054s | 0.1124s | 0.1379s | 0.1178s |

速度：

| method | forward speedup | online speedup | cached speedup |
| --- | ---: | ---: | ---: |
| learned_static_hier | 4.66x | 3.80x | 4.45x |

解释：

```text
只看 Qwen forward:
  0.5242 / 0.1124 = 4.66x

如果在线做 learned extractive 压缩:
  0.5242 / 0.1379 = 3.80x

如果 summary memory 已经离线缓存:
  0.5242 / 0.1178 = 4.45x
```

learned extractive 的在线压缩成本很小：

```text
compression = 20ms
tokenize = 5ms
forward = 112ms
```

所以即使把在线压缩算进去，它仍然比 full_raw forward 快约 3.8x。

## 3. Teacher/generative summary：cached timing

### 3.1 设置

数据：

```text
4 texts
samples = 8
```

使用已有 teacher summary cache：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/teacher_summary_lora_4texts_s1000_20260703/teacher_summary_cache.jsonl
```

输出目录：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/compression_timing_teacher_cached_4texts_20260703
```

### 3.2 结果

| method | samples | PPL | input tokens | compression/cache lookup | tokenize | forward | online total | cached total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_raw | 8 | 23.0376 | 8320.0 | 0.0000s | 0.0000s | 0.5464s | 0.5464s | 0.5464s |
| teacher_static_hier | 8 | 25.9538 | 1247.0 | 0.0085s | 0.0042s | 0.0750s | 0.0878s | 0.0793s |

速度：

| method | forward speedup | online speedup | cached speedup |
| --- | ---: | ---: | ---: |
| teacher_static_hier | 7.28x | 6.23x | 6.89x |

解释：

```text
如果 teacher summaries 已经离线预生成：
  teacher_static_hier 的 forward 很快，约 75ms。

加上 cache lookup + prompt tokenize:
  total online = 87.8ms。

对比 full_raw forward 546ms:
  online speedup = 6.23x
  cached speedup = 6.89x
```

这说明 teacher summary 的部署价值主要来自“离线压缩、在线复用”。

## 4. Teacher/generative summary：cold online generation timing

### 4.1 设置

数据：

```text
4 texts
samples = 4
```

这里故意不使用已有 teacher cache，让 Qwen3-4B-Instruct 在线生成 block summaries。

输出目录：

```text
/home/fdong/ymluo/projects/learned_hierarchical_summary_memory/outputs/compression_timing_teacher_cold_4texts_1sample_20260703
```

### 4.2 结果

| method | samples | PPL | input tokens | compression | tokenize | forward | online total | cached total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_raw | 4 | 22.2322 | 8320.0 | 0.0000s | 0.0000s | 0.5842s | 0.5842s | 0.5842s |
| teacher_static_hier cold | 4 | 23.8925 | 1284.2 | 90.2119s | 0.0074s | 0.0939s | 90.3131s | 0.1013s |

速度：

| method | forward speedup | online speedup | cached speedup |
| --- | ---: | ---: | ---: |
| teacher_static_hier cold | 6.22x | 0.006x | 5.77x |

解释：

```text
只看 compressed forward:
  0.5842 / 0.0939 = 6.22x

如果在线调用 4B teacher 生成 summary:
  compression = 90.21s
  total online = 90.31s
  online speedup = 0.006x
```

也就是说，teacher/generative summary 绝对不能在每次 query 时 cold generate。它必须是：

```text
offline:
  raw block -> teacher summaries -> cache

online:
  load cached summary memory -> Qwen3-0.6B forward
```

## 5. 总结

### 5.1 learned extractive 可以在线压缩

learned extractive 的压缩很便宜：

```text
compression = 20ms
forward = 112ms
online total = 138ms
full_raw forward = 524ms
```

所以它即使在线做压缩，也有：

```text
online speedup = 3.80x
```

这对 streaming 或临时长上下文是有意义的。

### 5.2 teacher/generative summary 必须离线预计算

teacher cold generation 太慢：

```text
compression = 90s / sample
```

但如果 summary 已经缓存：

```text
forward = 75ms
total cached = 79ms
cached speedup = 6.89x
```

所以 teacher summary 的合理部署方式不是“query 来了再压缩”，而是：

```text
文档 ingest 阶段生成 summary memory；
推理阶段只读 summary memory。
```

### 5.3 和 full attention 的直接比较

| 方法 | 是否在线压缩 | compression | forward | total | 相对 full_raw |
| --- | --- | ---: | ---: | ---: | ---: |
| full_raw | 无 | 0.000s | 0.524s-0.584s | 0.524s-0.584s | 1.00x |
| learned_static_hier | 在线 | 0.020s | 0.112s | 0.138s | 3.80x |
| learned_static_hier | 预计算 | 0.000s | 0.112s | 0.118s | 4.45x |
| teacher_static_hier | cold 在线生成 | 90.212s | 0.094s | 90.313s | 0.006x |
| teacher_static_hier | 预计算/cache | 0.008s | 0.075s | 0.088s | 6.23x |

最终判断：

```text
learned extractive:
  在线压缩也划算。

teacher/generative:
  在线 cold generate 不划算；
  离线预计算后非常快。
```

