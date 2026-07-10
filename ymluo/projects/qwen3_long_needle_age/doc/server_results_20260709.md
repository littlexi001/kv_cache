# Qwen3-0.6B 长上下文年龄 needle 服务器结果

日期：2026-07-09

## 运行环境

服务器：

```text
fdong@10.176.37.31
hostname: CISL-NF5468M5
GPU: NVIDIA RTX 3090 24GB
python: /home/fdong/miniconda3/envs/moe/bin/python
model: /home/fdong/hrj/prove/Qwen3-0.6B
```

项目路径：

```text
/home/fdong/ymluo/projects/qwen3_long_needle_age
```

本轮实现：

```text
src/run_long_needle_age.py
scripts/run_long_needle_age_smoke_server.sh
scripts/run_long_needle_age_phase1_server.sh
scripts/run_long_needle_age_phase2_server.sh
scripts/reclassify_existing_results.py
```

判分规则已修正为只看第一段直接答案。模型有时会输出：

```text
10岁
答案：9岁
```

这种情况按第一段答案算 wrong，不能因为后文出现 `9岁` 就算 correct。

## Phase 1: 8k/16k/32k native

输出目录：

```text
/home/fdong/ymluo/projects/qwen3_long_needle_age/outputs/long_needle_age_phase1_20260709
```

配置：

```text
lengths = 8192, 16384, 32768
depths = 10, 50, 90
seeds = 0, 1, 2, 3, 4
rope = native
compute_attention = true
```

结果：

| length | depth | cases | accuracy | miss | wrong | answer PPL | evidence mass |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8192 | 10 | 5 | 1.00 | 0.00 | 0.00 | 13.751542 | 0.01159901 |
| 8192 | 50 | 5 | 0.80 | 0.00 | 0.20 | 10.532277 | 0.01650042 |
| 8192 | 90 | 5 | 1.00 | 0.00 | 0.00 | 3.415447 | 0.01339390 |
| 16384 | 10 | 5 | 0.20 | 0.00 | 0.80 | 18.178558 | 0.01097204 |
| 16384 | 50 | 5 | 1.00 | 0.00 | 0.00 | 15.424324 | 0.01199565 |
| 16384 | 90 | 5 | 1.00 | 0.00 | 0.00 | 3.000427 | 0.02212904 |
| 32768 | 10 | 5 | 0.40 | 0.00 | 0.60 | 61.156675 | 0.00848741 |
| 32768 | 50 | 5 | 0.40 | 0.00 | 0.60 | 73.736207 | 0.01097817 |
| 32768 | 90 | 5 | 1.00 | 0.00 | 0.00 | 5.011567 | 0.02309377 |

观察：

1. `depth=90%` 最稳，8k/16k/32k 都是 5/5 correct。
2. `depth=10%` 随长度变长明显退化：8k 为 5/5，16k 为 1/5，32k 为 2/5。
3. 32k 的 `depth=10/50%` PPL 明显升高到 61-74，和 accuracy 下降一致。
4. evidence mass 在 `depth=90%` 通常更高，说明近尾部证据更容易被最后 query 关注。

## 64k seed0

输出目录：

```text
/home/fdong/ymluo/projects/qwen3_long_needle_age/outputs/long_needle_age_64k_seed0_20260709
```

配置：

```text
length = 65536
depths = 10, 50, 90
seed = 0
rope = YaRN factor 2.0
compute_attention = true
CUDA_VISIBLE_DEVICES = 0,1
```

结果：

| length | depth | cases | accuracy | miss | wrong | answer PPL | evidence mass |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 65536 | 10 | 1 | 0.00 | 0.00 | 1.00 | 6.462810 | 0.00808180 |
| 65536 | 50 | 1 | 0.00 | 0.00 | 1.00 | 5.782634 | 0.01373214 |
| 65536 | 90 | 1 | 1.00 | 0.00 | 0.00 | 3.464140 | 0.01349627 |

生成现象：

```text
64k@10%: 10岁 / 11岁 / 12岁...
64k@50%: 10岁, then later 答案：9岁...
64k@90%: 9岁...
```

因此 64k@50% 虽然 PPL 低且后文出现 `9岁`，但第一段答案是 `10岁`，按 wrong 处理。

## 128k seed0

输出目录：

```text
/home/fdong/ymluo/projects/qwen3_long_needle_age/outputs/long_needle_age_128k_seed0_20260709
```

配置：

```text
length = 131072
depths = 10, 50, 90
seed = 0
rope = YaRN factor 4.0
compute_attention = true
CUDA_VISIBLE_DEVICES = 0,1
```

结果：

| length | depth | cases | accuracy | miss | wrong | answer PPL | evidence mass |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 131072 | 10 | 1 | 0.00 | 1.00 | 0.00 | 111.273583 | 0.00019033 |
| 131072 | 50 | 1 | 0.00 | 1.00 | 0.00 | 80.729068 | 0.00011675 |
| 131072 | 90 | 1 | 0.00 | 1.00 | 0.00 | 92.638966 | 0.00011402 |

观察：

1. 128k 三个位置全部回答“无法确定”类答案。
2. PPL 全部很高，说明目标答案 `九岁` 的条件概率已经明显掉下去。
3. evidence mass 从 8k-64k 的约 `1e-2` 降到 `1e-4`，证据几乎没有被最后 query 稳定关注。

## 256k feasibility

输出目录：

```text
/home/fdong/ymluo/projects/qwen3_long_needle_age/outputs/long_needle_age_256k_feasibility_20260709
```

配置：

```text
length = 262144
depth = 50
seed = 0
rope = YaRN factor 8.0
compute_attention = false
CUDA_VISIBLE_DEVICES = 0,1
```

结果：当前服务器资源下 OOM，未得到质量指标。

错误摘要：

```text
CUDA out of memory. Tried to allocate 928 MiB.
GPU 1 total 23.56 GiB, free 905 MiB.
This process already used 22.65 GiB.
```

解释：即使关闭 evidence attention mass，256k full-context KV cache 在当前两张 3090 且已有其他进程占用的情况下仍不足。需要更空的多卡窗口、80GB GPU，或显式 CPU/offload 策略。

## 当前结论

1. Qwen3-0.6B 在这个中文 needle 任务上存在明显位置偏置：证据越靠近尾部越容易回答正确。
2. 原生 32k 内并不稳定，尤其 `depth=10%` 和 `depth=50%` 会明显失败。
3. 64k 用 YaRN factor 2 后，seed0 只有尾部 `90%` 成功；前中部生成会偏向 `10岁/11岁/12岁` 这类错误模式。
4. 128k 用 YaRN factor 4 后，seed0 三个 depth 全部 miss，且 evidence mass 降到 `1e-4` 量级。
5. 当前结果支持“长上下文失败主要来自证据 attention mass 下降 / 证据检索失败”的解释，尤其 128k 的 PPL 与 attention mass 同步恶化。

## Pivot 表

说明：8k/16k/32k 是 5 个 seed 的平均；64k/128k 当前只有 seed0，先作为长上下文 smoke。

### Accuracy

| length | 10% | 50% | 90% |
|---:|---:|---:|---:|
| 8k | 1.00 | 0.80 | 1.00 |
| 16k | 0.20 | 1.00 | 1.00 |
| 32k | 0.40 | 0.40 | 1.00 |
| 64k | 0.00 | 0.00 | 1.00 |
| 128k | 0.00 | 0.00 | 0.00 |

### Answer PPL

| length | 10% | 50% | 90% |
|---:|---:|---:|---:|
| 8k | 13.751542 | 10.532277 | 3.415447 |
| 16k | 18.178558 | 15.424324 | 3.000427 |
| 32k | 61.156675 | 73.736207 | 5.011567 |
| 64k | 6.462810 | 5.782634 | 3.464140 |
| 128k | 111.273583 | 80.729068 | 92.638966 |

### Evidence Attention Mass

| length | 10% | 50% | 90% |
|---:|---:|---:|---:|
| 8k | 0.01159901 | 0.01650042 | 0.01339390 |
| 16k | 0.01097204 | 0.01199565 | 0.02212904 |
| 32k | 0.00848741 | 0.01097817 | 0.02309377 |
| 64k | 0.00808180 | 0.01373214 | 0.01349627 |
| 128k | 0.00019033 | 0.00011675 | 0.00011402 |

## 后续建议

1. 补跑 64k/128k 的 seeds 1-4，得到更稳的统计结论。
2. 256k 需要等待 GPU 更空或换 80GB 卡；当前 2x3090 不够。
3. 可以新增一个更强输出约束 prompt，例如“只输出年龄，不要输出多轮答案”，避免生成 `10岁\n答案：9岁` 这种循环格式。
4. 后续 KV 方法评估时，应优先看 `depth=10/50%`，因为这些位置最容易暴露长程检索失败。
