# Cross-language needle/filler server results

日期：2026-07-09

## 实验设置

在原始 `zh prompt + zh filler` 实验之外，本轮新增三组：

| 条件 | 插入事实与提示词 | 其它 haystack 文本 |
|---|---|---|
| `zh_prompt + en_filler` | 中文 | 英文 |
| `en_prompt + zh_filler` | 英文 | 中文 |
| `en_prompt + en_filler` | 英文 | 英文 |

中文事实与问题：

```text
小明今年是九岁。
问题：小明今年是几岁？
标准答案：九岁
```

英文事实与问题：

```text
Xiaoming is nine years old this year.
Question: How old is Xiaoming this year?
Target answer: nine years old
```

注意：英文条件下的 PPL 是对 `nine years old` 计算的条件 PPL；模型有时生成完整句子 `Xiaoming is nine years old...`，因此英文 PPL 和中文 `九岁` 的 PPL 不能直接横向比较，只适合同一语言条件内部比较。

## Phase 1: 8k/16k/32k, 5 seeds

三个条件均完整跑了：

```text
lengths = 8192, 16384, 32768
depths = 10, 50, 90
seeds = 0, 1, 2, 3, 4
compute_attention = true
```

### zh prompt + en filler

输出目录：

```text
/home/fdong/ymluo/projects/qwen3_long_needle_age/outputs/lang_zhprompt_enfiller_phase1_20260709
```

| length | depth | cases | accuracy | miss | wrong | answer PPL | evidence mass |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8k | 10 | 5 | 1.00 | 0.00 | 0.00 | 2.207167 | 0.02484347 |
| 8k | 50 | 5 | 1.00 | 0.00 | 0.00 | 2.784198 | 0.02398312 |
| 8k | 90 | 5 | 1.00 | 0.00 | 0.00 | 1.719993 | 0.02414639 |
| 16k | 10 | 5 | 0.60 | 0.00 | 0.40 | 4.742376 | 0.02008108 |
| 16k | 50 | 5 | 0.80 | 0.00 | 0.20 | 5.257407 | 0.01798205 |
| 16k | 90 | 5 | 1.00 | 0.00 | 0.00 | 1.699546 | 0.02706125 |
| 32k | 10 | 5 | 0.40 | 0.00 | 0.60 | 7.200064 | 0.01693000 |
| 32k | 50 | 5 | 1.00 | 0.00 | 0.00 | 4.754842 | 0.01967757 |
| 32k | 90 | 5 | 1.00 | 0.00 | 0.00 | 1.617127 | 0.03235651 |

### en prompt + zh filler

输出目录：

```text
/home/fdong/ymluo/projects/qwen3_long_needle_age/outputs/lang_enprompt_zhfiller_phase1_20260709
```

| length | depth | cases | accuracy | miss | wrong | answer PPL | evidence mass |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8k | 10 | 5 | 1.00 | 0.00 | 0.00 | 372.156684 | 0.01143598 |
| 8k | 50 | 5 | 0.80 | 0.20 | 0.00 | 478.480900 | 0.01153009 |
| 8k | 90 | 5 | 1.00 | 0.00 | 0.00 | 288.333236 | 0.01126145 |
| 16k | 10 | 5 | 1.00 | 0.00 | 0.00 | 557.907365 | 0.01237983 |
| 16k | 50 | 5 | 1.00 | 0.00 | 0.00 | 723.985269 | 0.00522085 |
| 16k | 90 | 5 | 1.00 | 0.00 | 0.00 | 181.917435 | 0.01638348 |
| 32k | 10 | 5 | 1.00 | 0.00 | 0.00 | 626.413494 | 0.01459480 |
| 32k | 50 | 5 | 1.00 | 0.00 | 0.00 | 649.691040 | 0.00692302 |
| 32k | 90 | 5 | 1.00 | 0.00 | 0.00 | 188.579573 | 0.01731449 |

### en prompt + en filler

输出目录：

```text
/home/fdong/ymluo/projects/qwen3_long_needle_age/outputs/lang_enprompt_enfiller_phase1_20260709
```

| length | depth | cases | accuracy | miss | wrong | answer PPL | evidence mass |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8k | 10 | 5 | 1.00 | 0.00 | 0.00 | 943.728206 | 0.00953842 |
| 8k | 50 | 5 | 1.00 | 0.00 | 0.00 | 643.760806 | 0.01065312 |
| 8k | 90 | 5 | 1.00 | 0.00 | 0.00 | 763.341471 | 0.00776893 |
| 16k | 10 | 5 | 1.00 | 0.00 | 0.00 | 1097.461470 | 0.00923206 |
| 16k | 50 | 5 | 1.00 | 0.00 | 0.00 | 1235.047017 | 0.00403580 |
| 16k | 90 | 5 | 1.00 | 0.00 | 0.00 | 443.431699 | 0.01091575 |
| 32k | 10 | 5 | 1.00 | 0.00 | 0.00 | 1607.332647 | 0.00898351 |
| 32k | 50 | 5 | 1.00 | 0.00 | 0.00 | 1314.995749 | 0.00370440 |
| 32k | 90 | 5 | 0.80 | 0.20 | 0.00 | 284.593637 | 0.00915495 |

## 64k seed0

64k 使用 YaRN factor 2.0，三个 depth 均为 seed0 smoke。

| condition | depth | accuracy | miss | wrong | answer PPL | evidence mass |
|---|---:|---:|---:|---:|---:|---:|
| zh prompt + en filler | 10 | 1 | 0 | 0 | 1.919504 | 0.01676228 |
| zh prompt + en filler | 50 | 1 | 0 | 0 | 2.139591 | 0.02002816 |
| zh prompt + en filler | 90 | 1 | 0 | 0 | 1.679035 | 0.02987793 |
| en prompt + zh filler | 10 | 1 | 0 | 0 | 227.801811 | 0.01371939 |
| en prompt + zh filler | 50 | 1 | 0 | 0 | 507.158998 | 0.00880761 |
| en prompt + zh filler | 90 | 1 | 0 | 0 | 205.546906 | 0.01656272 |
| en prompt + en filler | 10 | 1 | 0 | 0 | 631.106126 | 0.01241446 |
| en prompt + en filler | 50 | 1 | 0 | 0 | 731.458688 | 0.00570772 |
| en prompt + en filler | 90 | 1 | 0 | 0 | 382.008737 | 0.00790069 |

## 128k seed0

128k 使用 YaRN factor 4.0，三个 depth 均为 seed0 smoke。

| condition | depth | accuracy | miss | wrong | answer PPL | evidence mass |
|---|---:|---:|---:|---:|---:|---:|
| zh prompt + en filler | 10 | 0 | 0 | 1 | 12.430914 | 0.00873173 |
| zh prompt + en filler | 50 | 0 | 0 | 1 | 73.939031 | 0.00102112 |
| zh prompt + en filler | 90 | 0 | 0 | 1 | 54.228416 | 0.00228660 |
| en prompt + zh filler | 10 | 0 | 1 | 0 | 267.417987 | 0.00184719 |
| en prompt + zh filler | 50 | 1 | 0 | 0 | 53.568356 | 0.01830343 |
| en prompt + zh filler | 90 | 0 | 1 | 0 | 214.733867 | 0.00638995 |
| en prompt + en filler | 10 | 1 | 0 | 0 | 130.187011 | 0.00786673 |
| en prompt + en filler | 50 | 1 | 0 | 0 | 165.717384 | 0.02002239 |
| en prompt + en filler | 90 | 1 | 0 | 0 | 52.987647 | 0.02305208 |

## 256k feasibility

使用 4 张 RTX 3090 跑 `zh prompt + en filler, 256k@50%, seed0, compute_attention=false`，仍然 OOM：

```text
CUDA out of memory. Tried to allocate 976 MiB.
GPU 3 total 23.56 GiB, free 905 MiB.
This process already used 22.65 GiB.
```

因此本轮不再对其它两个语言组合重复 256k；它们的 token 长度和 KV 规模相同，当前 4x3090 资源不足。

## 观察

1. 中文事实/中文提示 + 英文背景在 8k-64k 上很强，PPL 和 evidence mass 都明显优于原始中文背景条件；英文 filler 反而降低了干扰。
2. 英文事实/英文提示 + 中文背景在 8k-64k 上 accuracy 很高，但 answer PPL 很大，说明模型常生成完整英文句子而不是精确的 `nine years old` 片段。
3. 英文事实/英文提示 + 英文背景在 8k-64k 上也很稳，128k seed0 三个位置全部正确，是本轮 128k 最好的语言组合。
4. 128k 下中文事实/中文提示 + 英文背景全部 wrong；这与原始中文背景下全部 miss 不同，说明失败模式会随背景语言变化。
5. evidence mass 对 128k 的成败仍有解释力：`en prompt + en filler` 在 128k 的 mass 约 `0.0079-0.0231`，而原始中文背景 128k 只有 `1e-4` 量级。

## 后续建议

1. 对 128k 三组语言条件补 seeds 1-4，确认 seed0 现象是否稳定。
2. 英文 PPL 应新增一个辅助目标 `Xiaoming is nine years old this year.`，避免只用 `nine years old` 低估模型概率。
3. 如果后续要比较跨语言 PPL，应该报告 token-normalized NLL 和目标答案形式，并避免直接把中文 `九岁` 和英文 `nine years old` 的 PPL 当作同一量纲。
