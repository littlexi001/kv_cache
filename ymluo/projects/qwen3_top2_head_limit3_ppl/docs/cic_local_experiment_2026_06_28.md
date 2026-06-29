# CIC / Pairwise-CIC 本地实验记录（2026-06-28）

## 目标

本轮目标不是继续微调 `fulll25landmarkr4096s64attn` 这类固定层数 landmark baseline，而是寻找更像论文方法的方向：

```text
用模型输出质量的反事实影响，决定哪些层可以压缩、哪些层必须保留 full attention。
```

当前候选方法命名：

- `CIC`: Counterfactual Influence Cache
- `Pairwise-CIC`: 显式测量层间组合交互的 CIC
- `CIC-SKV` / `PCIC-SKV`: 后续推荐主线，即反事实预算 + 合成 KV 补偿 + rescue gate

当前实验仍使用 `recent=512 + landmark stride=64` 作为压缩 fallback。它不是最终创新点，只是用来验证“哪些层可压缩”这个反事实预算思想是否成立。

## 代码与数据

新增/使用的核心脚本：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_cic_layer_budget_local.py
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_cic_layer_budget_local.ps1
ymluo/projects/qwen3_top2_head_limit3_ppl/src/run_cic_combo_local.py
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_cic_combo_local.ps1
```

真实文本数据：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/data/war_and_peace_pg2600.txt
ymluo/projects/qwen3_top2_head_limit3_ppl/data/count_monte_cristo_pg1184.txt
```

本地环境：

```text
Windows + AMD Radeon RX 7900 XTX + ROCm PyTorch
Python: .\.venv-qabs-rocm\Scripts\python.exe
model: Qwen/Qwen3-0.6B
dtype: bfloat16
attn_implementation: eager
prefill_tokens: 4096
eval_tokens: 128
fallback: recent=512, stride=64
```

注意：Windows ROCm + eager PyTorch 的时间噪声较大，速度只看趋势；PPL 结论更可信。

## 已修复问题

`run_cic_layer_budget_local.py` 初版忘记安装 Qwen3 attention patch，导致 mode 实际没有生效，PPL 差值全为 0。现在已修复：

- 加载模型后调用 `install_qwen3_attention_patch()`
- 设置 `model.config.use_cache = True`
- 同一模型进程共享 prefill cache，减少实验时间

## 实验一：单层 CIC

### War and Peace

输出目录：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/local_cic_war4096_r512_patched
```

Baseline：

```text
2.0913s / PPL 31.9181
```

最安全单层：

| rank | layer | delta_ppl | speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0 | -0.1287 | +0.11% |
| 2 | 13 | +0.0263 | +19.80% |
| 3 | 7 | +0.0866 | +16.87% |
| 4 | 9 | +0.0986 | +4.58% |
| 5 | 8 | +0.1079 | +14.03% |
| 6 | 12 | +0.1149 | +16.92% |
| 7 | 6 | +0.1591 | +22.57% |

最危险层：

| layer | delta_ppl |
| ---: | ---: |
| 27 | +11.6000 |
| 4 | +4.8592 |
| 25 | +4.5981 |
| 26 | +3.9619 |
| 3 | +3.9248 |

贪心组合：

| compressed layers | delta_ppl | speedup |
| --- | ---: | ---: |
| `0` | -0.1287 | +24.74% |
| `0,13` | -0.4909 | +10.84% |
| `0,13,7` | +0.3944 | +15.86% |
| `0,13,7,9` | +0.7928 | +16.58% |
| `0,13,7,9,8,12` | +2.5052 | -9.79% |

结论：单层安全不代表组合安全，层间 coupling 很强。

### Monte Cristo

输出目录：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/local_cic_monte4096_r512_patched
```

Baseline：

```text
1.9760s / PPL 17.6653
```

最安全单层：

| rank | layer | delta_ppl | speedup |
| ---: | ---: | ---: | ---: |
| 1 | 2 | -0.3124 | +11.74% |
| 2 | 0 | -0.2622 | +4.45% |
| 3 | 7 | -0.2279 | +2.98% |
| 4 | 12 | -0.1129 | +5.97% |
| 5 | 13 | -0.0346 | +9.93% |
| 6 | 25 | -0.0046 | +12.27% |

贪心组合：

| compressed layers | delta_ppl | speedup |
| --- | ---: | ---: |
| `2` | -0.3124 | -4.93% |
| `2,0` | -0.3266 | -1.64% |
| `2,0,7` | -0.5592 | +6.83% |
| `2,0,7,12` | -0.8929 | +10.29% |
| `2,0,7,12,13,25` | -0.7557 | +5.16% |
| `2,0,7,12,13,25,1,15` | +0.2711 | +5.04% |

结论：Monte 上 CIC 更强，4 层压缩 `2,0,7,12` 同时更快、PPL 更好；但继续加层会崩。

## 实验二：Pairwise-CIC

新增脚本 `run_cic_combo_local.py` 支持显式组合、pairwise 组合、singleton 和 prefix 组合，用于验证层间交互。

### War top7 pairwise

输出目录：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/local_cic_pairwise_war_top7
```

Baseline：

```text
2.8169s / PPL 31.9181
```

强组合：

| compressed layers | delta_ppl | speedup |
| --- | ---: | ---: |
| `7,6` | -0.9366 | +4.00% |
| `0,13` | -0.4909 | +12.96% |
| `0,6` | -0.3366 | +42.81% |
| `7,12` | -0.3350 | +0.28% |
| `0,12` | -0.1608 | +41.17% |
| `0,7` | -0.1192 | +41.63% |
| `0,9` | -0.1128 | +42.79% |

关键观察：

- 单层排名的贪心 `0,13,7` 变差：`delta_ppl=+0.3944`
- pair `7,6` 明显优于贪心前缀，说明二阶交互必须建模
- `0,*` 组合经常快很多，但不同 partner 的 PPL 差异很大

### Monte top7 pairwise

输出目录：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/local_cic_pairwise_monte_top7
```

Baseline：

```text
2.1334s / PPL 17.6653
```

强组合：

| compressed layers | delta_ppl | speedup |
| --- | ---: | ---: |
| `2,0,7,12,13` | -1.0215 | -8.03% |
| `2,0,7,12` | -0.8929 | +4.34% |
| `2,0,7,12,13,25` | -0.7557 | +4.25% |
| `7,13` | -0.6571 | +3.08% |
| `2,7` | -0.4844 | +7.83% |
| `2,0` | -0.3266 | +10.69% |
| `0,25` | -0.3220 | +11.95% |

关键观察：

- Pairwise-CIC 明显比单层贪心更有信息量
- Monte 的 4 层组合 `2,0,7,12` 是目前最好的质量/速度折中
- 更大组合可能 PPL 更好但速度不稳，说明需要 kernel 级优化和更干净的 benchmark

## 实验三：跨文本迁移

将 War 上的 top4 `0,13,7,9` 迁移到 Monte：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/local_cic_transfer_war_top4_to_monte
```

| mode | seconds | PPL |
| --- | ---: | ---: |
| baseline | 1.8282 | 17.6653 |
| War top4 -> Monte | 2.1835 | 18.6727 |

结论：

- PPL 变差 `+1.0074`
- 速度也更慢
- 固定全局 layer list 不能作为最终方法

## 实验四：32-token 短校准迁移

目的：验证“prefill 后用 32 token 校准出安全层，然后应用到后续 128 token”是否可行。

### War

校准输出：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/local_cic_cal32_war4096
```

校准窗口内 top4：

| compressed layers | delta_ppl | speedup |
| --- | ---: | ---: |
| `7` | -0.2630 | +16.76% |
| `7,13` | -0.3822 | +28.08% |
| `7,13,18` | -0.7047 | +25.58% |
| `7,13,18,14` | -0.6966 | +22.65% |

后续 128-token 验证：

| setting | PPL | delta_ppl | seconds |
| --- | ---: | ---: | ---: |
| baseline | 26.4148 | 0 | 2.0473 |
| cal32 top2 | 28.2559 | +1.8411 | 2.2405 |
| cal32 top4 | 30.6177 | +4.2029 | 2.3637 |

### Monte

校准输出：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/local_cic_cal32_monte4096
```

校准窗口内 top4：

| compressed layers | delta_ppl | speedup |
| --- | ---: | ---: |
| `15` | -0.8572 | +39.87% |
| `15,16` | -1.5216 | +19.97% |
| `15,16,10` | -1.7099 | +34.17% |
| `15,16,10,18` | -1.6073 | +23.79% |

后续 128-token 验证：

| setting | PPL | delta_ppl | seconds |
| --- | ---: | ---: | ---: |
| baseline | 16.7183 | 0 | 1.7794 |
| cal32 top2 | 19.2677 | +2.5494 | 2.3762 |
| cal32 top4 | 21.3764 | +4.6581 | 2.3507 |

结论：

- 32-token 短校准在校准窗口内有效，但不能直接迁移到后续窗口
- 这不是坏结果，反而说明最终方法不能是一劳永逸的静态预算
- 需要 blockwise / online budget、rescue gate，或者用多窗口校准降低过拟合

## 实验五：强组合后续窗口验证

目的：验证在 `4096-4223` token 窗口里选出的强组合，迁移到 `4224-4351` token 是否仍然有效。

### War

输出目录：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/local_cic_pairwise_war_validate_0_13_next128
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/local_cic_pairwise_war_validate_7_6_next128
```

| setting | PPL | delta_ppl | seconds |
| --- | ---: | ---: | ---: |
| baseline | 15.0882 | 0 | 2.1762 |
| `0,13` | 15.5099 | +0.4217 | 2.2529 |
| baseline | 15.0882 | 0 | 1.8813 |
| `7,6` | 15.8089 | +0.7208 | 2.3007 |

### Monte

输出目录：

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/local_cic_pairwise_monte_validate_2_0_7_12_next128
```

| setting | PPL | delta_ppl | seconds |
| --- | ---: | ---: | ---: |
| baseline | 11.5267 | 0 | 1.8629 |
| `2,0,7,12` | 11.7493 | +0.2226 | 2.3883 |

结论：

- 同一文本内的下一个窗口也会退化
- 静态“选一组层然后长期使用”不够稳
- 更合理的系统应是：每个 block 重新估计风险，默认压缩，风险高时 rescue

## 当前方法判断

`fulll25landmarkr4096s64attn`：

- 创新性弱，本质是固定 3 层 landmark fallback
- 可作为 baseline 或 ablation，不适合作为主方法

单层 CIC：

- 证明层的反事实影响差异很大
- 但单层贪心组合不稳定

Pairwise-CIC：

- 明显更有研究价值
- 能发现单层排名找不到的强组合，例如 War 的 `7,6`
- 但静态迁移仍不稳

32-token 校准：

- 当前形式失败
- 失败原因不是 CIC 思想错，而是校准窗口过短、目标过拟合、没有 token/block 级 rescue

## 推荐论文主线

建议把最终方法定义为：

```text
PCIC-SKV: Pairwise Counterfactual Influence Cache with Synthetic KV Compensation
```

核心组件：

1. Pairwise counterfactual budget：用单层 + 二层组合影响建模 layer coupling
2. Blockwise online calibration：每个 block 或若干 block 更新预算，不固定全局 layer list
3. Rescue gate：低 margin / 高 entropy / 高压缩风险 token 临时回退 full attention
4. Synthetic KV compensation：非 full 层不只保留 uniform landmark，而是构造少量补偿 K/V 来近似被丢弃上下文
5. Robust objective：优化 `speedup - lambda * PPL_risk - mu * instability`，避免只在一个窗口过拟合

最小下一步实验：

```text
1. 实现 blockwise Pairwise-CIC：每 64/128 tokens 重新选择压缩层
2. 加入简单 rescue gate：当 baseline/压缩 logits margin 差异过大时回退 full
3. 跑 War/Monte 多窗口平均，不再只看一个 128-token 窗口
4. 对比 qabs8cand3attn、fulll25landmark、单层 CIC、Pairwise-CIC、Pairwise-CIC+rescue
```

目前最有希望发论文的不是某个静态 layer list，而是“反事实影响驱动的在线预算 + 层间交互 + 补偿/救援”这个框架。
