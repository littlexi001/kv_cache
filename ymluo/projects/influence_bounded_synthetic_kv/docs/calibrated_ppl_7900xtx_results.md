# Influence-Bounded Synthetic KV 本地 PPL 初测

> 运行日期：2026-06-28  
> 机器：Windows + AMD Radeon RX 7900 XTX  
> 运行时：`.venv-qabs-rocm`，PyTorch ROCm `2.9.1+rocm7.2.1`  
> 模型：`Qwen/Qwen3-0.6B`  
> attention：`eager`

## 1. 已实现内容

本轮已经实现了真正的 calibration 版本，而不只是 oracle 或均值压缩。

流程：

1. dense prefill `512` tokens；
2. dense 跑一小段 calibration query；
3. 收集每层/head 的 calibration query 和最终 KV cache；
4. 为每层/head 生成少量 synthetic KV prototypes；
5. 用 ridge 初始化 synthetic K/V/bias；
6. 可选用 calibration full attention output 做 joint K/V/bias 优化；
7. 后续 eval 只使用：

```text
sink real KV + synthetic KV prototypes + recent real KV
```

不再使用真实 remote KV 做 attention。

主脚本：

```text
ymluo/projects/influence_bounded_synthetic_kv/src/run_calibrated_synthetic_kv_ppl.py
```

Windows 运行脚本：

```text
ymluo/projects/influence_bounded_synthetic_kv/scripts/run_calibrated_ppl_windows.ps1
```

## 2. 实验结果

### 2.1 Ridge 初始化版本

设置：

```text
dataset = science
prefill_tokens = 512
calib_tokens = 16
eval_tokens = 32
prototypes = 16
joint_steps = 0
protected_sink_tokens = 10
protected_recent_tokens = 10
```

结果：

| mode | PPL | PPL / baseline | eval seconds | fit seconds |
|---|---:|---:|---:|---:|
| baseline | 1.0167 | 1.0000 | 0.8721 | |
| synthkv_calibrated_ridge | 25.7186 | 25.2966 | 0.8515 | 6.4900 |

输出目录：

```text
ymluo/projects/influence_bounded_synthetic_kv/outputs/calibrated_ppl_science_p512_c16_e32_p16_dtypefix
```

### 2.2 Joint output-MSE 优化版本

设置：

```text
dataset = science
prefill_tokens = 512
calib_tokens = 32
eval_tokens = 32
prototypes = 32
joint_steps = 50
protected_sink_tokens = 10
protected_recent_tokens = 10
```

结果：

| mode | PPL | PPL / baseline | eval seconds | fit seconds |
|---|---:|---:|---:|---:|
| baseline | 1.0008 | 1.0000 | 0.8613 | |
| synthkv_calibrated_ridge + joint | 94.0742 | 94.0012 | 1.1395 | 13.3529 |

输出目录：

```text
ymluo/projects/influence_bounded_synthetic_kv/outputs/calibrated_joint_science_p512_c32_e32_p32_s50
```

## 3. 当前判断

这个方向仍然值得继续，但当前实现暴露了一个关键问题：

```text
单层/head 局部 output MSE 下降，不代表替换所有层 remote KV 后 PPL 稳定。
```

已经观察到：

- `mass oracle` 版本几乎贴住 baseline，说明“少量原型表示远程 attention contribution”本身有希望；
- 直接 mean synthetic KV 非常差；
- calibration ridge / joint 虽然能降低 calibration 层内 output MSE，但 PPL 仍然明显崩；
- 更大的 `calib_tokens/prototypes/joint_steps` 没有改善，反而更差，说明问题不是单纯容量不足。

更可能的问题是：

1. 每层独立拟合 full attention output，但下一层看到的是 synthetic 改过的 hidden state，分布漂移会逐层放大；
2. 当前 K 拟合用 bias 辅助预测 chunk logsumexp，虽然校准误差下降，但 decode query 的 attention routing 泛化差；
3. 一次性替换全部 28 层太激进，应该先做 layer ablation；
4. calibration query 太局部，无法覆盖后续 decode query 分布；
5. 只约束 output MSE，没有约束 logits KL 或最终 LM loss。

## 4. 下一步建议

优先做更小步的诊断，而不是继续盲目增大 prototype 数：

1. 只替换单层或少数层 remote KV，扫描哪几层最敏感；
2. 只替换后半层，保留前半层 dense remote KV；
3. 用 held-out calibration query 直接记录每层 output MSE 和 PPL 的相关性；
4. 把目标从每层 attention output MSE 改成短 unroll 的 logits KL；
5. 增加 influence bound，例如限制 synthetic key norm、bias 范围和最大 attention mass。

当前最重要的结论不是“方向失败”，而是：

```text
attention function approximation 需要考虑层间闭环和 decode 分布漂移。
仅做 per-layer independent reconstruction 不够。
```

## 5. Layer Ablation 新结果

为了定位 PPL 崩溃是否来自“全部层同时替换”，新增了 `--layer_sets` 支持。脚本现在可以在同一轮 calibration 后，分别评估不同层集合使用 synthetic KV，其余层仍保留 dense remote KV。

示例：

```powershell
powershell -ExecutionPolicy Bypass -File ymluo\projects\influence_bounded_synthetic_kv\scripts\run_calibrated_ppl_windows.ps1 `
  -OutputDir ymluo\projects\influence_bounded_synthetic_kv\outputs\layer_ablation_science_p512_c16_e32_p16_replay `
  -CalibTokens 16 `
  -EvalTokens 32 `
  -Prototypes 16 `
  -LayerSets 'all;0-6;7-13;14-20;21-27;0;7;14;21;27'
```

### 5.1 四段层扫描

设置：

```text
dataset = science
prefill_tokens = 512
calib_tokens = 16
eval_tokens = 32
prototypes = 16
joint_steps = 0
```

| synthetic layers | 层数 | PPL | PPL / baseline |
|---|---:|---:|---:|
| baseline | 0 | 1.0167 | 1.0000 |
| all 0-27 | 28 | 25.4779 | 25.0598 |
| 0-6 | 7 | 1.3632 | 1.3408 |
| 7-13 | 7 | 1.0466 | 1.0294 |
| 14-20 | 7 | 1.1179 | 1.0995 |
| 21-27 | 7 | 2.9838 | 2.9348 |

输出目录：

```text
ymluo/projects/influence_bounded_synthetic_kv/outputs/layer_ablation_science_p512_c16_e32_p16_replay
```

结论：

```text
不是所有层都同样敏感。
7-13 这一段相对最安全，0-6 和 21-27 更敏感。
```

### 5.2 敏感段单层扫描

单层结果显示，很多单层替换几乎不伤 PPL：

| layer | PPL / baseline |
|---:|---:|
| 0 | 1.0902 |
| 1 | 1.0178 |
| 2 | 1.0340 |
| 3 | 1.0042 |
| 4 | 1.0003 |
| 5 | 1.0013 |
| 6 | 1.0907 |
| 21 | 1.0205 |
| 22 | 0.9991 |
| 23 | 1.0024 |
| 24 | 0.9982 |
| 25 | 0.9988 |
| 26 | 1.0025 |
| 27 | 1.0011 |

输出目录：

```text
ymluo/projects/influence_bounded_synthetic_kv/outputs/layer_ablation_sensitive_singles_science_p512_c16_e32_p16
```

结论：

```text
单层替换大多可承受。
严重退化主要来自多层误差叠加，而不是每一层 synthetic KV 都完全不可用。
```

### 5.3 安全组合扫描

基于单层和四段结果，测试了一些更保守的组合：

| synthetic layers | 层数 | PPL | PPL / baseline |
|---|---:|---:|---:|
| baseline | 0 | 1.0167 | 1.0000 |
| 4,5 | 2 | 1.0190 | 1.0023 |
| 7-14 | 8 | 1.0460 | 1.0288 |
| 22-27 | 6 | 1.0679 | 1.0504 |
| 4,5,7-14 | 10 | 1.0529 | 1.0357 |
| 7-14,22-27 | 14 | 1.1919 | 1.1723 |
| 4,5,7-14,22-27 | 16 | 1.2684 | 1.2476 |

输出目录：

```text
ymluo/projects/influence_bounded_synthetic_kv/outputs/layer_ablation_safe_combos_science_p512_c16_e32_p16
```

当前最有价值的方向变成：

```text
不要全层替换。
先做 layer-selected synthetic KV，只替换中间较安全层。
```

短期推荐配置：

```text
synthetic layers = 4,5,7-14
synthetic layer count = 10 / 28
PPL ratio = 1.0357
```

这个结果已经比“全层替换 PPL ratio 25x”有本质改善，说明 layer selection 是必要组成部分。

## 6. 下一步方向更新

后续优先级应调整为：

1. 在更多文本和 needle PPL 上验证 `4,5,7-14` 是否稳定；
2. 扫描 `calib_tokens/prototypes` 对 `7-14` 和 `4,5,7-14` 的影响；
3. 对敏感层保留真实 remote KV，对安全层使用 synthetic KV；
4. 再考虑对 safe layers 做 joint output-MSE 或 logits-KL 微调；
5. 如果 safe-layer 策略稳定，再和 QABS/SparQ 在相同 remote KV budget 下比较。
