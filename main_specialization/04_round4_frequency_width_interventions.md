# Round4 Intervention Evidence

## 0. 目的

本文件回答：

> Zipf 训练造成的 tail 落后，是不是不可逆的表征/参数空间损伤？

核心假设：

1. 如果 tail failure 主要来自训练曝光不足，那么 loss reweight 或 uniform fine-tune 应该显著恢复 tail。
2. 如果 tail failure 主要来自不可逆表征污染，那么这些 intervention 的恢复会很有限。
3. 如果 width 有独立作用，那么 intervention 后 h96 仍应优于 h64。

## 1. Zipf + inverse-frequency loss reweight

### 1.1 实验设定

- train distribution: Zipf, `alpha=1.3`
- loss weighting: `inverse_sqrt`
- eval distribution: uniform
- hidden size: 64 / 96

主要结果文件：

- `fdong/experiments/frequency-width-reweight-inverse_sqrt-analysis.json`
- baseline: `fdong/experiments/frequency-width-dense-five-analysis.json`

### 1.2 关键结果

| run | head loss | middle loss | tail loss | tail-head gap | tail acc |
|---|---:|---:|---:|---:|---:|
| baseline zipf h64 | 0.2413 | 0.4008 | 0.5111 | 0.2697 | 0.8972 |
| reweight h64 | 0.2443 | 0.3316 | 0.3763 | 0.1320 | 0.9036 |
| baseline zipf h96 | 0.2253 | 0.3641 | 0.4450 | 0.2197 | 0.9006 |
| reweight h96 | 0.2286 | 0.3075 | 0.3467 | 0.1181 | 0.9030 |

现象：

- h64 tail loss 改善 `0.1348`；
- h96 tail loss 改善 `0.0982`；
- head loss 基本没有明显损伤；
- tail-head gap 明显缩小。

LM margin 同步恢复：

| run | head margin | middle margin | tail margin |
|---|---:|---:|---:|
| baseline zipf h64 | 5.5623 | 4.5416 | 4.0865 |
| reweight h64 | 5.5160 | 4.9973 | 4.7285 |
| baseline zipf h96 | 6.3751 | 5.4523 | 5.0403 |
| reweight h96 | 6.4119 | 5.9043 | 5.6601 |

### 1.3 结论

结论：

> Tail failure 有很强的有效梯度曝光 / loss 权重成分。

支持证据：

- reweight 显著降低 tail loss；
- reweight 显著提高 tail margin；
- head loss 基本不受损。

同时：

> Reweight 没有完全替代 width。

支持证据：

- reweight 后 h96 tail loss `0.3467` 仍优于 h64 tail loss `0.3763`；
- reweight 后 h96 tail margin `5.6601` 仍优于 h64 tail margin `4.7285`。

## 2. Zipf 训练后 uniform fine-tune

### 2.1 实验设定

- source checkpoint: `frequency-width-dense-zipf-h{64,96}/1000.pth`
- fine-tune distribution: uniform
- fine-tune steps: 300
- eval distribution: uniform

主要结果文件：

- `fdong/experiments/frequency-width-zipf-to-uniform-analysis.json`

### 2.2 关键结果

| run | step | head loss | middle loss | tail loss | tail-head gap |
|---|---:|---:|---:|---:|---:|
| source zipf h64 | 1000 | 0.2413 | 0.4008 | 0.5111 | 0.2697 |
| fine-tune h64 | 50 | 0.2904 | 0.3112 | 0.3543 | 0.0638 |
| fine-tune h64 | 300 | 0.2731 | 0.2821 | 0.2913 | 0.0181 |
| source zipf h96 | 1000 | 0.2253 | 0.3641 | 0.4450 | 0.2197 |
| fine-tune h96 | 50 | 0.2811 | 0.2818 | 0.3047 | 0.0236 |
| fine-tune h96 | 300 | 0.2662 | 0.2691 | 0.2739 | 0.0076 |

现象：

- h64 tail-head gap 从 `0.2697` 在 50 steps 内降到 `0.0638`，300 steps 后降到 `0.0181`。
- h96 tail-head gap 从 `0.2197` 在 50 steps 内降到 `0.0236`，300 steps 后降到 `0.0076`。
- h96 恢复更快、更彻底。

Final LM margin：

| run | head margin | middle margin | tail margin |
|---|---:|---:|---:|
| fine-tune h64 | 6.0010 | 5.5525 | 5.4544 |
| fine-tune h96 | 6.9338 | 6.5281 | 6.4327 |

### 2.3 结论

结论：

> Zipf 造成的 tail 落后不是不可逆的。恢复 uniform 数据续训可以快速缓解甚至基本解决 tail gap。

支持证据：

- h64/h96 的 tail-head gap 都在 50 steps 内大幅缩小；
- 300 steps 后 gap 接近 uniform baseline；
- tail margin 显著恢复。

同时：

> Width 仍影响恢复速度和最终均衡程度。

支持证据：

- h96 50-step gap `0.0236` 明显小于 h64 的 `0.0638`；
- h96 300-step gap `0.0076` 小于 h64 的 `0.0181`。

## 3. 本文件结论

Round4 intervention 结论是：

1. Tail failure 很大程度来自训练分布和有效梯度曝光，而不是完全不可逆的表征损伤。
2. Loss reweight 和 uniform fine-tune 都可以显著恢复 tail。
3. Width 仍有独立作用：reweight 后 h96 仍更好，fine-tune 中 h96 恢复更快。

因此，当前机制应表述为：

> 训练动力学是主因；宽度是缓解 long-tail 学习瓶颈的结构条件，而不是唯一原因。
