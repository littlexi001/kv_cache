# MHA 下 ValueSketch 补偿的速度成本

## 结论

在真实 MHA 模型上，ValueSketch 补偿不会破坏长上下文加速，但不是免费的。相同候选条件下，它使 QKSieve 的稳态整模型 decode 延迟增加约 `21.6%--24.1%`。QKSieve-Robust 在 32K、64K、128K 仍分别达到 `1.32x`、`2.22x`、`3.98x`；去掉补偿的 QKSieve-Fast 分别达到 `1.63x`、`2.73x`、`4.84x`。

结合既有质量消融，Fast 是速度上界和部署选项，Robust 才是论文中面向通用高保真的主方法。

## 方法

QKSieve 先用低比特 QK 索引选出最多 1,280 个候选 token/head，再读取候选的原始 FP16 K/V 做精确 attention。Fast 丢弃未选 token。Robust 使用 rank-16、block-256、INT4 ValueSketch 近似未选 token 的 softmax 分母和加权 Value 分子，并与候选结果合并。

实验固定 QK-balanced 坐标、Key 位宽、有限样本分位数、候选预算、原始 K/V 和模型输入。Fast 与 Robust 在 15 组 attention 测试中的阈值、候选数和候选集合全部完全一致。

## 实验

Attention 子系统使用一张 RTX 3090、MHA 32Q/32KV、head dimension 128，覆盖 8K--128K，三个随机种子，CUDA events 独立计时。真实 decode 使用 `Yarn-Llama-2-7b-128k`，FP16，32K 两卡、64K 三卡、128K 八卡；每组 greedy 生成 64 token，后 48 token 计算稳态均值。

### Attention 子系统

| 长度 | Fast 加速 | Robust 加速 |
|---:|---:|---:|
| 8K | 1.27x | 1.08x |
| 16K | 1.67x | 1.41x |
| 32K | 2.65x | 2.09x |
| 64K | 4.55x | 3.36x |
| 128K | 6.37x | 4.12x |

补偿的主要开销来自 selector 扫描时同时累计尾部概率质量和 Value 低秩系数。128K 时，Robust 比 Fast 多 `0.2035 ms/layer`，其中 `0.1765 ms/layer` 来自该扫描。

### 与 FIER 的同路径比较

FIER RTN-1 g32 与 Fast 使用相同 active-token 日程和相同精确 sparse-attention consumer。Robust 额外支付 ValueSketch 成本。数值均为三个 seed 的中位数；`Fast/FIER` 和 `Robust/FIER` 大于一表示 QKSieve 更快。

| 长度 | FIER ms | Fast ms | Robust ms | Fast/FIER | Robust/FIER |
|---:|---:|---:|---:|---:|---:|
| 8K | .1816 | .1327 | .1560 | 1.37x | 1.16x |
| 16K | .2951 | .1900 | .2250 | 1.55x | 1.31x |
| 32K | .4647 | .2324 | .2945 | 2.00x | 1.58x |
| 64K | .7329 | .2646 | .3576 | 2.77x | 2.05x |
| 128K | 1.3322 | .3722 | .5757 | 3.58x | 2.31x |

Fast/FIER 隔离的是 packed selector 成本；Robust/FIER 是包含尾部补偿的完整 attention 路径比较。该结果只支持当前审计实现和 RTX 3090，不等价于对未实测厂商优化实现的速度声明。

### 真实稳态 Decode

| 长度 | Full | Fast | Robust | Fast 加速 | Robust 加速 |
|---:|---:|---:|---:|---:|---:|
| 32K | 84.18 ms | 51.56 ms | 63.97 ms | 1.63x | 1.32x |
| 64K | 144.75 ms | 52.97 ms | 65.25 ms | 2.73x | 2.22x |
| 128K | 268.17 ms | 55.41 ms | 67.38 ms | 4.84x | 3.98x |

Robust 的一次性索引构建为 32K `1.375 s`、64K `1.488 s`、128K `1.839 s`。对只生成 64 token 的单次请求，含构建后的加速为 `0.81x/1.31x/2.15x`；若同一上下文在多轮问答或 agent 中复用，构建成本只支付一次，随后使用稳态加速。

## 解释

Fast 更快，但既有 32K/96K teacher-forced 消融显示它会出现文本级质量失败；Robust 将合并质量恢复到约 100%。因此补偿不是为了提高 selector top-k 召回，而是恢复大量未选低权重 token 的聚合 Value 贡献。

本次结果支持两个明确部署点：

1. `QKSieve-Robust`：论文主方法，优先质量稳健性；长上下文仍有明显加速。
2. `QKSieve-Fast`：无补偿速度消融或允许质量风险的部署模式。

## 边界与下一项不确定性

本报告只覆盖一个 MHA Llama 模型和 RTX 3090。下一项速度证据应冻结实现后，在 H100 上用相同口径复测，并增加至少一个 MHA 模型；不能用延迟分解替代真实运行。当前结果也没有证明 8K/16K 整模型会加速，attention 表只显示该长度下的子系统行为。

原始结果保存在本目录 `artifacts/`。Attention 正式结果来自 `attention/seed20260809.json`、`seed20260810.json`、`seed20260811.json`；整模型结果使用 `decode_clean32/`、`decode_clean64/` 和 `decode_strict/n131072/`。
