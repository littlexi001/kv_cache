# Value-mediated causal closure v2: unquantized BF16 smoke

> **状态：已被 8-seed 复核取代。** 正式统计与可引用口径见 `../value_mediated_singleton_v2_bf16_noop_8seed_gpu67/README.md`；本文件只保留最初的 2-seed smoke 记录。

## 结论

**机制证据为 Preliminary GO；不是部署方法。** 严格同路径 `epsilon=0` no-op 消除了旧 NF4 smoke 的共同 replay 漂移；在两个 8K seeds、8 个 target singletons 上，一阶预测与真实答案 margin 变化高度一致：Pearson `0.988`、Spearman `0.881`、sign accuracy `75%`。matched-random 的 Pearson 只有 `0.184`、Spearman `0.286`。

这说明，在小幅单 score 干预的局部范围内，模型的精确梯度能够预测“该 attention score 经 Value/residual 写入后会怎样改变最终答案 margin”。它支持跨层因果链中的一个关键局部环节，但样本仍太少，且 candidate ranking 使用 answer-margin gradient，因此不能成为推理时 selector。

## 实验协议

- Qwen3-8B，**未量化 BF16**；
- 8K，seed 0/1，物理 GPU 6/7；
- 每个 seed 从 gold、conflict、lexical、filler 各冻结 top-1 target；
- 每个 target 配一个同 layer/head/class 的随机 token；
- 所有 singleton 都只把一个 pre-softmax score 增加 `0.25`；
- 每个 case 先运行完全同代码路径的 `epsilon=0` no-op；
- 所有实际 delta 只相对 no-op 计算。

两个 case 的 replay audit 均通过：no-op 恰好一条且改动计数为 0；每个 target/random pair 完整且位置不同；每条 intervention 只改一个 score；全部 delta 由 no-op 重算；prefix cache 保持不变。

## 共同路径漂移被消除

| Seed | Native margin | Instrumented margin | No-op margin | No-op − instrumented | No-op − native |
|---:|---:|---:|---:|---:|---:|
| 0 | 2.0662 | 2.0508 | 2.0508 | **0.0000** | -0.0154 |
| 1 | 9.8780 | 9.7942 | 9.7942 | **0.0000** | -0.0838 |

no-op 精确复现 instrumented baseline，因此旧实验中 target/random 共同出现约 `+0.23` 的现象确实是比较了不同执行路径，而不是单 token 干预的真实因果效果。native 与 instrumented 仍有少量 BF16 路径差异，但它不再进入 intervention delta。

## 一阶预测闭合

| Plan | n | Mean predicted Δmargin | Mean actual Δmargin | Pearson | Spearman | Sign | Mean abs closure error |
|---|---:|---:|---:|---:|---:|---:|---:|
| Target | 8 | -0.03768 | -0.03835 | **0.988** | **0.881** | 75% | 0.0251 |
| Matched random | 8 | +0.00181 | -0.01161 | 0.184 | 0.286 | 75% | 0.0212 |

target 的平均绝对实际效应为 `0.0876`，random 为 `0.0203`，即 target 约大 `4.31×`；8 个配对中有 7 个 target 的绝对效应更大。这个差异符合 candidate ranking 使用 `|suppression × exact sensitivity|` 的设计。

逐点结果：

| Seed | Class | Predicted | Target actual | Random actual |
|---:|---|---:|---:|---:|
| 0 | Gold | -0.0466 | -0.0700 | -0.0101 |
| 0 | Conflict | -0.1070 | -0.1192 | -0.0285 |
| 0 | Lexical | -0.0216 | -0.0128 | -0.0357 |
| 0 | Filler | -0.0014 | +0.0424 | -0.0087 |
| 1 | Gold | +0.1045 | +0.1283 | -0.0420 |
| 1 | Conflict | -0.2352 | -0.2875 | +0.0242 |
| 1 | Lexical | +0.0051 | +0.0264 | +0.0107 |
| 1 | Filler | +0.0008 | -0.0144 | -0.0027 |

两个 conflict target 都降低 gold-vs-conflict margin；但两个 gold target 一正一负。这再次说明“token 属于真实证据”不等于“提高它在任意 layer/head 的 attention 一定帮助答案”。真正有预测力的是该具体 score coordinate 的下游敏感度。

## 主张边界与下一步

当前结果可以支持：

> 对同一路径、未量化 BF16 的小幅 singleton intervention，局部一阶 `d margin / d score` 可以预测 Value/residual 通道对最终答案 margin 的真实影响。

当前不能支持：

- 完整的 phase→QK→mass→Value→margin 中介比例已经闭合；
- 该量在不同长度、模型和自然任务上稳定；
- answer-gradient ranking 是可部署 selector；
- gold span 中所有 token 的作用方向相同；
- 该结果证明了一种新的 RoPE 方法。

正式论文至少应扩到 8 seeds、8K/32K、多模型，并加入 phase-only、等价 additive-logit、mass-preserve、Value-write patch 与 residual patch。`certificate_reconstruction_error_max` 仍为 0.25/0.375，也应在 FP32 小模型或 FP32 attention-score control 中继续压低。

数据：`merged/case_rows.csv`、`merged/singleton_prediction_summary.csv`、`merged/value_samples.csv`；逐 case 审计在两个 shard 的 `raw/*_result.json`。
