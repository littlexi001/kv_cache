# Value-mediated singleton intervention smoke

## 结论

**NO-GO。** 当前 NF4/BF16 replay 协议不能验证单个 attention-score 提升经 Value 写入预测最终答案 margin 的一阶因果闭环。

实验在 8K、seed 0 上对 64 个目标 `(layer, head, token)` 和 64 个匹配随机位置分别施加 `+0.25` score 干预。原始 case rows 没有重复；旧汇总中 10 行变 20 行，是单长度结果同时输出 `8192` 和内容相同的 `all`。runner 已修复为只有多长度时才生成 `all`。

## 共同 replay 漂移

| 路径 | FP32 pair margin | 原生 logit margin | Gold NLL |
|---|---:|---:|---:|
| Native baseline | -1.5193 | -1.500 | 1.7025 |
| Instrumented gradient baseline | -1.6687 | -1.750 | 1.9113 |

任何 singleton 干预前，instrumented 路径已相对 native 漂移 `-0.1494`。随后目标 singleton 的平均实际 margin 变化为 `+0.2373`，matched-random 也为 `+0.2310`；因此约 `+0.23` 主要是共同执行路径偏移，不能解释成目标 token 的特异因果效果。

当前 artifact 没有“完全相同 inference/custom-attention replay、但 epsilon=0”的 no-op baseline，无法唯一分解该偏移来自 gradient/inference 模式、`inputs_embeds` 路径、额外 diagnostic forward，还是 NF4/BF16 的数值跳变。原生 margin 只有约 0.125 的离散粒度，certificate reconstruction 最大误差也达到 0.375。

## 配对消除共同偏移后的结果

对每个 `pair_id` 使用：

$$
\Delta_{\mathrm{paired}}
=
m_{\mathrm{target}}-m_{\mathrm{random}}.
$$

| 类别 | 配对 actual gap | 配对 Spearman | 配对 sign accuracy |
|---|---:|---:|---:|
| All | +0.00636 | 0.379 | 60.9% |
| Gold | +0.00647 | 0.676 | 50.0% |
| Conflict | +0.03127 | 0.444 | 68.8% |
| Lexical | -0.02688 | 0.224 | 56.3% |
| Filler | +0.01457 | 0.435 | 68.8% |

全体只有 `34/64` 个 target 优于其 matched-random，配对 gap 的近似 95% 区间为 `[-0.0139, 0.0266]`；Spearman、sign accuracy 与 symmetric closure error 均未通过预设门槛。

该结果不否定解析导数恒等式本身，只说明当前量化 replay 的有限干预没有形成可复核、target-specific 的实际预测。正式重启前必须先加入同路径 epsilon=0 基线，并在更高精度模型上证明 target 显著优于 write-norm 匹配随机控制。
