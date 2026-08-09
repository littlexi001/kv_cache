# 64K suppression-certificate safety smoke

这是 Qwen3-8B、64K、seed 0 的单样本 smoke，仅用于验证 64K grouped-GQA / read-only prefix 实现和决定是否值得扩大；不能作为正式统计结论。

## 有效性检查

- schema version：2。
- 64K untouched-native 路径按协议显式跳过：`native_baseline_status = skipped_context_exceeds_native_max`。
- 主比较基线：`instrumented_none`。
- 运行完成且未 OOM；prefix prefill 约 292.3 秒。
- certificate 的 BF16 最大重构误差为 1.0 QK logit，因此阈值附近的单点结论不可用；本次被选中干预的平均 score delta 约为 8.75--9.56，远大于该误差。

## 单 seed 结果

| 干预类别 | Gold PPL | Gold-vs-conflict margin 变化 |
|---|---:|---:|
| 无干预 | 376.885 | 0 |
| gold evidence | 533.490 | -0.375 |
| conflict evidence | 184.408 | +0.750 |
| lexical/format distractor | 343.029 | +0.125 |
| filler | 229.811 | +0.500 |

这里的干预是：每层、每个 query head 在指定类别中选择 suppression 最大的一个 sampled token，并把它搬到冻结的最优 anchor phase；总计 36x32=1152 个 score 位置。它不是部署方法，而是 safety counterfactual。

## 初步判断

单独的 pre/post-RoPE suppression 不是“真实证据证书”：在这个 smoke 中，gold 对 conflict 的 pooled AUROC 为 0.502，gold 对所有 non-gold 为 0.502，基本等于随机。更重要的是，增强 gold 反而降低答案 margin，而增强 conflict 或 filler 却提高了 margin。正式 8-seed 批次正在验证这是否可复现；若仍成立，应淘汰“只凭 suppression gap 触发 phase repair”的主方法。

