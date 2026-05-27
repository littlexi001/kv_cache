# Candidate Design: Expert Input

Expert input 决定被选中的 expert 实际处理哪一份 token state。它不一定要和 router input 相同。

1. **Full residual token vector：** 让 router 使用更 feature-selective 的输入，但 expert 仍处理完整的 `attention output + residual`。这是当前最稳的方案：它保留完整预测信息，同时允许 gate 在更合适的表征空间中做分发。
2. **Attention output without residual：** 让 expert 只处理 attention 聚合出的信息。它有时能提升 attention bucket 与 expert bucket 的重合度，但 NTP 通常不如 full residual expert input 稳定。
3. **Layer input / q / k / v：** 让 expert 处理更早期或更局部的投影表征。已有 synthetic 结果显示，这类 expert input 往往明显伤 NTP，因此不应作为当前主线。
4. **Same as router input：** router 和 expert 使用同一表征。这个设计更“纯”，但容易把 routing 诊断问题和 expert 表达能力问题混在一起，实验解释更困难。
