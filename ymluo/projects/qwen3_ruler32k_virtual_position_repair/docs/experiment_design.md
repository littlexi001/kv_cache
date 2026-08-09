# 实验设计

## 第一阶段：已知样例 smoke

模型为 Qwen3-8B、NF4 权重、BF16 计算；数据复用 Qwen tokenizer 生成的 RULER-32K seed 42 缓存。所有条件共享同一 prompt、一次前缀 prefill、同一 2% selector 规则和 greedy 解码。位置修复会改变后层 Query，因此后层的实际支持位置允许发生内生变化。

样例：

- `niah_multivalue_32768_0`：当前 postscore 能从 exact Top-2% 的 0.75 恢复到 1.00，用于检查修复会继续改善还是破坏。
- `niah_multikey_3_32768_0`：此前 blend25 将正确 UUID 后缀写错，用于检查位置修复的多 token 风险。

条件：

- `native_full`：原生 Full reference。
- `rope_top2`：exact post-RoPE Top-2%。
- `local_global_postscore`：alpha=0。
- `local_global_rephase25/50/75/100`：逐步移动到虚拟位置。

如果 25%--100% 的粗粒度冒烟扫描破坏已知 UUID 正例，则追加
2%、5%、10%、15% 的保守插值扫描。该追加扫描只用于诊断；若进入
26 样本扩展，必须在查看其余样本前冻结一个位置修复强度。

第一阶段通过：预算和重复位置审计错误为 0，且至少一个 alpha 在不损害 UUID 的情况下保持或提高 multivalue；否则不扩展。selection hash 记录完整轨迹，但不要求跨 alpha 相同。

## 第二阶段：26 条配对 pilot

若 smoke 通过，在刚才同一批 13 任务 × 2 条 RULER-32K 上运行。GPU6–7 各承担一个 shard，每个任务每卡 1 条。

主指标：RULER 官方 13 任务宏平均。辅助指标：

- paired score delta 与 bootstrap 95% CI；
- 改善/退化/不变样本数；
- NIAH 答案值 token attention mass；
- gold 与 non-gold 远程候选的平均 QK score delta；
- 完整 UUID/string-match，而不是只看首 token；
- 支持集 hash、预算、重复位置和旋转重构审计。

## 解释规则

- mass 与 score 同时上升：支持“位置相位压低了已召回证据”。
- mass 上升但 score 下降：说明虚拟位置也放大了干扰或破坏了后续层熟悉的相位结构。
- alpha 呈中间值最优：说明需要部分位置修复，而不是完全删除远程位置信息。
- 所有 alpha 不优于 0：只否定当前“动态、逐 token 压缩”的实现，不否定其他块级或训练期位置修复。
