# Real-Corpus Training Effect

问题：模型结构与训练范式如何影响 specialization？

当前 real-corpus 结论：

1. **Load-balance loss：** 它能显著提升 expert usage 的均匀性，但不一定直接带来 feature specialization。依据是：加入 load-balance 以后，effective expert count 明显上升，说明流量分布被显著拉平；但与此同时，routing 与 feature 对齐相关的指标变化很小，expert purity 也没有同步提升。也就是说，load-balance loss 主要改变的是“token 是否更平均地分到各个 expert”，而不是“expert 是否更清楚地按 feature 分工”。
2. **残差链接：** 残差在这里更像是在帮助 gate，而不是干扰 gate。依据是：当 gate 使用标准的 residual-plus-normalized 表征时，feature 更容易被线性读出，最终 routing 与 feature 的对齐也更好；而当 gate 只看 pure attention output 时，这两个结果都会下降。也就是说，在当前 ordinary MoE 设定里，残差路径并没有明显削弱 gate 对 feature 的识别，相反，它更可能给 gate 提供了一个更容易利用的输入表示。
3. **Attention：** 没有证据表明存在某个 head 会让 token 几乎只在同一 feature 内部 attend。依据是：在本地 `qwen3-0.6B` 的正式 attention 分析里，用整层 attention output 做 feature probe，最好的 layer 也只有 `0.0688`，最好的单头 probe 只有 `0.0656`，整体绝对值仍然偏低；同时，直接看 attention pattern 时，表现最强的 head 对同 feature token 的偏好仍然很弱，而且这种偏好只在大约 `21%` 的位置上出现，远达不到“某个 head 基本只在同一 feature 内 attend”的程度。也就是说，当前 attention 最多只能说明它弱地捕捉到了一部分 feature relation，还不能说明它已经形成了清晰而强的 feature-internal attention structure。
