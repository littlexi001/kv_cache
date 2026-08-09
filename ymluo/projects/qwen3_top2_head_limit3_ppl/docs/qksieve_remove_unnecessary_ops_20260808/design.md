# QKSieve 解码冗余操作裁剪：设计

## 问题与可证伪假设

当前 64K QKSieve 的公平 CUDA Graph 延迟为 24.689 ms/token，原生 GQA、零拷贝 Full KV 基线为 25.891 ms/token，只有 1.049x。目标是在不改变 1,280 token/head 上限和 QK-balanced 主索引的前提下，删除对最终候选或输出贡献很小的操作。

可证伪假设：阈值估计不需要扫描约 3,328 个样本，且 ValueSketch 不需要在全部 36 层执行；较小的固定样本和部分层补偿可以降低延迟，同时保持自然文本上的 Full KV top-1 一致率。

## 先验与数学模型

### 阈值样本

每个 head 用代理分数样本估计保留比例为 p 的分位点。若把样本是否超过真实阈值写成 Bernoulli 变量，m 个独立样本的比例标准差近似为

```text
sqrt(p * (1 - p) / m).
```

这里不要求精确恢复分位点，只要求候选容量 1,280 内的阈值误差不明显改变最终输出。因此把样本数从约 3,328 降到 512 是可检验的实现，而不是理论保证；候选数量、PPL、top-1 和 KL 必须同时检查。

### ValueSketch

被选候选集合记为 S，未选 token 的 softmax 分子近似为

```text
u_tail = sum(i not in S) exp(score_i) * value_i.
```

ValueSketch 用低秩、INT4 表示近似该项。它增加约 1.61% Full-KV 等价辅助索引，并在每层增加尾部补偿 kernel。若某些层对输出扰动不敏感，则只在层集合 L_vs 上计算补偿，其余层直接使用候选 attention；这是层选择消融，不声称所有模型都具有相同敏感层。

### Exact attention 分块

最终候选 attention 的输出为

```text
softmax(q * K_S^T) * V_S.
```

分块数控制候选 kernel 的并行度。减少分块只会减少 launch/merge 工作，但也会降低 GPU 并行度，所以必须通过真实 kernel 延迟决定，不能按“操作更少必然更快”判断。

## 实现合同

输入：64K 历史 KV、当前 query、QK-balanced 压缩 Key 索引、GPU 常驻完整 K/V。

参数：

- `QKSIEVE_MIN_QUANTILE_TAIL_SAMPLES=8`：使每 head 的实际阈值样本为 512。
- `QKSIEVE_VALUE_SKETCH_LAYERS`：逗号分隔的 ValueSketch 层号；空集合等价关闭补偿。
- `QKSIEVE_DEBUG_DISABLE_VALUE_SKETCH=1`：关闭全部 ValueSketch 的调试开关。
- `packed_attention_split_counts`：候选 attention 的分块配置；主路径保持自动选择。

步骤：

1. 从 512 个代理分数样本估计候选阈值。
2. 扫描压缩 Key 索引并写入最多 1,280 个候选。
3. 从 GPU 常驻精确 K/V 读取候选，执行 exact attention。
4. 对 `QKSIEVE_VALUE_SKETCH_LAYERS` 中的层合并 ValueSketch 尾部贡献。
5. 对动态新增 suffix 追加精确候选，保证生成过程中新增 token 不丢失。

输出和调试证据：`summary.json` 中的 PPL、top-1、KL、实际候选数、辅助索引比例、Graph wall/CUDA 延迟、Graph/Eager token 一致性和 logit 最大差。

通过条件：Graph 与 Eager 的贪心 token 完全一致；四类自然文本平均 top-1 尽量不低于完整 ValueSketch 的 99.61%；64K Graph 延迟明显低于 23.88 ms/token。

失败条件：PPL 或 top-1 在多个主题同时退化；候选溢出；Graph/Eager token 不一致；减少分块后延迟上升。

## 当前边界

当前证据只覆盖 Qwen3-4B-Instruct-2507、64K、RTX 3090、四条各 64 token 的自然文本流和单个混合窗口。它不能替代完整 LongBench、RULER、多模型、多长度与多 seed 结论。
