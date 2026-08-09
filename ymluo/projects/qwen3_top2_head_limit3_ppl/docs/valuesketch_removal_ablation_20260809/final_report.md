# ValueSketch 是否应从 QKSieve 删除

## 问题

QKSieve 用低比特 QK 索引找出少量历史 token，再对候选的原始 FP16 K/V 做精确 attention。待验证的问题是：如果候选已经覆盖主要 attention mass，是否可以完全忽略未选 token，从而删除 ValueSketch，降低存储和延迟。

可证伪假设为：在当前 QK-balanced、OAS、sampled-512 selector 下，ValueSketch 的平均质量收益小于 0.3%，且不会修复超过 1% 的单流质量下降。

## 数学模型

设候选集合为 `S`，未选集合为 `T`。无补偿输出为：

`y_S = sum_{i in S} exp(s_i) v_i / sum_{i in S} exp(s_i)`。

完整输出还包含 `T` 的 softmax 分母和 Value 分子。ValueSketch 用 rank-16、block-256、INT4 表征近似这两项，再与 `S` 上的精确结果在同一 softmax 标尺合并。若 `T` 的概率质量或加权 Value 合力不可忽略，即使候选 token 本身找得合理，删除补偿仍会改变层输出。

## 实现合同

两种版本使用完全相同的 request-local QK-balanced 坐标、240-bit 混合位宽 Key 索引、OAS 位宽分配、512-sample 分位数阈值、1,280 候选上限和原始 FP16 K/V。唯一开关是 `QKSIEVE_DEBUG_DISABLE_VALUE_SKETCH`。实验没有 router、任务规则或 Full Attention fallback。

CUDA v15 缓存最初与当前持久化调用契约不一致，导致 contiguous 检查失败。将扩展升级为 `qksieve_valuesketch_attention_20260809_v16_contiguous_contract` 后，4K 最小复现通过，再运行正式对照。该修复不改变公式、候选或预算。

## 实验与结果

Llama-3.1-8B-Instruct 在三条文本上进行 teacher-forced 配对测试。32K 每条 64 个 token，96K 每条 32 个 token。32K 单卡，96K 双卡；同长度内所有方法硬件一致。

| 长度 | 方法 | 合并质量保持 | 稳态 ms/token | 相对 Full | 辅助存储 |
|---:|---|---:|---:|---:|---:|
| 32K | Full | 100% | 85.55 | 1.00x | 0% |
| 32K | 无补偿 | 98.578% | 39.79 | 2.15x | 5.79% |
| 32K | ValueSketch | **100.517%** | 52.43 | **1.63x** | 7.40% |
| 96K | Full | 100% | 211.85 | 1.00x | 0% |
| 96K | 无补偿 | 95.099% | 52.62 | 4.03x | 5.78% |
| 96K | ValueSketch | **99.768%** | 68.33 | **3.10x** | 7.39% |

最重要的失败是 96K《基督山伯爵》：无补偿只保持 Full 的 86.892%，补偿后为 100.296%。这说明未选 token 的聚合 Value 贡献在部分长文本中不是小扰动。

## 失败解释

删除补偿减少了约 1.61% 的辅助存储，并让稳态路径快约 23%--24%，但它把“候选覆盖主要高分 token”错误地等同于“未选 token 的总输出贡献可忽略”。大量单个权重较小的 token 仍可能在 softmax 分母或 Value 方向上形成不可忽略的合力。96K 失败窗口正是该假设的反例。

## 冻结决策

- **论文主方法：QKSieve-Robust**，保留 ValueSketch。它在本诊断中达到 32K 100.517%、96K 99.768% 的合并质量保持。
- **速度消融：QKSieve-Fast**，删除 ValueSketch。它说明 selector 本身的速度上限，但不能作为通用高保真方法。
- 不引入按长度、文本或任务选择版本的 router；这会掩盖失败条件并削弱方法的数值可解释性。

## 声明边界与下一项实验

本实验足以否定“ValueSketch 可以无条件删除”，但三条文本不足以证明 Robust 在所有模型和任务上都达到 99.5%。下一项不再改方法：冻结 Robust 配置，完成同路径 LongBench、RULER、多模型质量以及 MHA attention/decode 实测。速度表必须分别报告索引构建、selector、ValueSketch、候选 attention 和整模型 decode，不能用延迟分解替代独立 CUDA 测量。
