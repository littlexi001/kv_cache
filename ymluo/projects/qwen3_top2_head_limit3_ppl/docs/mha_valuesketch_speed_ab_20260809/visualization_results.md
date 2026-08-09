# MHA 下 ValueSketch 有无补偿速度对照：结果

## 实验设置

比较三个条件：标准 Full attention、只使用候选 token 的 QKSieve-Fast、在相同候选上增加 ValueSketch 尾部补偿的 QKSieve-Robust。模型为 `Yarn-Llama-2-7b-128k`，32 个 Query heads、32 个 KV heads、head dimension 128，属于真实 MHA。Fast 与 Robust 都最多保留 1,280 token/head，不使用 router 或 Full fallback。

`Attention 加速`是单层 Full MHA CUDA 时间除以稀疏完整路径 CUDA 时间。`稳态 Decode 加速`是 Full 的 ms/token 除以相应方法跳过前 16 token 后的平均 ms/token。`64-token 在线加速`还包含一次性索引构建，但不包含 prompt prefill 和部署前 CUDA JIT 编译。

## 公平性检查

8K、16K、32K、64K、128K 各跑三个随机种子。全部 15 组满足：

- 32 个 head 的分位数阈值最大绝对差为 `0`；
- 候选数量最大差为 `0`；
- 排序后候选 token 集合逐 head 完全相同；
- 没有 candidate overflow、CUDA OOM 或运行异常。

因此下面 Fast 与 Robust 的区别只来自 ValueSketch 统计和尾部合并，不来自不同候选。

## Attention 子系统

下表是三个随机种子的中位数，单位为单层 `ms`。完整路径包含 Query 投影量化、候选扫描和候选 attention；Robust 还包含未选 Value 的低秩统计与合并。

| 历史长度 | Full | Fast | Robust | Fast 加速 | Robust 加速 | Robust 相对 Fast 延迟 |
|---:|---:|---:|---:|---:|---:|---:|
| 8K | 0.1691 | 0.1327 | 0.1560 | 1.27x | 1.08x | +17.45% |
| 16K | 0.3173 | 0.1900 | 0.2250 | 1.67x | 1.41x | +19.09% |
| 32K | 0.6163 | 0.2324 | 0.2945 | 2.65x | 2.09x | +26.73% |
| 64K | 1.2027 | 0.2646 | 0.3576 | 4.55x | 3.36x | +35.14% |
| 128K | 2.3708 | 0.3722 | 0.5757 | 6.37x | 4.12x | +57.75% |

Robust 仍随长度获得更高加速，但补偿必须扫描全部历史，因此它相对 Fast 的 attention 成本随长度增长。

## Attention 组件

| 长度 | Query 准备 | Fast 扫描 | Robust 扫描+尾部统计 | Fast 候选 attention | Robust 候选+尾部合并 |
|---:|---:|---:|---:|---:|---:|
| 8K | 0.0339 | 0.0332 | 0.0540 | 0.0441 | 0.0452 |
| 16K | 0.0338 | 0.0424 | 0.0759 | 0.0900 | 0.0914 |
| 32K | 0.0339 | 0.0612 | 0.1211 | 0.1155 | 0.1158 |
| 64K | 0.0310 | 0.1063 | 0.1967 | 0.1082 | 0.1094 |
| 128K | 0.0305 | 0.1887 | 0.3652 | 0.1332 | 0.1570 |

主要新增成本不是最后的 Value 合并，而是扫描未选 token 时累计 softmax 分母和 rank-16 Value 系数。128K 时，Robust 比 Fast 多 `0.2035 ms/layer`，其中扫描部分增加 `0.1765 ms/layer`。

## 真实模型稳态 Decode

每个条件生成 64 token，跳过前 16 token，报告后 48 token 的平均值。32K 的 Fast/Robust 和 64K 的 Fast 使用无并发干扰重测值；128K 三条件均使用八卡顺序运行。

| 长度 | Full ms/token | Fast ms/token | Robust ms/token | Fast 加速 | Robust 加速 | Robust 相对 Fast 延迟 |
|---:|---:|---:|---:|---:|---:|---:|
| 32K | 84.18 | 51.56 | 63.97 | 1.63x | 1.32x | +24.07% |
| 64K | 144.75 | 52.97 | 65.25 | 2.73x | 2.22x | +23.17% |
| 128K | 268.17 | 55.41 | 67.38 | 4.84x | 3.98x | +21.60% |

结论是：补偿在真实 decode 中稳定花费约 22%--24% 延迟，但 Robust 在 64K 和 128K 仍分别达到 `2.22x` 和 `3.98x`。

## 一次性索引与 64-token 在线速度

| 长度 | Fast 构建 | Robust 构建 | ValueSketch 单独构建 | Fast 64-token 在线加速 | Robust 64-token 在线加速 |
|---:|---:|---:|---:|---:|---:|
| 32K | 0.768 s | 1.375 s | 0.580 s | 1.01x | 0.81x |
| 64K | 0.774 s | 1.488 s | 0.713 s | 1.64x | 1.31x |
| 128K | 0.743 s | 1.839 s | 1.051 s | 2.75x | 2.15x |

用稳态节省估计，Fast 相对 Full 的构建成本分别需约 `24/9/4` 个生成 token 摊平；Robust 需约 `69/19/10` 个 token。因此，32K Robust 对只生成 64 token 的冷请求仍较慢；在多轮问答或 agent 复用同一索引时，这项成本只支付一次。

## 失败解释

最初烟测没有通过候选一致性，原因有两个：MHA 无补偿路径使用了另一套 generic scanner；微基准还把原始目标比例传给 Robust、把有限样本修正后的比例传给 Fast。统一为同一个 plain MHA scanner 和同一个有限样本阈值后，所有候选一致性检查通过。

64K 首轮 Fast 与 Full 并发运行时测得 `64.91 ms/token`，单独重跑后为 `52.97 ms/token`。最终表只采用单独重测值，说明多 GPU 任务并发会明显污染整模型测速。

## 结论边界

这些结果证明当前 CUDA 实现在 RTX 3090、真实 MHA Llama、resident FP16 K/V 下的速度。它不等价于 H100 结果，也不提供跨模型速度置信区间。Attention 表是 synthetic MHA-shaped 单层测量；Decode 表是真实模型测量。质量收益应引用独立的 ValueSketch 去留消融，不能从本速度表推断。
