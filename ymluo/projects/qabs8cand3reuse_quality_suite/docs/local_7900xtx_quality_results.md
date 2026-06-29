# QABS8Cand3Reuse 本地质量测试结果

本文档整理 Windows 本地机器上对 `qabs8cand3reuse` 的第一轮质量测试结果。测试目标是比较：

```text
baseline
qabs8cand3reuse
sparqfast8cand3
```

在不同主题文本 PPL 和大海捞针文本 PPL 上的差距。

## 1. 运行配置

- 机器：Windows 本地工作站
- 显卡：AMD Radeon RX 7900 XTX
- 运行时：PyTorch ROCm，检测结果为 `torch 2.9.1+rocm7.2.1`
- 模型：`Qwen/Qwen3-0.6B`
- attention 实现：`eager`
- QABS CUDA final kernel：关闭，因为本地是 AMD ROCm，不是 NVIDIA CUDA
- PPL 设置：`prefill_tokens=512`，`eval_tokens=32`
- chunk 设置：`chunk_size=8`，`eval_chunk_size=1`
- 保护 token：`protected_sink_tokens=10`，`protected_recent_tokens=10`
- 自身 token：`always_keep_self=true`
- 原始结果：`outputs_fast/quality_suite_combined.csv`

注意：这是一轮短上下文 smoke-quality 测试，主要用于确认方法质量趋势和本地运行链路，不应当视为最终 benchmark。

## 2. 平均结果

| 数据组 | 方法 | 样本数 | 平均 PPL / baseline | 平均 PPL 差值 | 平均耗时 / baseline |
|---|---:|---:|---:|---:|---:|
| 大海捞针 PPL | `qabs8cand3reuse` | 3 | 1.287095 | 5.188400 | 5.796 |
| 大海捞针 PPL | `sparqfast8cand3` | 3 | 4.546752 | 61.554379 | 3.396 |
| 主题文本 PPL | `qabs8cand3reuse` | 6 | 1.055661 | 0.055873 | 5.988 |
| 主题文本 PPL | `sparqfast8cand3` | 6 | 2.022085 | 1.025148 | 3.009 |

## 3. 逐数据集结果

| 数据组 | 数据集 | baseline PPL | qabs8cand3reuse PPL | qabs 比例 | sparqfast8cand3 PPL | sparq 比例 |
|---|---|---:|---:|---:|---:|---:|
| 大海捞针 PPL | `niah_len1000_depth0` | 20.9428 | 28.6774 | 1.369322 | 72.6409 | 3.468540 |
| 大海捞针 PPL | `niah_len1000_depth25` | 21.8435 | 24.1503 | 1.105606 | 68.5438 | 3.137946 |
| 大海捞针 PPL | `niah_len1000_depth50` | 14.2970 | 19.8207 | 1.386357 | 100.5618 | 7.033771 |
| 主题文本 PPL | `finance` | 1.0008 | 1.0542 | 1.053402 | 2.2504 | 2.248668 |
| 主题文本 PPL | `history` | 1.0097 | 1.0287 | 1.018764 | 1.5272 | 1.512430 |
| 主题文本 PPL | `literature` | 1.0005 | 1.0393 | 1.038785 | 1.8273 | 1.826309 |
| 主题文本 PPL | `mixed_qa` | 1.0014 | 1.0384 | 1.036928 | 1.7132 | 1.710830 |
| 主题文本 PPL | `science` | 1.0166 | 1.0697 | 1.052276 | 1.5732 | 1.547608 |
| 主题文本 PPL | `software` | 1.0009 | 1.1348 | 1.133809 | 3.2895 | 3.286663 |

## 4. 结论

### 4.1 质量结论

`qabs8cand3reuse` 明显优于 `sparqfast8cand3`。

在主题文本上：

- `qabs8cand3reuse` 平均 PPL 是 baseline 的 `1.055661x`
- `sparqfast8cand3` 平均 PPL 是 baseline 的 `2.022085x`

在大海捞针 PPL 上：

- `qabs8cand3reuse` 平均 PPL 是 baseline 的 `1.287095x`
- `sparqfast8cand3` 平均 PPL 是 baseline 的 `4.546752x`

这说明在当前设置下，`qabs8cand3reuse` 的质量退化远小于 SparQ 对照组。

### 4.2 与 baseline 的关系

本轮短上下文测试中，`qabs8cand3reuse` 没有超过 dense baseline。

主题文本上它平均比 baseline 差约 `5.6%` PPL；大海捞针 PPL 上平均比 baseline 差约 `28.7%` PPL。

因此当前结论应表述为：

```text
qabs8cand3reuse 明显强于 sparqfast8cand3，但这轮本地短上下文测试还没有达到 dense baseline 质量。
```

### 4.3 速度结论

当前 PyTorch/ROCm 路径不是速度优化实现。

本轮中：

- `qabs8cand3reuse` 平均约为 baseline 的 `6x` 耗时
- `sparqfast8cand3` 平均约为 baseline 的 `3x` 耗时

这个速度结果主要反映当前 prototype 的 PyTorch 小算子、动态 mask、gather、topk 和调度开销，不代表方法理论上无法加速。

## 5. 大海捞针生成测试

我尝试过 1 个 case 的生成式大海捞针 probe，仍然使用：

```text
baseline,qabs8cand3reuse,sparqfast8cand3
```

但该 probe 在本地 ROCm 环境下超过 10 分钟仍未完成，因此已停止。

所以本文档只汇报 PPL 结果。生成式大海捞针准确率还需要单独优化运行路径，例如：

- 减小 prompt 长度
- 减少生成 token 数
- 避免在 prefill 阶段触发不适合的 sparse path
- 或者把生成评测拆成更小的 batch/case

## 6. 下一步建议

如果要把这个结果推进成正式结论，建议下一轮做：

1. 增大 `prefill_tokens`，例如 `2048`、`4096`、`8192`
2. 增大 `eval_tokens`，例如 `128` 或 `256`
3. 覆盖更多大海捞针长度和深度
4. 单独跑 `qabs8cand3reuse` 与 baseline，减少 SparQ 对整体耗时的影响
5. 如果继续在 AMD 7900XTX 本地跑，优先把评测规模分批，避免单次长任务不易观察进度

