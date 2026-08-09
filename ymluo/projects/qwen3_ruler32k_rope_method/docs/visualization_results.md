# RULER-32K 可视化与结果

![RULER-32K task scores](../outputs/merged/task_scores.png)

图中每个任务只有 2 条样本，共 26 条，因此柱高是 pilot 点估计，不是正式 RULER 排行分数。

## 总体结果

| 方法 | 13 任务宏平均 | 相对 exact Top-2% | paired 95% CI | 改善/变差/不变样本 |
|---|---:|---:|---:|---:|
| Native Full | 85.19 | +1.67 | [0.00, 4.62] | 2 / 0 / 24 |
| exact post-RoPE Top-2% | 83.53 | 0 | — | — |
| local/global postscore | **86.15** | **+2.63** | **[0.00, 6.15]** | **3 / 0 / 23** |
| local/global blend25 | 82.69 | -0.83 | [-10.19, 5.64] | 4 / 1 / 21 |

`local/global postscore` 相对 Native Full 为 +0.96 分，95% CI 为 [-0.77, 3.27]。它在这个 pilot 上的点估计最高，但相对 exact 的区间下界为 0，相对 Full 的区间跨 0；因此结论是“有正向信号，尚未证明稳定提升”。

## 差异来自哪些任务

| 任务 | Full | exact Top-2% | postscore | blend25 |
|---|---:|---:|---:|---:|
| CWE | 70.00 | 65.00 | 70.00 | 75.00 |
| FWE | 100.00 | 83.33 | 100.00 | 100.00 |
| NIAH multivalue | 87.50 | 87.50 | 100.00 | 100.00 |
| NIAH multikey-3 UUID | 100.00 | 100.00 | 100.00 | 50.00 |

其余 9 个任务四种主方法在这两条样本上的任务均分相同：6 个 NIAH/VT 为 100，SQuAD 为 50，HotpotQA 为 0。postscore 的 +2.63 分来自 3 条样本：CWE +0.10、FWE +0.333、NIAH multivalue +0.25；没有观察到相对 exact 的退化样本。

## 证据代理

对 8 类 NIAH 的 16 条样本，将答案值在 context 中的精确 token span 作为诊断代理：

| 方法 | 答案值 token recall | 答案值 attention mass |
|---|---:|---:|
| exact Top-2% | 23.43% | 1.277% |
| postscore | 23.62% | 1.338% |
| blend25 | 24.13% | 1.369% |

总体 recall 只增加 0.19 个百分点，并不是普遍提升。postscore 在 `niah_single_1` 上增加 9.35 点，在 `single_2`、`multiquery` 和 `multivalue` 上小幅增加；但在三个 multikey、UUID single-3 上下降 1.36–3.10 点。方法对证据召回的影响具有明显任务结构依赖。

## 代表性样例

改善样例 `niah_multivalue_32768_0`：

- gold 有 4 个数字。
- exact Top-2% 只生成 3 个，score 0.75。
- postscore 和 blend25 都恢复第 4 个数字，score 1.00。
- 答案值 recall 从 35.30% 升到 39.58% / 40.67%，与输出改善方向一致。

失败样例 `niah_multikey_3_32768_0`：

- exact 和 postscore 都完整生成 gold UUID，score 1.00。
- blend25 的 UUID 前缀正确，但后半段变为另一个字符串，score 0。
- blend 的首答案 token NLL 反而更低。这说明“首 token 更有信心”不保证多 token 实体完整正确，直接修改消费分数会放大后续解码风险。

## 审计结果

- 所有稀疏观测的 2% 支持集预算错误率为 0，重复位置错误率为 0。
- `full_rope_replay` 与 Native Full 的 26 条官方分数完全一致，但最大 logit 误差均值为 0.328、最大值为 0.625。NF4/BF16 下自定义 eager/GQA 路径并非严格逐 logit 等价；exact 与两种方法之间仍使用同一自定义路径，可做相对比较。
- Native Full 平均 query 0.105 秒；exact、postscore、blend 分别约 0.269、0.298、0.300 秒。当前 Python 原型仍完整扫描 K 来构造 pre/post 分数，不代表优化后速度。
- raw prompt 为 28,768–32,656 token；共享 prefill 平均 31.5 秒。

## 结论

最保守结论是：在 26 条 RULER-32K pilot 上，`local/global postscore` 比 exact post-RoPE Top-2% 高 2.63 分，且观察到 3 个改善、0 个退化；但样本量过小，bootstrap 下界为 0。25% 分数混合不稳定，虽然提高若干任务和平均首 token 置信度，却会破坏长 UUID 的后续 token，因此当前不应作为主方法。

下一步最值得扩样的是差异任务 `cwe`、`fwe`、`niah_multivalue` 和风险任务 `niah_multikey_3`，每类至少 20–50 条；不必立刻把 13 个任务全部扩大到同样规模。

