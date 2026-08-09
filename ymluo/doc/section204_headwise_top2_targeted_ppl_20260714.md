# Section 204：体育与医学上的逐 Head Top-2% Attention PPL

日期：2026-07-14

## 1. 实验问题

此前 32K 普通文本续写实验中，体育和医学是共享 2K KV 检索最难处理的两个主题。本文验证一个更强但计算昂贵的诊断上界：不让所有 head 共用一套 token，而是让每个 query head 在每一步独立选取 attention logit 最高的 2% 历史 token，再进行 softmax 和 Value 聚合。

这个实验回答的是：质量退化究竟来自“2% 太少”，还是来自“不同 head 被迫共享同一批 token”。

## 2. 精确定义

对第 `l` 层、第 `h` 个 query head、当前 token `t`，先计算所有历史 token 的完整 QK logit：

```text
s(l,h,t,j) = q(l,h,t)^T k(l,h,j) / sqrt(d),  j < t
k(t) = ceil(0.02 * t)
S(l,h,t) = TopK_j(s(l,h,t,j), k(t)) union {t}
```

然后只在 `S(l,h,t)` 上归一化：

```text
o(l,h,t) = sum_{j in S(l,h,t)} softmax_{S(l,h,t)}(s(l,h,t,j)) * v(l,h,j)
```

Llama-3.1-8B-Instruct 使用 GQA。实验先把 8 个 KV heads 映射到 32 个 query heads，再让每个 query head 独立选 token。当前 token 无条件保留；没有额外保留 sink、recent 或检索 block，因此结果没有混入保护策略。

## 3. 评测协议

- 模型：Llama-3.1-8B-Instruct。
- 数据：20 Newsgroups 的 `rec.sport.baseball` 与 `sci.med`。
- 每个主题 3 个不重叠窗口，共 6 个 case。
- 每个 case：31,744-token full prefill + 256-token query + 256-token PPL target。
- remote prefill 使用标准 full attention；query 和 target 才启用逐 head top-2%，与 KV-cache 在线压缩的评测边界一致。
- PPL 严格因果；每个 target token 只使用其之前的 token。
- 32K 时每个 head 约保留 640 个历史 token和当前 token，attention link ratio 为 `641/32000 = 2.0031%`。
- Full 与 top-2% 使用完全相同的数据窗口、tokenizer 和 target。

## 4. 聚合结果

| 主题 | Full PPL | Head top-2% PPL | PPL/Full | Delta NLL |
|---|---:|---:|---:|---:|
| 体育 | 8.0679 | **8.3369** | **1.0333** | +0.0328 |
| 医学 | 8.7336 | **8.8821** | **1.0170** | +0.0169 |
| 合并 | 8.3941 | **8.6052** | **1.0251** | +0.0248 |

Full 聚合值与 Section 202/203 的既有结果一致，说明数据窗口和 PPL 对齐没有漂移。

## 5. 逐窗口结果

| Case | Full PPL | Head top-2% PPL | PPL/Full |
|---|---:|---:|---:|
| 体育 0 | 8.7902 | 9.3963 | 1.0690 |
| 体育 1 | 2.5838 | 2.5908 | 1.0027 |
| 体育 2 | 23.1219 | 23.8027 | 1.0294 |
| 医学 0 | 10.2239 | 10.4068 | 1.0179 |
| 医学 1 | 7.6943 | 7.7294 | 1.0046 |
| 医学 2 | 8.4682 | 8.7111 | 1.0287 |

最差的体育窗口 0 仍有 6.9% PPL 退化，但已经远小于共享 2K KV 方法。其余 5 个窗口均在 3% 左右以内，其中体育 1 和医学 1 约为 0.3% 与 0.5%。

## 6. 与现有实用方法的关系

既有共享 KV 方法在两个主题上的结果为：

| 方法 | 体育 PPL/Full | 医学 PPL/Full | 合并 PPL/Full |
|---|---:|---:|---:|
| 原始 2K LPCM | 1.3725 | 1.6110 | - |
| 冻结 queryless controller v2 | 1.1593 | 1.1877 | 1.1735 |
| 逐 head exact top-2% | **1.0333** | **1.0170** | **1.0251** |

该结果支持以下判断：体育和医学并非天然需要稠密 attention。主要瓶颈是共享 token 集合不能同时覆盖不同 head 的功能需求。逐 head 的 640-token 集合虽然每个都很小，但能保留实体、重复模板、局部句法和远程 continuation 等不同类型的信息。

## 7. 速度与可实现性边界

当前实现是质量上界，不是可部署加速版本：

1. 为了知道真实 top-2%，仍然计算了全部 QK logits。
2. KV cache 仍保存全部 32K token，没有实现 2% 的物理 KV 内存。
3. 只把 softmax 和 Value 聚合限制到约 2% links。
4. 为保证逐 token 精确选择，当前 PyTorch/Hugging Face harness 使用 token-by-token eager forward。

因此当前测得每个 case 的 top-2% online 时间约 36.35 秒，而 SDPA full 约 0.83 秒。这个数字反映的是 oracle selector 与 Python/eager 开销，不能用于宣称加速。若没有能避免 full QK 扫描的候选索引和 ragged/paged KV kernel，top-2% 只证明质量可达，不是完整系统。

此外，`2.0031%` 是 head-token attention links 的比例，不是整层 KV token union。不同 query heads 选中的 token 并不相同，union 可能明显高于 2%。

## 8. 结论与下一步

这是一个重要的正结果：仅保留每个 head 的 top-2% 历史 attention links，体育和医学合并 PPL 只比 Full 高 2.51%，即 `PPL/Full = 1.0251`。它把问题从“怎样继续调共享 2K router”转化成了“怎样低成本预测每个 head 的 top-2% 候选”。

下一步优先级：

1. 统计每层、每个 KV head 的 query-head token union，确定真实物理 KV 下界。
2. 测量 LPCM/低维 QK/相邻步复用候选对 exact top-2% 的 recall，尤其分析体育窗口 0。
3. 训练 head-conditioned candidate router，而不是只预测一个全局共享 budget。
4. 在候选 recall 足够后实现 block-ragged 或 paged sparse attention kernel，再报告真实速度和 KV 内存。
5. 把 head-wise 模式作为 frozen controller 的高风险动作，而不是对所有样本无条件启用。

## 9. 产物

- 实验脚本：`src/run_head_top2_targeted_ppl_20260714.py`
- 启动脚本：`scripts/launch_head_top2_targeted_ppl_20260714.sh`
- 单元测试：`tests/test_head_top2_targeted_ppl.py`
- 原始结果：`results/20260714_head_top2_targeted_32k_w3/`
- 聚合表：`results/20260714_head_top2_targeted_32k_w3/combined_summary.csv`
