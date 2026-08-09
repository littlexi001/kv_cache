# QKSieve、FIER 与 ValueSketch 的 64K 配对实验

## 实验口径

- 模型：Qwen3-4B-Instruct-2507，FP16，RTX 3090。
- 历史长度：65,536 token；评测 64 个连续 token。
- 数据：冻结的本地代码文本流，与 2026-08-07 长度测速使用同一文本。
- 每个版本独立进程运行，四个版本在 GPU 0--3 上轮换，共 3 次重复。
- `Full ms/token` 和 `Sparse ms/token` 是完整模型稳态 decode，包含 selector、top-k、精确稀疏 attention、MLP 和其他层计算，不包含 prefill 与一次性索引构建。
- QKSieve 两个版本均设置 `QKSIEVE_DEBUG_DISABLE_VALUE_SKETCH=1`；运行记录中的 ValueSketch ratio 为 0。
- FIER 是按论文实现的 audited reproduction，不是官方代码。

## 四个原始版本

| 版本 | PPL质量保持 | Top-1一致率 | 实际token/head | 辅助索引/Full KV | 固定开销 | Sparse ms/token | 配对加速 |
|---|---:|---:|---:|---:|---:|---:|---:|
| QKSieve top1280，无ValueSketch | 101.435% | 98.438% | 1,277.4 | 5.766% | 0.559 s | **51.815** | **3.114x** |
| FIER RTN-1 g32 top1280 | **104.052%** | 98.438% | 1,280.0 | 6.250% | **0.040 s** | 56.734 | 2.959x |
| FIER RTN-1 g32 top512，原始unsplit | 102.774% | 96.875% | 512.0 | 6.250% | 0.041 s | 76.113 | 2.154x |
| QKSieve top512，无ValueSketch | 102.209% | **98.438%** | 508.9 | **5.766%** | 0.561 s | 54.139 | **3.162x** |

这里的 PPL 质量超过 100% 表示稀疏扰动在这一个文本流上降低了 PPL，不能外推为一般任务上优于 Full。

## 直接结论

1. **ValueSketch 在该 64K 流上没有兑现质量收益，且稳态成本明显。** 昨日相同文本、含 ValueSketch 的 top1280 路径为 72.13 ms/token；关闭后为 51.82 ms/token，稀疏路径快约 1.39x。质量从昨日的 99.398% 变为本次的 101.435%。这支持在主线中将 ValueSketch 设为可选安全模块，但还不能仅凭一个 PPL 流将其删除，仍需在 LCC、QMSum 等弱任务上做配对消融。

2. **同为约 1,280 候选时，QKSieve 稳态快于当前 FIER reproduction。** QKSieve 为 51.82 ms/token，FIER 为 56.73 ms/token，前者快约 9.5%，且索引小约 7.7%。FIER 的一次性索引开销仅约 0.04 s，而 QKSieve 的请求级 QK-balanced 变换与编码约 0.56 s；按当前数字，生成约 106 token 后 QKSieve 才能靠稳态优势追回固定成本。缓存前缀复用场景中，索引已存在，因此没有这项回本等待。

3. **把候选从 1,280 降到 512 并没有让 QKSieve 明显更快。** 51.82 与 54.14 ms/token 基本处于同一档，说明 64K 下主要成本已经不是精确消费 768 个额外候选，而是低比特全历史扫描、top-k、模型非 attention 部分和 kernel 调度。top512 不能作为主要提速手段。

4. **FIER top512 的原始慢速是 kernel 选择问题。** 原实现对 top512 使用 unsplit exact attention。独立 CUDA 计时显示，512 候选时 split1 为 0.1355 ms/layer，split16 为 0.0386 ms/layer，后者快 3.51x；1,280 候选时对应提升为 4.89x。

## FIER top512 kernel 修复

加入只改变归约并行度、不改变 selector 和候选集合的 split override 后：

| FIER top512路径 | Sparse ms/token | 相对原路径 | 整模型decode加速 | PPL质量保持 | Top-1一致率 |
|---|---:|---:|---:|---:|---:|
| 原始split1 | 76.113 | 1.000x | 2.154x | 102.774% | 96.875% |
| split8 | 56.850 | **1.339x** | 2.792x | 102.668% | 96.875% |
| split16 | **56.639** | **1.344x** | **2.794x** | 103.115% | 96.875%（中位数） |

split8 与 split16 的速度差只有约 0.4%，split8 的 Top-1 在三次运行中更稳定，因此当前更适合作为 FIER top512 的保守实现。

修复后，QKSieve top512 仍从 56.85 降到 54.14 ms/token，约快 5.0%，索引小 7.7%，Top-1 一致率高 1.56 个百分点；FIER 在该文本流上的 PPL 更低。QKSieve 的固定索引成本更高，若每个请求都重新建索引，需要约 192 个生成 token 才能超过 split8 FIER；在多轮问答或 Agent 的缓存前缀复用场景中，QKSieve 从第一步 decode 就有稳态优势。

## 当前判断

- 速度主线应冻结为 **QKSieve sampled top1280/top512，无默认 ValueSketch**；512 与 1,280 可视为质量预算点，而不是两个明显不同的速度点。
- ValueSketch 应降级为面向已识别弱任务的可选补偿，必须用独立弱任务证据决定是否启用。
- 论文中的 FIER 对照必须使用 split8 修复后的版本；继续报告原始 split1 会不公平地低估 FIER。
- 下一项最有信息量的实验不是继续调候选数，而是对 LCC、QMSum、MultiNews、PassageCount 做有/无 ValueSketch 的严格配对质量实验，再决定主方法是否彻底删除 ValueSketch。

## 结果文件

- `results/20260808_qksieve_fier_budget_ab_64k/isolated_summary.json`
- `results/20260808_qksieve_fier_budget_ab_64k/ragged_attention_split.json`
- `results/20260808_qksieve_fier_budget_ab_64k/fier_split8_summary.json`
- `results/20260808_qksieve_fier_budget_ab_64k/fier_split16_summary.json`
