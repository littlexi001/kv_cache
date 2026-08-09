# 基于 PCA64 INT4 的层次化 Per-Head KV 检索

更新时间：2026-07-17

## 1. 问题

前期实验发现：长上下文中，每个 query head 只保留真实 attention score 最高的2%历史 token，仍能保持接近 Full Attention 的质量。

在体育和医学各3个32K窗口上的结果为：

| 主题 | Full PPL | Per-head exact top-2% PPL | 相对质量保持 |
|---|---:|---:|---:|
| 体育 | 8.0679 | 8.3369 | 96.78% |
| 医学 | 8.7336 | 8.8821 | 98.33% |
| 合并 | -- | -- | **97.55%** |

这说明模型并不需要对全部历史 token 做 attention，关键在于找到每个 head 真正需要的少量 token。

但上述 exact top-2% 是诊断上界：为了得到真实排名，仍然计算了完整 QK：

```text
score(head h, token i) = dot(query_h, key_h,i),  i = 1 ... N
```

因此，问题被定义为：

> **如何不计算完整 QK，就低成本找到每个 query head 的 top-2% 高 attention token？**

注意：现有结果支持“接近 Full”，不支持“稳定超越 Full”。个别任务可能高于 Full，但体育和医学的合并 PPL 比 Full 高2.51%。

## 2. 新发现

我们对每层、每个 KV head 的 post-RoPE K 建立 PCA64 低维空间，并将64维投影结果量化为 INT4。当前 query 投影到同一空间后，用低维内积近似 QK score。

索引只占完整 FP16 K+V 的 **6.64%**。在体育和医学的真实32K Q/K trace上，共测试320个 layer-head-query case：

| 指标 | PCA64 INT4 |
|---|---:|
| Exact top-2% token recall | **68.11%** |
| Token recall p10 | 57.61% |
| Oracle top-2% attention mass recall | **98.27%** |
| Mass recall p10 | **95.81%** |
| Mass recall minimum | 87.34% |

核心发现是：

> **PCA64 INT4 只召回68.11%的 exact top-2% token，却召回98.27%的 oracle attention mass。**

原因是 top-2% 内部并不等权。低维索引遗漏的主要是低权重尾部 token，而高 attention mass token 更容易保持高分。

另一个发现是，不同 query heads 需要不同历史证据。128K实验中，让同一GQA组共享一套候选时，质量保持率只有78.07%；改为每个 query head 独立检索后，质量恢复到96.34%。

## 3. 设计的方法

基于上述发现，我们设计了层次化 Per-Head KV Retrieval：

1. **GPU全局索引**：保存每个历史 K 的 PCA64 INT4 表征，用于扫描和定位候选；
2. **CPU完整K/V**：完整 FP16 K/V 保存在 pinned memory，不做不可逆删除；
3. **Per-query-head检索**：每个 query head 独立选择1%--2.5%的历史 token；
4. **GPU精确热缓存**：保存3.2%--4.1%的精确 FP16 K/V，cache miss才从CPU搬运；
5. **精确稀疏attention**：最终使用被选中的原始 FP16 K/V，而不是PCA近似值；
6. **Residency-windowed GQA**：每次只处理一个或两个 query heads，复用同一批GPU cache slots，避免逐头候选并集膨胀。

整体流程为：

```text
Query
  -> PCA64投影
  -> INT4全局索引扫描
  -> 每个query head独立选择候选
  -> 查询GPU精确K/V热缓存
  -> 从CPU填充cache miss
  -> 在精确FP16 K/V上计算稀疏attention
```

短上下文中检索开销不能摊销，因此当前使用纯长度gate：

```text
少于16K：Full Attention
16K及以上：层次化Per-Head KV Retrieval
```

该方法不读取答案，不使用任务标签、oracle或测试集router。

## 4. 最终效果

| 场景 | 质量保持率 | GPU KV ratio | Decode speedup | Protocol/E2E speedup |
|---|---:|---:|---:|---:|
| LongBench 16任务，3750样本 | **96.66%** | **10.62%** | 旧协议不报告 | 旧协议不报告 |
| LongBench新协议，160样本 | **98.59%** | **10.66%** | 0.283x raw sparse | 长度gate后等于Full |
| 128K speed-first | **96.34%** | **9.99%** | **2.706x** | **1.126x** |
| 128K quality-first | **99.84%** | **11.00%** | **2.383x** | **1.069x** |

其中，128K的2.706x是整模型decode加速，已经包含PCA64 INT4扫描、top-k、cache查询、PCIe搬运、精确稀疏attention、MLP及其他模型计算。

可以用于汇报的总结是：

> **逐head exact top-2%实验首先证明了长上下文attention具有极强稀疏性。随后，我们发现PCA64 INT4索引能以68.11%的token召回恢复98.27%的oracle attention mass。基于这一特性设计的层次化KV检索系统，在完整LongBench上用10.62% GPU KV保持96.66%质量，并在一个128K开发窗口上用9.99% GPU KV保持96.34%质量、取得2.706x整模型decode加速。**

当前边界：128K速度结果仍来自单主题、单窗口；完整LongBench证明了质量，但其旧协议不能用于速度结论。
