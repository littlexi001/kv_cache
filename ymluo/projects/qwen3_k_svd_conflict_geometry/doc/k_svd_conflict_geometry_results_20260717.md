# 正确/矛盾推理链的 K-SVD 几何实验

日期：2026-07-17  
模型：Qwen3-0.6B  
主实验：64 seeds，8,265-token prompt，2-hop gold chain + 可选 2-hop conflict chain

## 一句话结论

**K 的高能低秩空间主要保存 gold/conflict 共有的编号和规则结构；真正与答案 query 对齐、用于区分正确和矛盾证据的方向，显著更多地落在长尾。**

在主分析 rank=16 时，Top-16 已解释全上下文 K 谱能量的 88.63%，但：

- 答案位置 Q 只有 15.41% 的能量在 Top-16；
- 冲突引起的 Q 方向变化只有 12.91% 在 Top-16，即 87.09% 在长尾；
- code 的 Top-16 `q·(K_gold-K_conflict)` 为 -0.220，反而偏向 conflict；长尾为 +0.335，反转并恢复了对 gold 的偏好；
- 相同编号的数字 sub-token 也有同样模式：Top-16 为 -0.058，长尾为 +0.353。

因此，若外部 KV retrieval 或 KV compression 只保留 K 的 PCA/SVD 主方向，很可能保留了“这两个 token 看起来属于同一规则/编号模板”，却丢掉“哪一个证据是正确的”这一部分。

## 1. 实验设计

### 1.1 严格配对

每个 seed 构造一对 prompt：

1. `filler + gold chain`
2. 完全相同的 `filler + gold chain + conflict chain`

两者均为 8,265 tokens，gold 的位置不变；conflict 通过替换 filler 加入，不改变 prompt 长度或答案 query 的绝对位置。32 个 seed 的 conflict 在 gold 前，另外 32 个在 gold 后。

另跑了短上下文对照，但加入 conflict 会改变短 prompt 的长度和 query 位置，因此主结论只使用严格长度/位置匹配的 8K 条件。

### 1.2 K、Q 和 SVD 的定义

- K：`k_proj -> k_norm -> RoPE` 后、真正进入 attention dot product 的历史 key；不包含最终 query token 自己的 K。
- Q：答案提示最后一个 token 的 `q_proj -> q_norm -> RoPE` 后 query。
- 每层、每个 KV head 单独构造 clean/conflict 共用的未中心化二阶矩：

  `C = (K_clean^T K_clean + K_conflict^T K_conflict) / (N_clean + N_conflict)`

- 对 `C` 做特征分解；Top-r 是最大 r 个特征值对应的主子空间，tail 是其正交补。
- 使用未中心化矩阵是因为 attention 使用原点相关的 `q^T k`，而不是减均值后的距离。
- 扫描 `r = 4, 8, 16, 32, 64`，预注册主分析为 `r=16`。

所有 QK 数值均为实际 attention scale 下的 `q^T k / sqrt(128)`。定义：

`QK advantage = q^T K_gold - q^T K_conflict`

正值表示偏向 gold，负值表示偏向 conflict。

### 1.3 三种 token 集合

- `rule span`：完整的两条规则文本；
- `code only`：规则中的 antecedent/consequent 编号 token，避免公共模板文字主导结果；
- `same-number token`：gold T0 和 conflict X0 中完全相同的起始编号，严格逐 sub-token 配对；其中数字 sub-token 单独汇总。

## 2. 主子空间与长尾

### 2.1 Top-16 能量并不等于判别信息

| 指标 | Top-16 | Tail | 解释 |
|---|---:|---:|---|
| 全上下文 K 谱能量 | 88.629% ± 0.003% | 11.371% | K 本身高度低秩 |
| conflict prompt 的 Q 能量 | 15.411% ± 0.029% | 84.589% | Q 主要不在 K 的高能方向 |
| gold/conflict code 差向量能量 | 54.904% ± 0.117% | 45.096% | 差异远比 K 总能量更偏尾部 |
| 相同数字 token 差向量能量 | 50.277% ± 0.103% | 49.723% | 同编号的角色差异近乎一半在 tail |
| conflict 引起的 Q 差向量能量 | 12.907% ± 0.142% | 87.093% | query 的变化几乎全在 tail |

这里最关键的对照是：Top-16 占 K 总能量 88.6%，却只占 query 变化 12.9%。所以不能用“解释了多少 K variance”来判断是否保留了任务所需方向。

### 2.2 gold 和 conflict 的方向相似度

| 对象 | RoPE 前 raw cosine | RoPE 后 raw cosine | Top-16 cosine | Tail cosine |
|---|---:|---:|---:|---:|
| 所有 code token 均值 | 0.9771 ± 0.0003 | 0.8630 ± 0.0004 | 0.9084 ± 0.0005 | 0.4805 ± 0.0016 |
| 相同编号的数字 sub-token 均值 | 0.9530 ± 0.0006 | 0.8334 ± 0.0006 | 0.8970 ± 0.0004 | 0.4439 ± 0.0046 |

主方向上的 gold/conflict 很相似，长尾方向明显分开。RoPE 前 cosine 更高，说明实际 attention K 的一部分差异来自位置旋转；但 RoPE 前仍不是 1，尤其在中后层，说明规则标签、前文和链角色也确实改变了相同编号的内容表征。

### 2.3 与 query 方向的关系

| token 集合 | Top-16 QK advantage | Tail QK advantage | 合计 |
|---|---:|---:|---:|
| 完整 rule span | -0.180 ± 0.039 | **+0.372 ± 0.009** | +0.192 |
| code only | -0.220 ± 0.037 | **+0.335 ± 0.012** | +0.115 |
| 相同编号数字 token | -0.058 ± 0.015 | **+0.353 ± 0.049** | +0.295 |

Top-16 不仅没有提供正确证据优势，在总体平均上还偏向 conflict；gold 优势来自长尾的反向修正。code tail advantage 在 64/64 个 seed 中为正；code Top-16 在 40/64 个 seed 中为负。

方向 cosine 也一致：`K_gold-K_conflict` 与 Q 在 Top-16 的 cosine 为 -0.0196，而在 tail 为 +0.0216；完整 rule span 为 -0.0143 对 +0.0281。绝对值小是 128 维向量的正常现象，但符号在 seed 级平均上稳定。

## 3. Rank 扫描

| Rank | K 谱能量 | Q 能量 | code 差向量能量 | 主空间 QK advantage | Tail QK advantage |
|---:|---:|---:|---:|---:|---:|
| 4 | 75.456% | 4.258% | 22.263% | -0.094 | +0.209 |
| 8 | 82.096% | 8.063% | 37.499% | -0.291 | +0.406 |
| 16 | 88.629% | 15.411% | 54.904% | -0.220 | +0.335 |
| 32 | 94.116% | 27.112% | 74.984% | +0.009 | +0.106 |
| 64 | 98.130% | 45.823% | 91.753% | +0.138 | -0.023 |

总 QK advantage 对 rank 分割是不变的；变化的是它被分配到主空间还是 tail。直到 rank 约 32，主空间才不再总体偏向 conflict。这说明仅看很低 rank 的 K 主方向，会系统性遗漏正确/矛盾判别方向。

## 4. Query 本身如何变化

加入 conflict 后：

- clean/conflict Q 的 raw cosine 为 0.98645 ± 0.00029；
- 相对变化范数为 0.12471 ± 0.00128；
- 该变化只有 12.91% 在 Top-16，87.09% 在 tail。

也就是说，conflict 没有让 query 完全换方向，而是在一个总体仍高度相似的 Q 上加入了以 tail 为主的偏转。这个偏转不是简单地朝 `K_gold-K_conflict` 移动：两者在 Top-16/tail 的 cosine 分别为 -0.0649/-0.0148，说明 query 的变化还包含传播到其他 token/功能头的间接效应。

## 5. 相同编号 token 的特别分析

每个 prompt 中，gold T0 和 conflict X0 使用同一个起始编号，例如同一个 `GA...-...`。平均每个编号为 7 个 sub-token，其中 5 个是数字 sub-token。

主要结果：

- RoPE 前，同数字 token 的 K cosine 为 0.9530；RoPE 后降到 0.8334；
- Top-16 cosine 仍高达 0.8970，tail 只有 0.4439；
- 同数字 token 的方向差异约一半在 Top-16、一半在 tail，但与 Q 有关的 gold 优势主要来自 tail。

这表明“token ID 相同”并不等于“在 KV cache 中可互换”。它们共享一个强的低秩身份/格式成分，但层内上下文化、规则角色和 RoPE 位置会在长尾产生很不同的实际 attention 几何。

位置顺序进一步说明它不是一个固定 token 属性：

| conflict 位置 | gold K 的 clean→conflict 相对变化 | Q 相对变化 | 同数字 Top-16 advantage | 同数字 Tail advantage |
|---|---:|---:|---:|---:|
| conflict 在 gold 后 | **0.000** | 0.12465 | -0.0648 | +0.7375 |
| conflict 在 gold 前 | 0.1800 | 0.12477 | -0.0509 | -0.0319 |

当 conflict 在 gold 后时，causal mask 保证 gold K 完全不变，实验确实得到精确的 0；但最终 Q 仍变化约 12.5%。同编号 tail 的 gold 优势主要出现在“更晚的 decoy 重复了同一编号”时，像是在抵消后出现的矛盾副本，而不是编码一个与位置无关的“正确标签”。

## 6. Head 异质性

448 个 layer/query-head 中，按 code 总 QK advantage 的符号：

- 251 个总体偏 gold；
- 197 个总体偏 conflict。

极端 conflict-favoring 主要集中在 Layer 7，例如：

| Head | KV head | Top-16 | Tail | 合计 |
|---|---:|---:|---:|---:|
| L07H03 | 1 | -11.990 | +0.562 | **-11.428** |
| L07H02 | 1 | -11.033 | -0.377 | **-11.410** |
| L07H10 | 5 | -10.946 | +0.074 | **-10.872** |

相反，Layer 4 有一组强 gold-favoring heads：

| Head | KV head | Top-16 | Tail | 合计 |
|---|---:|---:|---:|---:|
| L04H04 | 2 | +4.314 | +1.256 | **+5.570** |
| L04H05 | 2 | +4.620 | +0.858 | **+5.478** |

此前 attention-evidence 实验中的三个保守证据 head，在本实验中的表现为：

| Head | rule advantage | code advantage | 相同数字 advantage |
|---|---:|---:|---:|
| L26H02 | +0.076 | -0.148 | +0.215 |
| L17H08 | +0.387 | -0.007 | +0.025 |
| L21H13 | +0.401 | +0.248 | +0.429 |

这支持“不同 head 功能和失败模式不同”，但也说明不能只凭某个 head 是否看 evidence span 就推断它能正确区分 gold/conflict；例如它可能关注整条规则，却在 code 层面近乎中性或偏向 conflict。

## 7. 对外部检索方案的直接含义

1. **不要只用 K 的 top PCA/SVD 坐标做 retrieval。** 它能很好地召回同格式、同编号、同规则模板，却可能把 gold 和 conflict 一起召回。
2. **需要 tail-aware 或 query-aware 的打分。** 可以保留一个小的主空间通道做高召回，再用 head-specific tail projection 或轻量判别器重排。
3. **相同 token ID 不能共享一个静态 KV。** 至少要带上规则角色、局部上下文和位置；相同数字在 gold/decoy 中的 pre-RoPE K 已有差异，RoPE 后差异更大。
4. **按 head 分配检索方法是有必要的。** Layer 7 的冲突偏置与 Layer 4 的 gold 偏置方向相反；一个全局统一的相似度会把这两类需求平均掉。
5. **Top-2% oracle 的近似目标不应是重构 K variance。** 更合适的训练目标是复现每个 head 的 QK 排序或 gold-vs-conflict margin，并显式加入 hard-negative conflict chains。

## 8. 限制和下一步最关键实验

- 当前结论来自 Qwen3-0.6B 和合成的显式 `VERIFIED/DECOY RULE`；需要在更大模型和真实证据文本复现。
- 这是强配对几何证据，不等同于因果干预。最关键的后续实验是把实际 attention 的 Q/K 分别投影到 Top-r 或 tail 后重新跑 accuracy/PPL，验证删除 tail 是否会特异性加重 conflict failure。
- 同编号 gold/conflict 位于不同位置。这里保留这种设计是因为它对应真实 cache 中实际参与 QK 的向量，并额外报告了 RoPE 前结果；若要完全分离“规则角色”和“位置”，下一步应做等位置替换/标签交换实验。

## 9. 文件

- 主实验脚本：`src/run_k_svd_conflict_geometry.py`
- 汇总脚本：`src/summarize_k_svd_conflict_geometry.py`
- 本地汇总：`outputs/k_svd_geometry_64seed_final_20260717/final_results/`
- 远程原始 shard：`/home/fdong/ymluo/projects/qwen3_k_svd_conflict_geometry/outputs/k_svd_geometry_64seed_final_20260717/`

