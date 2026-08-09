# Qwen3-0.6B 不同 Head 功能结果总结

日期：2026-07-16  
范围：28 层 × 16 query heads，共 448 个 query heads

## 1. 总体结论

- 只有 172/448（38.4%）个 head 通过严格专门化阈值；276/448（61.6%）属于混合或通用。不能把每个 head 当作永久固定的单功能模块。
- 强制主签名用于完整枚举：self 55、previous-token 56、local-recent 15、sink 80、punctuation 63、lexical-copy 28、syntax 22、structural-anchor 27、semantic-evidence 102。
- 功能轮廓整体可复现：稳定偏置 273、中等稳定 112、上下文敏感 63。attention 主类与局部输出因果主类的一致率为 74.8%。
- 置信度为高 127、中 168、低 153。全部低置信 head 中绝大多数属于混合/通用，不适合直接绑定固定检索器。
- 精确 Oracle Top-2% 在 512-token 长测中将 PPL 从 25.3298 降至 25.1366，证明 head-specific 稀疏选择存在真实上界；当前外部检索还不能稳定复现这一收益。

## 2. 九类功能

| 功能 | 保守标签数 / 强制主签名数 | 层分布与代表 head | 因果/证据结果 | KV Retrieval 建议 |
|---|---:|---|---|---|
| 当前 token / self | 48 / 55 | 各层都有，早层较多；L00H03、L04H04、L26H09 | L00H03 删除 self link 后 ΔNLL +2.437，是最强正因果结果 | self 永久保留，不交给外部检索 |
| 前一 token | 38 / 56 | 早中层为主；L00H02、L01H03、L15H03 | L00H02 ΔNLL +0.076；其余代表 head 较弱 | 固定保留 previous/recent，避免远程检索替换 |
| 局部近期上下文 | 6 / 15 | 5/6 个专门化 head 在 L0–L8；L01H04、L02H06 | 代表 head ΔNLL 约 +0.003～+0.005 | recent window；通常不需要内容检索 |
| 序列起点 / sink | 9 / 80 | 专门化 head 7/9 在 L19–L27；L27H03、L19H04 | L27H03 ΔNLL +0.087 | 少量 sink 永久保留，与 recent 组合 |
| 标点与边界 | 19 / 63 | 15/19 在早层；L01H12、L02H07、L10H06 | 端到端 ΔNLL 接近 0，更多体现边界/格式组织 | 标点、换行、段落边界索引；保守使用 |
| 同词回指 / 复制 | 12 / 28 | 早中层；L02H03、L02H11、L06H13 | L02H03 ΔNLL +0.015；L06H13 在冲突下 gold-vs-decoy 比值很强 | BM25/倒排表、exact match、重复 span 检索 |
| 句法依赖 | 14 / 22 | 中层为主；L10H10、L12H04、L11H05 | 三个代表 head 的后继-token ΔNLL 为负，尚无正端到端支持 | 只作为实验路由；需要实体/依存检索和 full fallback |
| 结构锚点 | 17 / 27 | 中层为主；L11H14、L03H03、L08H00 | 有局部输出因果证据；因锚点位于样本尾部，未构造端到端 NLL | 标题、字段名、分隔符、缩进/段落锚点检索 |
| 语义证据 | 9 / 102 | 9/9 都在 L19–L27；L21H13、L24H11、L24H12 | ΔNLL 分别 +0.919、+0.085、+0.026，语义功能因果证据最强 | 独立句向量/证据 span 检索，再做 token 级精排 |

## 3. 混合/通用 Head

276 个混合/通用 head 中，中置信 125、低置信 151，没有高置信单功能结论。它们并非“不重要”：真实证据关注排名靠前的 L26H02、L17H08、L26H10、L25H07 等很多都在这一组。这说明受控探针的九类标签没有覆盖全部自然任务行为。

对这组 head，不应按静态功能分配检索器。更合理的是：以 head 图谱作为 prior，再让 query-conditioned gate 在 recent、lexical、semantic、format、full/QK fallback 之间动态选择。

## 4. 真实证据与冲突数据

### 无冲突

- gold selectivity 最高：L26H02=0.480、L26H10=0.334、L25H07=0.295、L21H05=0.272、L15H11=0.252。
- Top-2% gold token recall 最高：L17H08=0.435、L21H13=0.371、L19H12=0.345、L17H01=0.332、L18H02=0.304。

### 四条竞争证据链

- gold attention mass 最高：L21H13=0.052、L17H08=0.042、L26H02=0.039、L26H00=0.037、L24H14=0.035。
- gold-vs-decoy log2 density ratio 最高：L04H00=1.858、L06H13=1.251、L26H02=1.183、L04H01=0.663、L06H10=0.417。
- 冲突造成 gold mass 降幅最大：L17H08=-0.0105、L21H13=-0.0080、L24H15=-0.0062、L21H01=-0.0058、L24H14=-0.0049。

L21H13 的语义因果作用和冲突场景 gold mass 都很强，但也明显受 decoy 干扰。L26H02 的无冲突 selectivity 第一，强冲突判别仍居前三，是目前更均衡的真实证据 head。

## 5. 稀疏 PPL 与外部检索结果

### 精确 Oracle 上界

| 测试窗口 | Full PPL | 每 head Oracle Top-2% PPL | 相对变化 |
|---:|---:|---:|---:|
| 64 token | 46.3085 | 43.7253 | -5.58% |
| 512 token | 25.3298 | 25.1366 | -0.76% |

Oracle 每个 head 平均只保留约 330–335 个历史 token，但需要先计算完整 QK，因此它是质量上界，不是低成本方案。

### 功能门控外部记忆

16K prefill、64-token 小样本：

| 外部记忆策略 | 激活 heads | PPL |
|---|---:|---:|
| recent-500 | 0 | 48.6790 |
| 所有 head 使用分层检索 | 448 | 57.1037 |
| 仅 semantic-evidence | 9 | 48.4223 |
| 仅 structural-anchor | 15 | 48.8560 |
| 仅 lexical-copy | 12 | 51.1389 |

语义组在 64-token 小样本上看似略优，但 512-token 长测为 30.7213，recent-500 为 30.7007，优势没有复现。结论是：功能门控是必要条件，但当前 mean-token-embedding 检索器过弱；功能标签本身不能替代 query-aware 检索质量。

## 6. 可直接用于系统设计的路由规则

1. self、previous、local-recent、sink：使用确定性位置规则，不消耗外部语义检索预算。
2. lexical-copy：使用倒排表/BM25/repeated-span；当前类别级 PPL 失败意味着还需要逐 head gate，不能把 12 个 head 一起开启。
3. punctuation、structural-anchor：检索标题、字段名、换行、分隔符和段落边界；只在结构化输入启用。
4. semantic-evidence：使用独立 sentence/paragraph encoder 找块，再做证据 span 与 token 精排；优先验证 L21H13、L24H11、L24H12，但必须加入冲突/decoy 惩罚。
5. syntax：当前没有正端到端支持，默认 recent/full fallback。
6. mixed/common：使用 head prior + query-conditioned gate，不静态绑定检索器。

## 7. 文件索引

- 全部 448 个 head 逐层卡片：`all_448_head_cards_20260715.md`
- 完整数值表：`../outputs/head_function_atlas.csv`
- 原研究报告：`qwen3_0p6b_head_function_atlas_20260715.md`
- 分层记忆与 Oracle PPL：`../../qwen3_per_head_hierarchical_memory/docs/results_20260716_zh.md`

