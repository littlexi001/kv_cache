# MoR-KV 相关工作与 Novelty Audit（2026-07-11）

## 1. 审计结论

“不同 head 分配不同 KV budget”本身已经不够新。若论文只做到 retrieval/streaming 二分类、head importance 排名或动态预算，很容易被认为是 HeadKV、RazorAttention、DuoAttention、Task-KV、REAL、CompilerKV、PolyKV 或 HARD-KV 的直接变体。

MoR-KV 唯一有希望形成强 novelty 的主张是：

> 现有 head/layer-aware compression 主要路由保留量；MoR-KV 在每个 query 上、以 query-head/GQA-group 为粒度路由检索算子本身，并用 specialist-preserving objective 防止多数共识删除少数专业 head 发现的证据。

如果最终系统没有做到“query-conditioned operator identity + GQA physical realization + minority evidence objective”，则不应投稿为这个题目。

## 2. 最接近工作

### RazorAttention 与 DuoAttention

- RazorAttention 保留 retrieval heads 的完整 cache，非 retrieval heads 只保留 local/sink，并加入 compensation token。
- DuoAttention 用优化方法识别 retrieval/streaming heads；retrieval heads 仍保留 full KV。
- 边界：MoR-KV 不把远程 head 等同于 full cache，而是在多个远程 retrieval operators 中按 query 路由，目标是继续压缩 retrieval heads。

Primary sources:

- RazorAttention: https://openreview.net/forum?id=tkiZQlL04w
- DuoAttention: https://arxiv.org/abs/2410.10819

### HeadKV / HeadKV-R2

HeadKV 按 contextual retrieval/reasoning importance 在 heads 间重新分配 cache size。边界仍是 budget/importance allocation，不是每个 query 的 operator identity。

Primary source: https://openreview.net/forum?id=FJFVmeXusW

### Task-KV

Task-KV 用 semantic separator 区分 heterogeneous heads，并对不同任务动态感知 head semantic difference；heterogeneous heads 获得 full cache，其他 heads 保留 recent/sink 和 middle activation。

边界：MoR-KV 必须证明同一个远程 head 在不同 query 下需要不同 operator，而不只是 heterogeneous/full 与 homogeneous/local 的动态划分。

Primary source: https://arxiv.org/abs/2501.15113

### REAL

REAL 用 attention weight confusion matrix 描述 diverse attention behaviors，再用 inference score 做 head-wise dynamic budget allocation。该工作非常接近“attention behavior”叙事。

边界：MoR-KV 的核心不能只是行为分类，必须落在 heterogeneous retrieval operators、少数 specialist nomination 和 GQA block union。

Primary source: https://openreview.net/forum?id=XCqrMBh1Uj

### CompilerKV

CompilerKV 将 per-head reliability 和 prompt risk 编译为 offline tables，并显示跨 corpus head ranking 高稳定性。它直接覆盖“head prior + prompt risk”的组合。

边界：MoR-KV 的 offline compilation 应只作为组件；主创新是 operator portfolio，而不是 reliability table。

Primary source: https://arxiv.org/abs/2602.08686

### PolyKV

PolyKV 已经在 layer 级联合选择 compression method 和 non-uniform budget，是最危险的 novelty overlap。

必须保持四个差异：

1. PolyKV 是 layer-wise；MoR-KV 是 query-head/GQA-group；
2. PolyKV 的 layer action 是部署前配置；MoR-KV 每个 query 动态选择；
3. MoR-KV 路由的是 retrieval semantic（streaming/lexical/structure/QK/dense），不是一般 compression method grid；
4. MoR-KV 有专业 head 少数证据被共识删除的 failure mechanism 和 group-saturating objective。

Primary source: https://arxiv.org/abs/2606.15157

### HARD-KV

HARD-KV 解决 dynamic head budgeting 与 CUDA Graph/PagedAttention 静态布局之间的矛盾，并提供 contiguous physical rewrite。MoR-KV 必须正面比较其 runtime template 思路。

Primary source: https://arxiv.org/abs/2606.28831

### RedKnot

RedKnot 将 KV cache 视为按 KV heads 分解的结构化 serving object，支持 reuse、compression、hot/cold placement 和 distributed management。

边界：RedKnot 更偏 serving substrate；MoR-KV 是 query-conditioned sparse KV content access policy。系统实验应说明二者可组合。

Primary source: https://arxiv.org/abs/2606.06256

### KVzip

KVzip 用 context reconstruction 做 query-agnostic compression，并强调 compressed cache 在 multi-query 场景的复用。MoR-KV 的 query-aware优势必须在 multi-query cache reuse 成本下评估，不能默认每个 query 免费重建 index/cache。

Primary source: https://arxiv.org/abs/2505.23416

## 3. 当前已有的独立证据

1. Section 129：head attention-pattern ranking 在同关系改写下稳定，但 lexical/structure/semantic 跨域稳定性明显下降，支持“stable prior + query activation”。
2. Section 128：all-head Top-16 gold candidate union 为 71.88%，但多数共识 Recall@39 只有 28.13%；86 个被丢弃 gold blocks 平均仅获 2.45 heads 支持。
3. MoR-KV v1：错误路由显著降低 test utility；正确 score-signature route 在所有测试预算上超过单 operator/global hybrid 的预注册 utility。

## 4. Reviewer 最可能的拒稿理由

1. “这只是 PolyKV 从 layer 改成 head。”
2. “BM25 + QK 是普通 hybrid retrieval，不是 KV cache 方法。”
3. “Router 在人工模板上 100%，自然数据不会成立。”
4. “只报告 recall，没有真实 attention output、generation quality 和 kernel speed。”
5. “GQA heads 共享 K/V，所谓 per-head policy 无法物理实现。”
6. “方法 action search 依赖 task labels 或大量 calibration，线上不可用。”
7. “hard-negative utility 的 lambda 是人为选择。”

论文必须逐项用实验和实现关闭这些问题，而不是只在文字中辩解。
