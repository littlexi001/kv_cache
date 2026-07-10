# Section 120: Graph-bridge multihop evidence selection

## 背景

v86-v88 的结果说明，HotpotQA 和 MuSiQue 不能简单用固定 2048 budget 替代 full fallback：

- HotpotQA：0.4008 / 100% keep -> 0.2517 / 52.3% keep
- MuSiQue：0.3000 / 100% keep -> 0.1834 / 45.3% keep

这说明压缩空间存在，但当前 block scorer 没有稳定找全多跳证据链。

## 新机制

新增 `graph_bridge` 模块：

1. 用 query-aware scorer 选出若干 top seed pages。
2. 从 seed page 中提取长尾实体/数字/专名。
3. 在全上下文里召回共享这些稀有实体的 second-hop pages。
4. 按 `entity_link_idf + seed_score + page_score` 给 evidence pair 排序。
5. 在预算内成对保留 seed page 和 linked page。

它和原来的 bridge 不同：原 bridge 只在某个页面被主循环选中后做局部扩展；graph bridge 在 regular fill 前主动做二跳图搜索，目标是避免第一跳选错后无法补回第二跳证据。

## 实现字段

新增配置：

- `graph_bridge`
- `graph_bridge_budget_fraction`
- `graph_bridge_seed_pages`
- `graph_bridge_max_terms`
- `graph_bridge_min_score`

输出字段：

- `ours_graph_bridge_active`
- `ours_graph_bridge_pairs`
- `ours_graph_bridge_tokens`

## v92 设置

配置：

```text
configs/riskkv_task_policy_v92_graph_bridge_multihop_20260709.json
```

关键设置：

- HotpotQA：3072 budget，45% graph-bridge reserve。
- MuSiQue：3072 budget，48% graph-bridge reserve。
- Qasper：沿用 v81 的 2048 bridge budget。

启动：

```bash
SAMPLES=20 TASKS=hotpotqa,musique,qasper GPUS=5 \
  nohup bash scripts/run_riskkv_v92_graph_bridge_20260709.sh \
  > outputs/logs/run_riskkv_v92_graph_bridge_20260709.nohup.log 2>&1 &
```

## 判定

如果 v92 在 HotpotQA/MuSiQue 上接近 v81 full fallback 分数，同时 keep 明显低于 100%，则可以把论文故事从“预算 router”升级为：

```text
Risk-conditioned evidence graph memory selection
```

这比普通 RAG 或 top-k KV pruning 更有区分度，因为它不检索外部文档，也不只压缩注意力分数，而是在已预填充上下文的 KV cache 上做结构化证据链选择。

## 结果

v92 原版 targeted m20：

| 任务 | Score | KV keep | 备注 |
| --- | ---: | ---: | --- |
| HotpotQA | 0.2558 | 44.61% | 低于 v81 的 0.4008 |
| MuSiQue | 0.1333 | 43.94% | 低于 v81 的 0.3000 |
| Qasper | 0.5331 | 46.94% | 与 v81 持平 |
| Overall | 0.3074 | 45.16% | KV 很低，但 multi-hop 质量不够 |

v92 query-seeded graph bridge targeted m20：

| 任务 | Score | KV keep | 备注 |
| --- | ---: | ---: | --- |
| HotpotQA | 0.2558 | 44.61% | 与原 v92 基本一致 |
| MuSiQue | 0.1833 | 45.98% | 比原 v92 好，但仍低于 v81 |
| Qasper | 0.5331 | 46.94% | 与 v81 持平 |
| Overall | 0.3241 | 45.84% | 有改善，但不能作为主线 |

结论：graph bridge 能保持 Qasper，但没有解决 HotpotQA/MuSiQue 的核心错误。当前 multi-hop 低预算失败不是因为没有足够二跳候选，而是 block scorer/证据链选择仍不能稳定定位必要证据。后续不应继续盲目压 HotpotQA/MuSiQue；更合理的是把它们作为 high-risk family，用 full 或更高预算保护，同时从其他任务和长上下文场景兑现压缩收益。
