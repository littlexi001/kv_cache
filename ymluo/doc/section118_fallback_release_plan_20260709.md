# Section 118: Full-fallback release sweep

## 动机

当前最稳主线 v72/v81 的整体 KV keep ratio 仍然在 56%-60% 左右，主要原因不是基础检索预算过高，而是若干 LongBench 任务仍然使用 `full_fallback`。这会让方法更稳，但作为 KV cache 压缩论文主结果偏保守。

## 诊断

v81 的局部结果显示：

- `qasper` 已经从 full fallback 改成 1024/2048 adaptive 预算，局部 keep 约 42%-47%。
- `hotpotqa`、`musique`、`trec`、`passage_count`、`repobench-p` 仍然是 full fallback 或近似 full keep，是当前全局 KV ratio 的主要瓶颈。
- 直接做全局低预算已经失败过，原因是不同任务的最小安全预算差异很大。

截至 2026-07-09 12:56，完整 m20 结果：

| 方法 | Samples | Score | KV keep | 说明 |
| --- | ---: | ---: | ---: | --- |
| v72 | 320 | 0.37990 | 61.62% | qasper 仍 full fallback |
| v81 | 320 | 0.38037 | 58.31% | qasper budgeted，质量略升且多省约 3.31 个百分点 KV |

结论：qasper release 是正向结果，但只释放 qasper 还不够。下一步必须处理剩余 full-fallback 任务，否则全局 keep 很难降到 30%-40% 论文主结果区间。

## 新一轮实验

新增 v86-v90，目标是逐个释放 full fallback，寻找每个任务的最小安全动作。

| 配置 | 改动 | 目的 |
| --- | --- | --- |
| v86 | 只释放 `hotpotqa`，2048 budget + bridge + certificate + risk escalation | 判断 HotpotQA 是否真的需要 full KV |
| v87 | 只释放 `musique`，2048 budget + bridge + certificate + risk escalation | 判断 MuSiQue 是否真的需要 full KV |
| v88 | 同时释放 `hotpotqa` 和 `musique` | 测试两个 multi-hop QA 同时压缩时是否可叠加 |
| v89 | 释放 `trec`、`passage_count`、`repobench-p` | 测试分类、计数、代码任务能否脱离 full fallback |
| v90 | 释放所有上述 full-fallback 任务，并保留 qasper adaptive | 探索低 KV 主线候选 |

## 判定标准

短期先跑 m20 targeted sweep：

- 如果单任务分数接近 v81/full，并显著降低 keep ratio，则把该任务的 release 策略并入下一版主线。
- 如果单任务明显掉分，则保留 full fallback，并把它作为论文里的 hard case/negative evidence。
- 最值得关注的是 v90：如果它能把 targeted tasks 的 keep ratio 明显压低且整体不崩，下一步跑完整 LongBench m50/m100。

## 运行脚本

```bash
nohup bash scripts/run_riskkv_v86_v90_fallback_release_sweep_20260709.sh \
  > outputs/logs/run_riskkv_v86_v90_fallback_release_sweep_20260709.nohup.log 2>&1 &
```

本轮已在服务器启动，限制使用 GPU 5/6/7：

```bash
SAMPLES=20 GPUS=5,6,7 TASKS=hotpotqa,musique,trec,passage_count,repobench-p,qasper \
  STAMP=20260709_fallback_release_sweep \
  nohup bash scripts/run_riskkv_v86_v90_fallback_release_sweep_20260709.sh \
  > outputs/logs/run_riskkv_v86_v90_fallback_release_sweep_20260709.nohup.log 2>&1 &
```

当前运行状态：

- `v86_hotpot2048` 已完成 targeted m20。
- `v87_musique2048` 已完成 targeted m20。
- `v88_hotpot_musique2048` 已完成 targeted m20。
- `v89_static_release` 首次运行在 GPU 竞争下 OOM，已用 `v89_static_release_retry` 重跑。
- `v90_full_release_adaptive` 正在继续跑。

## 自动选择器

新增脚本：

```bash
scripts/select_fallback_release_candidates_20260709.py
```

它会以 v81 m20 为 baseline，对每个 targeted task 选择满足质量约束的最低 KV keep 候选动作：

```bash
python scripts/select_fallback_release_candidates_20260709.py \
  --baseline-log logs/riskkv_v81_v72_qasper_budgeted_m20_20260709.log \
  --quality-floor 0.02 \
  v86=outputs/logs/riskkv_v19_v86_hotpot2048_20260709_fallback_release_sweep_m20_bDyn_pDyn.log \
  v87=outputs/logs/riskkv_v19_v87_musique2048_20260709_fallback_release_sweep_m20_bDyn_pDyn.log \
  v88=outputs/logs/riskkv_v19_v88_hotpot_musique2048_20260709_fallback_release_sweep_m20_bDyn_pDyn.log \
  v89=outputs/logs/riskkv_v19_v89_static_release_20260709_fallback_release_sweep_m20_bDyn_pDyn.log \
  v90=outputs/logs/riskkv_v19_v90_full_release_adaptive_20260709_fallback_release_sweep_m20_bDyn_pDyn.log
```

## 已完成的 targeted 结果

| 方法 | Overall score | KV keep | 结论 |
| --- | ---: | ---: | --- |
| v86 hotpot2048 | 0.41161 | 82.40% | 只释放 HotpotQA 会显著拉低 HotpotQA 分数，不可直接并入 |
| v87 musique2048 | 0.41703 | 81.24% | 只释放 MuSiQue 会显著拉低 MuSiQue 分数，不可直接并入 |
| v88 hotpot+musique2048 | 0.39217 | 73.28% | KV 降低明显，但 multi-hop QA 分数同时掉，说明需要样本级风险路由 |

关键单任务结果：

| 方法 | HotpotQA score/keep | MuSiQue score/keep | Qasper score/keep |
| --- | --- | --- | --- |
| v81 baseline | 0.4008 / 100.00% | 0.3000 / 100.00% | 0.5332 / 46.94% |
| v86 | 0.2517 / 52.26% | 0.3000 / 100.00% | 0.5014 / 42.13% |
| v87 | 0.4008 / 100.00% | 0.1834 / 45.30% | 0.5014 / 42.13% |
| v88 | 0.2517 / 52.26% | 0.1834 / 45.30% | 0.5014 / 42.13% |

这说明 HotpotQA/MuSiQue 不是“无脑可以压”的任务。固定 2048 的 release 会丢失关键多跳证据；下一步必须做样本级风险预算，而不是任务级固定预算。

## 后续 fixed/adaptive release 结果

v90 full-release adaptive targeted m20 已完成：

| 任务 | Score | KV keep | 对 v81 的结论 |
| --- | ---: | ---: | --- |
| HotpotQA | 0.2517 | 52.26% | 明显掉分 |
| MuSiQue | 0.1833 | 45.30% | 明显掉分 |
| TREC | 0.7000 | 29.38% | 低于 v81 的 0.7500 |
| PassageCount | 0.0100 | 28.58% | 明显掉分 |
| RepoBench-P | 0.4094 | 39.53% | 低于 v81 的 0.5167 |
| Qasper | 0.5013 | 42.13% | 低于 v81 的 0.5332 |

Overall：Score 0.3426，KV keep 39.53%。

结论：v90 虽然把 KV keep 压到 40% 左右，但质量不可接受，不能作为主线。

static-only retry 完整结果也不理想：

- TREC：20 samples，Score 0.7000，KV keep 29.38%，低于 v81 的 0.7500。
- PassageCount：20 samples，Score 0.0100，KV keep 28.58%，明显低于 v81 的 0.1500。
- RepoBench-P：20 samples，Score 0.4094，KV keep 39.53%，低于 v81 的 0.5167。
- Overall：Score 0.3731，KV keep 32.50%。

因此，当前可安全并入主线的 release 仍然主要是 qasper budgeted；其他 full-fallback 任务需要继续保护或重新设计更强证据选择机制。
