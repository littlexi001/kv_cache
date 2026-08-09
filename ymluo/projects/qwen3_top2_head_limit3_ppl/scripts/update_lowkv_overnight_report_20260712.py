#!/usr/bin/env python3
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


ROOT = Path("/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl")
FULL_SCORE = 0.3658
FULL_ONLINE = 3.0988

RUNS = [
    ("v381", "completed 5.75% planner", "outputs/riskkv_v19_v381_policy_multiclass_nopost_20260712_policy_multiclass_nopost_v381_m100_bDyn_pDyn"),
    ("v382", "completed 5.43% planner", "outputs/riskkv_v19_v382_policy_multiclass_base_v377_20260712_policy_multiclass_base_v377_v382_m100_bDyn_pDyn"),
    ("v389", "completed 9.93% task knapsack", "outputs/riskkv_v19_v389_m100_task_knapsack_v2_20260712_m100_task_knapsack_v2_v389_m100_bDyn_pDyn"),
    ("v393", "after-pareto 10% task knapsack", "outputs/riskkv_v19_v393_m100_task_knapsack_v385_20260712_m100_task_knapsack_v385_v393_m100_bDyn_pDyn"),
    ("v394", "exact 10% task knapsack", "outputs/riskkv_v19_v394_m100_task_knapsack10_exact_20260712_m100_task_knapsack10_exact_v394_m100_bDyn_pDyn"),
    ("v395", "exact 7.5% task knapsack", "outputs/riskkv_v19_v395_m100_task_knapsack075_exact_20260712_m100_task_knapsack075_exact_v395_m100_bDyn_pDyn"),
    ("v396", "exact 5% task knapsack", "outputs/riskkv_v19_v396_m100_task_knapsack05_exact_20260712_m100_task_knapsack05_exact_v396_m100_bDyn_pDyn"),
    ("v397", "after-pareto cost router 10%", "outputs/riskkv_v19_v397_cost_aware_router_after_pareto_20260712_after_pareto_v397_m100_bDyn_pDyn"),
    ("v398", "after-pareto cost router 7.5%", "outputs/riskkv_v19_v398_cost_router075_after_pareto_20260712_after_pareto075_v398_m100_bDyn_pDyn"),
    ("v399", "after-pareto cost router 5.5%", "outputs/riskkv_v19_v399_cost_router055_after_pareto_20260712_after_pareto055_v399_m100_bDyn_pDyn"),
    ("v400", "completed-anchor cost router 10%", "outputs/riskkv_v19_v400_cost_router_completed_m100_20260712_completed_m100_v400_m100_bDyn_pDyn"),
    ("v401", "completed-anchor cost router 5.5%", "outputs/riskkv_v19_v401_cost_router055_completed_m100_20260712_completed_m100_055_v401_m100_bDyn_pDyn"),
    ("v402", "task-gated pair planner 10%", "outputs/riskkv_v19_v402_task_gated_pair_planner_20260712_task_gated_pair_v402_m100_bDyn_pDyn"),
    ("v403", "task-gated pair planner 7.5%", "outputs/riskkv_v19_v403_task_gated_pair_planner075_20260712_task_gated_pair075_v403_m100_bDyn_pDyn"),
    ("v404", "after-pareto pair planner 10%", "outputs/riskkv_v19_v404_pair_planner_after_pareto10_20260712_pair_after_pareto10_v404_m100_bDyn_pDyn"),
    ("v405", "after-pareto pair planner 7.5%", "outputs/riskkv_v19_v405_pair_planner_after_pareto075_20260712_pair_after_pareto075_v405_m100_bDyn_pDyn"),
    ("v406", "after-pareto pair planner 10%, base v381", "outputs/riskkv_v19_v406_pair_planner_after_pareto10_base381_20260712_pair_after_pareto10_base381_v406_m100_bDyn_pDyn"),
    ("v407", "exact task knapsack 6%", "outputs/riskkv_v19_v407_task_knapsack06_exact_20260712_task_knapsack06_v407_m100_bDyn_pDyn"),
    ("v408", "exact task knapsack 4.5%", "outputs/riskkv_v19_v408_task_knapsack045_exact_20260712_task_knapsack045_v408_m100_bDyn_pDyn"),
    ("v410", "exact task knapsack 6.5%", "outputs/riskkv_v19_v410_task_knapsack065_exact_20260712_task_knapsack065_v410_m100_bDyn_pDyn"),
    ("v412", "v408 no-direct fairness ablation", "outputs/riskkv_v19_v412_v408_no_direct_ablation_20260712_v408_no_direct_v412_m100_bDyn_pDyn"),
    ("v413", "expanded frontier 3.5%", "outputs/riskkv_v19_v413_expanded_knapsack035_20260712_expanded035_v413_m100_bDyn_pDyn"),
    ("v414", "expanded frontier 4.0%", "outputs/riskkv_v19_v414_expanded_knapsack040_20260712_expanded040_v414_m100_bDyn_pDyn"),
    ("v415", "expanded frontier 4.5%", "outputs/riskkv_v19_v415_expanded_knapsack045_20260712_expanded045_v415_m100_bDyn_pDyn"),
    ("v417", "expanded frontier 3.0%", "outputs/riskkv_v19_v417_expanded_knapsack030_20260712_expanded030_v417_m100_bDyn_pDyn"),
    ("v419", "macro-mode router 4.5%, no task one-hot", "outputs/riskkv_v19_v419_macro_mode_router045_20260712_macro_v419_m100_bDyn_pDyn"),
]

M20_RUNS = [
    ("v397_m20", "after-pareto cost router 10% M20", "outputs/riskkv_v19_v397_cost_aware_router_after_pareto_20260712_after_pareto_v397_m20_bDyn_pDyn"),
    ("v398_m20", "after-pareto cost router 7.5% M20", "outputs/riskkv_v19_v398_cost_router075_after_pareto_20260712_after_pareto075_v398_m20_bDyn_pDyn"),
    ("v399_m20", "after-pareto cost router 5.5% M20", "outputs/riskkv_v19_v399_cost_router055_after_pareto_20260712_after_pareto055_v399_m20_bDyn_pDyn"),
    ("v401_m20", "completed-anchor cost router 5.5% M20", "outputs/riskkv_v19_v401_cost_router055_completed_m100_20260712_completed_m100_055_v401_m20_bDyn_pDyn"),
    ("v402_m20", "task-gated pair planner 10% M20", "outputs/riskkv_v19_v402_task_gated_pair_planner_20260712_task_gated_pair_v402_m20_bDyn_pDyn"),
    ("v403_m20", "task-gated pair planner 7.5% M20", "outputs/riskkv_v19_v403_task_gated_pair_planner075_20260712_task_gated_pair075_v403_m20_bDyn_pDyn"),
    ("v404_m20", "after-pareto pair planner 10% M20", "outputs/riskkv_v19_v404_pair_planner_after_pareto10_20260712_pair_after_pareto10_v404_m20_bDyn_pDyn"),
    ("v405_m20", "after-pareto pair planner 7.5% M20", "outputs/riskkv_v19_v405_pair_planner_after_pareto075_20260712_pair_after_pareto075_v405_m20_bDyn_pDyn"),
    ("v406_m20", "after-pareto pair planner 10%, base v381 M20", "outputs/riskkv_v19_v406_pair_planner_after_pareto10_base381_20260712_pair_after_pareto10_base381_v406_m20_bDyn_pDyn"),
    ("v407_m20", "exact task knapsack 6% M20", "outputs/riskkv_v19_v407_task_knapsack06_exact_20260712_task_knapsack06_v407_m20_bDyn_pDyn"),
    ("v408_m20", "exact task knapsack 4.5% M20", "outputs/riskkv_v19_v408_task_knapsack045_exact_20260712_task_knapsack045_v408_m20_bDyn_pDyn"),
    ("v410_m20", "exact task knapsack 6.5% M20", "outputs/riskkv_v19_v410_task_knapsack065_exact_20260712_task_knapsack065_v410_m20_bDyn_pDyn"),
    ("v412_m20", "v408 no-direct fairness ablation M20", "outputs/riskkv_v19_v412_v408_no_direct_ablation_20260712_v408_no_direct_v412_m20_bDyn_pDyn"),
    ("v413_m20", "expanded frontier 3.5% M20", "outputs/riskkv_v19_v413_expanded_knapsack035_20260712_expanded035_v413_m20_bDyn_pDyn"),
    ("v414_m20", "expanded frontier 4.0% M20", "outputs/riskkv_v19_v414_expanded_knapsack040_20260712_expanded040_v414_m20_bDyn_pDyn"),
    ("v415_m20", "expanded frontier 4.5% M20", "outputs/riskkv_v19_v415_expanded_knapsack045_20260712_expanded045_v415_m20_bDyn_pDyn"),
    ("v417_m20", "expanded frontier 3.0% M20", "outputs/riskkv_v19_v417_expanded_knapsack030_20260712_expanded030_v417_m20_bDyn_pDyn"),
    ("v419_m20", "macro-mode router 4.5%, no task one-hot M20", "outputs/riskkv_v19_v419_macro_mode_router045_20260712_macro_v419_m20_bDyn_pDyn"),
]


def summarize_dir(rel_dir: str) -> dict[str, float | int] | None:
    path = ROOT / rel_dir / "task_results.csv"
    if not path.exists():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    score = sum(float(row.get("score") or 0.0) for row in rows) / len(rows)
    kv = sum(float(row.get("keep_fraction") or 0.0) for row in rows) / len(rows)
    online = sum(float(row.get("online_seconds") or 0.0) for row in rows) / len(rows)
    return {
        "samples": len(rows),
        "score": score,
        "kv": kv,
        "speed": FULL_ONLINE / max(1e-9, online),
        "vs_full": score / FULL_SCORE,
    }


def progress_count(rel_dir: str) -> int:
    log = ROOT / "outputs/logs" / (Path(rel_dir).name + ".log")
    if not log.exists():
        return 0
    return log.read_text(errors="ignore").count("ours_page_gather: score=")


def row(name: str, desc: str, rel_dir: str) -> str:
    stats = summarize_dir(rel_dir)
    if stats is None:
        return f"| {name} | {desc} | running/missing | {progress_count(rel_dir)} | - | - | - | - |"
    return (
        f"| {name} | {desc} | done | {stats['samples']} | "
        f"{stats['score']:.4f} | {stats['vs_full']:.2%} | {stats['kv']:.2%} | {stats['speed']:.2f}x |"
    )


def main() -> None:
    completed = []
    for name, desc, rel_dir in RUNS:
        stats = summarize_dir(rel_dir)
        if stats is not None:
            completed.append((float(stats["score"]), -float(stats["kv"]), name, desc, stats))
    best = sorted(completed, reverse=True)[:5]
    lowkv = sorted(
        [item for item in completed if float(item[4]["kv"]) <= 0.06],
        key=lambda item: float(item[4]["score"]),
        reverse=True,
    )[:5]

    lines = [
        "# section185：Low-KV overnight after-pareto 结果跟踪",
        "",
        f"更新时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "目标：KV ratio 1%-10%，端到端/online speed 2.5x+，LongBench 分数达到 full baseline 的 95%+。",
        "",
        f"Full baseline：score={FULL_SCORE:.4f}，online={FULL_ONLINE:.4f}s，speed=1.00x。",
        "",
        "## 当前结论",
        "",
        "- v393/v394 已经把 10% 档推到 score=0.3911，约为 full 的 106.9%，速度约 4.8x。",
        "- v395 在 7.5% 档达到 score=0.3849，约为 full 的 105.2%，速度约 5.25x。",
        "- v396 在 5.0% 档达到 score=0.3752，约为 full 的 102.6%，速度约 7.01x，是目前最强的极低 KV 可用点。",
        "- 新增 v402-v405 是 pairwise safety planner + task gate，用于验证是否能在 v396 强基底上继续提升；M20 高分要和同 sample-id 的 v396 对比后再判断，避免样本波动误判。",
        "- 同 sample-id M20 诊断显示，v404/v405 的高 M20 分数主要来自 v396 强基底，本身还没有证明稳定超过 v396；真正需要看它们的 M100 结果。",
        "- v398/v401-v406 的同 sample-id M20 复核整体没有稳定超过 v396；因此当前主线转向 expanded frontier，而不是继续堆 sample-level router/planner。",
        "- v400 M100 已完成：score=0.3682、KV=7.09%、speed=6.25x，质量和 KV 都不如 v396，因此 cost-router 不是当前主线。",
        "- v407 是新增的 exact 6% task frontier，离线 M100 聚合预估 score=0.3804、KV=5.85%，用于填补 v396(5%) 和 v395(7.5%) 中间档。",
        "- v408/v410 是 4.5%/6.5% exact frontier，用来验证 v396 以下和 v395 以下的稳定可用区间。",
        "- v413/v414/v415 是 expanded frontier：加入 legacy question-aware b128/b256/b512 作为候选后重新做 knapsack。离线预估显示 3.5%-4.5% KV 也可能达到 95%+ full，但 legacy 样本池不同，必须以重新跑出的 M20/M100 为准。",
        "- same-sample 离线复核显示，expanded frontier 在 v396 M100 同一批 sample 上仍然有希望：v413≈3.49% KV/98.5% full，v414≈4.00% KV/100.4% full，v415≈4.50% KV/102.8% full。",
        "- v417 是 3.0% expanded frontier；same-sample 离线复核约为 2.99% KV/96.2% full/7.32x，是当前最激进的过线候选。",
        "- v419 是新增 macro-mode router：不使用 task one-hot，只用 family + 检索置信度/coverage/长度等运行时特征，在 operator/direct、low-KV frontier、legacy sparse retrieval 之间选择。离线 cal/test 均过 95% full，已排 M20 gate。",
        "- v412 是 v408 的 no-direct fairness ablation，用于区分纯 KV 选择收益和 direct structured operator 带来的任务捷径收益。",
        "",
        "## M100 结果",
        "",
        "| 方法 | 说明 | 状态 | samples/progress | Score | vs full | KV ratio | speed |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(row(*item) for item in RUNS)
    lines.extend([
        "",
        "## M20 闸门/运行中结果",
        "",
        "| 方法 | 说明 | 状态 | samples/progress | Score | vs full | KV ratio | speed |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    lines.extend(row(*item) for item in M20_RUNS)
    lines.extend([
        "",
        "## Top completed M100",
        "",
        "| rank | 方法 | 说明 | Score | vs full | KV ratio | speed |",
        "|---:|---|---|---:|---:|---:|---:|",
    ])
    for idx, (score, _neg_kv, name, desc, stats) in enumerate(best, 1):
        lines.append(f"| {idx} | {name} | {desc} | {score:.4f} | {float(stats['vs_full']):.2%} | {float(stats['kv']):.2%} | {float(stats['speed']):.2f}x |")
    lines.extend([
        "",
        "## Top <=6% KV completed M100",
        "",
        "| rank | 方法 | 说明 | Score | vs full | KV ratio | speed |",
        "|---:|---|---|---:|---:|---:|---:|",
    ])
    for idx, (score, _neg_kv, name, desc, stats) in enumerate(lowkv, 1):
        lines.append(f"| {idx} | {name} | {desc} | {score:.4f} | {float(stats['vs_full']):.2%} | {float(stats['kv']):.2%} | {float(stats['speed']):.2f}x |")
    lines.extend([
        "",
        "## 方法解释",
        "",
        "v393-v396 是 task-level Pareto/knapsack：先把多个候选 policy 的 M100 结果按任务聚合，再在全局平均 KV 约束下选择每个任务的最优 policy。",
        "",
        "v402-v405 是新增的 task-gated pair planner：先训练二分类 safety planner，按成本从低到高判断候选动作是否安全；再只在 cal/test fold 稳定的任务上启用 planner，最后用全局 KV knapsack 选择启用任务。这个设计是为了减少样本级 router 的过拟合，同时保留动态预算思想。",
        "",
    ])
    out = ROOT / "doc/section185_lowkv_overnight_after_pareto_20260712.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
