from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def pick(rows: list[dict[str, Any]], **conditions: Any) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in conditions.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one match for {conditions}, got {len(matches)}")
    return matches[0]


def percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def number(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--llama_logit", type=Path, required=True)
    parser.add_argument("--qwen_logit", type=Path, required=True)
    parser.add_argument("--crossing", type=Path, required=True)
    parser.add_argument("--qk_spectrum", type=Path, required=True)
    parser.add_argument("--fixed_basis", type=Path, required=True)
    parser.add_argument("--actual_budget", type=Path, required=True)
    parser.add_argument("--long_speed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    comparison = load_json(args.comparison)
    llama_logit = load_json(args.llama_logit)
    qwen_logit = load_json(args.qwen_logit)
    crossing = load_json(args.crossing)
    qk_spectrum = load_json(args.qk_spectrum)
    fixed_basis = load_json(args.fixed_basis)
    actual_budget = load_json(args.actual_budget)
    long_speed = load_json(args.long_speed)

    longbench = [
        row for row in comparison["countcap"] if row["subset"] == "longbench16"
    ]
    by_task = comparison["countcap_by_task"]
    baseline_rows = comparison["same_environment_baselines"]
    crossing_row = pick(
        crossing["candidate_overall"],
        method="production_pca48_int4k_int8q",
        fraction=0.04,
    )
    sampled_threshold_row = pick(
        crossing["candidate_overall"],
        method=(
            "production_pca48_int4k_int8q_"
            "sampled_quantile_uncapped"
        ),
        fraction=0.04,
    )
    logit_rows = {
        "Llama-3.1-8B-Instruct": pick(llama_logit, topic="ALL"),
        "Qwen3-4B-Instruct": pick(qwen_logit, topic="ALL"),
    }
    qk_rows = {row["model"]: row for row in qk_spectrum["by_model"]}

    lines = [
        "# Direct CountCap：理论与双模型正式结果",
        "",
        "更新时间：2026-07-26",
        "",
        "## 1. 冻结方法",
        "",
        "当前方法在首个 2048-token prefill chunk 上建立 sampled uncentered "
        "PCA48，随后固定；再使用 grouped log-scale INT4 Key + "
        "INT8 Query + 256 点 sampled-quantile + direct sparse attention。"
        "它不使用训练式 router、任务标签、exact-QK 重排或 Full 回退。",
        "",
        "目标预算为：",
        "",
        "$$",
        "B(N)=\\min\\left(N,1280,\\max\\left(256,\\left\\lceil0.06N\\right\\rceil\\right)\\right).",
        "$$",
        "",
        "当 $N\\le256$ 时，$B(N)=N$，此时精确消费全部可用历史。"
        "这是预算公式的饱和边界，不是风险、任务或成本触发的 Full fallback。",
        "",
        "`1280` 是分位数目标，不是当前 kernel 的绝对 hard cap；正式方法说明见 "
        "`docs/20260726_direct_countcap_frozen_method_reproduction_zh.md`。",
        "",
        "## 2. LongBench 16 任务",
        "",
        "| 模型 | Full Macro | CountCap Macro | 质量保持率 | 95% CI（Macro 差值） | 平均目标 token/head | Decode ms/token speed | Online ms/token speed | 整样本 Total latency speed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in longbench:
        lines.append(
            "| {model} | {full} | {sparse} | {retention} | "
            "[{low}, {high}] | {tokens:.1f} | {decode:.3f}x | "
            "{online:.3f}x | {total:.3f}x |".format(
                model=row["model"],
                full=number(row["full_macro"]),
                sparse=number(row["countcap_macro"]),
                retention=percent(row["quality_retention"]),
                low=number(row["macro_delta_ci95_low"]),
                high=number(row["macro_delta_ci95_high"]),
                tokens=float(row["mean_attention_tokens"]),
                decode=float(row["paired_decode_per_token_speedup"]),
                online=float(row["paired_online_per_token_speedup"]),
                total=float(row["paired_total_speedup"]),
            )
        )

    lines.extend(
        [
            "",
            "`Decode/Online speed` 按各方法实际生成 token 数归一化；`Total latency "
            "speed` 是整条样本延迟比，会受输出长度差异影响，只作为应用延迟补充，"
            "不作为纯 decode kernel 加速结论。",
            "",
            "### sampled-quantile 实际消费审计",
            "",
            "| 模型 | 样本 | 平均 prompt | 目标比例 | 实际比例 | "
            "样本内 p95 比例均值 | 最大比例 | 实际 token/head 均值 | "
            "p95 token 均值 | 最大 token | capacity overflow head 比例 | "
            "host-side proxy top-k fallback 比例 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in actual_budget["overall"]:
        lines.append(
            "| {model} | {samples} | {prompt:.1f} | {target} | {actual} | "
                "{p95_fraction} | {max_fraction} | {count:.1f} | "
                "{p95_count:.1f} | {max_count:.0f} | {overflow} | "
                "{fallback} |".format(
                model=row["model"],
                samples=int(row["samples"]),
                prompt=float(row["prompt_tokens_mean"]),
                target=percent(row["target_fraction_mean"]),
                actual=percent(row["actual_fraction_mean"]),
                p95_fraction=percent(row["actual_fraction_p95_mean"]),
                max_fraction=percent(row["actual_fraction_max"]),
                count=float(row["actual_count_mean"]),
                p95_count=float(row["actual_count_p95_mean"]),
                max_count=float(row["actual_count_max"]),
                overflow=percent(
                    row["candidate_overflow_head_fraction_mean"]
                ),
                fallback=percent(
                    row["sampled_quantile_fallback_rate_mean"]
                ),
            )
        )

    lines.extend(
        [
            "",
            "实际消费来自开启诊断的独立 m4 审计，不使用目标预算代替执行数量。"
            "该诊断会增加开销，因此其时间不用于速度主张。",
            "冻结的 qprojscan/qkvfused 是异步 sampled-quantile 路径：极少数"
            "超过 capacity 的 head 由 CUDA kernel 在 capacity 处截断，不触发"
            "host-side proxy top-k。表中的 fallback 为 0；即使非异步变体触发"
            "该字段，它也只表示完整低比特 proxy top-k，不是 Full Attention。",
            "",
            "7.5K 左右的短上下文结果用于验证质量，不构成长序列加速主张。"
            "该 runner 保留完整 GPU FP16 K/V。",
            "",
            "### 64K/128K 冻结方法配对测速",
            "",
            "| 模型 | 历史长度 | 配对 case | Full PPL | CountCap PPL | "
            "PPL 保持率 | 实际 token/head | per-head 范围 | 实际比例 | Full ms/token | "
            "CountCap ms/token | $\\Delta T_{fixed}$ (s) | break-even token | "
            "Decode speed | Prefill+decode speed |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in long_speed:
        actual_min = row.get("actual_attention_tokens_min")
        actual_max = row.get("actual_attention_tokens_max")
        actual_range = (
            "-"
            if actual_min is None or actual_max is None
            else f"{float(actual_min):.0f}-{float(actual_max):.0f}"
        )
        fixed_seconds = row.get("additional_fixed_seconds_per_case")
        break_even = row.get("break_even_decode_steps")
        fixed_text = (
            "-" if fixed_seconds is None else f"{float(fixed_seconds):+.3f}"
        )
        break_even_text = (
            "-"
            if break_even is None
            else f"{math.ceil(float(break_even))}"
        )
        lines.append(
            "| {model} | {length} | {cases} | {full:.4f} | "
            "{direct:.4f} | {retention} | {tokens:.1f} | {actual_range} | {ratio} | "
            "{full_ms:.2f} | {direct_ms:.2f} | {fixed} | {break_even} | "
            "{decode:.3f}x | "
            "{protocol:.3f}x |".format(
                model=row["model"],
                length=int(row["history_tokens"]),
                cases=int(row["paired_cases"]),
                full=float(row["full_ppl"]),
                direct=float(row["direct_ppl"]),
                retention=percent(row["ppl_retention"]),
                tokens=float(row["actual_attention_tokens"]),
                actual_range=actual_range,
                ratio=percent(row["actual_attention_ratio"]),
                full_ms=float(row["full_milliseconds_per_step"]),
                direct_ms=float(row["direct_milliseconds_per_step"]),
                fixed=fixed_text,
                break_even=break_even_text,
                decode=float(row["decode_speedup"]),
                protocol=float(row["protocol_speedup"]),
            )
        )

    worst_long = min(long_speed, key=lambda row: float(row["ppl_retention"]))
    lines.extend(
        [
            "",
            "该表在相同文本窗口、相同预测 token 数下配对；`Decode speed` 包含 "
            "PCA/INT4 检索、sampled threshold、候选 gather 和精确稀疏 "
            "attention。`Prefill+decode speed` 还计入 dense prefill 与索引构建。",
            "",
            "令 $F$ 为每个请求的 prefill/索引固定成本，$t$ 为每个 decode "
            "step 的在线成本，则摊销交叉点按配对计时定义为",
            "",
            "$$",
            "n^*=\\max\\left(0,"
            "\\frac{F_{\\mathrm{CountCap}}-F_{\\mathrm{Full}}}"
            "{t_{\\mathrm{Full}}-t_{\\mathrm{CountCap}}}\\right).",
            "$$",
            "",
            (
                "最差长文本 PPL 保持率为 "
                f"{percent(worst_long['ppl_retention'])}"
                f"（{worst_long['model']}，"
                f"{int(worst_long['history_tokens']) // 1000}K）。"
                + (
                    "这构成“PCA/量化尾部扰动对任意自然文本都可忽略”的反例；"
                    "LongBench 任务质量与通用连续文本 PPL 必须分别陈述。"
                    if float(worst_long["ppl_retention"]) < 0.95
                    else "当前有限长文本窗口未观察到低于 95% 的 PPL 保持率，"
                    "但这仍不是无条件保证。"
                )
            ),
            "",
            "### 分任务结果",
            "",
            "| 任务 | Llama Full | Llama CountCap | 保持率 | Qwen Full | Qwen CountCap | 保持率 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    task_names = sorted({row["task"] for row in by_task})
    for task in task_names:
        llama = pick(
            by_task,
            model="Llama-3.1-8B-Instruct",
            task=task,
        )
        qwen = pick(
            by_task,
            model="Qwen3-4B-Instruct",
            task=task,
        )
        lines.append(
            "| {task} | {lf} | {lc} | {lr} | {qf} | {qc} | {qr} |".format(
                task=task,
                lf=number(llama["full_score"]),
                lc=number(llama["countcap_score"]),
                lr=percent(llama["quality_retention"]),
                qf=number(qwen["full_score"]),
                qc=number(qwen["countcap_score"]),
                qr=percent(qwen["quality_retention"]),
            )
        )

    published_rows = comparison["published_reference"][
        "RaBitQCache_paper_LongBench13_table"
    ]
    rabitq_rows = [
        row for row in comparison["countcap"] if row["subset"] == "rabitq13"
    ]
    llama_rabitq_subset = pick(
        rabitq_rows,
        model="Llama-3.1-8B-Instruct",
    )
    self_index_rows = [
        row
        for row in comparison["countcap"]
        if row["subset"] == "selfindex11"
    ]
    llama_self_index_subset = pick(
        self_index_rows,
        model="Llama-3.1-8B-Instruct",
    )
    published_self_index = comparison["published_reference"][
        "SelfIndexingKVCache_Llama31_8B_LongBench11"
    ]
    lines.extend(
        [
            "",
            "### 与 RaBitQCache 公开结果的协议级参考",
            "",
            "| 来源 | Full（×100） | 稀疏方法 | 稀疏分数（×100） | 平均目标/预算 |",
            "|---|---:|---|---:|---:|",
            "| 本次同 runner 的 13-task 子集 | "
            f"{number(100.0 * llama_rabitq_subset['full_macro'], 2)} | CountCap | "
            f"{number(100.0 * llama_rabitq_subset['countcap_macro'], 2)} | "
            f"{percent(llama_rabitq_subset['mean_attention_fraction'])} target |",
            "",
            "| RaBitQCache 论文中的方法 | 设置 | 分数（×100） | 实际预算 |",
            "|---|---|---:|---:|",
        ]
    )
    for row in published_rows:
        budget = (
            "-"
            if row["budget_ratio"] is None
            else percent(row["budget_ratio"])
        )
        lines.append(
            f"| {row['method']} | {row['setting']} | "
            f"{number(row['score'], 2)} | {budget} |"
        )
    lines.extend(
        [
            "",
            "RaBitQCache 行是论文公开数字，不是本仓库同环境复现。其前两层使用 Full，"
            "硬件、推理框架、prompt 与 stop policy 也不同，只能比较相对保持率和预算量级。",
            "",
            "### 与 Self-Indexing KVCache 公开结果的协议级参考",
            "",
            "| 来源 | Full（×100） | 稀疏方法 | 稀疏分数（×100） | "
            "质量保持率 | 预算 |",
            "|---|---:|---|---:|---:|---:|",
            "| 本次同 runner 的 11-task 子集 | "
            f"{number(100.0 * llama_self_index_subset['full_macro'], 2)} | "
            "CountCap | "
            f"{number(100.0 * llama_self_index_subset['countcap_macro'], 2)} | "
            f"{percent(llama_self_index_subset['quality_retention'])} | "
            f"{llama_self_index_subset['mean_attention_tokens']:.1f} "
            "target token/head |",
            "| Self-Indexing 论文 | "
            f"{number(published_self_index['full_score'], 2)} | "
            "Self-Indexing 16-bit | "
            f"{number(published_self_index['self_indexing_16bit_score'], 2)} | "
            f"{percent(published_self_index['self_indexing_16bit_score'] / published_self_index['full_score'])} | "
            f"{int(published_self_index['budget_tokens'])} token |",
            "| Self-Indexing 论文 | "
            f"{number(published_self_index['full_score'], 2)} | "
            "Self-Indexing 2-bit | "
            f"{number(published_self_index['self_indexing_2bit_score'], 2)} | "
            f"{percent(published_self_index['self_indexing_2bit_score'] / published_self_index['full_score'])} | "
            f"{int(published_self_index['budget_tokens'])} token |",
            "",
            "Self-Indexing 的 160-token 预算包含 64 个固定 full-precision sink "
            "和 96 个动态 token，并同时把 K/V 压到低比特；CountCap 不保留 "
            "sink 特判且完整 FP16 K/V 仍驻留 GPU。两者任务子集相同，但 runner、"
            "prompt、stop policy 与存储目标不同，所以只比较各自相对 Full 的"
            "保持率，不直接比较绝对分数。",
            "",
            "## 3. Llama 同环境基线",
            "",
            "| 方法 | Macro | 相对 Full | 预算 |",
            "|---|---:|---:|---:|",
        ]
    )
    baseline_full = next(
        float(row["score"])
        for row in baseline_rows
        if row.get("method") == "FullAttention"
    )
    for row in baseline_rows:
        method = row.get("method", "")
        score = float(row.get("macro_score", row.get("score", 0.0)))
        retention = score / baseline_full if baseline_full else 0.0
        budget = row.get(
            "mean_budget_tokens",
            row.get(
                "max_capacity_prompts",
                row.get(
                    "budget",
                    "1024"
                    if method in {"SnapKV", "AdaKV", "H2O"}
                    else "100%"
                    if method == "FullAttention"
                    else "-",
                ),
            ),
        )
        lines.append(
            f"| {method} | {score:.4f} | "
            f"{percent(retention)} | {budget} |"
        )

    lines.extend(
        [
            "",
            "## 4. QK 候选误差链",
            "",
            "| 指标 | 32K、4% 候选 |",
            "|---|---:|",
            f"| 普通 exact top-k recall | {percent(crossing_row['topk_recall_mean'])} |",
            f"| attention-mass weighted recall | {percent(crossing_row['attention_mass_weighted_topk_recall_mean'])} |",
            f"| 保留 full-attention mass | {percent(crossing_row['retained_attention_mass_mean'])} |",
            f"| 相对 Exact-QK top-k 的 mass regret | {percent(crossing_row['retained_attention_mass_regret_mean'])} |",
            f"| 确定性 mass 下界通过率 | {percent(crossing_row['deterministic_mass_bound_satisfied_mean'])} |",
            f"| attention 输出界通过率 | {percent(crossing_row['output_bound_satisfied_mean'])} |",
            "",
            "集合 recall 不是最终质量的充分统计量。真正与输出误差直接相连的是遗漏 "
            "attention mass，以及候选内外 Value 条件均值的差。",
            "",
            "真实 256 点 midpoint sampled-quantile 还会引入阈值估计误差。"
            "下表中的候选集未做人为截断，用于把阈值误差与 capacity 截断分开：",
            "",
            "| sampled-quantile 指标 | 32K、目标 4% |",
            "|---|---:|",
            f"| 实际选中比例均值 | {percent(sampled_threshold_row['sampled_selected_fraction_mean'])} |",
            f"| 实际选中比例 p90 | {percent(sampled_threshold_row['sampled_selected_fraction_p90'])} |",
            f"| 超过 production capacity 的比例 | {percent(sampled_threshold_row['sampled_candidate_overflow_mean'])} |",
            f"| 相对精确 proxy 分位点的绝对 score 误差 | {sampled_threshold_row['sampled_threshold_absolute_error_mean']:.5f} |",
            f"| 未截断候选保留的 full-attention mass | {percent(sampled_threshold_row['retained_attention_mass_mean'])} |",
            "",
            "固定 256 点分位数采样在极低目标比例下不是等相对精度的。"
            "若暂用独立同分布连续分数近似，样本数为 $m$、目标比例为 $f$，则",
            "",
            "$$",
            "\\operatorname{Std}(\\widehat f)"
            "\\approx\\sqrt{\\frac{f(1-f)}{m+2}},\\qquad"
            "\\frac{\\operatorname{Std}(\\widehat B)}{B}"
            "\\approx\\sqrt{\\frac{1-f}{(m+2)f}}.",
            "$$",
            "",
            "取 $m=256$ 时，32K/4%、64K/2%、128K/1% 的预算相对标准差"
            "近似为 30.5%、43.6%、61.9%。生产采样点是确定性均匀位置且彼此相关，"
            "所以这不是严格置信区间；它用于解释为什么长度增大、目标比例降低后，"
            "固定采样数会带来更强的 per-head 候选数量抖动。该误差必须与 "
            "PCA/SVD 子空间误差、INT4 量化误差和 capacity 截断分别统计。",
            "",
            "对确定性 midpoint 采样也不存在分布无关保证。若超过当前阈值的"
            "位置只形成 $C$ 个连续区间，则分层 midpoint tail-fraction 误差"
            "可由约 $2C/m$ 控制；若高分位置高度碎片化，$C$ 随长度增长，"
            "该界即失效。极端情况下可把高分全部放在未采样位置。因而真实"
            "候选数量与保留 attention mass 是必要指标，不能用 256 个固定"
            "样本替代验证。",
            "",
            "## 5. 双模型中心化 QK 奇异谱",
            "",
            "softmax 对每个 query 的分数整体平移不敏感。因此令 "
            "$H=I-\\mathbf1\\mathbf1^T/N$，以下主要分析 "
            "$S_c=SH=Q(HK)^T/\\sqrt d$，而不是可能被无效均值模态主导的原始 $S$。",
            "",
            "| 模型 | 原始 rank-1 能量 | 行中心化删除的无效能量 | "
            "第一右奇异向量与常数方向对齐 | 中心化 rank-1 能量 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in ("llama31_8b", "qwen3_4b"):
        row = qk_rows[model]
        lines.append(
            "| {model} | {raw} | {removed} | {alignment} | "
            "{centered} |".format(
                model=model,
                raw=percent(row["qk_rank1_energy_fraction_mean"]),
                removed=percent(
                    row[
                        "softmax_invariant_row_mean_energy_fraction_mean"
                    ]
                ),
                alignment=percent(
                    row[
                        "qk_top_right_vector_constant_alignment_mean"
                    ]
                ),
                centered=percent(
                    row["centered_qk_rank1_energy_fraction_mean"]
                ),
            )
        )

    lines.extend(
        [
            "",
            "| 模型 | K 有效秩 | 原始 QK 有效秩 | 中心化 QK 有效秩 | "
            "中心化 rank-16 | rank-32 | 最优 rank-48 | "
            "未中心化 Key-PCA48 | 最优性差距 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in ("llama31_8b", "qwen3_4b"):
        row = qk_rows[model]
        lines.append(
            "| {model} | {kr:.2f} | {qr:.2f} | {cqr:.2f} | "
            "{r16} | {r32} | {opt} | {qe} | {gap} |".format(
                model=model,
                kr=float(row["key_effective_rank_mean"]),
                qr=float(row["qk_effective_rank_mean"]),
                cqr=float(row["centered_qk_effective_rank_mean"]),
                r16=percent(
                    row[
                        "centered_qk_energy_retained_optimal_rank16_mean"
                    ]
                ),
                r32=percent(
                    row[
                        "centered_qk_energy_retained_optimal_rank32_mean"
                    ]
                ),
                qe=percent(
                    row[
                        "centered_qk_energy_retained_uncentered_key_pca48_mean"
                    ]
                ),
                opt=percent(
                    row[
                        "centered_qk_energy_retained_optimal_rank48_mean"
                    ]
                ),
                gap=percent(
                    row[
                        "centered_qk_uncentered_key_pca_optimality_gap_mean"
                    ]
                ),
            )
        )

    lines.extend(
        [
            "",
            "原始 QK 的极低有效秩部分来自 Key 均值造成的逐行常数，"
            "这个分量会被 softmax 精确抵消。中心化后有效秩仍远低于 128，"
            "且最优 rank-48 仍保留接近全部分数能量，说明低秩现象不是均值伪影；"
            "但 Key-only PCA 与最优 QK-SVD 之间仍有可测差距。",
            "",
            "对同一个 Key 矩阵，若 "
            "$K=U\\Sigma V^T$，则 "
            "$K^TK=V\\Sigma^2V^T$。因此 sampled uncentered PCA 与同样本 "
            "right-SVD 数学上完全等价；旧实验中两者的主子空间重合度为 "
            "1.0000，检索指标只存在约 $10^{-5}$ 的数值差异。"
            "直接分解中心化 QK 则同时利用了真实 Query，只能作为更强的"
            "解释上界。",
            "",
            "对任意正交 Key 投影 $P$，本文按残差定义 score fidelity：",
            "",
            "$$",
            "\\mathcal F_r(P)=1-"
            "\\frac{\\|Q(I-P)\\widetilde K^T\\|_F^2}"
            "{d\\|S_c\\|_F^2}.",
            "$$",
            "",
            "不能一般性地把 $\\|QP\\widetilde K^T\\|_F^2/d$ 直接叫作"
            "保留能量，因为保留项与残差项未必 Frobenius 正交。"
            "最优 QK-SVD 可写成 Key 协方差白化坐标中的 query-aware "
            "斜投影；它通常不是当前 Key-PCA 使用的正交投影。表中"
            "“最优性差距”严格等于实际残差超过 Eckart--Young 最优 "
            "rank-$r$ 尾谱残差的归一化部分。",
            "",
            "对应的精确三项分解是：",
            "",
            "$$",
            "K=\\mathbf1\\mu^T+\\widetilde KP+\\widetilde K(I-P).",
            "$$",
            "",
            "第一项产生逐 query 常数分数，softmax 精确忽略；第二项由低维索引"
            "保留；只有第三项是真正影响排序的谱尾误差。INT4/INT8 是投影坐标"
            "中的额外数值扰动。谱尾不能被称为“没有语义”，其可忽略性必须由"
            "边界 attention mass 与下游 logit 稳定共同验证。",
            "",
            "### 理想 full-history basis 与真实 first-2K basis",
            "",
            "| 模型 | Full uncentered-Key PCA48 中心化 fidelity | "
            "Full centered-Key PCA48 | Sampled-full 中心化 fidelity | "
            "真实 first-2K 中心化 fidelity | First-2K 中心化 cosine | "
            "First-2K/full 子空间 overlap |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model in ("llama31_8b", "qwen3_4b"):
        row = qk_rows[model]
        lines.append(
            "| {model} | {full} | {centered} | {sampled} | {prefix} | "
            "{cosine:.4f} | {overlap} |".format(
                model=model,
                full=percent(
                    row["centered_full_key_pca_qk_fidelity_mean"]
                ),
                centered=percent(
                    row["centered_key_pca_qk_fidelity_mean"]
                ),
                sampled=percent(
                    row["centered_sampled_full_pca_qk_fidelity_mean"]
                ),
                prefix=percent(
                    row[
                        "centered_production_prefix_pca_qk_fidelity_mean"
                    ]
                ),
                cosine=float(
                    row[
                        "centered_production_prefix_pca_qk_score_cosine_mean"
                    ]
                ),
                overlap=percent(
                    row["production_prefix_pca_subspace_overlap_mean"]
                ),
            )
        )

    lines.extend(
        [
            "",
            "| 模型 | 原始 Key/Query 交换子 | 中心化 Key/Query 交换子 | "
            "全历史 stride-32/full Key 子空间 overlap | "
            "首 2K/full Key 子空间 overlap |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for model in ("llama31_8b", "qwen3_4b"):
        row = qk_rows[model]
        lines.append(
            "| {model} | {commutator:.4f} | {centered_commutator:.4f} | "
            "{sampled} | {prefix} |".format(
                model=model,
                commutator=float(
                    row["key_query_covariance_commutator_ratio_mean"]
                ),
                centered_commutator=float(
                    row[
                        "centered_key_query_covariance_commutator_ratio_mean"
                    ]
                ),
                sampled=percent(
                    row["sampled_full_pca_subspace_overlap_mean"]
                ),
                prefix=percent(
                    row["production_prefix_pca_subspace_overlap_mean"]
                ),
            )
        )

    llama_qk = qk_rows["llama31_8b"]
    qwen_qk = qk_rows["qwen3_4b"]
    lines.extend(
        [
            "",
            "中心化后交换子明显非零但较小，说明 centered-Key 与 Query 的"
            "主方向存在经验对齐，却不能逐方向同时对角化；所以 Key-PCA "
            "奇异向量不等于最优 QK-SVD 奇异向量。"
            "沿完整历史均匀采样时，Key 子空间仍较接近 full-history SVD；"
            "真实 first-2K basis 的分布失配明显更大。",
            "",
            "逐 layer/head 的描述性相关性也没有给出单一可靠代理："
            "删除的行均值能量与 raw QK rank-1 能量在两个模型上的 "
            "Spearman 为 0.800/0.866，但 centered commutator、first-2K "
            "subspace overlap 或 centered effective rank 都不能跨模型稳定"
            "预测生产 fidelity。因此谱量用于构造联合误差账本，不用于替代"
            "候选 attention mass 和最终输出验证。",
            "",
            "中心化 QK 最优 rank-48 的 p10 为 "
            f"{percent(llama_qk['centered_qk_energy_retained_optimal_rank48_p10'])}/"
            f"{percent(qwen_qk['centered_qk_energy_retained_optimal_rank48_p10'])}，"
            "但生产 first-2K fidelity 的 p10 只有 "
            f"{percent(llama_qk['centered_production_prefix_pca_qk_fidelity_p10'])}/"
            f"{percent(qwen_qk['centered_production_prefix_pca_qk_fidelity_p10'])}。"
            "所以 QK 低秩是稳定现象，首段 basis 对困难 head 的准确性不是；"
            "最终论证必须继续依赖 attention mass 与 logit/NLL。",
            "",
            "### Prefix 长度的纯数值消融",
            "",
            "| 模型 | 512 fidelity | 1K | 2K（冻结设置） | 4K | 8K |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model in ("llama31_8b", "qwen3_4b"):
        row = qk_rows[model]
        lines.append(
            "| {model} | {p512} | {p1024} | {p2048} | {p4096} | "
            "{p8192} |".format(
                model=model,
                p512=percent(
                    row["prefix512_pca_centered_qk_fidelity_mean"]
                ),
                p1024=percent(
                    row["prefix1024_pca_centered_qk_fidelity_mean"]
                ),
                p2048=percent(
                    row["prefix2048_pca_centered_qk_fidelity_mean"]
                ),
                p4096=percent(
                    row["prefix4096_pca_centered_qk_fidelity_mean"]
                ),
                p8192=percent(
                    row["prefix8192_pca_centered_qk_fidelity_mean"]
                ),
            )
        )

    lines.extend(
        [
            "",
            "该表只复用 Q/K trace 改变估计 basis 的首段长度，不改变冻结方法或 "
            "LongBench 结果。它用于判断 2K 是否已经进入子空间稳定平台。",
            "",
            "Key-PCA48 对 softmax 有效分数的误差严格等于：",
            "",
            "$$",
            "\\|(S-S_{48})H\\|_F^2="
            "\\frac{1}{d}\\|Q(I-P_{48})(HK)^T\\|_F^2.",
            "$$",
            "",
            "其中 $H=I-\\mathbf1\\mathbf1^T/N$。冻结实现使用首 2048 token 的 sampled "
            "basis，因此还存在 "
            "$Q(P_{48}-\\widehat P_{48})(HK)^T/\\sqrt d$ 的 "
            "prefix-basis 失配项。有效性必须同时来自 Key 尾部奇异值衰减、"
            "Query 尾部对齐能量较小，以及首段 basis 对后续历史仍有代表性，"
            "而不是“PCA 尾部没有语义”。",
            "",
            "RoPE 对不同位置使用不同正交旋转："
            "$C_K=N^{-1}\\sum_i R_i\\bar k_i\\bar k_i^TR_i^T$。"
            "因此 pre-RoPE 低秩不自动保证 post-RoPE 首段 basis 可迁移；"
            "prefix 长度消融直接检验的正是这项位置条件下的协方差漂移。",
            "",
            "令 $E=S-\\widehat S$，并令 $\\mathcal B_{\\gamma,t}$ 是第 $t$ 个 "
            "query 在精确 top-$k$ 阈值附近宽度为 $\\gamma$ 的边界带。"
            "对任意 $\\gamma>0$ 有确定性上界：",
            "",
            "$$",
            "\\frac{1}{MN}\\sum_t"
            "|S_t^\\star\\triangle\\widehat S_t|"
            "\\le"
            "\\frac{2}{MN}\\sum_t|\\mathcal B_{\\gamma,t}|"
            "+\\frac{2\\|E\\|_F^2}{MN\\gamma^2}.",
            "$$",
            "",
            "因此，QK 尾部奇异能量小只控制第二项；还必须同时验证 top-$k$ "
            "边界不承载大量 token/attention mass。完整证明见数学附录。",
            "",
            "## 6. Token-logit 稳定性",
            "",
            "| 模型 | Token 数 | Top-1 agreement | Margin certificate | KL | JS | 平均 NLL 差 | KL/NLL 界通过率 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for model, row in logit_rows.items():
        bound_rate = min(
            float(row["kl_range_bound_satisfied_mean"]),
            float(row["target_nll_range_bound_satisfied_mean"]),
        )
        lines.append(
            "| {model} | {tokens} | {agree} | {cert} | {kl:.6f} | "
            "{js:.6f} | {nll:+.6f} | {bound} |".format(
                model=model,
                tokens=int(row["tokens"]),
                agree=percent(row["top1_agreement_mean"]),
                cert=percent(row["margin_certificate_satisfied_mean"]),
                kl=float(row["kl_full_to_sparse_mean"]),
                js=float(row["js_divergence_mean"]),
                nll=float(row["target_nll_delta_mean"]),
                bound=percent(bound_rate),
            )
        )

    lines.extend(
        [
            "",
            "令 $d=z_{\\mathrm{sparse}}-z_{\\mathrm{full}}$ 且 "
            "$R=\\max d-\\min d$。当 Full top-1 margin 大于 $R$ 时预测不变；"
            "目标 NLL 变化不超过 $R$，并有 "
            "$D_{\\mathrm{KL}}(p_{\\mathrm{full}}\\|p_{\\mathrm{sparse}})"
            "\\le R^2/8$。",
            "",
            "## 7. 当前请求首段 PCA 基与跨请求固定基",
            "",
            "| 模型 | Online macro | Fixed macro | Fixed-Online 95% CI | "
            "Fixed/Online | Prediction agreement | "
            "固定基索引构建加速 | 固定基总耗时加速 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in fixed_basis["overall"]:
        lines.append(
            "| {model} | {online:.4f} | {fixed:.4f} | [{low:+.4f}, "
            "{high:+.4f}] | {relative} | "
            "{agreement} | {index:.3f}x | {total:.3f}x |".format(
                model=row["model"],
                online=float(row["online_macro"]),
                fixed=float(row["fixed_macro"]),
                low=float(
                    row["fixed_minus_online_macro_ci95_low"]
                ),
                high=float(
                    row["fixed_minus_online_macro_ci95_high"]
                ),
                relative=percent(row["fixed_relative"]),
                agreement=percent(row["prediction_agreement"]),
                index=float(row["fixed_index_build_speedup"]),
                total=float(row["fixed_total_speedup"]),
            )
        )

    lines.extend(
        [
            "",
            "这里的固定基由四个与测试样本分离的 LongBench 请求校准，并在全部测试请求中复用。"
            "Online 列使用当前请求首个 2048-token chunk 的 post-RoPE 基底；"
            "Fixed 列使用四个校准请求的首段 rank-48 二阶矩合并得到的跨请求基底。"
            "该消融用于检验 prefix-conditioned basis 是否降低跨请求子空间失配，"
            "而不是修改冻结方法。固定基的 Macro 接近 Online，但 prediction "
            "agreement 明显较低，而且校准请求仍来自 LongBench；在独立语料和"
            "未见模型上验证之前，它只能作为减少短上下文索引构建开销的候选优化，"
            "不能写成无需校准的通用替代。",
            "",
            "## 8. 结论与限制",
            "",
            "1. 结论仅支持自然输入分布上的条件稳定性，不是任意 query 的无条件证明。",
            "2. 7.5K LongBench 的主要作用是质量与跨模型验证；短文本速度仍是弱项。",
            "3. 当前 final-direct LongBench 保留完整 GPU K/V，不能把 attention 比例写成 GPU KV 比例。",
            "4. PCA/SVD top-k 本身不是新颖点；论文贡献必须落在当前请求首段条件化的"
            "低比特 direct retrieval、"
            "误差账本、无 Full 回退和真实长上下文系统实现的组合上。",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
