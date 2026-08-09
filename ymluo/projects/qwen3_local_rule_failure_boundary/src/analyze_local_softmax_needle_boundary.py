from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence


ATTENTION_METRICS = (
    "evidence_mass",
    "target_special_mass",
    "special_mass",
    "ordinary_background_mass",
    "evidence_logsumexp",
    "non_evidence_logsumexp",
    "evidence_vs_non_evidence",
    "evidence_vs_target_special",
    "evidence_label_mass",
    "special_label_mass",
    "evidence_hit_top20",
    "evidence_hit_top2pct",
    "outside_top20_mass",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def geomean(values: Iterable[float]) -> float:
    values = list(values)
    return math.exp(mean(math.log(max(value, 1e-300)) for value in values))


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    result = {
        "n": float(len(rows)),
        "candidate_accuracy": mean(float(row["scores"]["candidate_correct"]) for row in rows),
        "greedy_accuracy": mean(float(row["scores"]["greedy_correct"]) for row in rows),
        "gold_ppl_geomean": geomean(float(row["scores"]["gold_ppl"]) for row in rows),
        "candidate_margin": mean(float(row["scores"]["candidate_margin"]) for row in rows),
        "condition_label_margin": mean(
            float(row["scores"]["condition_label_margin"]) for row in rows
        ),
    }
    for metric in ATTENTION_METRICS:
        result[metric] = mean(
            float(row["attention"]["model_mean"][metric]) for row in rows
        )
    return result


def grouped(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str, int], dict[str, float]]:
    buckets: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[
            (row["placement"], row["condition"], int(row["filler_length"]))
        ].append(row)
    return {key: aggregate(group) for key, group in buckets.items()}


def bin_summaries(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[
            (
                row["placement"],
                row["condition"],
                int(row["filler_length"]),
                row["bin"],
            )
        ].append(row)
    output = []
    for (placement, condition, length, bin_name), group_rows in sorted(buckets.items()):
        output.append(
            {
                "placement": placement,
                "condition": condition,
                "filler_length": length,
                "bin": bin_name,
                **aggregate(group_rows),
            }
        )
    return output


def bin_boundaries(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in summaries:
        buckets[(row["placement"], row["condition"], row["bin"])].append(row)
    output = []
    log_half = -math.log(2.0)
    for (placement, condition, bin_name), group_rows in sorted(buckets.items()):
        ordered = sorted(group_rows, key=lambda row: int(row["filler_length"]))
        baseline = ordered[0]
        first_any = next(
            (
                int(row["filler_length"])
                for row in ordered
                if row["candidate_accuracy"] < 1.0
            ),
            None,
        )
        first_majority = next(
            (
                int(row["filler_length"])
                for row in ordered
                if row["candidate_accuracy"] <= 0.5
            ),
            None,
        )
        first_odds_half = next(
            (
                int(row["filler_length"])
                for row in ordered
                if row["evidence_vs_non_evidence"]
                - baseline["evidence_vs_non_evidence"]
                <= log_half
            ),
            None,
        )
        first_mass_half = next(
            (
                int(row["filler_length"])
                for row in ordered
                if row["evidence_mass"] <= 0.5 * baseline["evidence_mass"]
            ),
            None,
        )
        output.append(
            {
                "placement": placement,
                "condition": condition,
                "bin": bin_name,
                "first_any_output_failure": first_any,
                "first_accuracy_at_or_below_50pct": first_majority,
                "first_evidence_odds_half": first_odds_half,
                "first_evidence_mass_half": first_mass_half,
                "maximum_tested_length": max(int(row["filler_length"]) for row in ordered),
                "accuracy_trajectory": ",".join(
                    f"{int(row['filler_length'])}:{row['candidate_accuracy']:.2f}"
                    for row in ordered
                ),
                "margin_trajectory": ",".join(
                    f"{int(row['filler_length'])}:{row['candidate_margin']:.3f}"
                    for row in ordered
                ),
            }
        )
    return output


def exp_clip(value: float) -> float:
    return math.exp(max(-50.0, min(50.0, value)))


def decomposition(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups = grouped(rows)
    output: list[dict[str, Any]] = []
    for (placement, condition, length), current in sorted(groups.items()):
        zero = groups.get((placement, condition, 0))
        plain = groups.get((placement, "plain", length))
        if zero is None or plain is None:
            continue
        delta_evidence = current["evidence_logsumexp"] - zero["evidence_logsumexp"]
        delta_non_evidence = (
            current["non_evidence_logsumexp"] - zero["non_evidence_logsumexp"]
        )
        delta_log_odds = delta_evidence - delta_non_evidence
        denominator_loss = max(delta_non_evidence, 0.0)
        evidence_loss = max(-delta_evidence, 0.0)
        evidence_gain = max(delta_evidence, 0.0)
        unopposed = denominator_loss + evidence_loss

        delta_non_evidence_plain = (
            current["non_evidence_logsumexp"] - plain["non_evidence_logsumexp"]
        )
        delta_evidence_plain = (
            current["evidence_logsumexp"] - plain["evidence_logsumexp"]
        )
        output.append(
            {
                "placement": placement,
                "condition": condition,
                "filler_length": length,
                "candidate_accuracy": current["candidate_accuracy"],
                "gold_ppl_geomean": current["gold_ppl_geomean"],
                "candidate_margin": current["candidate_margin"],
                "evidence_mass": current["evidence_mass"],
                "target_special_mass": current["target_special_mass"],
                "evidence_logsumexp": current["evidence_logsumexp"],
                "non_evidence_logsumexp": current["non_evidence_logsumexp"],
                "evidence_log_odds": current["evidence_vs_non_evidence"],
                "delta_evidence_lse_vs_zero": delta_evidence,
                "delta_non_evidence_lse_vs_zero": delta_non_evidence,
                "delta_log_odds_vs_zero": delta_log_odds,
                "effective_evidence_multiplier_vs_zero": exp_clip(delta_evidence),
                "effective_competitor_multiplier_vs_zero": exp_clip(
                    delta_non_evidence
                ),
                "evidence_odds_multiplier_vs_zero": exp_clip(delta_log_odds),
                "denominator_loss": denominator_loss,
                "evidence_score_loss": evidence_loss,
                "evidence_score_gain": evidence_gain,
                "denominator_share_of_unopposed_loss": (
                    denominator_loss / unopposed if unopposed else 0.0
                ),
                "delta_evidence_lse_vs_plain": delta_evidence_plain,
                "delta_non_evidence_lse_vs_plain": delta_non_evidence_plain,
                "competitor_multiplier_vs_plain": exp_clip(
                    delta_non_evidence_plain
                ),
                "evidence_multiplier_vs_plain": exp_clip(delta_evidence_plain),
                "evidence_mass_ratio_vs_plain": current["evidence_mass"]
                / max(plain["evidence_mass"], 1e-300),
                "target_special_mass_ratio_vs_plain": current[
                    "target_special_mass"
                ]
                / max(plain["target_special_mass"], 1e-300),
                "ppl_ratio_vs_plain": current["gold_ppl_geomean"]
                / max(plain["gold_ppl_geomean"], 1e-300),
            }
        )
    return output


def layer_aggregates(
    rows: Sequence[dict[str, Any]],
) -> dict[tuple[str, str, int, int], dict[str, float]]:
    buckets: dict[
        tuple[str, str, int, int], dict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        prefix = (row["placement"], row["condition"], int(row["filler_length"]))
        for layer in row["attention"]["layer_mean"]:
            key = (*prefix, int(layer["layer"]))
            for metric in ATTENTION_METRICS:
                buckets[key][metric].append(float(layer[metric]))
    return {
        key: {metric: mean(values) for metric, values in metrics.items()}
        for key, metrics in buckets.items()
    }


def layer_contrasts(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    layers = layer_aggregates(rows)
    output: list[dict[str, Any]] = []
    for (placement, condition, length, layer), current in sorted(layers.items()):
        plain = layers.get((placement, "plain", length, layer))
        other_placement = "fixed_recent" if placement == "remote" else "remote"
        position_control = layers.get((other_placement, condition, length, layer))
        if plain is None:
            continue
        row: dict[str, Any] = {
            "placement": placement,
            "condition": condition,
            "filler_length": length,
            "layer": layer,
        }
        for metric in ATTENTION_METRICS:
            row[metric] = current[metric]
            row[f"delta_{metric}_vs_plain"] = current[metric] - plain[metric]
            if position_control is not None:
                row[f"delta_{metric}_vs_other_placement"] = (
                    current[metric] - position_control[metric]
                )
        output.append(row)
    return output


def trajectories(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "placement": row["placement"],
            "condition": row["condition"],
            "filler_length": int(row["filler_length"]),
            "concept_id": row["concept_id"],
            "lemma": row["lemma"],
            "bin": row["bin"],
            "candidate_correct": int(bool(row["scores"]["candidate_correct"])),
            "greedy_correct": int(bool(row["scores"]["greedy_correct"])),
            "gold_ppl": float(row["scores"]["gold_ppl"]),
            "candidate_margin": float(row["scores"]["candidate_margin"]),
            "condition_label_margin": float(
                row["scores"]["condition_label_margin"]
            ),
            **{
                metric: float(row["attention"]["model_mean"][metric])
                for metric in ATTENTION_METRICS
            },
        }
        for row in sorted(
            rows,
            key=lambda value: (
                value["placement"],
                value["condition"],
                value["concept_id"],
                int(value["filler_length"]),
            ),
        )
    ]


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_norm = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_norm = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    return numerator / (x_norm * y_norm) if x_norm and y_norm else 0.0


def metric_associations(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    scopes = [("all", list(rows))]
    scopes.extend(
        (placement, [row for row in rows if row["placement"] == placement])
        for placement in ("remote", "fixed_recent")
    )
    for scope, scoped in scopes:
        margins = [float(row["scores"]["candidate_margin"]) for row in scoped]
        failures = [row for row in scoped if not row["scores"]["candidate_correct"]]
        successes = [row for row in scoped if row["scores"]["candidate_correct"]]
        for metric in ATTENTION_METRICS:
            values = [
                float(row["attention"]["model_mean"][metric]) for row in scoped
            ]
            output.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "pearson_with_candidate_margin": pearson(values, margins),
                    "success_mean": mean(
                        float(row["attention"]["model_mean"][metric])
                        for row in successes
                    )
                    if successes
                    else None,
                    "failure_mean": mean(
                        float(row["attention"]["model_mean"][metric])
                        for row in failures
                    )
                    if failures
                    else None,
                    "n": len(scoped),
                    "failures": len(failures),
                }
            )
    return output


def transition_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["placement"], row["condition"], row["concept_id"])].append(
            row
        )
    output = []
    for (placement, condition, concept_id), group in sorted(groups.items()):
        ordered = sorted(group, key=lambda row: int(row["filler_length"]))
        transitions = []
        for left, right in zip(ordered, ordered[1:]):
            if bool(left["scores"]["candidate_correct"]) != bool(
                right["scores"]["candidate_correct"]
            ):
                transitions.append(
                    f"{int(left['filler_length'])}:{int(bool(left['scores']['candidate_correct']))}"
                    f"->{int(right['filler_length'])}:{int(bool(right['scores']['candidate_correct']))}"
                )
        failures = [
            int(row["filler_length"])
            for row in ordered
            if not row["scores"]["candidate_correct"]
        ]
        output.append(
            {
                "placement": placement,
                "condition": condition,
                "concept_id": concept_id,
                "lemma": ordered[0]["lemma"],
                "bin": ordered[0]["bin"],
                "tested_lengths": ",".join(
                    str(int(row["filler_length"])) for row in ordered
                ),
                "failure_lengths": ",".join(str(value) for value in failures),
                "first_observed_failure": failures[0] if failures else None,
                "transition_count": len(transitions),
                "transitions": ";".join(transitions),
                "minimum_margin": min(
                    float(row["scores"]["candidate_margin"]) for row in ordered
                ),
            }
        )
    return output


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def pct(value: float) -> str:
    return f"{value:.1%}"


def write_chinese_report(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    decomposed: Sequence[dict[str, Any]],
    layers: Sequence[dict[str, Any]],
    associations: Sequence[dict[str, Any]],
    transitions: Sequence[dict[str, Any]],
    bins: Sequence[dict[str, Any]],
    bin_boundary_rows: Sequence[dict[str, Any]],
) -> None:
    groups = grouped(rows)
    lengths = sorted({int(row["filler_length"]) for row in rows})
    maximum = max(lengths)

    def group(place: str, condition: str, length: int) -> dict[str, float] | None:
        return groups.get((place, condition, length))

    lines = [
        "# Qwen3-8B 本机长上下文语义针：Softmax 定量实验",
        "",
        f"- GPU：AMD Radeon RX 7900 XTX（24 GB）",
        f"- 当前已完成长度：{', '.join(str(value) for value in lengths)} filler tokens",
        f"- 样本：8 根简单语义针 × 3 类信息 × 2 种位置，共 {len(rows)} 条逐针结果",
        "",
        "## 一眼能看懂的结论",
        "",
        "1. **普通文本不会让所有针在同一长度失效。** 强针保持很大答案 margin，弱针会先接近 0，而且还可能短暂恢复，因此边界不是严格单调。",
        "2. **远程普通证据的退化由两部分共同造成。** Softmax 分母增长降低证据份额；同时证据本身的聚合 QK 得分下降。固定证据—查询距离后，第二项大幅减弱。",
        "3. **语义干扰不是简单地把整个分母放大。** 它主要把少数语义匹配 head 的质量搬到近邻条目；全 head 平均分母可能几乎不变，但弱针的答案 margin 已翻转。",
        "4. **矛盾信息最危险。** 它把同一语义直接绑定到错误标签，使关键 head/后层的错误标签通路形成；因此只需约 1K 上下文就能让部分针失败，比普通 filler 早得多。",
        "",
        "## Common 与 tail：谁更早被淹没？",
        "",
        "当前每组只有 4 根针，因此这是机制校准结论，不是总体显著性结论。整体趋势是：**tail 在语义干扰和矛盾下更早失败；纯普通 filler 下两组到最大已测长度都没有输出失败。**",
        "",
        "| 位置 | 信息 | Bin | 证据 odds 首次减半 | 首次任一针失败 | 准确率首次≤50% | Evidence mass 首次减半 | 最大已测 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in bin_boundary_rows:
        def boundary(value: Any) -> str:
            return str(value) if value is not None else f">{row['maximum_tested_length']}"

        lines.append(
            f"| {row['placement']} | {row['condition']} | {row['bin']} | "
            f"{boundary(row['first_evidence_odds_half'])} | "
            f"{boundary(row['first_any_output_failure'])} | "
            f"{boundary(row['first_accuracy_at_or_below_50pct'])} | "
            f"{boundary(row['first_evidence_mass_half'])} | "
            f"{row['maximum_tested_length']} |"
        )
    lines.extend(
        [
            "",
            "### 分组量化轨迹",
            "",
            "| 位置 | 信息 | Filler | Bin | 准确率 | Gold PPL | Margin | Evidence mass | Evidence log-odds |",
            "|---|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in bins:
        lines.append(
            f"| {row['placement']} | {row['condition']} | {row['filler_length']} | "
            f"{row['bin']} | {pct(row['candidate_accuracy'])} | "
            f"{row['gold_ppl_geomean']:.3f} | {row['candidate_margin']:+.3f} | "
            f"{row['evidence_mass']:.6f} | {row['evidence_vs_non_evidence']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "最干净的语义难度对照是 `fixed_recent + distractor`：common 在 1K/4K/16K 都是 100%，tail 在 1K 和 4K 是 75%，到 16K 又恢复为 100%。这说明 tail 更容易触边，但边界不是单调的固定长度。",
            "",
            "`remote + conflict` 的分离更强：1K 时 common 为 75%，tail 只有 25%；到 16K 时两组都已严重退化。tail 更早崩溃，但 common 中的 `salt` 也很难，tail 中的 `toolbox` 却一直较稳，所以频率标签不是唯一决定因素。",
            "",
        "## 准确率与 PPL",
        "",
        "| 位置 | 信息类型 | Filler | 候选准确率 | Greedy 准确率 | Gold PPL | 平均答案 margin | 证据 attention mass |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key, value in sorted(groups.items()):
        placement, condition, length = key
        lines.append(
            f"| {placement} | {condition} | {length} | "
            f"{pct(value['candidate_accuracy'])} | {pct(value['greedy_accuracy'])} | "
            f"{fmt(value['gold_ppl_geomean'])} | {value['candidate_margin']:+.3f} | "
            f"{value['evidence_mass']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Softmax 的严格分解",
            "",
            "对每个 layer/head，把真实证据 token 集合记为 E，其余历史 token 记为 N：",
            "",
            "`S_E = logsumexp(scores over E)`",
            "",
            "`S_N = logsumexp(scores over N)`",
            "",
            "`R = S_E - S_N`，且单个 head 的证据 attention mass 为 `sigmoid(R)`。",
            "",
            "从短上下文到长度 L：",
            "",
            "`ΔR = ΔS_E - ΔS_N`",
            "",
            "- `ΔS_N > 0`：Softmax 分母竞争增强；",
            "- `ΔS_E < 0`：真实证据的 QK 聚合得分也在变坏；",
            "- 二者都让证据 log-odds 下降。",
            "",
            "| 位置 | 信息 | Filler | ΔS_E | ΔS_N | ΔR | 证据倍率 | 竞争倍率 | 证据 odds 倍率 | 分母占未抵消损失 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in decomposed:
        if row["filler_length"] not in {0, maximum}:
            continue
        lines.append(
            f"| {row['placement']} | {row['condition']} | {row['filler_length']} | "
            f"{row['delta_evidence_lse_vs_zero']:+.3f} | "
            f"{row['delta_non_evidence_lse_vs_zero']:+.3f} | "
            f"{row['delta_log_odds_vs_zero']:+.3f} | "
            f"{row['effective_evidence_multiplier_vs_zero']:.2f}× | "
            f"{row['effective_competitor_multiplier_vs_zero']:.2f}× | "
            f"{row['evidence_odds_multiplier_vs_zero']:.3f}× | "
            f"{pct(row['denominator_share_of_unopposed_loss'])} |"
        )

    remote_plain = next(
        (
            row
            for row in decomposed
            if row["placement"] == "remote"
            and row["condition"] == "plain"
            and row["filler_length"] == maximum
        ),
        None,
    )
    fixed_plain = next(
        (
            row
            for row in decomposed
            if row["placement"] == "fixed_recent"
            and row["condition"] == "plain"
            and row["filler_length"] == maximum
        ),
        None,
    )
    if remote_plain and fixed_plain:
        lines.extend(
            [
                "",
                "### 普通 filler：位置与分母各有多大影响？",
                "",
                f"- 远程证据到 {maximum}: `ΔS_N={remote_plain['delta_non_evidence_lse_vs_zero']:+.3f}`，"
                f"即竞争项约 {remote_plain['effective_competitor_multiplier_vs_zero']:.2f}×；"
                f"`ΔS_E={remote_plain['delta_evidence_lse_vs_zero']:+.3f}`，"
                f"证据聚合得分倍率约 {remote_plain['effective_evidence_multiplier_vs_zero']:.2f}×。",
                f"- 固定近距离到 {maximum}: `ΔS_N={fixed_plain['delta_non_evidence_lse_vs_zero']:+.3f}`，"
                f"但 `ΔS_E={fixed_plain['delta_evidence_lse_vs_zero']:+.3f}`。"
                "这说明模型能部分增强近处证据来抵消更大的分母；远程证据没有得到这种补偿。",
            ]
        )

    lines.extend(
        [
            "",
            "## 为什么干扰/矛盾不只是“把分母变大”？",
            "",
            f"在最大已测长度 {maximum}，直接比较真实标签 token 与对应特殊块中的标签 token：",
            "",
            "| 位置 | 信息 | 真实标签 mass | 特殊块标签 mass | 对应特殊块 mass | Gold/特殊标签比 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for placement in ("remote", "fixed_recent"):
        for condition in ("plain", "distractor", "conflict"):
            value = group(placement, condition, maximum)
            if value is None:
                continue
            ratio = value["evidence_label_mass"] / max(
                value["special_label_mass"], 1e-300
            )
            lines.append(
                f"| {placement} | {condition} | "
                f"{value['evidence_label_mass']:.6f} | "
                f"{value['special_label_mass']:.6f} | "
                f"{value['target_special_mass']:.6f} | {ratio:.2f}× |"
            )
    lines.extend(
        [
            "",
            "如果只是均匀分母稀释，真实标签和错误标签会近似同比缩小；实际矛盾条件会让错误标签通道相对变强，所以答案 margin 可以在全 head 平均 `S_N` 只小幅变化时直接翻转。",
            "",
            "另一个重要现象是：失败样本的 `evidence_hit_top2pct` 不一定更低。因为 Top-2% 的 token 数随 N 增长，二值“命中”会变容易；它不能替代 evidence mass、标签通道比或答案 margin。",
        ]
    )

    lines.extend(
        [
            "",
            "## 哪些针先失败？",
            "",
            "| 位置 | 信息 | 针 | 首次观测失败 | 失败长度 | 状态翻转次数 | 最小 margin |",
            "|---|---|---|---:|---|---:|---:|",
        ]
    )
    for row in transitions:
        first_failure = (
            row["first_observed_failure"]
            if row["first_observed_failure"] is not None
            else "无"
        )
        lines.append(
            f"| {row['placement']} | {row['condition']} | {row['lemma']} | "
            f"{first_failure} | {row['failure_lengths'] or '无'} | "
            f"{row['transition_count']} | {row['minimum_margin']:+.3f} |"
        )

    all_assoc = [
        row for row in associations if row["scope"] == "all"
    ]
    all_assoc.sort(
        key=lambda row: abs(float(row["pearson_with_candidate_margin"])),
        reverse=True,
    )
    lines.extend(
        [
            "",
            "## 哪些内部量最接近答案 margin？",
            "",
            "这是描述性 Pearson 相关，不作因果结论。",
            "",
            "| 内部量 | 与答案 margin 的相关 | 正确样本均值 | 失败样本均值 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in all_assoc[:8]:
        lines.append(
            f"| {row['metric']} | {row['pearson_with_candidate_margin']:+.3f} | "
            f"{row['success_mean']:.6f} | {row['failure_mean']:.6f} |"
        )

    target_length = maximum
    target_layers = [
        row
        for row in layers
        if row["placement"] == "remote"
        and row["condition"] in {"distractor", "conflict"}
        and row["filler_length"] == target_length
    ]
    ranked = sorted(
        target_layers,
        key=lambda row: row["delta_evidence_vs_target_special_vs_plain"],
    )
    lines.extend(
        [
            "",
            f"## {target_length} 时最受语义竞争影响的层",
            "",
            "| 条件 | 层 | Δ(证据−对应特殊块 log-odds) | Δ证据 mass | Δ对应特殊块 mass |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in ranked[:10]:
        lines.append(
            f"| {row['condition']} | {row['layer']} | "
            f"{row['delta_evidence_vs_target_special_vs_plain']:+.4f} | "
            f"{row['delta_evidence_mass_vs_plain']:+.6f} | "
            f"{row['delta_target_special_mass_vs_plain']:+.6f} |"
        )

    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 这里的“候选准确率”是 A–H 中 gold 是否最高；“Greedy”是全词表首 token 是否就是 gold。当前协议下二者一致，说明错误发生在答案选择本身，不是答案抽取器。",
            "- `remote` 同时包含距离/RoPE 与 filler 竞争；`fixed_recent` 保持证据—查询距离基本不变，主要用于隔离分母和语义竞争。",
            "- 冲突块被明确标成 UNVERIFIED，问题也明确要求只信 VERIFIED；因此冲突失败不是 gold 定义含糊，而是模型未能稳定执行来源优先级。",
            "- 8 根针是机制校准集，不足以支持总体统计显著性；但非常适合定位失效边界和挑选后续大样本实验的难度层级。",
        ]
    )
    (output_dir / "analysis_zh.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.input_dir)
    rows = load_jsonl(output_dir / "rows.jsonl")
    decomposed = decomposition(rows)
    layers = layer_contrasts(rows)
    trajectory_rows = trajectories(rows)
    associations = metric_associations(rows)
    transitions = transition_rows(rows)
    bins = bin_summaries(rows)
    bin_boundary_rows = bin_boundaries(bins)
    write_csv(output_dir / "softmax_decomposition.csv", decomposed)
    write_csv(output_dir / "layer_contrasts.csv", layers)
    write_csv(output_dir / "needle_trajectories.csv", trajectory_rows)
    write_csv(output_dir / "metric_margin_correlations.csv", associations)
    write_csv(output_dir / "needle_transitions.csv", transitions)
    write_csv(output_dir / "common_tail_summary.csv", bins)
    write_csv(output_dir / "common_tail_boundaries.csv", bin_boundary_rows)
    write_chinese_report(
        output_dir,
        rows,
        decomposed,
        layers,
        associations,
        transitions,
        bins,
        bin_boundary_rows,
    )
    summary = {
        "rows": len(rows),
        "lengths": sorted({int(row["filler_length"]) for row in rows}),
        "decomposition_rows": len(decomposed),
        "layer_contrast_rows": len(layers),
        "trajectory_rows": len(trajectory_rows),
        "common_tail_rows": len(bins),
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
