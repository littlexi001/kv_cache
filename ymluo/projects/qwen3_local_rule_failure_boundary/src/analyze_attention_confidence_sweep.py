from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    x_mean = mean(xs)
    y_mean = mean(ys)
    numer = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_denom = sum((x - x_mean) ** 2 for x in xs)
    y_denom = sum((y - y_mean) ** 2 for y in ys)
    denom = math.sqrt(x_denom * y_denom)
    return numer / denom if denom > 0 else float("nan")


def ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    output = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0 + 1.0
        for position in order[cursor:end]:
            output[position] = rank
        cursor = end
    return output


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def slope_per_1k(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    x_mean = mean(xs)
    y_mean = mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom <= 0:
        return float("nan")
    return 1000.0 * sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def nearest_row(rows: Sequence[dict[str, Any]], target: int) -> dict[str, Any]:
    return min(rows, key=lambda row: abs(int(row["length"]) - target))


def fmt(value: float, digits: int = 4) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the Qwen3-8B attention/confidence length sweep.")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(
            (output_dir / "data").glob("length_*.json"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
    ]
    if not payloads:
        raise SystemExit("no result JSON files found")
    role_order = list(payloads[0]["attention"]["role_order"])
    role_index = {role: index for index, role in enumerate(role_order)}
    rows: list[dict[str, Any]] = []
    role_mechanism_rows: list[dict[str, Any]] = []
    head_series: dict[tuple[int, int], list[dict[str, float]]] = defaultdict(list)
    layer_series: dict[int, list[dict[str, float]]] = defaultdict(list)

    for payload in payloads:
        attention = payload["attention"]
        roles = attention["overall_role_mass"]
        length = int(payload["target_context_tokens"])
        key_length = int(attention["key_length"])
        top2pct_budget = max(1, int(math.ceil(0.02 * key_length)))
        gold_token_id = int(payload["answer"]["gold_token_scores"][0]["token_id"])
        next_token_top5 = payload["answer"].get("next_token_top5", [])
        top1 = next_token_top5[0] if next_token_top5 else {"token_id": -1, "token": "", "probability": float("nan")}
        gold_probability = math.exp(-float(payload["answer"]["gold_mean_nll"]))
        row = {
            "length": length,
            "key_length": key_length,
            "top2pct_budget": top2pct_budget,
            "prompt_tokens": int(payload["prompt_tokens"]),
            "gold_ppl": float(payload["answer"]["gold_ppl"]),
            "gold_mean_nll": float(payload["answer"]["gold_mean_nll"]),
            "gold_probability": gold_probability,
            "gold_is_top1": int(int(top1["token_id"]) == gold_token_id),
            "gold_in_top5": int(any(int(item["token_id"]) == gold_token_id for item in next_token_top5)),
            "top1_token": str(top1["token"]),
            "top1_probability": float(top1["probability"]),
            "top1_gold_log_gap": math.log(max(float(top1["probability"]), 1e-30)) - math.log(max(gold_probability, 1e-30)),
            "attention_entropy": float(attention["overall_entropy"]),
            "effective_tokens": float(attention["overall_effective_tokens"]),
            "recent512_mass": float(attention["overall_recent512_mass"]),
            "sink16_mass": float(attention["overall_sink16_mass"]),
        }
        for role, index in role_index.items():
            row[f"{role}_mass"] = float(roles[index])
            span_tokens = sum(
                max(0, int(end) - int(start))
                for start, end in payload["spans"].get(role, [])
            )
            uniform_mass = span_tokens / max(1, int(attention["key_length"]))
            row[f"{role}_uniform_enrichment"] = (
                row[f"{role}_mass"] / uniform_mass if uniform_mass > 0 else float("nan")
            )
        row["result_mass"] = row["hop1_result_mass"] + row["hop2_result_mass"]
        row["binding_mass"] = row["hop1_result_mass"] + row["hop2_input_mass"]
        diagnostic_hop2_logits: list[float] = []
        diagnostic_hop2_cosines: list[float] = []
        diagnostic_hop2_ranks: list[float] = []
        diagnostic_denominators: list[float] = []
        diagnostic_query_norms: list[float] = []
        diagnostic_max_logits: list[float] = []
        diagnostic_max_gaps: list[float] = []
        diagnostic_by_role: dict[str, dict[str, list[float]]] = {
            role: {"mass": [], "logit": [], "cosine": [], "rank": [], "max_gap": []}
            for role in ("hop1_result", "hop2_input", "hop2_result")
        }

        for layer_index, layer_heads in enumerate(attention["head_role_mass"]):
            layer_result_values: list[float] = []
            for head_index, head_roles in enumerate(layer_heads):
                head_row = {
                    "length": float(length),
                    "key_length": float(key_length),
                    "top2pct_budget": float(top2pct_budget),
                    "ppl": float(payload["answer"]["gold_ppl"]),
                    "hop1": float(head_roles[role_index["hop1_result"]]),
                    "hop2_input": float(head_roles[role_index["hop2_input"]]),
                    "hop2": float(head_roles[role_index["hop2_result"]]),
                }
                head_row["result"] = head_row["hop1"] + head_row["hop2"]
                if "head_role_logit_mean" in attention:
                    head_row.update(
                        {
                            "hop1_logit": float(attention["head_role_logit_mean"][layer_index][head_index][role_index["hop1_result"]]),
                            "hop2_input_logit": float(attention["head_role_logit_mean"][layer_index][head_index][role_index["hop2_input"]]),
                            "hop2_logit": float(attention["head_role_logit_mean"][layer_index][head_index][role_index["hop2_result"]]),
                            "hop1_rank": float(attention["head_role_best_rank"][layer_index][head_index][role_index["hop1_result"]]),
                            "hop2_input_rank": float(attention["head_role_best_rank"][layer_index][head_index][role_index["hop2_input"]]),
                            "hop2_rank": float(attention["head_role_best_rank"][layer_index][head_index][role_index["hop2_result"]]),
                            "hop1_cosine": float(attention["head_role_cosine_mean"][layer_index][head_index][role_index["hop1_result"]]),
                            "hop2_input_cosine": float(attention["head_role_cosine_mean"][layer_index][head_index][role_index["hop2_input"]]),
                            "hop2_cosine": float(attention["head_role_cosine_mean"][layer_index][head_index][role_index["hop2_result"]]),
                            "hop2_key_norm": float(attention["head_role_key_norm_mean"][layer_index][head_index][role_index["hop2_result"]]),
                            "query_norm": float(attention["head_query_norm"][layer_index][head_index]),
                            "max_logit": float(attention["head_max_logit"][layer_index][head_index]),
                            "logsumexp": float(attention["head_logsumexp"][layer_index][head_index]),
                        }
                    )
                    head_row["hop2_log_probability"] = head_row["hop2_logit"] - head_row["logsumexp"]
                    head_row["hop2_max_logit_gap"] = head_row["max_logit"] - head_row["hop2_logit"]
                    diagnostic_hop2_logits.append(head_row["hop2_logit"])
                    diagnostic_hop2_cosines.append(head_row["hop2_cosine"])
                    diagnostic_hop2_ranks.append(head_row["hop2_rank"])
                    diagnostic_denominators.append(head_row["logsumexp"])
                    diagnostic_query_norms.append(head_row["query_norm"])
                    diagnostic_max_logits.append(head_row["max_logit"])
                    diagnostic_max_gaps.append(head_row["hop2_max_logit_gap"])
                    for role, prefix in (
                        ("hop1_result", "hop1"),
                        ("hop2_input", "hop2_input"),
                        ("hop2_result", "hop2"),
                    ):
                        diagnostic_by_role[role]["mass"].append(head_row[prefix])
                        diagnostic_by_role[role]["logit"].append(head_row[f"{prefix}_logit"])
                        diagnostic_by_role[role]["cosine"].append(head_row[f"{prefix}_cosine"])
                        diagnostic_by_role[role]["rank"].append(head_row[f"{prefix}_rank"])
                        diagnostic_by_role[role]["max_gap"].append(head_row["max_logit"] - head_row[f"{prefix}_logit"])
                head_series[(layer_index, head_index)].append(head_row)
                layer_result_values.append(head_row["result"])
            layer_series[layer_index].append(
                {
                    "length": float(length),
                    "ppl": float(payload["answer"]["gold_ppl"]),
                    "result": mean(layer_result_values),
                }
            )
        if diagnostic_hop2_logits:
            row.update(
                {
                    "mean_hop2_logit": mean(diagnostic_hop2_logits),
                    "mean_hop2_cosine": mean(diagnostic_hop2_cosines),
                    "mean_hop2_rank": mean(diagnostic_hop2_ranks),
                    "hop2_top2_head_fraction": mean([float(rank <= 2) for rank in diagnostic_hop2_ranks]),
                    "hop2_top100_head_fraction": mean([float(rank <= 100) for rank in diagnostic_hop2_ranks]),
                    "hop2_top2pct_head_fraction": mean([float(rank <= top2pct_budget) for rank in diagnostic_hop2_ranks]),
                    "mean_hop2_rank_percentile": mean([rank / key_length for rank in diagnostic_hop2_ranks]),
                    "mean_head_logsumexp": mean(diagnostic_denominators),
                    "mean_query_norm": mean(diagnostic_query_norms),
                    "mean_head_max_logit": mean(diagnostic_max_logits),
                    "mean_hop2_max_logit_gap": mean(diagnostic_max_gaps),
                }
            )
            for role, values in diagnostic_by_role.items():
                prefix = {"hop1_result": "hop1", "hop2_input": "hop2_input", "hop2_result": "hop2"}[role]
                role_ranks = values["rank"]
                role_row = {
                    "length": length,
                    "role": role,
                    "key_length": key_length,
                    "top2pct_budget": top2pct_budget,
                    "mean_mass": mean(values["mass"]),
                    "mean_logit": mean(values["logit"]),
                    "mean_cosine": mean(values["cosine"]),
                    "mean_rank": mean(role_ranks),
                    "mean_rank_percentile": mean([rank / key_length for rank in role_ranks]),
                    "top2pct_head_fraction": mean([float(rank <= top2pct_budget) for rank in role_ranks]),
                    "top100_head_fraction": mean([float(rank <= 100) for rank in role_ranks]),
                    "mean_max_logit_gap": mean(values["max_gap"]),
                }
                role_mechanism_rows.append(role_row)
                row[f"mean_{prefix}_logit"] = role_row["mean_logit"]
                row[f"mean_{prefix}_cosine"] = role_row["mean_cosine"]
                row[f"mean_{prefix}_rank"] = role_row["mean_rank"]
                row[f"{prefix}_top2pct_head_fraction"] = role_row["top2pct_head_fraction"]
        rows.append(row)

    write_csv(output_dir / "analysis_summary.csv", rows)
    write_csv(output_dir / "role_mechanism_summary.csv", role_mechanism_rows)
    # Keep the model's native 40,960-token boundary visible rather than
    # averaging it into the whole 32K-64K region.
    bin_edges = (0, 8000, 32000, 41000, 64000, 96000, 128001)
    bin_rows: list[dict[str, Any]] = []
    for lower, upper in zip(bin_edges, bin_edges[1:]):
        members = [row for row in rows if lower <= int(row["length"]) < upper]
        if not members:
            continue
        bin_rows.append(
            {
                "length_bin": f"{lower}-{upper - 1}",
                "sample_count": len(members),
                "top1_accuracy": mean([float(row["gold_is_top1"]) for row in members]),
                "top5_recall": mean([float(row["gold_in_top5"]) for row in members]),
                "median_gold_ppl": statistics.median(float(row["gold_ppl"]) for row in members),
                "mean_gold_nll": mean([float(row["gold_mean_nll"]) for row in members]),
                "median_gold_probability": statistics.median(float(row["gold_probability"]) for row in members),
                "mean_result_mass": mean([float(row["result_mass"]) for row in members]),
            }
        )
    write_csv(output_dir / "length_bin_summary.csv", bin_rows)
    lengths = [float(row["length"]) for row in rows]
    ppls = [float(row["gold_ppl"]) for row in rows]
    nlls = [float(row["gold_mean_nll"]) for row in rows]
    correlations: list[dict[str, Any]] = []
    for metric in (
        "gold_ppl",
        "gold_mean_nll",
        "hop1_result_mass",
        "hop2_input_mass",
        "hop2_result_mass",
        "result_mass",
        "attention_entropy",
        "effective_tokens",
        "recent512_mass",
        "sink16_mass",
        "mean_hop2_logit",
        "mean_hop2_cosine",
        "mean_hop2_rank",
        "hop2_top2_head_fraction",
        "hop2_top100_head_fraction",
        "hop2_top2pct_head_fraction",
        "mean_hop2_rank_percentile",
        "mean_head_logsumexp",
        "mean_query_norm",
        "mean_head_max_logit",
        "mean_hop2_max_logit_gap",
    ):
        if metric not in rows[0]:
            continue
        values = [float(row[metric]) for row in rows]
        correlations.append(
            {
                "metric": metric,
                "pearson_with_length": pearson(lengths, values),
                "spearman_with_length": spearman(lengths, values),
                "slope_per_1k": slope_per_1k(lengths, values),
                "pearson_with_ppl": pearson(ppls, values),
                "spearman_with_ppl": spearman(ppls, values),
                "pearson_with_nll": pearson(nlls, values),
                "spearman_with_nll": spearman(nlls, values),
            }
        )
    write_csv(output_dir / "metric_correlations.csv", correlations)

    head_rows: list[dict[str, Any]] = []
    for (layer, head), series in sorted(head_series.items()):
        x = [row["length"] for row in series]
        result = [row["result"] for row in series]
        hop1 = [row["hop1"] for row in series]
        hop2 = [row["hop2"] for row in series]
        local_ppl = [row["ppl"] for row in series]
        head_summary = {
                "layer": layer,
                "head": head,
                "mean_result_mass": mean(result),
                "result_mass_slope_per_1k": slope_per_1k(x, result),
                "result_mass_pearson_length": pearson(x, result),
                "result_mass_spearman_length": spearman(x, result),
                "result_mass_pearson_ppl": pearson(local_ppl, result),
                "hop1_slope_per_1k": slope_per_1k(x, hop1),
                "hop2_slope_per_1k": slope_per_1k(x, hop2),
                "short_result_mass": result[0],
                "long_result_mass": result[-1],
                "long_short_ratio": result[-1] / max(result[0], 1e-30),
            }
        if "hop2_logit" in series[0]:
            for metric in (
                "hop1_logit", "hop2_input_logit", "hop2_logit",
                "hop1_rank", "hop2_input_rank", "hop2_rank",
                "hop1_cosine", "hop2_input_cosine", "hop2_cosine",
                "hop2_key_norm", "query_norm", "logsumexp",
                "hop2_log_probability", "hop2_max_logit_gap",
            ):
                values = [row[metric] for row in series]
                head_summary[f"{metric}_slope_per_1k"] = slope_per_1k(x, values)
                head_summary[f"{metric}_pearson_ppl"] = pearson(local_ppl, values)
                head_summary[f"short_{metric}"] = values[0]
                head_summary[f"long_{metric}"] = values[-1]
        head_rows.append(head_summary)
    write_csv(output_dir / "head_trends.csv", head_rows)

    retrieval_head_rows: list[dict[str, Any]] = []
    if head_series and "hop2_rank" in next(iter(head_series.values()))[0]:
        reference_length = min(lengths, key=lambda value: abs(value - 8000.0))

        def head_point(series: Sequence[dict[str, float]], target: float) -> dict[str, float]:
            return min(series, key=lambda item: abs(item["length"] - target))

        retrieval_heads = [
            key
            for key, series in head_series.items()
            if head_point(series, reference_length)["hop2_rank"] <= 100
        ]
        requested_anchors = (0, 1000, 8000, 32000, 64000, 96000, 128000)
        available_anchors = sorted({int(min(lengths, key=lambda value: abs(value - target))) for target in requested_anchors})
        for anchor in available_anchors:
            points = [head_point(head_series[key], float(anchor)) for key in retrieval_heads]
            retrieval_head_rows.append(
                {
                    "length": anchor,
                    "reference_length": int(reference_length),
                    "reference_retrieval_head_count": len(retrieval_heads),
                    "top100_retention_fraction": mean([float(point["hop2_rank"] <= 100) for point in points]),
                    "top2pct_retention_fraction": mean([float(point["hop2_rank"] <= point["top2pct_budget"]) for point in points]),
                    "top2pct_budget": int(points[0]["top2pct_budget"]) if points else 0,
                    "mean_hop2_mass": mean([point["hop2"] for point in points]),
                    "mean_hop2_logit": mean([point["hop2_logit"] for point in points]),
                    "mean_hop2_rank": mean([point["hop2_rank"] for point in points]),
                    "mean_hop2_max_logit_gap": mean([point["hop2_max_logit_gap"] for point in points]),
                }
            )
    write_csv(output_dir / "retrieval_head_retention.csv", retrieval_head_rows)

    layer_rows: list[dict[str, Any]] = []
    for layer, series in sorted(layer_series.items()):
        x = [row["length"] for row in series]
        result = [row["result"] for row in series]
        local_ppl = [row["ppl"] for row in series]
        layer_rows.append(
            {
                "layer": layer,
                "mean_result_mass": mean(result),
                "result_mass_slope_per_1k": slope_per_1k(x, result),
                "result_mass_pearson_length": pearson(x, result),
                "result_mass_pearson_ppl": pearson(local_ppl, result),
                "short_result_mass": result[0],
                "long_result_mass": result[-1],
            }
        )
    write_csv(output_dir / "layer_trends.csv", layer_rows)

    anchors = [nearest_row(rows, target) for target in (0, 1000, 8000, 32000, 64000, 96000, 128000)]
    corr_by_name = {row["metric"]: row for row in correlations}
    strongest_loss = sorted(head_rows, key=lambda row: row["result_mass_slope_per_1k"])[:10]
    strongest_gain = sorted(head_rows, key=lambda row: row["result_mass_slope_per_1k"], reverse=True)[:10]
    ppl_change = ppls[-1] - ppls[0]
    result_change = rows[-1]["result_mass"] - rows[0]["result_mass"]
    has_qk = "mean_hop2_logit" in rows[0]
    if ppl_change > 0 and result_change < 0:
        verdict = "该样本同时出现正确答案 PPL 上升与结果-token attention mass 下降；下面再用 pre-softmax logit、softmax 分母和 rank 判断下降来自哪里。"
    elif ppl_change > 0:
        verdict = "该样本复现了正确答案 PPL 上升，但结果-token attention mass 没有同步下降；仅用 attention dilution 不能解释。"
    else:
        verdict = "该单样本没有复现‘长度越长、正确答案 PPL 越高’；因此不能用它证明置信度下降机制，需要把单样本波动与旧表的多样本均值区分开。"

    lines = [
        "# Qwen3-8B 英文单-token clean 两跳链：长度退化机制扫描",
        "",
        f"- 长度点：{len(rows)}（{int(lengths[0])}–{int(lengths[-1])}，步长 500）",
        f"- 固定链：{' → '.join(payloads[0]['gold_codes'])}",
        f"- 证据位置：{payloads[0].get('placement', 'middle')}；查询：{payloads[0].get('query', {}).get('mode', 'full2')}；seed：{payloads[0].get('seed', 'unknown')}",
        f"- 结论：{verdict}",
        "",
        "## 关键长度点",
        "",
        "| target tokens | Gold PPL | top-1 correct | predicted token | hop1 result mass | hop2 input mass | hop2 result mass | entropy |",
        "|---:|---:|:---:|---|---:|---:|---:|---:|",
    ]
    for row in anchors:
        lines.append(
            f"| {int(row['length'])} | {fmt(row['gold_ppl'])} | {'yes' if row['gold_is_top1'] else 'no'} | "
            f"`{str(row['top1_token']).replace('|', '&#124;')}` | {fmt(row['hop1_result_mass'], 6)} | "
            f"{fmt(row['hop2_input_mass'], 6)} | {fmt(row['hop2_result_mass'], 6)} | "
            f"{fmt(row['attention_entropy'])} |"
        )
    if bin_rows:
        lines.extend(
            [
                "",
                "### 长度区间内的 top-1 / top-5 稳定性",
                "",
                "每个长度点仍是同一条链；这里把相邻长度点当作对 filler 截断位置的密集扰动，用区间命中率减少单点振荡的误导。答案严格为一个 token，因此 top-1 accuracy 等价于一步 greedy 答案正确率。",
                "",
                "| length bin | points | top-1 accuracy | top-5 recall | median PPL | mean NLL | mean result mass |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in bin_rows:
            lines.append(
                f"| {row['length_bin']} | {row['sample_count']} | {fmt(100.0 * row['top1_accuracy'], 1)}% | "
                f"{fmt(100.0 * row['top5_recall'], 1)}% | {fmt(row['median_gold_ppl'])} | "
                f"{fmt(row['mean_gold_nll'])} | {fmt(row['mean_result_mass'], 6)} |"
            )
    lines.extend(
        [
            "",
            "### 相对均匀分布的 enrichment",
            "",
            "| target tokens | hop1 result | hop2 input | hop2 result |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in anchors:
        lines.append(
            f"| {int(row['length'])} | {fmt(row['hop1_result_uniform_enrichment'], 2)}× | "
            f"{fmt(row['hop2_input_uniform_enrichment'], 2)}× | "
            f"{fmt(row['hop2_result_uniform_enrichment'], 2)}× |"
        )
    if role_mechanism_rows:
        role_labels = {
            "hop1_result": "第一跳结果",
            "hop2_input": "第二跳规则输入",
            "hop2_result": "最终结果",
        }
        lines.extend(
            [
                "",
                "### 第一跳与第二跳证据的 Q/K 分解",
                "",
                "同一个中间词在第一条规则的 consequent 与第二条规则的 antecedent 是两个不同位置，因此分别报告。",
                "",
                "| target tokens | role | mass | mean logit | mean rank | Top-2% heads | Q/K cosine | max competitor gap |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for anchor in anchors:
            for role in ("hop1_result", "hop2_input", "hop2_result"):
                role_row = min(
                    (item for item in role_mechanism_rows if item["role"] == role),
                    key=lambda item: abs(int(item["length"]) - int(anchor["length"])),
                )
                lines.append(
                    f"| {int(role_row['length'])} | {role_labels[role]} | {fmt(role_row['mean_mass'], 6)} | "
                    f"{fmt(role_row['mean_logit'])} | {fmt(role_row['mean_rank'], 1)} | "
                    f"{fmt(100.0 * role_row['top2pct_head_fraction'], 1)}% | "
                    f"{fmt(role_row['mean_cosine'], 4)} | {fmt(role_row['mean_max_logit_gap'])} |"
                )
    if has_qk:
        lines.extend(
            [
                "",
                "## Pre-softmax Q/K 诊断",
                "",
                "logit 反映 query 与目标 key 的直接匹配；logsumexp 是 softmax 分母的对数；rank 反映目标 key 在全部历史 token 中的相对名次。",
                "",
                "| target tokens | PPL | hop2 logit | max competitor gap | logsumexp | hop2 rank | rank≤Top-2% heads | rank≤100 heads | Q/K cosine | query norm |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in anchors:
            lines.append(
                f"| {int(row['length'])} | {fmt(row['gold_ppl'])} | {fmt(row['mean_hop2_logit'])} | "
                f"{fmt(row['mean_hop2_max_logit_gap'])} | {fmt(row['mean_head_logsumexp'])} | {fmt(row['mean_hop2_rank'], 1)} | "
                f"{fmt(100.0 * row['hop2_top2pct_head_fraction'], 1)}% | "
                f"{fmt(100.0 * row['hop2_top100_head_fraction'], 1)}% | "
                f"{fmt(row['mean_hop2_cosine'], 4)} | {fmt(row['mean_query_norm'])} |"
            )
        if retrieval_head_rows:
            lines.extend(
                [
                    "",
                    "### 8K 专门检索 heads 的身份保持率",
                    "",
                    f"把 8K 时 hop2 证据 rank≤100 的 {retrieval_head_rows[0]['reference_retrieval_head_count']} 个 layer-head 定义为该样本的检索 heads，再追踪同一批 heads；这避免被大量本来就不负责证据检索的 heads 稀释。",
                    "",
                    "| target tokens | Top-2% budget | Top-2% retention | Top100 retention | target mass | target logit | target rank | max competitor gap |",
                    "|---:|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            for row in retrieval_head_rows:
                lines.append(
                    f"| {int(row['length'])} | {int(row['top2pct_budget'])} | "
                    f"{fmt(100.0 * row['top2pct_retention_fraction'], 1)}% | "
                    f"{fmt(100.0 * row['top100_retention_fraction'], 1)}% | "
                    f"{fmt(row['mean_hop2_mass'], 6)} | {fmt(row['mean_hop2_logit'])} | "
                    f"{fmt(row['mean_hop2_rank'], 1)} | {fmt(row['mean_hop2_max_logit_gap'])} |"
                )

        delta_logit = rows[-1]["mean_hop2_logit"] - rows[0]["mean_hop2_logit"]
        delta_lse = rows[-1]["mean_head_logsumexp"] - rows[0]["mean_head_logsumexp"]
        delta_rank = rows[-1]["mean_hop2_rank"] - rows[0]["mean_hop2_rank"]
        delta_cosine = rows[-1]["mean_hop2_cosine"] - rows[0]["mean_hop2_cosine"]
        delta_max_gap = rows[-1]["mean_hop2_max_logit_gap"] - rows[0]["mean_hop2_max_logit_gap"]
        lines.extend(
            [
                "",
                "### 端点机制分解",
                "",
                f"- 目标 hop2 logit 变化：{fmt(delta_logit)}；Q/K cosine 变化：{fmt(delta_cosine, 4)}。负值表示目标匹配本身退化。",
                f"- logsumexp 变化：{fmt(delta_lse)}。正值表示竞争 token 增多或变强，形成 softmax 分母稀释。",
                f"- hop2 平均 rank 变化：{fmt(delta_rank, 1)}。正值表示目标证据被更多 token 超过。",
                f"- 最强竞争 token 相对 hop2 的 logit gap 变化：{fmt(delta_max_gap)}。正值表示极值竞争加剧。",
                "- 如果 logit 基本不变而 logsumexp 上升，主要是分母稀释；如果 logit/cosine 下降且 rank 变差，则还存在位置相关的 Q/K 检索退化；若 attention 诊断稳定但答案 PPL 仍变坏，则问题更可能位于 value 读取或后续两跳组合。",
            ]
        )
    lines.extend(
        [
            "",
            "## 全局相关性",
            "",
            f"- PPL vs length：Pearson {fmt(corr_by_name['gold_ppl']['pearson_with_length'])}，Spearman {fmt(corr_by_name['gold_ppl']['spearman_with_length'])}",
            f"- NLL=log(PPL) vs length：Pearson {fmt(corr_by_name['gold_mean_nll']['pearson_with_length'])}，Spearman {fmt(corr_by_name['gold_mean_nll']['spearman_with_length'])}",
            f"- hop1-result mass vs length：Pearson {fmt(corr_by_name['hop1_result_mass']['pearson_with_length'])}",
            f"- hop2-result mass vs length：Pearson {fmt(corr_by_name['hop2_result_mass']['pearson_with_length'])}",
            f"- total result mass vs PPL：Pearson {fmt(corr_by_name['result_mass']['pearson_with_ppl'])}",
            f"- total result mass vs NLL：Pearson {fmt(corr_by_name['result_mass']['pearson_with_nll'])}，Spearman {fmt(corr_by_name['result_mass']['spearman_with_nll'])}",
            f"- attention entropy vs PPL：Pearson {fmt(corr_by_name['attention_entropy']['pearson_with_ppl'])}",
            "",
            "这些是同一条样本沿长度变化的相关性，不是跨样本因果估计。",
            "",
            "## 实验口径与解释边界",
            "",
            "本次是 Qwen3-8B、固定 seed 0、固定英文单-token 链的高密度长度扫描。它用于同一样本内的机制定位，不等价于多样本平均准确率；结论还要结合位置和查询分解对照。",
            "",
            f"从 {int(lengths[0])} 到 {int(lengths[-1])} token，raw softmax mass 会受到 key 数量增长的机械影响，因此必须同时看 uniform enrichment、pre-softmax logit、logsumexp 和 rank，不能只凭 mass 判断模型是否‘忘记’证据。",
            "",
            "## 结果 attention 线性 slope 最负的 heads",
            "",
            f"注意：这是对全部 {len(rows)} 点的线性斜率排序；曲线可能非单调，所以 short/long 端点可能与斜率符号不同。",
            "",
            "| layer | head | slope / 1K | short mass | long mass | corr(PPL) |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in strongest_loss:
        lines.append(
            f"| {row['layer']} | {row['head']} | {fmt(row['result_mass_slope_per_1k'], 7)} | "
            f"{fmt(row['short_result_mass'], 6)} | {fmt(row['long_result_mass'], 6)} | "
            f"{fmt(row['result_mass_pearson_ppl'])} |"
        )
    lines.extend(
        [
            "",
            "## 结果 attention 线性 slope 最正的 heads",
            "",
            "| layer | head | slope / 1K | short mass | long mass | corr(PPL) |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in strongest_gain:
        lines.append(
            f"| {row['layer']} | {row['head']} | {fmt(row['result_mass_slope_per_1k'], 7)} | "
            f"{fmt(row['short_result_mass'], 6)} | {fmt(row['long_result_mass'], 6)} | "
            f"{fmt(row['result_mass_pearson_ppl'])} |"
        )
    (output_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"analyzed {len(rows)} lengths -> {output_dir / 'analysis_report.md'}")


if __name__ == "__main__":
    main()
