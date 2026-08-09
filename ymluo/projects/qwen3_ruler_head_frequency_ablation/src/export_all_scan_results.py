from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


INITIAL_STAGES = ("coarse", "refine", "cross", "finalists", "combinations")


def _compact(values: Iterable[int]) -> str:
    values = sorted(set(int(value) for value in values))
    if not values:
        return ""
    ranges: list[str] = []
    start = previous = values[0]
    for value in values[1:]:
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _query_heads(groups: Iterable[int]) -> str:
    heads: list[int] = []
    for group in groups:
        heads.extend(range(int(group) * 4, int(group) * 4 + 4))
    return _compact(heads)


def _atom_fields(spec: dict[str, Any]) -> dict[str, str]:
    atoms = list(spec.get("atoms", []))
    layers: list[int] = []
    groups: list[int] = []
    frequencies: list[int] = []
    for atom in atoms:
        layers.extend(atom.get("layers", []))
        groups.extend(atom.get("head_groups", []))
        frequencies.extend(atom.get("frequency_pairs", []))

    if not atoms:
        return {
            "layers": "",
            "kv_head_groups": "",
            "query_heads": "",
            "frequency_pairs": "",
            "method": "原生 RoPE",
            "frequency_scale": "",
            "warp_start": "",
            "warp_mode": "",
            "score_blend": "",
            "adaptive_gate": "",
            "adaptive_parameters": "",
            "atoms_json": "[]",
        }

    first = atoms[0]
    scale = float(first.get("frequency_scale", 0.0))
    start = first.get("position_warp_start")
    warp_mode = str(first.get("warp_mode", ""))
    blend = float(first.get("score_blend", 1.0))
    gate = str(first.get("adaptive_gate", ""))

    if gate == "semantic_topk":
        method = "pre-RoPE 语义 Top-k 选择后位置修复"
    elif gate:
        method = "Query 自适应门控的相对距离压缩"
    elif start is not None and warp_mode == "relative_distance":
        if int(start) == 0:
            method = "全距离 NoPE 分数连续混合"
        else:
            method = "超过阈值后的相对距离压缩"
    elif start is not None:
        method = "超过阈值后的绝对位置分段压缩"
    elif 0.0 < scale < 1.0:
        method = "全距离频率旋转削弱"
    elif scale == 0.0:
        method = "硬删除所选 RoPE 频率"
    else:
        method = "原生 RoPE"

    adaptive = {
        key: value
        for key, value in first.items()
        if key.startswith("adaptive_")
    }
    return {
        "layers": _compact(layers),
        "kv_head_groups": _compact(groups),
        "query_heads": _query_heads(groups),
        "frequency_pairs": _compact(frequencies),
        "method": method,
        "frequency_scale": str(scale),
        "warp_start": "" if start is None else str(start),
        "warp_mode": warp_mode,
        "score_blend": str(blend),
        "adaptive_gate": gate,
        "adaptive_parameters": json.dumps(adaptive, ensure_ascii=False, sort_keys=True),
        "atoms_json": json.dumps(atoms, ensure_ascii=False, sort_keys=True),
    }


def _base_row() -> dict[str, Any]:
    return {
        "experiment_family": "",
        "stage": "",
        "evaluation": "",
        "data_role": "",
        "precision": "",
        "context_length": "",
        "seeds": "",
        "samples": "",
        "variant": "",
        "status": "completed",
        "layers": "",
        "kv_head_groups": "",
        "query_heads": "",
        "frequency_pairs": "",
        "method": "",
        "frequency_scale": "",
        "warp_start": "",
        "warp_mode": "",
        "score_blend": "",
        "adaptive_gate": "",
        "adaptive_parameters": "",
        "official_score_pct": "",
        "official_delta_pp": "",
        "official_delta_ci95": "",
        "gold_mean_nll": "",
        "gold_ppl": "",
        "mean_gold_nll_improvement": "",
        "gold_nll_improvement_ci95": "",
        "score_improved": "",
        "score_degraded": "",
        "nll_improved": "",
        "nll_degraded": "",
        "qa_f1_pct": "",
        "em_pct": "",
        "contains_answer_pct": "",
        "first_token_accuracy_pct": "",
        "pg19_scored_tokens": "",
        "utility": "",
        "source_path": "",
        "atoms_json": "",
    }


def load_initial(project: Path) -> list[dict[str, Any]]:
    root = project / "outputs" / "deep_head_frequency_sweep_20260805"
    output: list[dict[str, Any]] = []
    for stage in INITIAL_STAGES:
        path = root / stage / "summary.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        for source in rows:
            row = _base_row()
            spec = source.get("spec", {})
            row.update(
                {
                    "experiment_family": "deep_head_frequency_sweep_20260805",
                    "stage": stage,
                    "evaluation": "RULER-32K",
                    "data_role": (
                        "发现集：6 条固定样本"
                        if stage in {"coarse", "refine", "cross"}
                        else "完整 26 条 probe；包含发现样本，不是独立测试"
                    ),
                    "precision": "4-bit weights + BF16 compute",
                    "context_length": 32768,
                    "samples": source.get("samples", ""),
                    "variant": source.get("variant", ""),
                    "official_score_pct": 100 * source["official_score_mean"],
                    "official_delta_pp": 100 * source["paired_official_delta"],
                    "gold_mean_nll": source.get("gold_answer_mean_nll", ""),
                    "gold_ppl": source.get("gold_answer_ppl_from_mean_nll", ""),
                    "mean_gold_nll_improvement": source.get("mean_nll_improvement", ""),
                    "score_improved": source.get("improved_score_samples", ""),
                    "score_degraded": source.get("degraded_score_samples", ""),
                    "nll_improved": source.get("improved_nll_samples", ""),
                    "nll_degraded": source.get("degraded_nll_samples", ""),
                    "first_token_accuracy_pct": 100 * source.get("first_token_accuracy", 0.0),
                    "utility": source.get("utility", ""),
                    "source_path": str(path.relative_to(project)).replace("\\", "/"),
                }
            )
            row.update(_atom_fields(spec))
            output.append(row)
    return output


def infer_spec(variant: str) -> dict[str, Any]:
    if variant == "native_rope":
        return {"atoms": []}
    atoms: list[dict[str, Any]] = []
    if "l25_g3_f46" in variant or "l25_g3_semantic" in variant:
        atom: dict[str, Any] = {"layers": [25], "head_groups": [3], "frequency_pairs": [46]}
        match = re.search(r"_a([0-9.]+)", variant)
        if match:
            atom["frequency_scale"] = float(match.group(1))
        blend = re.search(r"_b([0-9.]+)", variant)
        if blend:
            atom["score_blend"] = float(blend.group(1))
        start = re.search(r"_s(\d+)", variant)
        if start:
            atom["position_warp_start"] = int(start.group(1))
            atom["warp_mode"] = "relative_distance"
        if "nope_score_blend" in variant:
            atom.update(
                {
                    "frequency_scale": 0.0,
                    "position_warp_start": 0,
                    "warp_mode": "relative_distance",
                }
            )
        if "absolute_phase" in variant:
            atom.pop("position_warp_start", None)
            atom.pop("warp_mode", None)
        if "adaptive_" in variant:
            atom.update(
                {
                    "frequency_scale": 0.25,
                    "position_warp_start": 8192,
                    "warp_mode": "relative_distance",
                    "score_blend": 1.0,
                    "adaptive_gate": "remote_concentration",
                }
            )
            mass = re.search(r"_m([0-9.]+)", variant)
            topk = re.search(r"_k(\d+)", variant)
            concentration = re.search(r"_c([0-9.]+)", variant)
            if mass:
                atom["adaptive_remote_mass_scale"] = float(mass.group(1))
            if topk:
                atom["adaptive_topk"] = int(topk.group(1))
            if concentration:
                atom["adaptive_topk_mass_scale"] = float(concentration.group(1))
        if "semantic_" in variant:
            atom["adaptive_gate"] = "semantic_topk"
        atoms.append(atom)
    elif "l18_23_g4_f47" in variant:
        atom = {
            "layers": list(range(18, 24)),
            "head_groups": [4],
            "frequency_pairs": [47],
        }
        match = re.search(r"_a([0-9.]+)", variant)
        if match:
            atom["frequency_scale"] = float(match.group(1))
        start = re.search(r"_s(\d+)", variant)
        if start:
            atom["position_warp_start"] = int(start.group(1))
        if "relative_" in variant:
            atom["warp_mode"] = "relative_distance"
        atoms.append(atom)
    return {"atoms": atoms}


def _evaluation_metadata(relative: str, csv_row: dict[str, str]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "evaluation": "",
        "data_role": "",
        "precision": "BF16 full weights",
        "context_length": "",
        "seeds": csv_row.get("seeds", ""),
    }
    if "validation/combined" in relative:
        metadata.update(
            evaluation="RULER-32K",
            data_role="validation：seeds 43-44",
            context_length=32768,
        )
    elif "test/combined" in relative:
        metadata.update(
            evaluation="RULER-32K",
            data_role="冻结/独立 test",
            context_length=32768,
        )
    elif "long64" in relative:
        metadata.update(
            evaluation="RULER-64K",
            data_role="长度迁移：seeds 57-59",
            context_length=65536,
        )
    elif "longbench_hotpot" in relative:
        metadata.update(
            evaluation="LongBench HotpotQA",
            data_role="冻结跨任务样本",
        )
    elif "pg19_ppl" in relative:
        metadata.update(
            evaluation="PG19 PPL",
            data_role="8 本 held-out books",
            context_length=csv_row.get("context_length", ""),
        )
    return metadata


def load_multiseed(project: Path) -> list[dict[str, Any]]:
    root = project / "outputs" / "multiseed_frequency_scaling_20260806"
    output: list[dict[str, Any]] = []
    for path in sorted(root.rglob("summary.csv")):
        relative = str(path.relative_to(root)).replace("\\", "/")
        with path.open("r", encoding="utf-8", newline="") as handle:
            for source in csv.DictReader(handle):
                variant = source["variant"]
                row = _base_row()
                family = relative.split("/", 1)[0]
                row.update(
                    {
                        "experiment_family": family,
                        "stage": relative.rsplit("/summary.csv", 1)[0],
                        "variant": variant,
                        "samples": source.get("samples", source.get("sample_count", source.get("book_count", ""))),
                        "official_score_pct": _percent(source.get("official_score_mean")),
                        "official_delta_pp": _percent(source.get("paired_official_delta")),
                        "official_delta_ci95": source.get("official_delta_ci95", ""),
                        "gold_mean_nll": source.get("gold_answer_mean_nll", source.get("mean_nll", "")),
                        "gold_ppl": source.get("gold_answer_ppl", source.get("perplexity", "")),
                        "mean_gold_nll_improvement": source.get("mean_gold_nll_improvement", ""),
                        "gold_nll_improvement_ci95": source.get("gold_nll_improvement_ci95", ""),
                        "score_improved": source.get("official_improved", ""),
                        "score_degraded": source.get("official_degraded", ""),
                        "nll_improved": source.get("nll_improved", ""),
                        "nll_degraded": source.get("nll_degraded", ""),
                        "qa_f1_pct": source.get("qa_f1_percent", ""),
                        "em_pct": source.get("em_percent", ""),
                        "contains_answer_pct": source.get("contains_answer_percent", ""),
                        "first_token_accuracy_pct": source.get("first_token_accuracy_percent", ""),
                        "pg19_scored_tokens": source.get("scored_tokens", ""),
                        "source_path": str(path.relative_to(project)).replace("\\", "/"),
                    }
                )
                row.update(_evaluation_metadata(relative, source))
                row.update(_atom_fields(infer_spec(variant)))
                output.append(row)
    return output


def load_documented_results(project: Path) -> list[dict[str, Any]]:
    """Results whose raw remote tables are not currently mirrored locally.

    Every value below is copied from the named project report. These rows are
    kept separate from raw CSV-derived rows through the ``status`` field.
    """

    output: list[dict[str, Any]] = []

    def add(
        *,
        family: str,
        variant: str,
        precision: str,
        seeds: str,
        samples: int,
        score_pct: float,
        delta_pp: float,
        ppl: float,
        nll_improvement: float | None,
        ci95: str = "",
        score_improved: int | None = None,
        score_degraded: int | None = None,
        nll_improved: int | None = None,
        nll_degraded: int | None = None,
        source: str,
    ) -> None:
        row = _base_row()
        row.update(
            {
                "experiment_family": family,
                "stage": "documented_summary",
                "evaluation": "RULER-32K",
                "data_role": "独立/冻结测试；原始远程表尚未镜像到本地",
                "precision": precision,
                "context_length": 32768,
                "seeds": seeds,
                "samples": samples,
                "variant": variant,
                "status": "completed_documented_summary",
                "official_score_pct": score_pct,
                "official_delta_pp": delta_pp,
                "gold_mean_nll": math.log(ppl),
                "gold_ppl": ppl,
                "mean_gold_nll_improvement": "" if nll_improvement is None else nll_improvement,
                "gold_nll_improvement_ci95": ci95,
                "score_improved": "" if score_improved is None else score_improved,
                "score_degraded": "" if score_degraded is None else score_degraded,
                "nll_improved": "" if nll_improved is None else nll_improved,
                "nll_degraded": "" if nll_degraded is None else nll_degraded,
                "source_path": source,
            }
        )
        row.update(_atom_fields(infer_spec(variant)))
        output.append(row)

    stable_report = "docs/stable_frequency_solution_20260806.md"
    add(
        family="f47_stability_4bit",
        variant="native_rope",
        precision="4-bit weights + BF16 compute",
        seeds="45-47",
        samples=78,
        score_pct=82.31,
        delta_pp=0.0,
        ppl=31.89,
        nll_improvement=0.0,
        source=stable_report,
    )
    add(
        family="f47_stability_4bit",
        variant="l18_23_g4_f47_a0.5",
        precision="4-bit weights + BF16 compute",
        seeds="45-47",
        samples=78,
        score_pct=83.33,
        delta_pp=1.03,
        ppl=31.65,
        nll_improvement=math.log(31.89 / 31.65),
        ci95="跨过 0",
        source=stable_report,
    )
    add(
        family="f47_stability_4bit",
        variant="l18_23_g4_f47_a0.25",
        precision="4-bit weights + BF16 compute",
        seeds="45-47",
        samples=78,
        score_pct=83.33,
        delta_pp=1.03,
        ppl=31.03,
        nll_improvement=math.log(31.89 / 31.03),
        ci95="跨过 0",
        source=stable_report,
    )
    add(
        family="f47_stability_4bit",
        variant="l18_23_g4_f47_a0",
        precision="4-bit weights + BF16 compute",
        seeds="45-47",
        samples=78,
        score_pct=85.51,
        delta_pp=3.21,
        ppl=29.91,
        nll_improvement=math.log(31.89 / 29.91),
        ci95="[0.0225, 0.1122]",
        score_improved=3,
        score_degraded=0,
        source=stable_report,
    )
    add(
        family="f47_stability_bf16",
        variant="native_rope",
        precision="BF16 full weights",
        seeds="45-47",
        samples=78,
        score_pct=84.19,
        delta_pp=0.0,
        ppl=26.57,
        nll_improvement=0.0,
        source=stable_report,
    )
    add(
        family="f47_stability_bf16",
        variant="l18_23_g4_f47_a0",
        precision="BF16 full weights",
        seeds="45-47",
        samples=78,
        score_pct=82.91,
        delta_pp=-1.28,
        ppl=25.64,
        nll_improvement=math.log(26.57 / 25.64),
        ci95="[-0.0082, 0.0846]",
        score_improved=0,
        score_degraded=1,
        source=stable_report,
    )

    adaptive_report = "docs/adaptive_gate_results_20260806.md"
    adaptive = (
        ("native_rope", 84.70, 34.096, 0.0, "", None, None),
        (
            "l25_g3_f46_adaptive_m0.1_k8_c0.5",
            84.70,
            33.539,
            0.016484,
            "[0.007748, 0.025768]",
            57,
            21,
        ),
        (
            "l25_g3_f46_relative_s8192_a0.25_b1_fixed",
            84.70,
            33.472,
            0.018458,
            "[0.011574, 0.025997]",
            58,
            20,
        ),
    )
    for variant, score, ppl, improvement, ci95, improved, degraded in adaptive:
        add(
            family="f46_adaptive_gate_bf16",
            variant=variant,
            precision="BF16 full weights",
            seeds="60-62",
            samples=78,
            score_pct=score,
            delta_pp=0.0,
            ppl=ppl,
            nll_improvement=improvement,
            ci95=ci95,
            nll_improved=improved,
            nll_degraded=degraded,
            source=adaptive_report,
        )
    return output


def _percent(value: str | None) -> str | float:
    if value in (None, ""):
        return ""
    return 100.0 * float(value)


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_defined_configurations(project: Path, completed_variants: set[str]) -> list[dict[str, Any]]:
    sources = (
        ("make_stability_specs.py", "validation_specs"),
        ("make_piecewise_warp_specs.py", "piecewise_specs"),
        ("make_f47_distance_conditioned_specs.py", "make_specs"),
        ("make_f47_relative_distance_specs.py", "make_specs"),
        ("make_f46_relative_distance_specs.py", "make_specs"),
        ("make_f46_global_continuous_specs.py", "make_specs"),
        ("make_f46_weak_remote_specs.py", "make_specs"),
        ("make_f46_adaptive_gate_specs.py", "make_specs"),
        ("make_f46_semantic_topk_specs.py", "make_specs"),
    )
    rows: list[dict[str, Any]] = []
    for filename, function_name in sources:
        path = project / "src" / filename
        module = load_module(path)
        for config in getattr(module, function_name)():
            row = {
                "source_generator": filename,
                "stage": config.get("stage", ""),
                "variant": config["name"],
                "result_status": (
                    "本地结果表中已有至少一项评测"
                    if config["name"] in completed_variants
                    else "配置已定义；逐配置结果未同步到本地总表或尚未运行"
                ),
            }
            row.update(_atom_fields(config))
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = str(row[field])
        if value:
            groups[value].append(row)
    output = []
    for key, values in groups.items():
        output.append(
            {
                field: key,
                "config_count": len(values),
                "mean_utility": sum(float(row["utility"]) for row in values) / len(values),
                "mean_official_delta_pp": sum(float(row["official_delta_pp"]) for row in values) / len(values),
                "mean_nll_improvement": sum(float(row["mean_gold_nll_improvement"]) for row in values) / len(values),
            }
        )
    return sorted(output, key=lambda row: int(str(row[field]).split("-")[0]))


def write_report(path: Path, all_rows: list[dict[str, Any]], inventory_rows: list[dict[str, Any]]) -> None:
    initial = [
        row
        for row in all_rows
        if row["experiment_family"] == "deep_head_frequency_sweep_20260805"
    ]
    coarse = [row for row in initial if row["stage"] == "coarse" and row["variant"] != "native_rope"]
    finalists = [row for row in initial if row["stage"] == "finalists"]
    stage_counts = defaultdict(int)
    for row in initial:
        stage_counts[row["stage"]] += 1

    band_rows = _aggregate(coarse, "frequency_pairs")
    head_rows = _aggregate(coarse, "kv_head_groups")

    lines = [
        "# Qwen3-8B RoPE 层 × Head × 频率扫描：完整配置索引",
        "",
        "## 如何读取",
        "",
        "- `KV head-group g` 对应 Query heads `4g–4g+3`。例如 group 3 对应 Query heads 12–15。",
        "- 频率编号越小旋转越快；`F0–F7` 是最高频，`F56–F63` 是最低频。",
        "- `frequency_scale=0` 表示删除所选频率的 RoPE 旋转；`0<scale<1` 表示削弱旋转。",
        "- 正的 `official_delta_pp` 表示 RULER 官方分数提高；正的 `mean_gold_nll_improvement` 表示正确答案概率提高。",
        "- 6 条发现集上的结果只能用于提出候选，不能当作独立泛化结论。",
        "",
        "完整逐配置结果见 `all_result_records.csv`；后续方法网格及本地结果同步状态见 `defined_method_inventory.csv`。",
        "",
        "## 1. 地毯式扫描范围",
        "",
        "粗扫描测试了：",
        "",
        "- 层块：18–23、24–29、30–35；",
        "- KV head-group：0–7，即全部 8 个 KV heads / 全部 32 个 Query heads；",
        "- 频率带：0–7、8–15、16–23、24–31、32–39、40–47、48–55、56–63；",
        "- 方法：把选定层块、head-group、频率带的旋转直接设为单位旋转，即硬删除；",
        "- 配置数：3 × 8 × 8 = 192，外加原生 RoPE。",
        "",
        "各阶段结果记录数：",
        "",
        "| 阶段 | 记录数 | 数据用途 |",
        "|---|---:|---|",
    ]
    roles = {
        "coarse": "6 条发现样本上的粗扫描",
        "refine": "6 条发现样本上的单层/单频率细化",
        "cross": "6 条发现样本上的最佳层×最佳频率交叉",
        "finalists": "完整 26 条 probe 上的 8 个候选+原生基线",
        "combinations": "完整 26 条 probe 上的逐项累加组合+基线",
    }
    for stage in INITIAL_STAGES:
        lines.append(f"| {stage} | {stage_counts[stage]} | {roles[stage]} |")

    lines.extend(
        [
            "",
            "## 2. 粗扫描按频率带汇总",
            "",
            "下面是 24 个层块×head 配置在每个频带上的平均结果。它仍然只是 6 条发现集统计。",
            "",
            "| 频率对 | 配置数 | 平均 utility | 平均官方变化/pp | 平均 Gold NLL 改善 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in band_rows:
        lines.append(
            f"| {row['frequency_pairs']} | {row['config_count']} | {row['mean_utility']:.4f} | "
            f"{row['mean_official_delta_pp']:+.3f} | {row['mean_nll_improvement']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## 3. 粗扫描按 KV head-group 汇总",
            "",
            "| KV group | Query heads | 配置数 | 平均 utility | 平均官方变化/pp | 平均 Gold NLL 改善 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in head_rows:
        group = int(row["kv_head_groups"])
        lines.append(
            f"| {group} | {4*group}–{4*group+3} | {row['config_count']} | {row['mean_utility']:.4f} | "
            f"{row['mean_official_delta_pp']:+.3f} | {row['mean_nll_improvement']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## 4. 完整 26 条 probe 上的全部单项候选",
            "",
            "| 配置 | 层 | KV group / Query heads | 频率 | 方法 | 官方分数 | 变化/pp | Gold PPL | NLL 改善/退化 |",
            "|---|---:|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(finalists, key=lambda item: float(item["official_score_pct"]), reverse=True):
        head_label = (
            "—"
            if not row["kv_head_groups"]
            else f"G{row['kv_head_groups']} / Q{row['query_heads']}"
        )
        lines.append(
            f"| `{row['variant']}` | {row['layers'] or '—'} | "
            f"{head_label} | {row['frequency_pairs'] or '—'} | "
            f"{row['method']} | {float(row['official_score_pct']):.2f}% | "
            f"{float(row['official_delta_pp']):+.2f} | {float(row['gold_ppl']):.3f} | "
            f"{row['nll_improved']}/{row['nll_degraded']} |"
        )

    lines.extend(
        [
            "",
            "## 5. 后续删除、削弱和距离修复网格",
            "",
            "后续共定义了以下方法族。逐配置参数和本地结果状态全部保存在 `defined_method_inventory.csv`。",
            "",
            "| 方法族 | 配置数 | 操作 |",
            "|---|---:|---|",
        ]
    )
    by_generator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in inventory_rows:
        by_generator[row["source_generator"]].append(row)
    descriptions = {
        "make_stability_specs.py": "F46/F47/F40–47 的全距离硬删除或旋转缩放，以及 F46+F47 联合干预",
        "make_piecewise_warp_specs.py": "在绝对位置阈值后削弱旋转；该实现不是最终采用的相对距离定义",
        "make_f47_distance_conditioned_specs.py": "F47 的固定删除与分段位置压缩",
        "make_f47_relative_distance_specs.py": "F47 的相对距离阈值 8K/16K × scale 0/0.25/0.5",
        "make_f46_relative_distance_specs.py": "F46 的相对距离阈值 8K/16K × score blend，以及 scale=0.25",
        "make_f46_global_continuous_specs.py": "F46 全距离 NoPE 分数混合 10%–75% 与全距离旋转缩放",
        "make_f46_weak_remote_specs.py": "F46 远程弱修复对照",
        "make_f46_adaptive_gate_specs.py": "按远程 attention mass 和集中度控制 F46 修复强度",
        "make_f46_semantic_topk_specs.py": "按 pre-RoPE 内容分数选择 Top-0.5%–4% 远程 token，再修复 F46 或搬移完整 QK 分数；当前待运行",
    }
    for generator, rows in by_generator.items():
        lines.append(f"| `{generator}` | {len(rows)} | {descriptions.get(generator, '')} |")

    later_ruler = [
        row
        for row in all_rows
        if row["experiment_family"] != "deep_head_frequency_sweep_20260805"
        and str(row["evaluation"]).startswith("RULER")
    ]
    lines.extend(
        [
            "",
            "### 5.1 当前已同步的后续 RULER 结果",
            "",
            "同一方法在 validation、冻结 test 和 64K 长度迁移上会分别占一行。",
            "",
            "| 评测 | 数据角色 | 配置 | 方法 | 官方分数 | 变化/pp | Gold PPL | 平均 NLL 改善 | NLL 改善/退化 |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in later_ruler:
        score = "—" if row["official_score_pct"] == "" else f"{float(row['official_score_pct']):.2f}%"
        delta = "—" if row["official_delta_pp"] == "" else f"{float(row['official_delta_pp']):+.2f}"
        ppl = "—" if row["gold_ppl"] == "" else f"{float(row['gold_ppl']):.3f}"
        nll = (
            "—"
            if row["mean_gold_nll_improvement"] == ""
            else f"{float(row['mean_gold_nll_improvement']):+.5f}"
        )
        counts = (
            "—"
            if row["nll_improved"] == ""
            else f"{row['nll_improved']}/{row['nll_degraded']}"
        )
        lines.append(
            f"| {row['evaluation']} | {row['data_role']} | `{row['variant']}` | {row['method']} | "
            f"{score} | {delta} | {ppl} | {nll} | {counts} |"
        )

    later_cross = [
        row
        for row in all_rows
        if row["experiment_family"] != "deep_head_frequency_sweep_20260805"
        and row["evaluation"] in {"LongBench HotpotQA", "PG19 PPL"}
    ]
    lines.extend(
        [
            "",
            "### 5.2 当前已同步的 LongBench 与 PG19 结果",
            "",
            "| 评测 | 长度 | 配置 | QA-F1 | EM | Gold/LM PPL |",
            "|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in later_cross:
        length = row["context_length"] or "—"
        qa_f1 = "—" if row["qa_f1_pct"] == "" else f"{float(row['qa_f1_pct']):.2f}%"
        em = "—" if row["em_pct"] == "" else f"{float(row['em_pct']):.2f}%"
        ppl = "—" if row["gold_ppl"] == "" else f"{float(row['gold_ppl']):.4f}"
        lines.append(
            f"| {row['evaluation']} | {length} | `{row['variant']}` | {qa_f1} | {em} | {ppl} |"
        )

    lines.extend(
        [
            "",
            "## 6. 目前可以下的结论",
            "",
            "1. 所有 KV head-groups 和全部 64 个频率都经过了频带级粗扫描；不是只扫描了 F46/F47。",
            "2. 高频 F0–F7 的部分配置能改变官方答案，但完整 probe 上 Gold PPL 明显不稳；低频/中低频 F40–F47 的平均信号更强。",
            "3. 单独删除 L25/G3/F46 在 26 条 probe 上提供了最稳定的 Gold NLL 改善，但硬删除在独立 BF16 复验中不稳定。",
            "4. 当前最可靠的是 L25/G3/F46 的相对距离连续压缩：8K 内保持原生，8K 外额外距离按 25% 积累。",
            "5. 全局 NoPE 混合、简单减弱修复和 attention 集中度门控都没有稳定超过固定相对距离修复。",
            "",
            "## 7. 声明边界",
            "",
            "- `all_result_records.csv` 收录当前工作区已经同步的所有逐配置结果记录；同一配置在不同数据集上的结果会占多行。",
            "- 部分远程 validation 原始表和当前运行中的 64K/语义 Top-k 实验尚未同步，因此在 inventory 中明确标为未同步或未完成，不能填造数字。",
            "- 发现集结果只用于定位候选；论文结论应优先引用独立 seeds、LongBench 和 PG19 结果。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    output = args.output_dir or project / "outputs" / "all_scan_config_results_20260806"

    rows = load_initial(project) + load_multiseed(project) + load_documented_results(project)
    completed = {str(row["variant"]) for row in rows}
    inventory = load_defined_configurations(project, completed)
    write_csv(output / "all_result_records.csv", rows)
    write_csv(output / "defined_method_inventory.csv", inventory)
    write_report(project / "docs" / "all_scan_config_results_20260806.md", rows, inventory)
    print(
        json.dumps(
            {
                "result_records": len(rows),
                "defined_method_configs": len(inventory),
                "result_csv": str(output / "all_result_records.csv"),
                "inventory_csv": str(output / "defined_method_inventory.csv"),
                "report": str(project / "docs" / "all_scan_config_results_20260806.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
