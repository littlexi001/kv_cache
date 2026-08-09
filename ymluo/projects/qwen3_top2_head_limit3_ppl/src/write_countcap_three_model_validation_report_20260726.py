#!/usr/bin/env python3
"""Write the final three-model CountCap validation report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from summarize_final_direct_multimodel_comparison_20260726 import (
    read_csv,
    summarize_model,
)


def percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def pick(rows: list[dict[str, Any]], **conditions: Any) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if all(row.get(key) == value for key, value in conditions.items())
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one row for {conditions}, got {len(matches)}")
    return matches[0]


def optional_speed(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3f}x"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--qwen25_csv", type=Path, required=True)
    parser.add_argument("--long_speed", type=Path, required=True)
    parser.add_argument("--qwen25_spectrum", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()

    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    long_speed = json.loads(args.long_speed.read_text(encoding="utf-8"))
    qwen25_spectrum = (
        None
        if args.qwen25_spectrum is None
        else json.loads(args.qwen25_spectrum.read_text(encoding="utf-8"))
    )
    qwen_overall, qwen_tasks = summarize_model(
        read_csv(args.qwen25_csv),
        "Qwen2.5-7B-Instruct",
        args.bootstrap_samples,
        args.seed,
    )
    overall = [
        row
        for row in comparison["countcap"]
        if row["subset"] == "longbench16"
    ]
    overall.append(pick(qwen_overall, subset="longbench16"))
    self_index_rows = [
        row
        for row in comparison["countcap"]
        if row["subset"] == "selfindex11"
    ]
    self_index_rows.append(pick(qwen_overall, subset="selfindex11"))
    self_index_published = comparison["published_reference"][
        "SelfIndexingKVCache_Llama31_8B_LongBench11"
    ]
    tasks = list(comparison["countcap_by_task"]) + qwen_tasks

    lines = [
        "# Direct CountCap：三模型独立验证与长上下文边界",
        "",
        "更新时间：2026-07-26",
        "",
        "## 1. 冻结设置",
        "",
        "所有模型使用同一无训练配置：first-2K sampled uncentered PCA48、"
        "grouped log-scale INT4 Key、INT8 Query、256 点 sampled-quantile、"
        "候选内原始 FP16 Q/K/V direct sparse attention；不使用 exact-QK "
        "重排、任务 router、recent 特判、前层 Full 或 Full fallback。",
        "",
        "$$",
        "B(N)=\\min\\left(N,1280,"
        "\\max\\left(256,\\left\\lceil0.06N\\right\\rceil\\right)\\right).",
        "$$",
        "",
        "LongBench 使用 16 个英文任务、每任务 100 条；完整 prompt（含模板、"
        "上下文和问题）不超过 7500 tokens。每个样本均为 Full/CountCap 严格配对。",
        "",
        "## 2. LongBench 16 任务",
        "",
        "| 模型 | 配对样本 | Full Macro | CountCap Macro | 保持率 | "
        "Macro 差值 95% CI | 目标 token/head | 目标比例 | online/token speed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            "| {model} | {samples} | {full:.5f} | {sparse:.5f} | "
            "{retention} | [{low:+.5f}, {high:+.5f}] | {tokens:.1f} | "
            "{fraction} | {speed} |".format(
                model=row["model"],
                samples=int(row["paired_samples"]),
                full=float(row["full_macro"]),
                sparse=float(row["countcap_macro"]),
                retention=percent(row["quality_retention"]),
                low=float(row["macro_delta_ci95_low"]),
                high=float(row["macro_delta_ci95_high"]),
                tokens=float(row["mean_attention_tokens"]),
                fraction=percent(row["mean_attention_fraction"]),
                speed=optional_speed(
                    row.get("paired_online_per_token_speedup")
                ),
            )
        )

    lines.extend(
        [
            "",
            "短 LongBench 的 `online/token speed` 包含检索和生成，但索引固定开销"
            "在短 prompt/短输出上不能摊销；质量表和长上下文速度表必须分开解释。",
            "",
            "### 分任务",
            "",
            "| 任务 | Llama Full/CountCap | Qwen3 Full/CountCap | "
            "Qwen2.5 Full/CountCap |",
            "|---|---:|---:|---:|",
        ]
    )
    for task in sorted({row["task"] for row in tasks}):
        llama = pick(
            tasks, model="Llama-3.1-8B-Instruct", task=task
        )
        qwen3 = pick(tasks, model="Qwen3-4B-Instruct", task=task)
        qwen25 = pick(tasks, model="Qwen2.5-7B-Instruct", task=task)
        lines.append(
            "| {task} | {lf:.4f}/{ls:.4f} | {qf:.4f}/{qs:.4f} | "
            "{of:.4f}/{os:.4f} |".format(
                task=task,
                lf=float(llama["full_score"]),
                ls=float(llama["countcap_score"]),
                qf=float(qwen3["full_score"]),
                qs=float(qwen3["countcap_score"]),
                of=float(qwen25["full_score"]),
                os=float(qwen25["countcap_score"]),
            )
        )

    lines.extend(
        [
            "",
            "## 3. 64K/128K 连续文本 PPL 与速度",
            "",
            "| 模型 | 长度 | cases | PPL 保持率 | 实际 token/head | "
            "per-head 范围 | Decode speed | $\\Delta T_{fixed}$ | "
            "break-even token | 256-token protocol speed |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in long_speed:
        break_even = row.get("break_even_decode_steps")
        lines.append(
            "| {model} | {length} | {cases} | {retention} | {tokens:.1f} | "
            "{minimum:.0f}-{maximum:.0f} | {decode:.3f}x | "
            "{fixed:+.3f}s | {break_even} | {protocol:.3f}x |".format(
                model=row["model"],
                length=int(row["history_tokens"]),
                cases=int(row["paired_cases"]),
                retention=percent(row["ppl_retention"]),
                tokens=float(row["actual_attention_tokens"]),
                minimum=float(row["actual_attention_tokens_min"]),
                maximum=float(row["actual_attention_tokens_max"]),
                decode=float(row["decode_speedup"]),
                fixed=float(row["additional_fixed_seconds_per_case"]),
                break_even=(
                    "-"
                    if break_even is None
                    else math.ceil(float(break_even))
                ),
                protocol=float(row["protocol_speedup"]),
            )
        )

    worst = min(long_speed, key=lambda row: float(row["ppl_retention"]))
    lines.extend(
        [
            "",
            "Decode speed 包含 PCA/INT4 scan、sampled threshold、候选 gather "
            "和精确稀疏 attention；物理 FP16 K/V 仍完整常驻 GPU。最差连续文本 "
            f"PPL 保持率为 {percent(worst['ppl_retention'])}"
            f"（{worst['model']}，{int(worst['history_tokens']) // 1000}K）。"
            "因此不能把 LongBench 质量外推成通用 PPL 无损，也不能把 attention "
            "消费比例写成物理 KV 存储比例。",
        ]
    )

    next_section = 4
    if qwen25_spectrum is not None:
        spectrum_row = pick(
            qwen25_spectrum["by_model"],
            model="qwen25_7b",
        )
        lines.extend(
            [
                "",
                f"## {next_section}. Qwen2.5 的 centered QK 谱外推",
                "",
                "| 模型 | cases | Key 有效秩 | centered QK 有效秩 | "
                "最优 rank-48 fidelity | Full Key-PCA48 fidelity | "
                "First-2K PCA48 fidelity | First-2K score cosine |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
                "| Qwen2.5-7B | {cases} | {key_rank:.2f} | "
                "{qk_rank:.2f} | {optimal} | {full_key} | {prefix} | "
                "{cosine:.4f} |".format(
                    cases=int(spectrum_row["cases"]),
                    key_rank=float(
                        spectrum_row["key_effective_rank_mean"]
                    ),
                    qk_rank=float(
                        spectrum_row["centered_qk_effective_rank_mean"]
                    ),
                    optimal=percent(
                        spectrum_row[
                            "centered_qk_energy_retained_optimal_rank48_mean"
                        ]
                    ),
                    full_key=percent(
                        spectrum_row[
                            "centered_full_key_pca_qk_fidelity_mean"
                        ]
                    ),
                    prefix=percent(
                        spectrum_row[
                            "centered_production_prefix_pca_qk_fidelity_mean"
                        ]
                    ),
                    cosine=float(
                        spectrum_row[
                            "centered_production_prefix_pca_qk_score_cosine_mean"
                        ]
                    ),
                ),
                "",
                "该诊断只验证谱结构能否跨到第三个模型，不把 query-aware "
                "QK-SVD 上界当成可部署方法，也不替代 LongBench/PPL 结果。",
            ]
        )
        next_section += 1

    lines.extend(
        [
            "",
            f"## {next_section}. 与公开强基线的边界",
            "",
            "### Self-Indexing 11-task 子集的协议级对照",
            "",
            "| 来源/模型 | Full | 稀疏方法 | 稀疏分数 | 保持率 | "
            "attention 预算 | 物理 KV |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in self_index_rows:
        lines.append(
            "| 本次 runner/{model} | {full:.2f} | CountCap | "
            "{sparse:.2f} | {retention} | {tokens:.1f} token/head | "
            "100% FP16 |".format(
                model=row["model"],
                full=100.0 * float(row["full_macro"]),
                sparse=100.0 * float(row["countcap_macro"]),
                retention=percent(row["quality_retention"]),
                tokens=float(row["mean_attention_tokens"]),
            )
        )
    lines.extend(
        [
            "| Self-Indexing 论文/Llama-3.1-8B | "
            f"{self_index_published['full_score']:.2f} | 16-bit | "
            f"{self_index_published['self_indexing_16bit_score']:.2f} | "
            f"{percent(self_index_published['self_indexing_16bit_score'] / self_index_published['full_score'])} | "
            f"{int(self_index_published['budget_tokens'])} token | "
            "压缩 K/V |",
            "| Self-Indexing 论文/Llama-3.1-8B | "
            f"{self_index_published['full_score']:.2f} | 2-bit | "
            f"{self_index_published['self_indexing_2bit_score']:.2f} | "
            f"{percent(self_index_published['self_indexing_2bit_score'] / self_index_published['full_score'])} | "
            f"{int(self_index_published['budget_tokens'])} token | "
            "2-bit K/V |",
            "| Self-Indexing 论文/Qwen2.5-14B | "
            f"{self_index_published['qwen25_14b_full_score']:.2f} | "
            "16-bit | "
            f"{self_index_published['qwen25_14b_self_indexing_16bit_score']:.2f} | "
            f"{percent(self_index_published['qwen25_14b_self_indexing_16bit_score'] / self_index_published['qwen25_14b_full_score'])} | "
            f"{int(self_index_published['budget_tokens'])} token | "
            "压缩 K/V |",
            "| Self-Indexing 论文/Qwen2.5-14B | "
            f"{self_index_published['qwen25_14b_full_score']:.2f} | "
            "2-bit | "
            f"{self_index_published['qwen25_14b_self_indexing_2bit_score']:.2f} | "
            f"{percent(self_index_published['qwen25_14b_self_indexing_2bit_score'] / self_index_published['qwen25_14b_full_score'])} | "
            f"{int(self_index_published['budget_tokens'])} token | "
            "2-bit K/V |",
            "",
            "任务集合相同，但模型尺寸、prompt、stop policy、实现和硬件不同；"
            "因此只比较各自相对 Full 的保持率与资源目标，不比较绝对分数排名。",
            "",
            "- AdaKV 的 Llama-3.1-8B Table 5 是 question-aware、16 任务、固定 "
            "B=128--2048；本实验是长度预算且完整 prompt cap 不同。",
            "- RaBitQCache 在 Llama-3.1-8B LongBench 13 任务公开报告 "
            "Full 50.58、RaBitQ 50.63、平均预算 17.33%，且前两层 Full；"
            "CountCap 应只比较同 13 任务相对保持率与预算量级。",
            "- AAAI 2026 Self-Indexing KVCache 在压缩 Key 上直接检索，"
            "LongBench 使用 11-task、160-token 预算，其中 64 个固定 sink；"
            "这是 CountCap 必须正面对比的最近邻。",
            "- NeurIPS 2025 SALS 使用 RoPE-free latent Q/K 选择后仅重构候选；"
            "ICML 2025 RocketKV 结合永久驱逐与低维 top-k；ICLR 2026 "
            "ProxyAttn 使用代表 head 和动态预算。它们分别覆盖低秩、两阶段和"
            "跨 head 路线。",
            "- Loki 已使用离线 PCA 低维 top-k 和候选内完整维度 attention；"
            "LRQK 已使用联合 Q/K 低秩检索。STAR-KV 与 Thin Keys 还直接"
            "压缩隐藏维度。因此 PCA/SVD 低秩本身不是新颖点。",
            "",
            "公开论文数字来自不同硬件、框架、prompt 和 stop policy，不作为"
            "同环境排名。当前最可信的结论来自本报告内部的严格 Full 配对。",
            "",
            f"## {next_section + 1}. 结论",
            "",
            "三模型结果用于检验同一冻结配置能否跨架构迁移；64K/128K PPL 则用于"
            "暴露 LongBench 之外的失败模式。投稿时应同时保留正结果与负结果："
            "CountCap 是低预算、无 Full fallback 的 question-aware sparse "
            "attention 系统，但不是任意连续文本上的无条件等价替换。",
            "",
            "理论推导见 "
            "`docs/20260726_countcap_spectral_stability_mathematical_appendix_zh.md`；"
            "复现协议见 "
            "`docs/20260726_direct_countcap_frozen_method_reproduction_zh.md`。",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
