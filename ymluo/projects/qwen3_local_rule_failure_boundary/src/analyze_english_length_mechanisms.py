from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


PROBE_LABELS = (
    "prefix_full2",
    "recent_full2",
    "middle_hop1",
    "middle_oracle_hop2",
    "native40k_middle_full2",
    "yarn2_middle_full2",
    "middle_hop1_cloze",
    "middle_oracle_hop2_cloze",
)


def mean(values: Iterable[float]) -> float:
    rows = list(values)
    return statistics.fmean(rows) if rows else float("nan")


def median(values: Iterable[float]) -> float:
    rows = list(values)
    return statistics.median(rows) if rows else float("nan")


def load_payloads(directory: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(
            (directory / "data").glob("length_*.json"),
            key=lambda path: int(path.stem.split("_")[-1]),
        )
    ]


def summarize(label: str, payload: dict[str, Any]) -> dict[str, Any]:
    attention = payload["attention"]
    role_index = {role: index for index, role in enumerate(attention["role_order"])}
    query_mode = payload["query"]["mode"]
    target_role = "hop1_result" if query_mode == "hop1" else "hop2_result"
    target_index = role_index[target_role]
    key_length = int(attention["key_length"])
    top2pct_budget = max(1, int(math.ceil(0.02 * key_length)))
    gold_token_id = int(payload["answer"]["gold_token_scores"][0]["token_id"])
    top1 = payload["answer"]["next_token_top5"][0]
    logits: list[float] = []
    logsumexp: list[float] = []
    ranks: list[float] = []
    cosines: list[float] = []
    key_norms: list[float] = []
    query_norms: list[float] = []
    max_logits: list[float] = []
    for layer_index, layer_heads in enumerate(attention["head_role_mass"]):
        for head_index in range(len(layer_heads)):
            logits.append(float(attention["head_role_logit_mean"][layer_index][head_index][target_index]))
            logsumexp.append(float(attention["head_logsumexp"][layer_index][head_index]))
            ranks.append(float(attention["head_role_best_rank"][layer_index][head_index][target_index]))
            cosines.append(float(attention["head_role_cosine_mean"][layer_index][head_index][target_index]))
            key_norms.append(float(attention["head_role_key_norm_mean"][layer_index][head_index][target_index]))
            query_norms.append(float(attention["head_query_norm"][layer_index][head_index]))
            max_logits.append(float(attention["head_max_logit"][layer_index][head_index]))
    roles = attention["overall_role_mass"]
    top2pct_roles = attention.get("overall_mean_head_top2pct_role_mass")
    top2pct_target_mass = (
        float(top2pct_roles[target_index]) if top2pct_roles is not None else float("nan")
    )
    raw_target_mass = float(roles[target_index])
    return {
        "label": label,
        "length": int(payload["target_context_tokens"]),
        "placement": payload["placement"],
        "query_mode": query_mode,
        "rope_factor": float(payload.get("model_config", {}).get("rope_factor", float("nan"))),
        "target_role": target_role,
        "gold_answer": payload["answer"]["gold_answer"],
        "gold_ppl": float(payload["answer"]["gold_ppl"]),
        "gold_probability": math.exp(-float(payload["answer"]["gold_mean_nll"])),
        "gold_is_top1": int(int(top1["token_id"]) == gold_token_id),
        "top1_token": str(top1["token"]),
        "top1_probability": float(top1["probability"]),
        "attention_entropy": float(attention["overall_entropy"]),
        "hop1_result_mass": float(roles[role_index["hop1_result"]]),
        "hop2_input_mass": float(roles[role_index["hop2_input"]]),
        "hop2_result_mass": float(roles[role_index["hop2_result"]]),
        "mean_target_logit": mean(logits),
        "mean_head_logsumexp": mean(logsumexp),
        "mean_head_max_logit": mean(max_logits),
        "mean_competitor_gap": mean(maximum - target for maximum, target in zip(max_logits, logits)),
        "mean_target_log_probability": mean(logit - denominator for logit, denominator in zip(logits, logsumexp)),
        "mean_target_rank": mean(ranks),
        "median_target_rank": median(ranks),
        "target_top2_head_fraction": mean(float(rank <= 2) for rank in ranks),
        "target_top100_head_fraction": mean(float(rank <= 100) for rank in ranks),
        "top2pct_budget": top2pct_budget,
        "target_top2pct_head_fraction": mean(float(rank <= top2pct_budget) for rank in ranks),
        "mean_top2pct_kept_mass": float(attention.get("overall_mean_head_top2pct_kept_mass", float("nan"))),
        "top2pct_renormalized_target_mass": top2pct_target_mass,
        "top2pct_target_mass_boost": top2pct_target_mass / raw_target_mass if raw_target_mass > 0 else float("nan"),
        "mean_target_cosine": mean(cosines),
        "mean_target_key_norm": mean(key_norms),
        "mean_query_norm": mean(query_norms),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if math.isfinite(value) else "nan"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare length-failure mechanisms for English single-token rules.")
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--probes_root", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    baseline_dir = Path(args.baseline_dir)
    probes_root = Path(args.probes_root)
    output_dir = Path(args.output_dir)
    anchor_lengths = {8000, 32000, 64000, 96000, 128000}
    rows = [
        summarize("middle_full2", payload)
        for payload in load_payloads(baseline_dir)
        if int(payload["target_context_tokens"]) in anchor_lengths
    ]
    for label in PROBE_LABELS:
        rows.extend(summarize(label, payload) for payload in load_payloads(probes_root / label))
    rows.sort(key=lambda row: (int(row["length"]), str(row["label"])))
    write_csv(output_dir / "mechanism_comparison.csv", rows)

    by_key = {(row["label"], int(row["length"])): row for row in rows}
    short = by_key.get(("middle_full2", 8000))
    long = by_key.get(("middle_full2", 128000))
    prefix = by_key.get(("prefix_full2", 128000))
    recent = by_key.get(("recent_full2", 128000))
    hop1 = by_key.get(("middle_hop1", 128000))
    oracle = by_key.get(("middle_oracle_hop2", 128000))
    hop1_cloze = by_key.get(("middle_hop1_cloze", 128000))
    oracle_cloze = by_key.get(("middle_oracle_hop2_cloze", 128000))
    native8 = by_key.get(("native40k_middle_full2", 8000))
    native32 = by_key.get(("native40k_middle_full2", 32000))
    yarn2_32 = by_key.get(("yarn2_middle_full2", 32000))
    yarn2_64 = by_key.get(("yarn2_middle_full2", 64000))
    yarn4_32 = by_key.get(("middle_full2", 32000))
    yarn4_64 = by_key.get(("middle_full2", 64000))
    lines = [
        "# English single-token 128K length-failure mechanism study",
        "",
        "固定链：`river → window → basket`。所有 PPL 均按真实的 leading-space completion token 评分。",
        "",
        "## 稀疏锚点与干预",
        "",
        "| condition | length | RoPE factor | PPL | top-1 | target mass | Top-2% kept mass | Top-2% target mass | mean logit | max gap | mean logsumexp | mean rank | Top-2% head frac | Top100 head frac | cosine |",
        "|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        target_mass = row["hop1_result_mass"] if row["target_role"] == "hop1_result" else row["hop2_result_mass"]
        lines.append(
            f"| {row['label']} | {row['length']} | {fmt(row['rope_factor'], 1)} | {fmt(row['gold_ppl'])} | "
            f"{'yes' if row['gold_is_top1'] else 'no'} | {fmt(target_mass, 6)} | "
            f"{fmt(row['mean_top2pct_kept_mass'], 4)} | {fmt(row['top2pct_renormalized_target_mass'], 6)} | "
            f"{fmt(row['mean_target_logit'])} | {fmt(row['mean_competitor_gap'])} | {fmt(row['mean_head_logsumexp'])} | "
            f"{fmt(row['mean_target_rank'], 1)} | {fmt(row['target_top2pct_head_fraction'], 3)} | "
            f"{fmt(row['target_top100_head_fraction'], 3)} | "
            f"{fmt(row['mean_target_cosine'], 5)} |"
        )
    lines.extend(["", "## 机制判别", ""])
    if short and long:
        lines.append(
            f"- 中部证据 full2：8K→128K PPL {fmt(short['gold_ppl'])}→{fmt(long['gold_ppl'])}；"
            f"mean evidence logit 变化 {fmt(long['mean_target_logit'] - short['mean_target_logit'])}，"
            f"最强竞争 gap 变化 {fmt(long['mean_competitor_gap'] - short['mean_competitor_gap'])}，"
            f"log-sum-exp 增长 {fmt(long['mean_head_logsumexp'] - short['mean_head_logsumexp'])}。"
        )
    if long and recent:
        lines.append(
            f"- 把证据移到 query 附近：128K PPL {fmt(long['gold_ppl'])}→{fmt(recent['gold_ppl'])}。"
            "显著恢复说明中部位置/远距离检索是主要瓶颈之一；还需结合 prefix 区分相对距离和绝对位置。"
        )
    if long and prefix and recent:
        lines.append(
            f"- 128K 位置梯度：prefix={fmt(prefix['gold_ppl'])}，middle={fmt(long['gold_ppl'])}，"
            f"recent={fmt(recent['gold_ppl'])}。prefix 最好而 recent 次之，说明极端长度下不是距离越近越好，"
            "而是 absolute-position extrapolation 与 lost-in-the-middle 共同作用。"
        )
    if long and hop1 and oracle:
        lines.append(
            f"- 128K 分解：full2={fmt(long['gold_ppl'])}，hop1={fmt(hop1['gold_ppl'])}，"
            f"oracle-hop2={fmt(oracle['gold_ppl'])}。两个单跳相对 full2 有不同程度恢复但绝对 PPL 仍高，"
            "所以组合/状态更新是部分瓶颈，普通问答路径中的单跳读出也已退化。"
        )
    if hop1_cloze and oracle_cloze:
        lines.append(
            f"- 128K 纯 cloze 检索：hop1={fmt(hop1_cloze['gold_ppl'])}，"
            f"oracle-hop2={fmt(oracle_cloze['gold_ppl'])}。cloze 好而普通单跳差，说明普通单跳结果受 prompt 形式影响；"
            "cloze 也差才支持底层关联检索本身失败。"
        )
    top2_rows = [row for row in rows if math.isfinite(float(row["top2pct_renormalized_target_mass"]))]
    if top2_rows:
        strongest_top2 = max(top2_rows, key=lambda row: float(row["top2pct_target_mass_boost"]))
        lines.append(
            f"- Top-2% 后验估计：在 {strongest_top2['label']} / {strongest_top2['length']}，"
            f"每个 head 的 top-2% 平均保留原 softmax 质量 {fmt(strongest_top2['mean_top2pct_kept_mass'], 3)}，"
            f"目标证据的重归一化质量相对 full attention 放大 {fmt(strongest_top2['top2pct_target_mass_boost'], 2)}×。"
            "放大幅度很小，不能单独解释实际稀疏模型的大幅 PPL 改善；更可能还涉及跨层删除低质量 value、"
            "改变后续 query/residual 轨迹。这里是完整 attention 的单查询后验估计，不是再次执行稀疏模型得到的 PPL。"
        )
    if native8 and native32 and short and yarn4_32:
        lines.append(
            f"- RoPE 对照：8K native-factor1={fmt(native8['gold_ppl'])} vs factor4={fmt(short['gold_ppl'])}；"
            f"32K native-factor1={fmt(native32['gold_ppl'])} vs factor4={fmt(yarn4_32['gold_ppl'])}。"
            "优劣随长度反转，说明位置扩展配置会强烈改变单点行为，但不能概括成某个 factor 始终更好。"
        )
    if yarn2_32 and yarn2_64 and yarn4_32 and yarn4_64:
        lines.append(
            f"- YaRN 强度对照：32K factor2={fmt(yarn2_32['gold_ppl'])} vs factor4={fmt(yarn4_32['gold_ppl'])}；"
            f"64K factor2={fmt(yarn2_64['gold_ppl'])} vs factor4={fmt(yarn4_64['gold_ppl'])}。"
        )
    lines.extend(
        [
            "",
            "判别原则：evidence logit/cosine 与排名稳定、但 log-sum-exp 增长和 mass 下降，属于 softmax 分母稀释；"
            "若 logit/cosine 同时下降且排名恶化，则是 Q/K 检索几何本身退化；若检索指标稳定但答案 PPL 上升，"
            "则问题发生在 value 汇聚、跨层传递或两跳组合之后。",
            "",
            "这些结果来自固定词链和 seed 0，是机制干预证据，不应直接当作跨任务总体效应。",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "mechanism_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
