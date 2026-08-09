from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


CATEGORIES = (
    "self",
    "previous_token",
    "local_recent",
    "sink",
    "punctuation",
    "lexical_copy",
    "syntactic_dependency",
    "structural_anchor",
    "semantic_evidence",
)

ZH = {
    "self": "当前 token/self",
    "previous_token": "前一 token",
    "local_recent": "局部近期上下文",
    "sink": "序列起点/sink",
    "punctuation": "标点与边界",
    "lexical_copy": "同词回指/复制",
    "syntactic_dependency": "句法依赖",
    "structural_anchor": "结构锚点",
    "semantic_evidence": "语义证据",
    "mixed_or_common": "混合/通用",
}

FAMILY = {
    "self": "位置/局部",
    "previous_token": "位置/局部",
    "local_recent": "位置/局部",
    "sink": "位置/局部",
    "punctuation": "表面形式/结构",
    "lexical_copy": "词法/复制",
    "syntactic_dependency": "语言结构",
    "structural_anchor": "表面形式/结构",
    "semantic_evidence": "语义/证据",
}

CONCEPTUAL_RETRIEVER = {
    "self": "位置规则（保留当前位）",
    "previous_token": "位置规则（前一位）",
    "local_recent": "recent-window",
    "sink": "sink+位置规则",
    "punctuation": "格式/边界检索",
    "lexical_copy": "lexical+repeat 检索",
    "syntactic_dependency": "句法感知+语义检索（待实现）",
    "structural_anchor": "括号/标签/格式栈检索",
    "semantic_evidence": "语义证据检索",
}

OPERATOR_ZH = {
    "full": "全量",
    "streaming": "streaming",
    "uniform": "均匀块",
    "lexical_blocks": "词法块",
    "qk_top_blocks": "QK top-block",
    "mass_oracle": "mass oracle",
}

RETRIEVER_ZH = {
    "position": "位置",
    "lexical": "词法",
    "semantic": "语义",
    "format": "格式",
    "repeat": "重复片段",
    "hybrid_lexical": "位置+词法",
    "hybrid_semantic": "位置+语义",
    "hybrid_format": "位置+格式",
    "hybrid_repeat": "位置+重复片段",
    "random": "随机",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the complete Qwen3-0.6B head atlas")
    default_root = Path(__file__).resolve().parents[4]
    parser.add_argument("--repo_root", type=Path, default=default_root)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs",
    )
    parser.add_argument(
        "--docs_dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "docs",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, Any] | None, key: str, default: float = math.nan) -> float:
    if row is None or row.get(key, "") in ("", None):
        return default
    return float(row[key])


def index(rows: Iterable[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {tuple(row[key] for key in keys): row for row in rows}


def head_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["layer"]), int(row["head"])


def dominant_attention(row: dict[str, str]) -> tuple[str, float, float]:
    ranked = sorted(
        ((float(row[f"z_{category}"]), category) for category in CATEGORIES), reverse=True
    )
    best_z, best = ranked[0]
    return best, best_z, best_z - ranked[1][0]


def natural_operator_summary(rows: Sequence[dict[str, str]]) -> dict[tuple[int, int], dict[str, Any]]:
    groups: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if abs(float(row["threshold"]) - 0.05) < 1e-9:
            groups[(int(row["layer"]), int(row["query_head"]))].append(row)
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for key, group in groups.items():
        counts = Counter(row["chosen_action"] for row in group)
        action, count = counts.most_common(1)[0]
        result[key] = {
            "action": action,
            "agreement": count / len(group),
            "query_count": len(group),
            "mean_blocks": statistics.fmean(float(row["selected_blocks"]) for row in group),
            "mean_distortion": statistics.fmean(float(row["relative_output_l2"]) for row in group),
            "counts": ";".join(f"{name}:{value}" for name, value in sorted(counts.items())),
        }
    return result


def confidence(
    profile: dict[str, str],
    distortion: dict[str, str],
    dominant: str,
    nll: dict[str, str] | None,
) -> tuple[str, float, str]:
    score = 0.0
    reasons: list[str] = []
    if profile["primary_function"] != "mixed_or_common":
        score += 2.0
        reasons.append("通过保守专门化阈值")
    z = float(profile[f"z_{dominant}"])
    if z >= 2:
        score += 1.0
    elif z >= 1:
        score += 0.5
    stability = profile["stability_class"]
    if stability == "stable_bias":
        score += 1.5
        reasons.append("跨输入稳定")
    elif stability == "intermediate":
        score += 0.75
    causal_same = distortion["distortion_dominant_category"] == dominant
    causal_z = float(distortion[f"distortion_z_{dominant}"])
    if causal_same:
        score += 1.5
        reasons.append("attention 与局部因果主类一致")
    elif causal_z >= 1:
        score += 0.75
        reasons.append("局部因果效应高于同类 head 中位")
    if float(profile["domain_agreement"]) >= 0.75:
        score += 0.5
    if nll is not None:
        excess = float(nll["mean_target_minus_random_delta_nll"])
        if excess > 0 and float(nll["mean_target_delta_nll"]) > 0:
            score += 1.0
            reasons.append("端到端 NLL 正因果支持")
        elif excess < 0 and float(nll["mean_target_delta_nll"]) < 0:
            score -= 1.0
            reasons.append("端到端 NLL 未支持")
    if score >= 5:
        label = "高"
    elif score >= 3:
        label = "中"
    else:
        label = "低"
    return label, score, "；".join(reasons) if reasons else "仅有相对主签名"


def plot_maps(rows: Sequence[dict[str, Any]], plot_dir: Path) -> list[Path]:
    plot_dir.mkdir(parents=True, exist_ok=True)
    category_index = {category: idx for idx, category in enumerate(CATEGORIES)}
    function_map = np.zeros((28, 16), dtype=np.int64)
    confidence_map = np.zeros((28, 16), dtype=np.int64)
    evidence_map = np.full((28, 16), np.nan, dtype=np.float64)
    conf_index = {"低": 0, "中": 1, "高": 2}
    for row in rows:
        layer, head = int(row["layer"]), int(row["head"])
        function_map[layer, head] = category_index[row["dominant_signature"]]
        confidence_map[layer, head] = conf_index[row["confidence"]]
        evidence_map[layer, head] = float(row["clean_gold_selectivity"])

    paths: list[Path] = []
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(function_map, aspect="auto", interpolation="nearest", cmap="tab10")
    cbar = fig.colorbar(image, ax=ax, ticks=np.arange(len(CATEGORIES)))
    cbar.ax.set_yticklabels([ZH[item] for item in CATEGORIES])
    ax.set(title="Qwen3-0.6B: forced dominant functional signature", xlabel="Query head", ylabel="Layer")
    ax.set_xticks(range(16))
    ax.set_yticks(range(28))
    fig.tight_layout()
    path = plot_dir / "dominant_function_map.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(confidence_map, aspect="auto", interpolation="nearest", cmap="RdYlGn", vmin=0, vmax=2)
    cbar = fig.colorbar(image, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["低", "中", "高"])
    ax.set(title="Head functional-label confidence", xlabel="Query head", ylabel="Layer")
    ax.set_xticks(range(16))
    ax.set_yticks(range(28))
    fig.tight_layout()
    path = plot_dir / "confidence_map.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(evidence_map, aspect="auto", interpolation="nearest", cmap="magma")
    fig.colorbar(image, ax=ax, label="Clean gold-rule selectivity")
    ax.set(title="8K clean-evidence attention selectivity", xlabel="Query head", ylabel="Layer")
    ax.set_xticks(range(16))
    ax.set_yticks(range(28))
    fig.tight_layout()
    path = plot_dir / "clean_evidence_selectivity_map.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)
    return paths


def fmt(value: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return "—"
    number = float(value)
    if not math.isfinite(number):
        return "—"
    return f"{number:.{digits}f}"


def generate_catalog(rows: Sequence[dict[str, Any]], path: Path) -> None:
    lines = [
        "# Qwen3-0.6B 全部 448 个 Query Head 功能目录",
        "",
        "每个 head 都给出一个主签名，但请优先看“保守标签”和“置信度”。“弱→某类”表示该 head 没通过单功能专门化阈值，只是在九类探针中该类相对最强。局部因果一致指删除同类 attention links 后，最大输出变化类别是否与主签名一致。",
        "",
    ]
    by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_layer[int(row["layer"])].append(row)
    for layer in range(28):
        lines.extend(
            [
                f"## Layer {layer}",
                "",
                "| Head | 保守标签 | 完整主签名 | 多标签 | 稳定性 | 局部因果 | 外部检索（实测） | 自然任务安全算子 | 置信度 |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for row in sorted(by_layer[layer], key=lambda item: int(item["head"])):
            conservative = ZH[row["conservative_function"]]
            dominant = ZH[row["dominant_signature"]]
            if row["conservative_function"] == "mixed_or_common":
                dominant = "弱→" + dominant
            multi = row["multi_label_functions_zh"] or "—"
            causal = (
                "一致"
                if row["causal_category_agreement"] == "1"
                else "≠" + ZH[row["distortion_dominant_category"]]
            )
            lines.append(
                "| {head_id} | {conservative} | {dominant} | {multi} | {stability} | "
                "{causal} | {route} | {operator} ({agreement}) | {confidence} |".format(
                    head_id=row["head_id"],
                    conservative=conservative,
                    dominant=dominant,
                    multi=multi,
                    stability=row["stability_class"],
                    causal=causal,
                    route=RETRIEVER_ZH.get(row["empirical_retriever"], row["empirical_retriever"]),
                    operator=OPERATOR_ZH.get(row["natural_safe_operator"], row["natural_safe_operator"]),
                    agreement=fmt(row["natural_operator_agreement"], 2),
                    confidence=row["confidence"],
                )
            )
        lines.append("")
    lines.extend(
        [
            "## 字段边界",
            "",
            "- 功能标签描述的是本实验集合中的可观测 attention/输出签名，不是不可变的神经元语义。",
            "- 外部检索（实测）来自 War and Peace 4K 的 2% oracle-mask imitation；它衡量位置召回，不等于生成 PPL 已被验证。",
            "- 自然任务安全算子来自 64 个查询、相对 head-output L2≤0.05 的 teacher；agreement 越低，越应按 query 动态路由。",
            "- 逐 head 数值、冲突敏感性、NLL 干预和推荐方法请查 `../outputs/head_function_atlas.csv`。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_report(
    rows: Sequence[dict[str, Any]],
    nll_source_rows: Sequence[dict[str, Any]],
    path: Path,
) -> None:
    conservative = Counter(row["conservative_function"] for row in rows)
    dominant = Counter(row["dominant_signature"] for row in rows)
    confidence_counts = Counter(row["confidence"] for row in rows)
    stability = Counter(row["stability_class"] for row in rows)
    causal_agreement = statistics.fmean(int(row["causal_category_agreement"]) for row in rows)
    route_counts = Counter(row["empirical_retriever"] for row in rows)
    operator_counts = Counter(row["natural_safe_operator"] for row in rows)
    clean_selectivity_top = sorted(
        rows, key=lambda row: float(row["clean_gold_selectivity"]), reverse=True
    )[:5]
    conflict_mass_top = sorted(
        rows, key=lambda row: float(row["conflict4_gold_mass"]), reverse=True
    )[:5]
    conflict_discriminator_top = sorted(
        rows,
        key=lambda row: float(row["conflict4_gold_vs_decoy_log2_density_ratio"]),
        reverse=True,
    )[:5]
    conflict_mass_drop = sorted(
        rows, key=lambda row: float(row["conflict_minus_nonconflict_gold_mass"])
    )[:5]
    top_by_category: dict[str, list[dict[str, Any]]] = {}
    for category in CATEGORIES:
        candidates = [row for row in rows if row["dominant_signature"] == category]
        top_by_category[category] = sorted(
            candidates,
            key=lambda row: (float(row["confidence_score"]), float(row["attention_dominant_z"])),
            reverse=True,
        )[:5]

    lines = [
        "# Qwen3-0.6B 不同 Attention Head 功能图谱",
        "",
        "日期：2026-07-15  ",
        "模型：Qwen3-0.6B，28 层 × 16 query heads（8 KV heads），共 448 个 query heads。",
        "",
        "## 结论先行",
        "",
        "1. 不能把每个 head 可靠地压成一个永久功能。按严格的跨-head 专门化阈值，只有 172/448 个 head 得到单一保守标签，276/448 属于混合或通用；因此附录同时提供保守标签和用于完整枚举的强制主签名。",
        f"2. 受控输入中的功能轮廓整体可复现：稳定偏置 {stability['stable_bias']} 个、中等稳定 {stability['intermediate']} 个、上下文敏感 {stability['context_sensitive']} 个。attention 主签名与局部因果主类的一致率为 {causal_agreement:.1%}，说明只看 attention mass 会漏掉 V 向量方向与抵消效应。",
        "3. 新增端到端干预支持语义证据 head 的真实作用：L21H13 删除证据链接后平均 next-token NLL +0.919；语义前三 head 的平均 ΔNLL +0.343，相对等数量随机删除的额外损失 +0.354。self、sink、前一 token 也有正支持；句法关注在当前后继-token指标上未得到正因果支持。",
        "4. 外部 2% 检索仍远未贴近 oracle：逐-head 路由的测试位置召回 31.86%，同质 position 为 31.51%，只提升 0.34 个百分点，而 oracle mass recall 约 83.19%。这说明功能化检索方向成立，但当前语义/格式检索器过弱，不能直接替代 attention oracle。",
        "5. 自然任务上 head 的偏好不是完全静态。64-query teacher 在 5% 输出误差阈值下，head 内多数动作平均一致率约 0.762；所以最佳设计应是“head prior + query-conditioned gate”，而不是给每个 head 永久固定一个检索器。",
        "",
        "## 功能定义与标注规则",
        "",
        "九类探针为：self、前一 token、局部 recent、sink、标点、同词复制、句法依赖、结构锚点、语义证据。原始分数是 attention mass 相对因果可用 key 比例的 log2 enrichment，并在 32 个控制输入、改写对和多个领域上聚合。",
        "",
        "- 保守标签：主类 robust z≥1 且绝对 enrichment>0；否则标为混合/通用。",
        "- 强制主签名：九类 robust z 的最大值，确保 448 个 head 全部可枚举；它不是唯一功能声明。",
        "- 局部因果证据：删除指定类别 links、重新归一化 attention，精确计算单 head 输出向量变化；共 93,632 条记录。",
        "- 端到端证据：每类局部因果排名前三的 head 做精确模型 forward，对比目标 links 删除与等数量随机 links 删除；共 603 个干预。结构锚点位于样本结尾，没有自然后继 token，未人为构造 NLL target。",
        "- 置信度：综合保守专门化、跨输入稳定、attention/局部因果一致性、跨域一致性与（若有）端到端 NLL；低置信 head 仍列出主签名，但不应当作为固定路由依据。",
        "",
        "## 全局分布",
        "",
        "| 类别 | 保守单功能数 | 强制主签名数 |",
        "|---|---:|---:|",
    ]
    for category in CATEGORIES:
        lines.append(f"| {ZH[category]} | {conservative[category]} | {dominant[category]} |")
    lines.extend(
        [
            f"| 混合/通用 | {conservative['mixed_or_common']} | — |",
            "",
            f"置信度：高 {confidence_counts['高']}，中 {confidence_counts['中']}，低 {confidence_counts['低']}。",
            "",
            "## 端到端 NLL 因果结果（每类前三个 head）",
            "",
            "| 类别 | Head | 目标删除 ΔNLL | 相对随机删除额外 ΔNLL | 正效应样本比例 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in sorted(
        nll_source_rows,
        key=lambda item: (CATEGORIES.index(item["category"]), -float(item["mean_target_delta_nll"])),
    ):
        lines.append(
            f"| {ZH[row['category']]} | L{int(row['layer']):02d}H{int(row['head']):02d} | {fmt(row['mean_target_delta_nll'])} | {fmt(row['mean_target_minus_random_delta_nll'])} | {fmt(row['target_positive_fraction'], 2)} |"
        )
    lines.extend(
        [
            "",
            "句法三 head 的 ΔNLL 为负，只能说明它们在受控句子中有句法 attention 签名，不能据此声称该链接对当前 next-token 目标有正贡献。结构锚点只保留局部输出因果证据。",
            "",
            "## 每类最值得优先复核的 head",
            "",
            "以下按综合置信分和专门化强度列出，不是按单一 attention mass 排名。",
            "",
            "| 类别 | Top heads |",
            "|---|---|",
        ]
    )
    for category in CATEGORIES:
        labels = ", ".join(
            f"{row['head_id']}({row['confidence']}, z={fmt(row['attention_dominant_z'], 2)})"
            for row in top_by_category[category]
        )
        lines.append(f"| {ZH[category]} | {labels or '—'} |")
    lines.extend(
        [
            "",
            "## 对功能化 KV Retrieval 的直接含义",
            "",
            "建议把 head 图谱当作路由先验，而不是硬编码真值：位置/局部类先走 sink+recent；复制类加入 lexical/repeat；标点与结构类加入格式栈；语义证据类走语义检索；句法类需要新增依存/实体关系检索器。随后由轻量 query gate 在候选检索器之间动态选择，并保留 full/QK fallback。",
            "",
            f"War4K 实测路由分布为：{', '.join(f'{RETRIEVER_ZH.get(k,k)} {v}' for k,v in route_counts.most_common())}。目前大多数 head 仍选择位置基线，表明“不同功能应使用不同外部方法”尚未被现有弱检索器充分实现。",
            "",
            f"自然任务 5% 输出误差阈值下的多数安全算子分布为：{', '.join(f'{OPERATOR_ZH.get(k,k)} {v}' for k,v in operator_counts.most_common())}。完整逐 head 推荐见 CSV。",
            "",
            "## 冲突证据结果如何使用",
            "",
            "CSV 同时给出无冲突 gold evidence 的 mass/selectivity/top-2 recall，以及四条竞争链下的 conflict-minus-nonconflict 变化。不要只选无冲突 attention mass 最大的 head：例如高 coverage head 可能被 decoy 劫持。语义路由训练应同时优化 gold recall 和 conflict robustness，并把 decoy mass/选择性下降作为惩罚。",
            "",
            "| 场景与指标 | Top-5 heads |",
            "|---|---|",
            "| 无冲突：gold selectivity | "
            + ", ".join(f"{row['head_id']} ({fmt(row['clean_gold_selectivity'])})" for row in clean_selectivity_top)
            + " |",
            "| 四条竞争链：gold attention mass | "
            + ", ".join(f"{row['head_id']} ({fmt(row['conflict4_gold_mass'])})" for row in conflict_mass_top)
            + " |",
            "| 四条竞争链：gold-vs-decoy log2 density ratio | "
            + ", ".join(
                f"{row['head_id']} ({fmt(row['conflict4_gold_vs_decoy_log2_density_ratio'])})"
                for row in conflict_discriminator_top
            )
            + " |",
            "| 冲突造成 gold mass 降幅最大 | "
            + ", ".join(
                f"{row['head_id']} ({fmt(row['conflict_minus_nonconflict_gold_mass'])})"
                for row in conflict_mass_drop
            )
            + " |",
            "",
            "具体地，L21H13 在强冲突下仍有最高 gold mass，且语义证据删除的端到端 ΔNLL 最大，但它的 gold mass 也因冲突下降 0.008；L17H08 的强冲突 gold mass 很高，却是 gold mass 降幅最大的 head（-0.0105）。L26H02 在无冲突 selectivity 排名第一，并在强冲突 gold-vs-decoy 判别中仍居前三，是更均衡的证据 head。",
            "",
            "## 文件",
            "",
            "- `all_448_head_cards_20260715.md`：全部 448 个 head 的逐层功能目录。",
            "- `../outputs/head_function_atlas.csv`：完整数值、冲突敏感性、检索路由、自然算子和置信依据。",
            "- `../outputs/plots/dominant_function_map.png`：强制主签名热图。",
            "- `../outputs/plots/confidence_map.png`：置信度热图。",
            "- `../outputs/plots/clean_evidence_selectivity_map.png`：真实证据关注热图。",
            "",
            "## 局限与下一轮必要实验",
            "",
            "1. 32 个控制输入足以做第一版图谱，但不是开放域功能全集；应加入 induction、实体追踪、否定、算术、代码缩进、多语言长程依赖。",
            "2. 功能标签主要是 query-head 级；Qwen3-0.6B 的两个 query heads 共享一个 KV head，物理缓存路由最终必须在 GQA union 上评估。已有 War4K 的平均物理 union 为历史的 2.09%，约为单 query-head 2% budget 的 1.039 倍。",
            "3. 外部检索实验目前是 oracle mask imitation，不是实际稀疏 attention PPL。下一步应把候选 mask 真正注入生成，并在 LongBench/Needle/真实语料上测 PPL、任务准确率、延迟和显存。",
            "4. 冲突研究覆盖合成 8K 证据任务；需要在真实 RAG 噪声、矛盾文档和多跳证据上复现。",
            "5. 置信度是可解释的规则分数，不是校准概率。",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.repo_root
    profile_path = root / "ymluo/projects/qwen3_head_function_stability/outputs/full_v3/head_profiles.csv"
    distortion_path = root / "ymluo/projects/qwen3_head_function_stability/outputs/category_link_distortion_full_v1_20260715/head_link_distortion_profiles.csv"
    nll_path = root / "ymluo/projects/qwen3_head_function_stability/outputs/top_head_category_nll_full_v1_20260715/head_category_nll_ablation.csv"
    retrieval_path = root / "ymluo/projects/qwen3_head_routed_retrieval/outputs/pilot_war4k_hybrid_remote_20260714_1933/head_assignments.csv"
    evidence_path = root / "ymluo/projects/qwen3_local_rule_failure_boundary/outputs/head_evidence_attention_8k_20260714/head_summary_by_condition.csv"
    conflict_path = root / "ymluo/projects/qwen3_local_rule_failure_boundary/outputs/head_evidence_attention_8k_20260714/paired_conflict_effect_by_head.csv"
    operator_path = root / "ymluo/projects/mor_kv_operator_routing/outputs/head_distortion_teacher_64_v3/merged/compiled_actions.csv"

    profiles = read_csv(profile_path)
    distortions = {head_key(row): row for row in read_csv(distortion_path)}
    retrieval = {head_key(row): row for row in read_csv(retrieval_path)}
    nll_rows = read_csv(nll_path)
    nll_by_key_category = {
        (int(row["layer"]), int(row["head"]), row["category"]): row for row in nll_rows
    }
    nll_by_head: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in nll_rows:
        nll_by_head[(int(row["layer"]), int(row["head"]))].append(row)
    evidence_rows = read_csv(evidence_path)
    evidence = {
        (row["condition"], row["competitor_count"], int(row["layer"]), int(row["head"])): row
        for row in evidence_rows
    }
    conflict_rows = read_csv(conflict_path)
    conflict = {
        (row["competitor_count"], int(row["layer"]), int(row["head"])): row
        for row in conflict_rows
    }
    operators = natural_operator_summary(read_csv(operator_path))

    atlas: list[dict[str, Any]] = []
    for profile in sorted(profiles, key=head_key):
        layer, head = head_key(profile)
        key = (layer, head)
        distortion = distortions[key]
        route = retrieval[key]
        clean = evidence[("nonconflict", "0", layer, head)]
        conflict4 = evidence[("conflict", "4", layer, head)]
        delta4 = conflict[("4", layer, head)]
        operator = operators[key]
        dominant, dominant_z, attention_margin = dominant_attention(profile)
        matching_nll = nll_by_key_category.get((layer, head, dominant))
        label, confidence_score, confidence_basis = confidence(
            profile, distortion, dominant, matching_nll
        )
        tested = nll_by_head.get(key, [])
        representative_nll = matching_nll
        if representative_nll is None and tested:
            representative_nll = max(
                tested, key=lambda row: abs(float(row["mean_target_minus_random_delta_nll"]))
            )
        multi = [item for item in profile["multi_label_functions"].split(";") if item]
        causal_agreement = distortion["distortion_dominant_category"] == dominant
        row: dict[str, Any] = {
            "head_id": f"L{layer:02d}H{head:02d}",
            "layer": layer,
            "head": head,
            "kv_head": head // 2,
            "conservative_function": profile["primary_function"],
            "conservative_function_zh": ZH[profile["primary_function"]],
            "dominant_signature": dominant,
            "dominant_signature_zh": ZH[dominant],
            "function_family": FAMILY[dominant],
            "multi_label_functions": profile["multi_label_functions"],
            "multi_label_functions_zh": ";".join(ZH[item] for item in multi),
            "attention_dominant_z": dominant_z,
            "attention_dominant_margin": attention_margin,
            "attention_dominant_log2_enrichment": profile[f"score_{dominant}"],
            "profile_cosine_mean": profile["profile_cosine_mean"],
            "paired_paraphrase_cosine_mean": profile["paired_paraphrase_cosine_mean"],
            "primary_label_consistency": profile["primary_label_consistency"],
            "domain_agreement": profile["domain_agreement"],
            "stability_class": profile["stability_class"],
            "distortion_dominant_category": distortion["distortion_dominant_category"],
            "distortion_dominant_category_zh": ZH[distortion["distortion_dominant_category"]],
            "causal_category_agreement": int(causal_agreement),
            "dominant_category_relative_output_l2": distortion[f"distortion_{dominant}"],
            "dominant_category_distortion_z": distortion[f"distortion_z_{dominant}"],
            "nll_tested_category": representative_nll["category"] if representative_nll else "",
            "nll_target_delta": representative_nll["mean_target_delta_nll"] if representative_nll else "",
            "nll_random_delta": representative_nll["mean_random_delta_nll"] if representative_nll else "",
            "nll_target_minus_random": representative_nll["mean_target_minus_random_delta_nll"] if representative_nll else "",
            "nll_positive_fraction": representative_nll["target_positive_fraction"] if representative_nll else "",
            "nll_tested_categories": ";".join(sorted(row["category"] for row in tested)),
            "clean_gold_mass": clean["mean_gold_rule_mass"],
            "clean_gold_selectivity": clean["mean_gold_rule_selectivity"],
            "clean_gold_top2_token_recall": clean["mean_gold_top2_token_recall"],
            "clean_gold_top2_mass_recall": clean["mean_gold_top2_mass_recall"],
            "clean_gold_vs_decoy_log2_density_ratio": clean["mean_gold_vs_decoy_log2_density_ratio"],
            "conflict4_gold_mass": conflict4["mean_gold_rule_mass"],
            "conflict4_decoy_mass": conflict4["mean_decoy_rule_mass"],
            "conflict4_gold_selectivity": conflict4["mean_gold_rule_selectivity"],
            "conflict4_gold_top2_token_recall": conflict4["mean_gold_top2_token_recall"],
            "conflict4_gold_vs_decoy_log2_density_ratio": conflict4["mean_gold_vs_decoy_log2_density_ratio"],
            "conflict_minus_nonconflict_gold_mass": delta4["mean_delta_gold_rule_mass"],
            "conflict_minus_nonconflict_decoy_mass": delta4["mean_delta_decoy_rule_mass"],
            "conflict_minus_nonconflict_gold_selectivity": delta4["mean_delta_gold_rule_selectivity"],
            "conflict_minus_nonconflict_top2_recall": delta4["mean_delta_gold_top2_token_recall"],
            "empirical_retriever": route["train_best_method"],
            "empirical_retriever_zh": RETRIEVER_ZH.get(route["train_best_method"], route["train_best_method"]),
            "test_position_recall": route["test_position_recall"],
            "routed_test_position_recall": route.get("balanced_test_position_recall", ""),
            "test_remote_position_recall": route["test_remote_position_recall"],
            "diagnostic_oracle_best_retriever": route["diagnostic_test_oracle_best_method"],
            "conceptual_retriever_recommendation": CONCEPTUAL_RETRIEVER[dominant],
            "natural_safe_operator": operator["action"],
            "natural_safe_operator_zh": OPERATOR_ZH.get(operator["action"], operator["action"]),
            "natural_operator_agreement": operator["agreement"],
            "natural_operator_query_count": operator["query_count"],
            "natural_operator_mean_blocks": operator["mean_blocks"],
            "natural_operator_mean_relative_output_l2": operator["mean_distortion"],
            "natural_operator_action_counts": operator["counts"],
            "confidence": label,
            "confidence_score": confidence_score,
            "confidence_basis": confidence_basis,
        }
        for category in CATEGORIES:
            row[f"attention_z_{category}"] = profile[f"z_{category}"]
            row[f"attention_score_{category}"] = profile[f"score_{category}"]
            row[f"distortion_{category}"] = distortion[f"distortion_{category}"]
            row[f"distortion_z_{category}"] = distortion[f"distortion_z_{category}"]
        atlas.append(row)

    if len(atlas) != 448 or len({row["head_id"] for row in atlas}) != 448:
        raise RuntimeError("atlas must contain exactly 448 unique heads")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.docs_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "head_function_atlas.csv", atlas)
    plot_paths = plot_maps(atlas, args.output_dir / "plots")
    generate_catalog(atlas, args.docs_dir / "all_448_head_cards_20260715.md")
    generate_report(
        atlas,
        nll_rows,
        args.docs_dir / "qwen3_0p6b_head_function_atlas_20260715.md",
    )
    summary = {
        "head_count": len(atlas),
        "conservative_counts": dict(Counter(row["conservative_function"] for row in atlas)),
        "dominant_counts": dict(Counter(row["dominant_signature"] for row in atlas)),
        "confidence_counts": dict(Counter(row["confidence"] for row in atlas)),
        "stability_counts": dict(Counter(row["stability_class"] for row in atlas)),
        "causal_category_agreement": statistics.fmean(
            int(row["causal_category_agreement"]) for row in atlas
        ),
        "empirical_retriever_counts": dict(Counter(row["empirical_retriever"] for row in atlas)),
        "natural_safe_operator_counts": dict(Counter(row["natural_safe_operator"] for row in atlas)),
        "plot_paths": [str(path) for path in plot_paths],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
