from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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

GENERAL_CATEGORIES = CATEGORIES[:6]


@dataclass(frozen=True)
class ManualProbe:
    category: str
    query_span: tuple[int, int]
    key_span: tuple[int, int]


@dataclass(frozen=True)
class ControlledSample:
    sample_id: str
    domain: str
    pair_id: str
    text: str
    probes: tuple[ManualProbe, ...] = ()


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure per-head attention-function profiles and their stability across "
            "controlled Qwen3 inputs. Labels describe attention patterns, not causal roles."
        )
    )
    parser.add_argument(
        "--model_name_or_path",
        default="/home/fdong/hrj/prove/Qwen3-0.6B",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
    )
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--heads", default="all")
    parser.add_argument("--sample_limit", type=int, default=0)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--min_history", type=int, default=16)
    parser.add_argument("--sink_tokens", type=int, default=4)
    parser.add_argument("--recent_window", type=int, default=16)
    parser.add_argument(
        "--manual_query_tail",
        type=int,
        default=1,
        help=(
            "Use the last N tokens of each annotated query span. The default 1 measures "
            "the causal-LM position that predicts the next answer/closure token."
        ),
    )
    parser.add_argument("--exclude_local_from_typed", type=str2bool, default=True)
    parser.add_argument("--specialization_z_threshold", type=float, default=1.0)
    parser.add_argument("--primary_min_log2_enrichment", type=float, default=0.0)
    parser.add_argument("--score_clip", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--make_plots", type=str2bool, default=True)
    return parser.parse_args()


def parse_index_spec(spec: str, count: int, name: str) -> list[int]:
    if spec.strip().lower() == "all":
        return list(range(count))
    selected: set[int] = set()
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            left, right = item.split("-", 1)
            start, end = int(left), int(right)
            if start > end:
                raise ValueError(f"Invalid {name} range: {item}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(item))
    invalid = [value for value in sorted(selected) if value < 0 or value >= count]
    if invalid:
        raise ValueError(f"{name} values out of range [0, {count - 1}]: {invalid}")
    return sorted(selected)


def locate(text: str, needle: str, *, last: bool = False) -> tuple[int, int]:
    start = text.rfind(needle) if last else text.find(needle)
    if start < 0:
        raise ValueError(f"Could not locate {needle!r} in controlled sample")
    return start, start + len(needle)


def semantic_sample(
    case_id: str,
    records: Sequence[str],
    evidence_text: str,
    questions: Sequence[str],
) -> list[ControlledSample]:
    samples: list[ControlledSample] = []
    for variant, question in enumerate(questions):
        context = "Reference dossier (use the exact entry):\n" + "\n".join(
            f"- {record}" for record in records
        )
        query = f"Question: {question}\nAnswer:"
        text = context + "\n\n" + query
        samples.append(
            ControlledSample(
                sample_id=f"semantic_{case_id}_v{variant}",
                domain="semantic_qa",
                pair_id=f"semantic_{case_id}",
                text=text,
                probes=(
                    ManualProbe(
                        category="semantic_evidence",
                        query_span=locate(text, query, last=True),
                        key_span=locate(text, evidence_text),
                    ),
                ),
            )
        )
    return samples


def syntax_sample(
    sample_id: str,
    domain: str,
    pair_id: str,
    text: str,
    anchor: str,
    query: str,
) -> ControlledSample:
    return ControlledSample(
        sample_id=sample_id,
        domain=domain,
        pair_id=pair_id,
        text=text,
        probes=(
            ManualProbe(
                category="syntactic_dependency",
                query_span=locate(text, query, last=True),
                key_span=locate(text, anchor),
            ),
        ),
    )


def structure_sample(
    sample_id: str,
    domain: str,
    pair_id: str,
    text: str,
    opener: str,
    closer: str,
) -> ControlledSample:
    return ControlledSample(
        sample_id=sample_id,
        domain=domain,
        pair_id=pair_id,
        text=text,
        probes=(
            ManualProbe(
                category="structural_anchor",
                query_span=locate(text, closer, last=True),
                key_span=locate(text, opener),
            ),
        ),
    )


def build_controlled_samples() -> list[ControlledSample]:
    samples: list[ControlledSample] = []
    samples.extend(
        semantic_sample(
            "project_code",
            (
                "Project ORCHID has launch code 7319 and is supervised by Mira Chen.",
                "Project COBALT has launch code 2846 and is supervised by Tomas Reed.",
                "Project LANTERN has launch code 9052 and is supervised by Inez Park.",
                "Project HARBOR has launch code 4173 and is supervised by Omar Bell.",
            ),
            "7319",
            (
                "What is the launch code assigned to Project ORCHID?",
                "Which four-digit code belongs to the ORCHID project?",
                "For Project ORCHID, report its launch code.",
            ),
        )
    )
    samples.extend(
        semantic_sample(
            "archive_mineral",
            (
                "Archive ALPHA stores azurite samples in cabinet 14.",
                "Archive BETA stores calcite samples in cabinet 27.",
                "Archive GAMMA stores olivine samples in cabinet 33.",
                "Archive DELTA stores gypsum samples in cabinet 48.",
            ),
            "olivine",
            (
                "Which mineral is stored by Archive GAMMA?",
                "Name the sample material listed for the GAMMA archive.",
                "What substance does Archive GAMMA contain?",
            ),
        )
    )
    samples.extend(
        semantic_sample(
            "observatory_city",
            (
                "Dr. Lena Ortiz works at the North Observatory in Bergen.",
                "Dr. Pavel Singh works at the East Observatory in Jaipur.",
                "Dr. Amina Cole works at the South Observatory in Accra.",
                "Dr. Kenji Mori works at the West Observatory in Sendai.",
            ),
            "Jaipur",
            (
                "In which city does Dr. Pavel Singh work?",
                "Give the city associated with Pavel Singh's observatory.",
                "Where is the observatory employing Dr. Pavel Singh?",
            ),
        )
    )
    samples.extend(
        semantic_sample(
            "satellite_date",
            (
                "Satellite KESTREL began service on 12 March 2031.",
                "Satellite MERIDIAN began service on 7 October 2033.",
                "Satellite SOLACE began service on 21 June 2035.",
                "Satellite VANGUARD began service on 4 January 2037.",
            ),
            "4 January 2037",
            (
                "When did Satellite VANGUARD begin service?",
                "State the service-start date for VANGUARD.",
                "On what date was the VANGUARD satellite put into service?",
            ),
        )
    )

    samples.extend(
        (
            syntax_sample(
                "syntax_en_fox_v0",
                "syntax_en",
                "syntax_en_fox",
                (
                    "Grammar note: identify the subject that controls the final finite verb. "
                    "The amber fox, after waiting beside the river through several cold and "
                    "windy hours while the birds circled above the meadow, finally chases the rabbit."
                ),
                "The amber fox",
                "chases",
            ),
            syntax_sample(
                "syntax_en_fox_v1",
                "syntax_en",
                "syntax_en_fox",
                (
                    "Read the clause and track its long-distance subject. The amber fox, which "
                    "rested near the river as clouds crossed the wide valley and evening approached, "
                    "eventually chases the rabbit."
                ),
                "The amber fox",
                "chases",
            ),
            syntax_sample(
                "syntax_en_engineer_v0",
                "syntax_en",
                "syntax_en_engineer",
                (
                    "Long dependency example: The careful engineer, despite the repeated warnings "
                    "from every inspector who had examined the aging bridge during winter, approves "
                    "the revised design."
                ),
                "The careful engineer",
                "approves",
            ),
            syntax_sample(
                "syntax_en_engineer_v1",
                "syntax_en",
                "syntax_en_engineer",
                (
                    "Agreement test: The careful engineer, after reviewing the notes that several "
                    "inspectors had submitted about the old bridge and its supports, still approves "
                    "the revised design."
                ),
                "The careful engineer",
                "approves",
            ),
            syntax_sample(
                "syntax_zh_teacher_v0",
                "syntax_zh",
                "syntax_zh_teacher",
                "句法测试：那位年轻老师虽然在漫长的会议中认真听取了许多家长和学生的不同意见，最后仍然接受了新的课程方案。",
                "那位年轻老师",
                "接受",
            ),
            syntax_sample(
                "syntax_zh_teacher_v1",
                "syntax_zh",
                "syntax_zh_teacher",
                "请追踪长距离主谓关系：那位年轻老师在仔细比较所有家长、学生以及其他教师提出的建议以后，最终接受了新的课程方案。",
                "那位年轻老师",
                "接受",
            ),
            syntax_sample(
                "syntax_zh_team_v0",
                "syntax_zh",
                "syntax_zh_team",
                "句法分析样例：这支研究团队尽管连续几个月遭遇设备故障、数据缺失和计划调整，最终还是完成了关键实验。",
                "这支研究团队",
                "完成",
            ),
            syntax_sample(
                "syntax_zh_team_v1",
                "syntax_zh",
                "syntax_zh_team",
                "需要识别主语：这支研究团队在逐一解决仪器校准、样本污染和存储中断等困难之后，终于完成了关键实验。",
                "这支研究团队",
                "完成",
            ),
        )
    )

    json_v0 = (
        'Configuration object begins here: {\n  "service": "atlas",\n  "region": "north",\n'
        '  "retries": 7,\n  "notes": "retain this object through the long explanatory '
        'comment about validation, compatibility, migration, and rollback procedures"\n}'
    )
    json_v1 = (
        'Parse the following record as one complete value: {\n  "service": "atlas",\n'
        '  "owners": ["ana", "bo"],\n  "enabled": true,\n  "description": "the '
        'intervening fields deliberately make the matching boundary distant from its opener"\n}'
    )
    code_v0 = (
        "Evaluate the outer call boundary: compute_total(prepare(alpha, beta), "
        "normalize(gamma, delta), validate(epsilon, zeta), finalize(eta, theta))"
    )
    code_v1 = (
        "Track the first opening parenthesis until its mate: compute_total("
        "prepare(alpha, beta), normalize(gamma, delta), validate(epsilon, zeta), "
        "finalize(eta, theta), archive(iota, kappa))"
    )
    xml_v0 = (
        "Markup boundary test: <payload>header alpha; section beta with several descriptive "
        "phrases; section gamma with identifiers 17, 23, and 41; final checksum omega</payload>"
    )
    xml_v1 = (
        "Keep the element boundary active: <payload>metadata first; a long body describing "
        "transport, validation, recovery, compatibility, and audit requirements; trailer</payload>"
    )
    fence_v0 = (
        "A fenced block follows. ```python\nvalues = [3, 5, 8, 13]\n"
        "result = sum(v * v for v in values)\nprint(result)\n```"
    )
    fence_v1 = (
        "Preserve the Markdown fence boundary. ```text\nalpha beta gamma delta epsilon\n"
        "the middle line is intentionally verbose and contains many ordinary words\nend\n```"
    )
    samples.extend(
        (
            structure_sample("structure_json_v0", "structured_json", "structure_json", json_v0, "{", "}"),
            structure_sample("structure_json_v1", "structured_json", "structure_json", json_v1, "{", "}"),
            structure_sample("structure_code_v0", "structured_code", "structure_code", code_v0, "(", ")"),
            structure_sample("structure_code_v1", "structured_code", "structure_code", code_v1, "(", ")"),
            structure_sample("structure_xml_v0", "structured_markup", "structure_xml", xml_v0, "<payload>", "</payload>"),
            structure_sample("structure_xml_v1", "structured_markup", "structure_xml", xml_v1, "<payload>", "</payload>"),
            structure_sample("structure_fence_v0", "structured_markup", "structure_fence", fence_v0, "```python", "```"),
            structure_sample("structure_fence_v1", "structured_markup", "structure_fence", fence_v1, "```text", "```"),
        )
    )

    prose = (
        (
            "prose_science",
            "Natural prose describes a field expedition. The researchers crossed the plateau at "
            "dawn, measured the temperature near each spring, recorded the cloud cover, and returned "
            "to camp before the afternoon storm reached the northern ridge.",
        ),
        (
            "prose_history",
            "A short historical account follows. Merchants first used the harbor as a seasonal stop, "
            "but warehouses, workshops, and public offices gradually turned the settlement into a "
            "permanent trading city with several distinct neighborhoods.",
        ),
        (
            "prose_dialogue",
            'Nora asked, "Did the package arrive before noon?" Elias checked the ledger, paused for a '
            'moment, and replied, "Yes, but the blue envelope was delivered separately after lunch."',
        ),
        (
            "prose_zh",
            "自然文本样例：清晨的车站还很安静，工作人员依次检查信号、站台和车厢。太阳升起以后，第一批旅客穿过大厅，广播开始提醒大家准备检票。",
        ),
    )
    for sample_id, text in prose:
        samples.append(
            ControlledSample(
                sample_id=sample_id,
                domain="natural_prose",
                pair_id=sample_id,
                text=text,
            )
        )
    return samples


def spans_to_token_indices(
    offsets: Sequence[tuple[int, int]], span: tuple[int, int]
) -> list[int]:
    start, end = span
    return [
        index
        for index, (token_start, token_end) in enumerate(offsets)
        if token_end > token_start and token_start < end and token_end > start
    ]


def token_piece(text: str, offset: tuple[int, int]) -> str:
    start, end = offset
    return text[start:end] if end > start else ""


def is_punctuation_piece(piece: str) -> bool:
    stripped = piece.strip()
    if not stripped:
        return bool(piece)
    return all(unicodedata.category(char).startswith("P") for char in stripped)


def lexical_key(piece: str) -> str:
    return "".join(char.lower() for char in piece if char.isalnum())


def add_probe(
    probes: dict[str, list[tuple[int, tuple[int, ...]]]],
    category: str,
    query: int,
    keys: Iterable[int],
) -> None:
    unique_keys = tuple(sorted({key for key in keys if 0 <= key <= query}))
    if unique_keys:
        probes[category].append((query, unique_keys))


def build_token_probes(
    sample: ControlledSample,
    offsets: Sequence[tuple[int, int]],
    *,
    min_history: int,
    sink_tokens: int,
    recent_window: int,
    manual_query_tail: int,
    exclude_local_from_typed: bool,
) -> dict[str, list[tuple[int, tuple[int, ...]]]]:
    token_count = len(offsets)
    pieces = [token_piece(sample.text, offset) for offset in offsets]
    lexical = [lexical_key(piece) for piece in pieces]
    punctuation = [is_punctuation_piece(piece) for piece in pieces]
    probes: dict[str, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)

    for query in range(max(1, min_history), token_count):
        add_probe(probes, "self", query, (query,))
        add_probe(probes, "previous_token", query, (query - 1,))
        add_probe(
            probes,
            "local_recent",
            query,
            range(max(0, query - recent_window), max(0, query - 1)),
        )
        add_probe(probes, "sink", query, range(min(sink_tokens, query + 1)))

        typed_end = max(0, query - recent_window) if exclude_local_from_typed else query
        typed_start = min(sink_tokens, typed_end) if exclude_local_from_typed else 0
        add_probe(
            probes,
            "punctuation",
            query,
            (index for index in range(typed_start, typed_end) if punctuation[index]),
        )
        if lexical[query]:
            add_probe(
                probes,
                "lexical_copy",
                query,
                (
                    index
                    for index in range(typed_start, typed_end)
                    if lexical[index] == lexical[query]
                ),
            )

    for manual in sample.probes:
        query_indices = spans_to_token_indices(offsets, manual.query_span)
        key_indices = spans_to_token_indices(offsets, manual.key_span)
        if manual_query_tail > 0:
            query_indices = query_indices[-manual_query_tail:]
        for query in query_indices:
            add_probe(
                probes,
                manual.category,
                query,
                (key for key in key_indices if key <= query),
            )
    return probes


def score_attention_category(
    attention: torch.Tensor,
    query_key_pairs: Sequence[tuple[int, tuple[int, ...]]],
    *,
    epsilon: float = 1e-8,
    score_clip: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Return per-head mean mass, mean log2 enrichment, availability, query count."""
    masses: list[torch.Tensor] = []
    scores: list[torch.Tensor] = []
    availabilities: list[float] = []
    for query, keys in query_key_pairs:
        if not keys:
            continue
        key_tensor = torch.tensor(keys, dtype=torch.long)
        mass = attention[:, query, :].index_select(-1, key_tensor).sum(dim=-1)
        availability = len(keys) / float(query + 1)
        enrichment = torch.log2((mass + epsilon) / (availability + epsilon))
        masses.append(mass)
        scores.append(enrichment.clamp(-score_clip, score_clip))
        availabilities.append(availability)
    if not masses:
        raise ValueError("Cannot score an empty category")
    return (
        torch.stack(masses).mean(dim=0).numpy(),
        torch.stack(scores).mean(dim=0).numpy(),
        float(statistics.fmean(availabilities)),
        len(masses),
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def robust_center_scale(values: Sequence[float]) -> tuple[float, float]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return 0.0, 1.0
    center = float(np.median(finite))
    mad = float(np.median(np.abs(finite - center)))
    scale = max(1.4826 * mad, 1e-6)
    return center, scale


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if a.size < 3 or b.size != a.size:
        return float("nan")
    rank_a, rank_b = rankdata(a), rankdata(b)
    if np.std(rank_a) < 1e-12 or np.std(rank_b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def cosine_on_common(
    profile_a: dict[str, float], profile_b: dict[str, float], minimum: int = 3
) -> float:
    common = sorted(set(profile_a) & set(profile_b))
    if len(common) < minimum:
        return float("nan")
    a = np.asarray([profile_a[key] for key in common], dtype=np.float64)
    b = np.asarray([profile_b[key] for key in common], dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denominator)


def finite_mean(values: Iterable[float]) -> float:
    selected = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(selected)) if selected else float("nan")


def finite_median(values: Iterable[float]) -> float:
    selected = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(selected)) if selected else float("nan")


def analyze_features(
    feature_rows: Sequence[dict[str, Any]],
    *,
    selected_layers: Sequence[int],
    selected_heads: Sequence[int],
    specialization_z_threshold: float,
    primary_min_log2_enrichment: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    head_keys = [(layer, head) for layer in selected_layers for head in selected_heads]
    by_head_category: dict[tuple[int, int, str], list[float]] = defaultdict(list)
    by_head_sample: dict[tuple[int, int, str], dict[str, float]] = defaultdict(dict)
    sample_meta: dict[str, tuple[str, str]] = {}
    for row in feature_rows:
        layer, head = int(row["layer"]), int(row["head"])
        category = str(row["category"])
        sample_id = str(row["sample_id"])
        score = float(row["log2_enrichment"])
        by_head_category[(layer, head, category)].append(score)
        by_head_sample[(layer, head, sample_id)][category] = score
        sample_meta[sample_id] = (str(row["domain"]), str(row["pair_id"]))

    aggregate: dict[tuple[int, int, str], dict[str, float]] = {}
    category_head_means: dict[str, list[tuple[tuple[int, int], float]]] = defaultdict(list)
    for layer, head in head_keys:
        for category in CATEGORIES:
            values = by_head_category.get((layer, head, category), [])
            if not values:
                continue
            item = {
                "mean": finite_mean(values),
                "median": finite_median(values),
                "std": float(np.std(values)),
                "positive_rate": float(np.mean(np.asarray(values) > 0.0)),
                "strong_rate": float(np.mean(np.asarray(values) > 1.0)),
                "n_samples": float(len(values)),
            }
            aggregate[(layer, head, category)] = item
            category_head_means[category].append(((layer, head), item["mean"]))

    specialization: dict[tuple[int, int, str], float] = {}
    ranking_rows: list[dict[str, Any]] = []
    for category in CATEGORIES:
        items = category_head_means.get(category, [])
        center, scale = robust_center_scale([value for _, value in items])
        ranked: list[tuple[tuple[int, int], float, float]] = []
        for head_key, value in items:
            z_score = (value - center) / scale
            specialization[(*head_key, category)] = z_score
            ranked.append((head_key, value, z_score))
        ranked.sort(key=lambda item: (item[2], item[1]), reverse=True)
        for rank, (head_key, value, z_score) in enumerate(ranked, start=1):
            stats = aggregate[(*head_key, category)]
            ranking_rows.append(
                {
                    "category": category,
                    "head_rank": rank,
                    "layer": head_key[0],
                    "head": head_key[1],
                    "mean_log2_enrichment": value,
                    "specialization_z": z_score,
                    "sample_std": stats["std"],
                    "positive_rate": stats["positive_rate"],
                    "strong_rate": stats["strong_rate"],
                    "n_samples": int(stats["n_samples"]),
                }
            )

    category_global_center_scale = {
        category: robust_center_scale(
            [
                float(row["log2_enrichment"])
                for row in feature_rows
                if row["category"] == category
            ]
        )
        for category in CATEGORIES
    }
    normalized_profiles: dict[tuple[int, int, str], dict[str, float]] = {}
    for key, profile in by_head_sample.items():
        normalized_profiles[key] = {
            category: (score - category_global_center_scale[category][0])
            / category_global_center_scale[category][1]
            for category, score in profile.items()
        }

    domain_profiles: dict[tuple[int, int, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (layer, head, sample_id), profile in normalized_profiles.items():
        domain = sample_meta[sample_id][0]
        for category, value in profile.items():
            domain_profiles[(layer, head, domain)][category].append(value)

    profile_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for layer, head in head_keys:
        available_categories = [
            category for category in CATEGORIES if (layer, head, category) in aggregate
        ]
        sorted_categories = sorted(
            available_categories,
            key=lambda category: specialization.get((layer, head, category), -math.inf),
            reverse=True,
        )
        best = sorted_categories[0] if sorted_categories else "mixed_or_common"
        best_z = specialization.get((layer, head, best), float("nan"))
        best_mean = (
            aggregate[(layer, head, best)]["mean"]
            if best != "mixed_or_common"
            else float("nan")
        )
        primary = (
            best
            if best_z >= specialization_z_threshold
            and best_mean > primary_min_log2_enrichment
            else "mixed_or_common"
        )
        second_z = (
            specialization.get((layer, head, sorted_categories[1]), float("nan"))
            if len(sorted_categories) > 1
            else float("nan")
        )
        multi_labels = [
            category
            for category in sorted_categories
            if specialization.get((layer, head, category), -math.inf)
            >= specialization_z_threshold
            and aggregate[(layer, head, category)]["mean"]
            > primary_min_log2_enrichment
        ]

        sample_items = [
            (sample_id, normalized_profiles[(layer, head, sample_id)])
            for sample_id in sample_meta
            if (layer, head, sample_id) in normalized_profiles
        ]
        all_cosines: list[float] = []
        paired_cosines: list[float] = []
        for left in range(len(sample_items)):
            left_id, left_profile = sample_items[left]
            for right in range(left + 1, len(sample_items)):
                right_id, right_profile = sample_items[right]
                similarity = cosine_on_common(left_profile, right_profile)
                if math.isfinite(similarity):
                    all_cosines.append(similarity)
                    if sample_meta[left_id][1] == sample_meta[right_id][1]:
                        paired_cosines.append(similarity)
                        pair_rows.append(
                            {
                                "layer": layer,
                                "head": head,
                                "pair_id": sample_meta[left_id][1],
                                "sample_a": left_id,
                                "sample_b": right_id,
                                "profile_cosine": similarity,
                            }
                        )

        label_matches: list[bool] = []
        if primary != "mixed_or_common":
            for _, sample_profile in sample_items:
                if primary not in sample_profile:
                    continue
                sample_primary = max(sample_profile, key=sample_profile.get)
                label_matches.append(sample_primary == primary)

        domain_labels: list[str] = []
        for (domain_layer, domain_head, _domain), values_by_category in domain_profiles.items():
            if domain_layer != layer or domain_head != head:
                continue
            domain_labels.append(
                max(values_by_category, key=lambda category: finite_mean(values_by_category[category]))
            )
        domain_agreement = (
            float(np.mean([label == primary for label in domain_labels]))
            if domain_labels and primary != "mixed_or_common"
            else float("nan")
        )
        profile_cosine = finite_mean(all_cosines)
        paired_cosine = finite_mean(paired_cosines)
        label_consistency = float(np.mean(label_matches)) if label_matches else float("nan")
        if (
            math.isfinite(profile_cosine)
            and profile_cosine >= 0.70
            and (not math.isfinite(label_consistency) or label_consistency >= 0.60)
        ):
            stability_class = "stable_bias"
        elif (
            (math.isfinite(profile_cosine) and profile_cosine < 0.40)
            or (math.isfinite(label_consistency) and label_consistency < 0.40)
        ):
            stability_class = "context_sensitive"
        else:
            stability_class = "intermediate"

        row: dict[str, Any] = {
            "layer": layer,
            "head": head,
            "primary_function": primary,
            "multi_label_functions": ";".join(multi_labels),
            "primary_specialization_z": best_z,
            "specialization_margin": best_z - second_z,
            "profile_cosine_mean": profile_cosine,
            "paired_paraphrase_cosine_mean": paired_cosine,
            "primary_label_consistency": label_consistency,
            "domain_agreement": domain_agreement,
            "stability_class": stability_class,
        }
        for category in CATEGORIES:
            stats = aggregate.get((layer, head, category))
            row[f"score_{category}"] = stats["mean"] if stats else float("nan")
            row[f"z_{category}"] = specialization.get(
                (layer, head, category), float("nan")
            )
        profile_rows.append(row)

    category_stability_rows: list[dict[str, Any]] = []
    head_order = head_keys
    for category in CATEGORIES:
        sample_vectors: dict[str, list[float]] = {}
        for sample_id in sample_meta:
            values: list[float] = []
            complete = True
            for layer, head in head_order:
                profile = by_head_sample.get((layer, head, sample_id), {})
                if category not in profile:
                    complete = False
                    break
                values.append(profile[category])
            if complete:
                sample_vectors[sample_id] = values
        correlations: list[float] = []
        paired_correlations: list[float] = []
        sample_ids = sorted(sample_vectors)
        for left in range(len(sample_ids)):
            for right in range(left + 1, len(sample_ids)):
                correlation = spearman(
                    sample_vectors[sample_ids[left]], sample_vectors[sample_ids[right]]
                )
                if math.isfinite(correlation):
                    correlations.append(correlation)
                    if sample_meta[sample_ids[left]][1] == sample_meta[sample_ids[right]][1]:
                        paired_correlations.append(correlation)
        category_stability_rows.append(
            {
                "category": category,
                "n_samples": len(sample_ids),
                "n_sample_pairs": len(correlations),
                "head_rank_spearman_mean": finite_mean(correlations),
                "head_rank_spearman_median": finite_median(correlations),
                "paired_paraphrase_spearman_mean": finite_mean(paired_correlations),
                "n_paraphrase_pairs": len(paired_correlations),
            }
        )
    return profile_rows, ranking_rows, category_stability_rows, pair_rows


def safe_number(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def make_plots(output_dir: Path, profile_rows: Sequence[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
    except ImportError:
        return []

    layers = sorted({int(row["layer"]) for row in profile_rows})
    heads = sorted({int(row["head"]) for row in profile_rows})
    row_map = {(int(row["layer"]), int(row["head"])): row for row in profile_rows}
    label_order = ("mixed_or_common",) + CATEGORIES
    label_to_index = {label: index for index, label in enumerate(label_order)}
    label_grid = np.zeros((len(layers), len(heads)), dtype=np.int64)
    stability_grid = np.full((len(layers), len(heads)), np.nan, dtype=np.float64)
    for layer_index, layer in enumerate(layers):
        for head_index, head in enumerate(heads):
            row = row_map[(layer, head)]
            label_grid[layer_index, head_index] = label_to_index[row["primary_function"]]
            stability_grid[layer_index, head_index] = row["profile_cosine_mean"]

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    colors = ["#d9d9d9", "#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b", "#eeca3b", "#b279a2", "#ff9da6", "#9d755d"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(label_order) + 0.5), cmap.N)
    fig, ax = plt.subplots(figsize=(max(9, len(heads) * 0.55), max(7, len(layers) * 0.3)))
    image = ax.imshow(label_grid, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(len(heads)), heads)
    ax.set_yticks(range(len(layers)), layers)
    ax.set_xlabel("Attention head")
    ax.set_ylabel("Layer")
    ax.set_title("Primary attention-pattern specialization")
    colorbar = fig.colorbar(image, ax=ax, ticks=range(len(label_order)), fraction=0.03, pad=0.02)
    colorbar.ax.set_yticklabels(label_order)
    fig.tight_layout()
    path = plot_dir / "primary_function_map.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(max(9, len(heads) * 0.55), max(7, len(layers) * 0.3)))
    image = ax.imshow(stability_grid, aspect="auto", cmap="viridis", vmin=-0.2, vmax=1.0)
    ax.set_xticks(range(len(heads)), heads)
    ax.set_yticks(range(len(layers)), layers)
    ax.set_xlabel("Attention head")
    ax.set_ylabel("Layer")
    ax.set_title("Mean cross-input function-profile cosine")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    path = plot_dir / "cross_input_stability_map.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, axes = plt.subplots(
        len(CATEGORIES),
        1,
        figsize=(max(9, len(heads) * 0.55), 2.4 * len(CATEGORIES)),
        sharex=True,
    )
    for axis, category in zip(axes, CATEGORIES):
        grid = np.asarray(
            [
                [row_map[(layer, head)][f"z_{category}"] for head in heads]
                for layer in layers
            ],
            dtype=np.float64,
        )
        image = axis.imshow(grid, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
        axis.set_ylabel("Layer")
        axis.set_yticks(range(0, len(layers), max(1, len(layers) // 6)))
        axis.set_yticklabels(layers[:: max(1, len(layers) // 6)])
        axis.set_title(category, loc="left")
        fig.colorbar(image, ax=axis, fraction=0.02, pad=0.01)
    axes[-1].set_xticks(range(len(heads)), heads)
    axes[-1].set_xlabel("Attention head")
    fig.suptitle("Per-category head specialization z-scores", y=0.999)
    fig.tight_layout()
    path = plot_dir / "category_specialization_maps.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def write_report(
    output_dir: Path,
    profile_rows: Sequence[dict[str, Any]],
    ranking_rows: Sequence[dict[str, Any]],
    category_stability_rows: Sequence[dict[str, Any]],
    sample_count: int,
) -> None:
    primary_counts = Counter(row["primary_function"] for row in profile_rows)
    stability_counts = Counter(row["stability_class"] for row in profile_rows)
    lines = [
        "# Head function and stability report",
        "",
        f"Controlled inputs: {sample_count}; analyzed heads: {len(profile_rows)}.",
        "Scores are observational attention-pattern enrichments and do not by themselves prove a causal circuit role.",
        "",
        "## Primary pattern counts",
        "",
        "| pattern | heads |",
        "| --- | ---: |",
    ]
    for label, count in sorted(primary_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {label} | {count} |")
    lines.extend(
        [
            "",
            "## Stability classes",
            "",
            "| class | heads |",
            "| --- | ---: |",
        ]
    )
    for label, count in sorted(stability_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {label} | {count} |")
    lines.extend(
        [
            "",
            "## Top specialized heads",
            "",
            "| pattern | top heads (layer/head, z, mean log2 enrichment) |",
            "| --- | --- |",
        ]
    )
    for category in CATEGORIES:
        selected = [
            row
            for row in ranking_rows
            if row["category"] == category and int(row["head_rank"]) <= 5
        ]
        value = ", ".join(
            f"L{row['layer']}/H{row['head']} ({float(row['specialization_z']):.2f}, "
            f"{float(row['mean_log2_enrichment']):.2f})"
            for row in selected
        )
        lines.append(f"| {category} | {value} |")
    lines.extend(
        [
            "",
            "## Does the same head stay specialized across inputs?",
            "",
            "Spearman measures whether the ranking of all heads is preserved between input pairs. "
            "The paired column restricts the comparison to controlled paraphrases of the same relation.",
            "",
            "| pattern | inputs | all-pair Spearman | paired-paraphrase Spearman |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in category_stability_rows:
        all_pair = row["head_rank_spearman_mean"]
        paired = row["paired_paraphrase_spearman_mean"]
        lines.append(
            f"| {row['category']} | {row['n_samples']} | "
            f"{all_pair:.3f} | {paired:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- `primary_function` means strongest relative attention-pattern specialization, not the only thing the head can do.",
            "- `mixed_or_common` means no category exceeds the configured robust z threshold; it does not mean the head is useless.",
            "- High paired stability but lower all-input stability indicates a reusable bias that is modulated by task or domain.",
            "- A causal claim requires targeted head/link ablation and downstream loss or accuracy measurement.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = build_controlled_samples()
    if args.sample_limit > 0:
        samples = samples[: args.sample_limit]
    if not samples:
        raise ValueError("No controlled samples selected")

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)
    print(f"[load] tokenizer={args.model_name_or_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        use_fast=True,
    )
    print(f"[load] model={args.model_name_or_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()
    num_layers = int(model.config.num_hidden_layers)
    num_heads = int(model.config.num_attention_heads)
    selected_layers = parse_index_spec(args.layers, num_layers, "layer")
    selected_heads = parse_index_spec(args.heads, num_heads, "head")

    feature_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    started = time.time()
    for sample_index, sample in enumerate(samples, start=1):
        encoded = tokenizer(
            sample.text,
            add_special_tokens=False,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=args.max_seq_length,
        )
        offsets_tensor = encoded.pop("offset_mapping")[0]
        offsets = [(int(item[0]), int(item[1])) for item in offsets_tensor.tolist()]
        model_inputs = {key: value.to(device) for key, value in encoded.items()}
        probes = build_token_probes(
            sample,
            offsets,
            min_history=args.min_history,
            sink_tokens=args.sink_tokens,
            recent_window=args.recent_window,
            manual_query_tail=args.manual_query_tail,
            exclude_local_from_typed=args.exclude_local_from_typed,
        )
        with torch.inference_mode():
            outputs = model(
                **model_inputs,
                use_cache=False,
                output_attentions=True,
                return_dict=True,
            )
        if outputs.attentions is None or any(item is None for item in outputs.attentions):
            raise RuntimeError(
                "Model did not return attention weights. Use --attn_implementation eager "
                "with a Transformers version that supports Qwen3 output_attentions."
            )
        token_count = int(model_inputs["input_ids"].shape[1])
        applicable = sorted(category for category in CATEGORIES if probes.get(category))
        sample_rows.append(
            {
                "sample_id": sample.sample_id,
                "domain": sample.domain,
                "pair_id": sample.pair_id,
                "token_count": token_count,
                "applicable_categories": ";".join(applicable),
                "text": sample.text,
            }
        )
        for layer in selected_layers:
            attention = outputs.attentions[layer][0].detach().float().cpu()
            if attention.shape[0] != num_heads:
                raise RuntimeError(
                    f"Layer {layer} returned {attention.shape[0]} heads, config says {num_heads}"
                )
            for category in applicable:
                mean_mass, mean_score, availability, query_count = score_attention_category(
                    attention,
                    probes[category],
                    score_clip=args.score_clip,
                )
                for head in selected_heads:
                    feature_rows.append(
                        {
                            "sample_id": sample.sample_id,
                            "domain": sample.domain,
                            "pair_id": sample.pair_id,
                            "layer": layer,
                            "head": head,
                            "category": category,
                            "attention_mass": float(mean_mass[head]),
                            "key_availability": availability,
                            "log2_enrichment": float(mean_score[head]),
                            "query_count": query_count,
                            "token_count": token_count,
                        }
                    )
        del outputs
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"[sample {sample_index:02d}/{len(samples):02d}] {sample.sample_id} "
            f"tokens={token_count} categories={','.join(applicable)}",
            flush=True,
        )

    feature_fields = (
        "sample_id",
        "domain",
        "pair_id",
        "layer",
        "head",
        "category",
        "attention_mass",
        "key_availability",
        "log2_enrichment",
        "query_count",
        "token_count",
    )
    write_csv(output_dir / "per_sample_head_features.csv", feature_rows, feature_fields)
    write_csv(
        output_dir / "controlled_samples.csv",
        sample_rows,
        ("sample_id", "domain", "pair_id", "token_count", "applicable_categories", "text"),
    )

    profile_rows, ranking_rows, category_stability_rows, pair_rows = analyze_features(
        feature_rows,
        selected_layers=selected_layers,
        selected_heads=selected_heads,
        specialization_z_threshold=args.specialization_z_threshold,
        primary_min_log2_enrichment=args.primary_min_log2_enrichment,
    )
    profile_fields = (
        "layer",
        "head",
        "primary_function",
        "multi_label_functions",
        "primary_specialization_z",
        "specialization_margin",
        "profile_cosine_mean",
        "paired_paraphrase_cosine_mean",
        "primary_label_consistency",
        "domain_agreement",
        "stability_class",
        *[f"score_{category}" for category in CATEGORIES],
        *[f"z_{category}" for category in CATEGORIES],
    )
    write_csv(output_dir / "head_profiles.csv", profile_rows, profile_fields)
    write_csv(
        output_dir / "category_head_rankings.csv",
        ranking_rows,
        (
            "category",
            "head_rank",
            "layer",
            "head",
            "mean_log2_enrichment",
            "specialization_z",
            "sample_std",
            "positive_rate",
            "strong_rate",
            "n_samples",
        ),
    )
    write_csv(
        output_dir / "category_stability.csv",
        category_stability_rows,
        (
            "category",
            "n_samples",
            "n_sample_pairs",
            "head_rank_spearman_mean",
            "head_rank_spearman_median",
            "paired_paraphrase_spearman_mean",
            "n_paraphrase_pairs",
        ),
    )
    write_csv(
        output_dir / "paired_input_stability.csv",
        pair_rows,
        ("layer", "head", "pair_id", "sample_a", "sample_b", "profile_cosine"),
    )
    plot_paths = make_plots(output_dir, profile_rows) if args.make_plots else []
    write_report(
        output_dir,
        profile_rows,
        ranking_rows,
        category_stability_rows,
        len(samples),
    )

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "num_model_layers": num_layers,
        "num_model_attention_heads": num_heads,
        "selected_layers": selected_layers,
        "selected_heads": selected_heads,
        "sample_count": len(samples),
        "domain_counts": dict(Counter(sample.domain for sample in samples)),
        "categories": list(CATEGORIES),
        "feature_definition": "mean clipped log2(attention_mass / causal_key_availability)",
        "specialization_z_threshold": args.specialization_z_threshold,
        "primary_min_log2_enrichment": args.primary_min_log2_enrichment,
        "primary_function_counts": dict(Counter(row["primary_function"] for row in profile_rows)),
        "stability_class_counts": dict(Counter(row["stability_class"] for row in profile_rows)),
        "category_stability": [
            {key: safe_number(value) for key, value in row.items()}
            for row in category_stability_rows
        ],
        "runtime_seconds": time.time() - started,
        "plot_paths": plot_paths,
        "causal_claim": False,
        "interpretation": (
            "Labels are relative attention-pattern specializations. Targeted ablation is required "
            "before calling a head causally necessary for a function."
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"[done] output={output_dir} runtime={summary['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
