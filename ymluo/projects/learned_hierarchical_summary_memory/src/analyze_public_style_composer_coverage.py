from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_latent_scorer_raw_span_smoke import (  # noqa: E402
    candidate_remote_pages,
    expand_pages,
    page_recall,
    selected_indices_for_pages,
)
from run_real_qwen_seq_ae_search_trace import make_case, split_case_family  # noqa: E402
from run_recent_plus_kv_native_smoke import selected_text  # noqa: E402
from run_supervised_latent_page_ranker_smoke import (  # noqa: E402
    adaptive_halo_pages_for_case,
    adaptive_top_pages_for_case,
    compose_evidence_text,
)


@dataclass(frozen=True)
class SelectionConfig:
    page_tokens: int
    recent_tokens: int
    exclude_sink_pages: int
    exclude_recent_from_latent: bool


def parse_csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


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


def token_len(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def family_of(case_name: str) -> str:
    family, _ = split_case_family(case_name)
    return family


def aggregate(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(tuple(row[key] for key in keys), []).append(row)

    out: list[dict[str, Any]] = []
    for bucket_key, bucket_rows in sorted(buckets.items()):
        item = {key: value for key, value in zip(keys, bucket_key)}
        item.update(
            {
                "cases": len(bucket_rows),
                "mean_selected_pages": sum(row["selected_page_count"] for row in bucket_rows) / len(bucket_rows),
                "mean_expanded_pages": sum(row["expanded_page_count"] for row in bucket_rows) / len(bucket_rows),
                "mean_raw_tokens": sum(row["raw_token_count"] for row in bucket_rows) / len(bucket_rows),
                "mean_composed_tokens": sum(row["composed_token_count"] for row in bucket_rows) / len(bucket_rows),
                "mean_center_page_recall": sum(row["center_page_recall"] for row in bucket_rows) / len(bucket_rows),
                "mean_span_page_recall": sum(row["span_page_recall"] for row in bucket_rows) / len(bucket_rows),
                "raw_answer_coverage": sum(1.0 for row in bucket_rows if row["answer_in_raw"]) / len(bucket_rows),
                "composed_answer_coverage": sum(1.0 for row in bucket_rows if row["answer_in_composed"])
                / len(bucket_rows),
                "mean_compression_ratio_vs_context": sum(row["composed_token_count"] / row["context_tokens"] for row in bucket_rows)
                / len(bucket_rows),
                "mean_compression_ratio_vs_raw": sum(
                    row["composed_token_count"] / max(1, row["raw_token_count"]) for row in bucket_rows
                )
                / len(bucket_rows),
            }
        )
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze answer coverage of composed evidence from saved public-style page rankings."
    )
    parser.add_argument("--rank_output_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default="")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--rankers", type=parse_csv_tuple, default=("attention", "supervised"))
    parser.add_argument("--top_pages", type=parse_int_tuple, default=(1, 2))
    parser.add_argument("--include_adaptive", action="store_true")
    parser.add_argument("--composer_max_tokens", type=int, default=96)
    parser.add_argument("--composer_extra_halo_pages", type=int, default=1)
    args = parser.parse_args()

    rank_dir = Path(args.rank_output_dir)
    output_dir = Path(args.output_dir)
    summary = json.loads((rank_dir / "summary.json").read_text(encoding="utf-8"))
    rankings = json.loads((rank_dir / "page_rankings.json").read_text(encoding="utf-8"))
    config = summary["config"]
    model_name = args.model_name_or_path or config["model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=args.local_files_only)

    prompt_tokens = int(config["prompt_tokens"])
    page_tokens = int(config["page_tokens"])
    recent_tokens = int(config["recent_tokens"])
    page_halo_pages = int(config["page_halo_pages"])
    select_config = SelectionConfig(
        page_tokens=page_tokens,
        recent_tokens=recent_tokens,
        exclude_sink_pages=int(config["exclude_sink_pages"]),
        exclude_recent_from_latent=bool(config["exclude_recent_from_latent"]),
    )

    rows: list[dict[str, Any]] = []
    eval_cases = tuple(summary["eval_cases"])
    for case_name in eval_cases:
        case = make_case(tokenizer, case_name, prompt_tokens, page_tokens)
        for ranker in args.rankers:
            ranking = rankings[ranker][case_name]
            candidates = candidate_remote_pages(ranking["ranked_pages"], len(case.context_ids), select_config)
            policies: list[tuple[str, int, int]] = [
                (f"top{top}_halo{page_halo_pages}", top, page_halo_pages) for top in args.top_pages
            ]
            if args.include_adaptive:
                policies.append(
                    (
                        "adaptive_family_budget",
                        adaptive_top_pages_for_case(case_name),
                        adaptive_halo_pages_for_case(case_name),
                    )
                )
            for policy_name, top, halo in policies:
                selected = sorted(candidates[: min(top, len(candidates))])
                expanded = expand_pages(selected, len(case.context_ids), page_tokens, halo)
                raw_indices = selected_indices_for_pages(
                    selected,
                    len(case.context_ids),
                    page_tokens,
                    recent_tokens,
                    halo,
                )
                composer_indices = selected_indices_for_pages(
                    selected,
                    len(case.context_ids),
                    page_tokens,
                    recent_tokens,
                    halo + args.composer_extra_halo_pages,
                )
                raw = selected_text(tokenizer, case.context_ids, raw_indices)
                composed = compose_evidence_text(
                    tokenizer,
                    case.context_ids,
                    composer_indices,
                    case.query,
                    args.composer_max_tokens,
                )
                rows.append(
                    {
                        "case": case_name,
                        "family": family_of(case_name),
                        "ranker": ranker,
                        "policy": policy_name,
                        "top_pages": top,
                        "halo_pages": halo,
                        "answer": case.answer,
                        "selected_pages": json.dumps(selected),
                        "expanded_pages": json.dumps(expanded),
                        "evidence_pages": json.dumps(list(case.evidence_pages)),
                        "selected_page_count": len(selected),
                        "expanded_page_count": len(expanded),
                        "context_tokens": len(case.context_ids),
                        "raw_token_count": token_len(tokenizer, raw + "\n\n" + case.query),
                        "composed_token_count": token_len(tokenizer, composed + "\n\n" + case.query),
                        "center_page_recall": page_recall(selected, case.evidence_pages),
                        "span_page_recall": page_recall(expanded, case.evidence_pages),
                        "answer_in_raw": case.answer in raw,
                        "answer_in_composed": case.answer in composed,
                        "composed_preview": composed.replace("\n", " ")[:240],
                    }
                )

    method_summary = aggregate(rows, ("ranker", "policy"))
    family_summary = aggregate(rows, ("family", "ranker", "policy"))
    write_csv(output_dir / "composer_coverage_rows.csv", rows)
    write_csv(output_dir / "composer_coverage_summary.csv", method_summary)
    write_csv(output_dir / "composer_coverage_family_summary.csv", family_summary)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "rank_output_dir": str(rank_dir),
                "config": config,
                "eval_cases": eval_cases,
                "composer_max_tokens": args.composer_max_tokens,
                "composer_extra_halo_pages": args.composer_extra_halo_pages,
                "method_summary": method_summary,
                "family_summary": family_summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("ranker,policy,cases,span_recall,raw_answer,composed_answer,composed_tokens")
    for row in method_summary:
        print(
            f"{row['ranker']},{row['policy']},{row['cases']},"
            f"{row['mean_span_page_recall']:.4f},{row['raw_answer_coverage']:.4f},"
            f"{row['composed_answer_coverage']:.4f},{row['mean_composed_tokens']:.1f}"
        )
    print(f"wrote outputs to {output_dir}")


if __name__ == "__main__":
    main()
