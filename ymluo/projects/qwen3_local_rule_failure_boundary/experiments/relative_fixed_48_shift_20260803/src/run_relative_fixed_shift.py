from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


HERE = Path(__file__).resolve()
SHARED_SRC = HERE.parents[3] / "src"
if str(SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(SHARED_SRC))

import run_attention_confidence_sweep_8b as attention_runner  # noqa: E402
import run_local_rule_failure_boundary as base  # noqa: E402


ATOMIC_ROLES = ("start_key", "hop1_result", "hop2_input", "hop2_result")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--model_name_or_path",
        default=(
            "/home/fdong/.cache/huggingface/hub/models--Qwen--Qwen3-8B/"
            "snapshots/b968826d9c46dd6066d109eabc6255188de91218"
        ),
    )
    parser.add_argument("--shift_tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefill_chunk_size", type=int, default=128)
    parser.add_argument("--max_top", type=int, default=100)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="none")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--original_max_position_embeddings", type=int, default=40960)
    parser.add_argument("--global_max_position", type=int, default=130000)
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def reconstruct_prompt_ids(original: dict[str, Any]) -> list[int]:
    rows = sorted(original["token_table"], key=lambda row: int(row[0]))
    positions = [int(row[0]) for row in rows]
    expected = list(range(int(original["prompt_tokens"])))
    if positions != expected:
        raise RuntimeError("original token table does not cover every prompt position")
    return [int(row[1]) for row in rows]


def select_insert_ids(tokenizer: Any, count: int, seed: int) -> list[int]:
    pool = base.build_filler_ids(tokenizer, count + 512, 1_700_000 + seed)
    for start in range(1, len(pool) - count + 1):
        first = tokenizer.decode([pool[start]], clean_up_tokenization_spaces=False)
        last = tokenizer.decode([pool[start + count - 1]], clean_up_tokenization_spaces=False)
        if first[:1].isspace() and "\n" in last:
            return [int(token_id) for token_id in pool[start : start + count]]
    raise RuntimeError("could not find a boundary-safe filler window")


def shift_spans(
    spans: dict[str, list[list[int]]], insert_at: int, amount: int
) -> dict[str, list[list[int]]]:
    shifted: dict[str, list[list[int]]] = {}
    for role, role_spans in spans.items():
        shifted[role] = []
        for start, end in role_spans:
            if int(start) < insert_at < int(end):
                raise RuntimeError(f"insertion splits role span {role}: {(start, end)}")
            delta = amount if int(start) >= insert_at else 0
            shifted[role].append([int(start) + delta, int(end) + delta])
    return shifted


def make_conditions(
    original: dict[str, Any], prompt_ids: list[int], insert_ids: list[int]
) -> list[dict[str, Any]]:
    body_tokens = int(original["body_tokens"])
    spans = {role: [[int(a), int(b)] for a, b in role_spans] for role, role_spans in original["spans"].items()}
    evidence_start = min(start for start, _ in spans["rule1_line"])
    amount = len(insert_ids)
    return [
        {
            "name": "original_47",
            "prompt_ids": prompt_ids,
            "body_tokens": body_tokens,
            "spans": spans,
            "insert_at": None,
        },
        {
            "name": "gap_plus_48",
            "prompt_ids": prompt_ids[:body_tokens] + insert_ids + prompt_ids[body_tokens:],
            "body_tokens": body_tokens + amount,
            "spans": spans,
            "insert_at": body_tokens,
        },
        {
            "name": "co_shift_plus_48",
            "prompt_ids": prompt_ids[:evidence_start] + insert_ids + prompt_ids[evidence_start:],
            "body_tokens": body_tokens + amount,
            "spans": shift_spans(spans, evidence_start, amount),
            "insert_at": evidence_start,
        },
    ]


def atomic_mass(attention: dict[str, Any], layer_start: int | None = None, layer_end: int | None = None) -> float:
    role_indices = [attention["role_order"].index(role) for role in ATOMIC_ROLES]
    if layer_start is None:
        return float(sum(attention["overall_role_mass"][index] for index in role_indices))
    values: list[float] = []
    stop = len(attention["head_role_mass"]) if layer_end is None else layer_end
    for layer in attention["head_role_mass"][layer_start:stop]:
        for head in layer:
            values.append(sum(float(head[index]) for index in role_indices))
    return float(sum(values) / len(values))


def role_qk_mean(attention: dict[str, Any], layer_start: int | None = None, layer_end: int | None = None) -> float:
    role_indices = [attention["role_order"].index(role) for role in ATOMIC_ROLES]
    start = 0 if layer_start is None else layer_start
    stop = len(attention["head_role_logit_mean"]) if layer_end is None else layer_end
    values: list[float] = []
    for layer in attention["head_role_logit_mean"][start:stop]:
        for head in layer:
            values.extend(float(head[index]) for index in role_indices)
    return float(sum(values) / len(values))


def condition_geometry(condition: dict[str, Any]) -> dict[str, Any]:
    query_position = len(condition["prompt_ids"]) - 1
    evidence_positions = {
        role: int(condition["spans"][role][0][0]) for role in ATOMIC_ROLES
    }
    return {
        "prompt_tokens": len(condition["prompt_ids"]),
        "query_position": query_position,
        "evidence_positions": evidence_positions,
        "query_minus_evidence": {
            role: query_position - position for role, position in evidence_positions.items()
        },
    }


def full_vocab_margin(answer: dict[str, Any]) -> dict[str, Any]:
    gold_id = int(answer["gold_token_scores"][0]["token_id"])
    gold_probability = float(answer["gold_token_scores"][0]["probability"])
    wrong = next(row for row in answer["next_token_top5"] if int(row["token_id"]) != gold_id)
    wrong_probability = float(wrong["probability"])
    return {
        "strongest_wrong_token_id": int(wrong["token_id"]),
        "strongest_wrong_token": wrong["token"],
        "strongest_wrong_probability": wrong_probability,
        "gold_vs_strongest_wrong_margin": math.log(gold_probability) - math.log(wrong_probability),
    }


def verify_geometry(rows: list[dict[str, Any]], shift: int) -> dict[str, Any]:
    by_name = {row["name"]: row["geometry"] for row in rows}
    original = by_name["original_47"]
    gap = by_name["gap_plus_48"]
    co_shift = by_name["co_shift_plus_48"]
    checks: dict[str, bool] = {
        "extended_prompt_lengths_equal": gap["prompt_tokens"] == co_shift["prompt_tokens"],
        "co_shift_query_plus_48": co_shift["query_position"] == original["query_position"] + shift,
    }
    for role in ATOMIC_ROLES:
        checks[f"gap_{role}_relative_distance_plus_48"] = (
            gap["query_minus_evidence"][role]
            == original["query_minus_evidence"][role] + shift
        )
        checks[f"co_shift_{role}_relative_distance_fixed"] = (
            co_shift["query_minus_evidence"][role]
            == original["query_minus_evidence"][role]
        )
        checks[f"co_shift_{role}_absolute_position_plus_48"] = (
            co_shift["evidence_positions"][role]
            == original["evidence_positions"][role] + shift
        )
    return {"all_passed": all(checks.values()), "checks": checks}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    original = json.loads(Path(args.original_json).read_text(encoding="utf-8"))

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    original_ids = reconstruct_prompt_ids(original)
    insert_ids = select_insert_ids(tokenizer, args.shift_tokens, args.seed)
    conditions = make_conditions(original, original_ids, insert_ids)

    load_args = SimpleNamespace(
        model_name_or_path=args.model_name_or_path,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        original_max_position_embeddings=args.original_max_position_embeddings,
    )
    rope_factor = base.rope_factor_for_length(
        args.global_max_position, args.original_max_position_embeddings
    )
    model, tokenizer = base.load_model_and_tokenizer(
        load_args, args.global_max_position, rope_factor
    )

    rows: list[dict[str, Any]] = []
    for condition in conditions:
        started = time.perf_counter()
        prompt = torch.tensor(condition["prompt_ids"], dtype=torch.long).view(1, -1)
        prompt_len_minus_one = int(prompt.shape[1]) - 1
        base_cache, prefill_seconds = base.prefill_sequence(
            model, prompt[:, :-1], args.prefill_chunk_size
        )
        query_cache = base.cache_from_legacy(base_cache)
        del base_cache
        query_output, captured, query_seconds = attention_runner.capture_query_states(
            model, query_cache, prompt[:, -1:], prompt_len_minus_one
        )
        spans_tuple = {
            role: [(int(start), int(end)) for start, end in role_spans]
            for role, role_spans in condition["spans"].items()
        }
        attention, needed_positions = attention_runner.summarize_attention(
            model, query_output, captured, spans_tuple, args.max_top
        )
        answer = attention_runner.score_gold_from_query_output(
            model,
            tokenizer,
            query_output,
            int(prompt.shape[1]),
            "basket",
            completion_text=" basket",
            candidate_texts=[" basket", " Let", " river", " window"],
        )
        answer.update(full_vocab_margin(answer))
        needed_positions.update(range(len(condition["prompt_ids"])))
        token_table = attention_runner.build_token_table(
            tokenizer,
            condition["prompt_ids"],
            needed_positions,
            int(condition["body_tokens"]),
            spans_tuple,
        )
        summary = {
            "gold_probability": float(answer["gold_token_scores"][0]["probability"]),
            "gold_ppl": float(answer["gold_ppl"]),
            "gold_vs_strongest_wrong_margin": float(answer["gold_vs_strongest_wrong_margin"]),
            "strongest_wrong_token": answer["strongest_wrong_token"],
            "strongest_wrong_probability": float(answer["strongest_wrong_probability"]),
            "atomic_evidence_mass_all": atomic_mass(attention),
            "atomic_evidence_mass_l30_l33": atomic_mass(attention, 30, 34),
            "atomic_evidence_qk_mean_all": role_qk_mean(attention),
            "atomic_evidence_qk_mean_l30_l33": role_qk_mean(attention, 30, 34),
        }
        row = {
            "name": condition["name"],
            "geometry": condition_geometry(condition),
            "insert_at": condition["insert_at"],
            "spans": condition["spans"],
            "summary": summary,
            "answer": answer,
            "attention": attention,
            "token_table_columns": ["position", "token_id", "text", "role"],
            "token_table": token_table,
            "timing": {
                "prefill_seconds": prefill_seconds,
                "query_seconds": query_seconds,
                "total_seconds": time.perf_counter() - started,
            },
        }
        rows.append(row)
        write_json(output_dir / "conditions" / f"{condition['name']}.json", row)
        print(
            f"{condition['name']}: P(basket)={summary['gold_probability']:.6f} "
            f"wrong={summary['strongest_wrong_token']!r}:{summary['strongest_wrong_probability']:.6f} "
            f"margin={summary['gold_vs_strongest_wrong_margin']:.6f}",
            flush=True,
        )
        del query_cache, query_output, captured, attention
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    geometry_check = verify_geometry(rows, args.shift_tokens)
    if not geometry_check["all_passed"]:
        raise RuntimeError(f"geometry invariants failed: {geometry_check}")
    result = {
        "schema_version": 1,
        "model": "Qwen3-8B",
        "source_original_json": str(args.original_json),
        "shift_tokens": args.shift_tokens,
        "insert_token_ids": insert_ids,
        "insert_text": tokenizer.decode(insert_ids, clean_up_tokenization_spaces=False),
        "geometry_check": geometry_check,
        "conditions": [
            {
                "name": row["name"],
                "geometry": row["geometry"],
                "insert_at": row["insert_at"],
                "summary": row["summary"],
                "timing": row["timing"],
            }
            for row in rows
        ],
    }
    write_json(output_dir / "results.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
