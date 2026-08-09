from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any

import torch

import run_fixed300_age_distractor_qk_8b as age
import run_local_rule_failure_boundary as base


SUBJECT_NAMES = (
    "Alice",
    "Emma",
    "Grace",
    "Hannah",
    "Irene",
    "Julia",
    "Laura",
    "Mary",
    "Nora",
    "Olivia",
    "Paula",
    "Rose",
    "Sarah",
    "Tara",
    "Vera",
    "Wendy",
)

BROTHER_NAMES = (
    "Bob",
    "David",
    "Frank",
    "Henry",
    "Jack",
    "Liam",
    "Noah",
    "Peter",
    "Ryan",
    "Simon",
    "Thomas",
    "Victor",
    "Wyatt",
    "Aaron",
    "Caleb",
    "Ethan",
)

NUMBER_WORDS = age.NUMBER_WORDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Matched one-hop vs two-hop natural-language age QA at exactly "
            "50 tokenizer tokens with period-only filler."
        )
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--total-tokens", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--generation-max-new-tokens", type=int, default=12)
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--design-only", action="store_true")
    return parser.parse_args()


def rounded(value: float, digits: int = 10) -> float:
    return round(float(value), digits)


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return [
        int(token_id)
        for token_id in tokenizer(text, add_special_tokens=False).input_ids
    ]


def decode_token(tokenizer: Any, token_id: int) -> str:
    return tokenizer.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def build_pair(
    tokenizer: Any,
    answer_ids: dict[str, int],
    period_id: int,
    pair_index: int,
    total_tokens: int,
) -> list[dict[str, Any]]:
    subject = SUBJECT_NAMES[pair_index % len(SUBJECT_NAMES)]
    brother = BROTHER_NAMES[(pair_index * 5 + pair_index // len(SUBJECT_NAMES)) % len(BROTHER_NAMES)]
    gold_answer = NUMBER_WORDS[(pair_index * 7 + pair_index // len(NUMBER_WORDS)) % len(NUMBER_WORDS)]

    relation_text = (
        f"{subject} has exactly one older brother named {brother}.\n"
    )
    age_unit = "year" if gold_answer == "one" else "years"
    age_text = f"{brother} is {gold_answer} {age_unit} old.\n"
    direct_query = (
        f"\nQuestion: How old is {brother}? "
        "Reply with exactly one English number word and nothing else. Answer:"
    )
    composed_query = (
        f"\nQuestion: How old is {subject}'s only older brother? "
        "Reply with exactly one English number word and nothing else. Answer:"
    )

    relation_ids = token_ids(tokenizer, relation_text)
    evidence_ids = token_ids(tokenizer, age_text)
    direct_query_ids = token_ids(tokenizer, direct_query)
    composed_query_ids = token_ids(tokenizer, composed_query)
    local_age_span = age.local_word_span(tokenizer, age_text, gold_answer)
    if len(token_ids(tokenizer, f" {gold_answer}")) != 1:
        raise RuntimeError(f"answer is not one token: {gold_answer!r}")
    if answer_ids[gold_answer] != token_ids(tokenizer, f" {gold_answer}")[0]:
        raise AssertionError("answer vocabulary mismatch")

    cases: list[dict[str, Any]] = []
    for hop_count, relation_prefix, query_text, query_ids in (
        (1, [period_id] * len(relation_ids), direct_query, direct_query_ids),
        (2, relation_ids, composed_query, composed_query_ids),
    ):
        fixed_count = len(relation_prefix) + len(evidence_ids) + len(query_ids)
        filler_after_count = total_tokens - fixed_count
        if filler_after_count < 0:
            raise RuntimeError(
                f"pair {pair_index} hop {hop_count} needs {fixed_count} fixed "
                f"tokens, exceeding total={total_tokens}"
            )
        prompt_ids = (
            list(relation_prefix)
            + list(evidence_ids)
            + [period_id] * filler_after_count
            + list(query_ids)
        )
        if len(prompt_ids) != total_tokens:
            raise AssertionError("prompt length mismatch")
        evidence_start = len(relation_ids)
        evidence_span = (evidence_start, evidence_start + len(evidence_ids))
        gold_age_span = (
            evidence_start + local_age_span[0],
            evidence_start + local_age_span[1],
        )
        query_span = (total_tokens - len(query_ids), total_tokens)
        cases.append(
            {
                "case_id": f"pair_{pair_index:03d}_hop{hop_count}",
                "pair_index": pair_index,
                "hop_count": hop_count,
                "subject": subject,
                "brother": brother,
                "gold_answer": gold_answer,
                "gold_token_id": answer_ids[gold_answer],
                "total_tokens": total_tokens,
                "prompt_ids": prompt_ids,
                "prompt_text": tokenizer.decode(
                    prompt_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "relation_text": relation_text if hop_count == 2 else None,
                "relation_span": [0, len(relation_ids)],
                "relation_replaced_by_periods": hop_count == 1,
                "age_evidence_text": age_text,
                "age_evidence_span": list(evidence_span),
                "gold_age_span": list(gold_age_span),
                "query_text": query_text,
                "query_span": list(query_span),
                "prefix_period_count": len(relation_ids) if hop_count == 1 else 0,
                "filler_after_evidence_count": filler_after_count,
                "total_irrelevant_period_count": (
                    filler_after_count + (len(relation_ids) if hop_count == 1 else 0)
                ),
            }
        )
    if cases[0]["age_evidence_span"] != cases[1]["age_evidence_span"]:
        raise AssertionError("matched evidence positions diverged")
    if cases[0]["gold_age_span"] != cases[1]["gold_age_span"]:
        raise AssertionError("matched gold-token positions diverged")
    return cases


def build_dataset(
    tokenizer: Any,
    samples: int,
    total_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    answer_ids = age.validate_answer_vocabulary(tokenizer)
    period_id = age.one_token_id(tokenizer, ".", "period filler")
    cases: list[dict[str, Any]] = []
    for pair_index in range(samples):
        cases.extend(
            build_pair(
                tokenizer,
                answer_ids,
                period_id,
                pair_index,
                total_tokens,
            )
        )
    return cases, answer_ids, period_id


def normalize_generation(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_first_number_word(text: str) -> str | None:
    normalized = normalize_generation(text)
    pattern = r"\b(" + "|".join(NUMBER_WORDS) + r")\b"
    match = re.search(pattern, normalized)
    return match.group(1) if match else None


def load_model(args: argparse.Namespace) -> Any:
    from transformers import AutoModelForCausalLM

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": base.resolve_dtype(args.dtype),
    }
    if args.device_map.lower() != "none":
        kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        **kwargs,
    )
    if args.device_map.lower() == "none":
        model = model.to(args.device if torch.cuda.is_available() else "cpu")
    model.eval()
    model.config.use_cache = True
    return model


@torch.inference_mode()
def evaluate_cases(
    model: Any,
    tokenizer: Any,
    cases: list[dict[str, Any]],
    answer_ids: dict[str, int],
    batch_size: int,
    generation_max_new_tokens: int,
) -> list[dict[str, Any]]:
    device = base.input_device(model)
    candidate_words = list(NUMBER_WORDS)
    candidate_ids = torch.tensor(
        [answer_ids[word] for word in candidate_words],
        dtype=torch.long,
        device=device,
    )
    semantic_variant_ids: dict[str, list[int]] = {}
    for word in candidate_words:
        variants: list[int] = []
        for surface in (word, word.title()):
            ids = token_ids(tokenizer, f" {surface}")
            if len(ids) != 1:
                raise RuntimeError(
                    f"answer variant is not one token: {surface!r} -> {ids}"
                )
            if ids[0] not in variants:
                variants.append(ids[0])
        semantic_variant_ids[word] = variants
    records: list[dict[str, Any]] = []
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        input_ids = torch.tensor(
            [case["prompt_ids"] for case in batch],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.ones_like(input_ids)
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
        logits = output.logits[:, -1].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        probabilities = torch.softmax(logits, dim=-1)
        top_ids = torch.argmax(logits, dim=-1)
        candidate_logits = logits.index_select(-1, candidate_ids)
        candidate_predictions = torch.argmax(candidate_logits, dim=-1)

        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=generation_max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        new_tokens = generated[:, input_ids.shape[1] :]

        for index, case in enumerate(batch):
            gold_word = str(case["gold_answer"])
            gold_id = int(answer_ids[gold_word])
            gold_log_prob = float(log_probs[index, gold_id].item())
            semantic_gold_probability = float(
                probabilities[
                    index,
                    torch.tensor(
                        semantic_variant_ids[gold_word],
                        dtype=torch.long,
                        device=device,
                    ),
                ]
                .sum()
                .item()
            )
            top_id = int(top_ids[index].item())
            predicted_candidate = candidate_words[
                int(candidate_predictions[index].item())
            ]
            wrong_candidate_values = [
                float(candidate_logits[index, candidate_index].item())
                for candidate_index, word in enumerate(candidate_words)
                if word != gold_word
            ]
            gold_candidate_index = candidate_words.index(gold_word)
            gold_candidate_logit = float(
                candidate_logits[index, gold_candidate_index].item()
            )
            generated_text = tokenizer.decode(
                new_tokens[index],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            generated_normalized = normalize_generation(generated_text)
            generated_answer = extract_first_number_word(generated_text)
            record = {
                key: value
                for key, value in case.items()
                if key != "prompt_ids"
            }
            record.update(
                {
                    "gold_probability": rounded(
                        probabilities[index, gold_id].item()
                    ),
                    "gold_nll": rounded(-gold_log_prob),
                    "gold_ppl": rounded(math.exp(-gold_log_prob)),
                    "semantic_gold_probability": rounded(
                        semantic_gold_probability
                    ),
                    "semantic_gold_nll": rounded(
                        -math.log(semantic_gold_probability)
                    ),
                    "semantic_gold_ppl": rounded(
                        1.0 / semantic_gold_probability
                    ),
                    "top_token_id": top_id,
                    "top_token": decode_token(tokenizer, top_id),
                    "next_token_correct": top_id == gold_id,
                    "semantic_next_token_correct": (
                        top_id in semantic_variant_ids[gold_word]
                    ),
                    "candidate_prediction": predicted_candidate,
                    "candidate_correct": predicted_candidate == gold_word,
                    "candidate_margin": rounded(
                        gold_candidate_logit - max(wrong_candidate_values)
                    ),
                    "generated_text": generated_text,
                    "generated_normalized": generated_normalized,
                    "generation_answer": generated_answer,
                    "generation_answer_correct": generated_answer == gold_word,
                    "generation_strict_correct": generated_normalized == gold_word,
                }
            )
            records.append(record)
        del output, generated, new_tokens
    return records


def metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    nlls = [float(row["gold_nll"]) for row in rows]
    ppls = [float(row["gold_ppl"]) for row in rows]
    semantic_nlls = [float(row["semantic_gold_nll"]) for row in rows]
    semantic_ppls = [float(row["semantic_gold_ppl"]) for row in rows]
    margins = [float(row["candidate_margin"]) for row in rows]
    return {
        "cases": len(rows),
        "candidate_accuracy": rounded(
            statistics.fmean(bool(row["candidate_correct"]) for row in rows)
        ),
        "next_token_accuracy": rounded(
            statistics.fmean(bool(row["next_token_correct"]) for row in rows)
        ),
        "semantic_next_token_accuracy": rounded(
            statistics.fmean(
                bool(row["semantic_next_token_correct"]) for row in rows
            )
        ),
        "generation_answer_accuracy": rounded(
            statistics.fmean(
                bool(row["generation_answer_correct"]) for row in rows
            )
        ),
        "generation_strict_accuracy": rounded(
            statistics.fmean(
                bool(row["generation_strict_correct"]) for row in rows
            )
        ),
        "geometric_gold_ppl": rounded(math.exp(statistics.fmean(nlls))),
        "mean_gold_ppl": rounded(statistics.fmean(ppls)),
        "median_gold_ppl": rounded(statistics.median(ppls)),
        "geometric_semantic_gold_ppl": rounded(
            math.exp(statistics.fmean(semantic_nlls))
        ),
        "mean_semantic_gold_ppl": rounded(statistics.fmean(semantic_ppls)),
        "median_semantic_gold_ppl": rounded(
            statistics.median(semantic_ppls)
        ),
        "mean_candidate_margin": rounded(statistics.fmean(margins)),
        "median_candidate_margin": rounded(statistics.median(margins)),
    }


def paired_summary(records: list[dict[str, Any]], metric: str) -> dict[str, int]:
    grouped: dict[int, dict[int, dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["pair_index"]), {})[
            int(record["hop_count"])
        ] = record
    counts = {
        "both_correct": 0,
        "one_hop_only": 0,
        "two_hop_only": 0,
        "both_wrong": 0,
    }
    for pair in grouped.values():
        one = bool(pair[1][metric])
        two = bool(pair[2][metric])
        if one and two:
            counts["both_correct"] += 1
        elif one:
            counts["one_hop_only"] += 1
        elif two:
            counts["two_hop_only"] += 1
        else:
            counts["both_wrong"] += 1
    return counts


def write_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "pair_index",
        "hop_count",
        "subject",
        "brother",
        "gold_answer",
        "total_tokens",
        "age_evidence_span",
        "gold_age_span",
        "query_span",
        "prefix_period_count",
        "filler_after_evidence_count",
        "total_irrelevant_period_count",
        "gold_probability",
        "gold_nll",
        "gold_ppl",
        "semantic_gold_probability",
        "semantic_gold_nll",
        "semantic_gold_ppl",
        "top_token",
        "next_token_correct",
        "semantic_next_token_correct",
        "candidate_prediction",
        "candidate_correct",
        "candidate_margin",
        "generation_answer",
        "generation_answer_correct",
        "generation_strict_correct",
        "generated_normalized",
        "prompt_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    field: (
                        json.dumps(record[field], ensure_ascii=False)
                        if isinstance(record.get(field), (list, dict))
                        else record.get(field)
                    )
                    for field in fields
                }
            )


def percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def build_report(result: dict[str, Any]) -> str:
    one = result["summary"]["one_hop"]
    two = result["summary"]["two_hop"]
    lines = [
        "# 固定 50-token：单跳与两跳年龄问答",
        "",
        "## 实验控制",
        "",
        "- 模型：Qwen3-8B。",
        f"- 配对样例：{result['pair_count']} 对，共 {result['case_count']} 个 prompt。",
        "- 每个 prompt 严格为 50 tokenizer tokens。",
        "- 年龄证据及答案 token 在配对样例中的位置完全一致。",
        "- 一跳条件用等长句号替换关系证据；其余 filler 也全部为单-token句号。",
        "- 两跳条件需要先从人物关系找到哥哥，再从哥哥的年龄证据得到答案。",
        "",
        "## 准确率结果",
        "",
        "| 条件 | 候选准确率 | 语义首 token 准确率 | 抽取式生成准确率 | 小写精确首 token | 语义 Gold PPL |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| 一跳 | {percent(one['candidate_accuracy'])} | "
            f"{percent(one['semantic_next_token_accuracy'])} | "
            f"{percent(one['generation_answer_accuracy'])} | "
            f"{percent(one['next_token_accuracy'])} | "
            f"{one['geometric_semantic_gold_ppl']:.4f} |"
        ),
        (
            f"| 两跳 | {percent(two['candidate_accuracy'])} | "
            f"{percent(two['semantic_next_token_accuracy'])} | "
            f"{percent(two['generation_answer_accuracy'])} | "
            f"{percent(two['next_token_accuracy'])} | "
            f"{two['geometric_semantic_gold_ppl']:.4f} |"
        ),
        "",
        "指标定义：",
        "",
        "- 候选准确率：只在 one–ten 十个合法年龄词中选择。",
        "- 语义首 token 准确率：全词表最高 logit 是正确年龄，忽略首字母大小写。",
        "- 抽取式生成准确率：greedy 输出中抽取到的第一个年龄词正确。",
        "- 小写精确首 token：全词表最高 logit 必须是小写年龄 token。",
        "- 语义 Gold PPL：将正确答案的小写和首字母大写概率相加后计算，避免把大小写偏好误判为推理错误。",
        "",
        "## 配对统计",
        "",
        "```json",
        json.dumps(result["paired"], ensure_ascii=False, indent=2),
        "```",
        "",
        "## 样例",
        "",
    ]
    for example in result["examples"]:
        lines.extend(
            [
                f"### {example['case_id']}",
                "",
                "```text",
                example["prompt_text"],
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("samples must be positive")
    if args.total_tokens <= 0:
        raise ValueError("total-tokens must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    cases, answer_ids, period_id = build_dataset(
        tokenizer,
        args.samples,
        args.total_tokens,
    )
    design = {
        "schema_version": 1,
        "experiment": "fixed50_onehop_twohop_age",
        "model_name_or_path": args.model_name_or_path,
        "pair_count": args.samples,
        "case_count": len(cases),
        "total_tokens": args.total_tokens,
        "filler_token_id": period_id,
        "filler_token_text": decode_token(tokenizer, period_id),
        "answer_token_ids": answer_ids,
        "paired_controls": {
            "same_total_tokens": True,
            "same_age_evidence_position": True,
            "same_gold_age_token_position": True,
            "same_final_answer_position": True,
            "only_period_filler": True,
            "one_hop_relation_replaced_by_equal_period_count": True,
        },
        "cases": [
            {
                key: value
                for key, value in case.items()
                if key != "prompt_ids"
            }
            for case in cases
        ],
    }
    write_json(output_dir / "design.json", design)
    if args.design_only:
        print(json.dumps(design, ensure_ascii=False))
        return

    started = time.perf_counter()
    model = load_model(args)
    records = evaluate_cases(
        model,
        tokenizer,
        cases,
        answer_ids,
        args.batch_size,
        args.generation_max_new_tokens,
    )
    one_rows = [row for row in records if int(row["hop_count"]) == 1]
    two_rows = [row for row in records if int(row["hop_count"]) == 2]
    result = {
        "schema_version": 1,
        "experiment": "fixed50_onehop_twohop_age",
        "model_name_or_path": args.model_name_or_path,
        "pair_count": args.samples,
        "case_count": len(records),
        "total_tokens": args.total_tokens,
        "elapsed_seconds": rounded(time.perf_counter() - started),
        "summary": {
            "one_hop": metric_summary(one_rows),
            "two_hop": metric_summary(two_rows),
        },
        "paired": {
            metric: paired_summary(records, metric)
            for metric in (
                "candidate_correct",
                "next_token_correct",
                "semantic_next_token_correct",
                "generation_answer_correct",
                "generation_strict_correct",
            )
        },
        "examples": [
            next(row for row in records if row["pair_index"] == 0 and row["hop_count"] == hop)
            for hop in (1, 2)
        ],
    }
    write_json(output_dir / "result.json", result)
    write_records_csv(output_dir / "records.csv", records)
    (output_dir / "report.md").write_text(build_report(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
