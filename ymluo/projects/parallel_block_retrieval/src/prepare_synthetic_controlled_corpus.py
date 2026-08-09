from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer


TASKS = ["lexical", "semantic_paraphrase", "hard_negative", "multihop"]
COLORS = [
    "amber",
    "azure",
    "bronze",
    "cobalt",
    "coral",
    "crimson",
    "emerald",
    "golden",
    "indigo",
    "ivory",
    "jade",
    "lavender",
    "maroon",
    "ochre",
    "pearl",
    "scarlet",
    "silver",
    "teal",
    "umber",
    "violet",
]
TOKENS = [
    "anchor",
    "badger",
    "cedar",
    "dolphin",
    "falcon",
    "garden",
    "harbor",
    "island",
    "juniper",
    "lantern",
    "meadow",
    "nebula",
    "orchid",
    "pioneer",
    "quartz",
    "raven",
    "summit",
    "thistle",
    "voyager",
    "willow",
    "zephyr",
    "citadel",
    "compass",
    "meridian",
    "tempest",
]
SYLLABLES = [
    "avel",
    "bren",
    "cora",
    "dorin",
    "elva",
    "faren",
    "gala",
    "hestor",
    "iona",
    "jorin",
    "kela",
    "luma",
    "maren",
    "nora",
    "orin",
    "pela",
    "quorin",
    "rhea",
    "sela",
    "torin",
    "ulma",
    "vesper",
    "wren",
    "xara",
    "yoren",
    "zella",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a controlled semi-synthetic block-retrieval corpus with explicit evidence."
    )
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seq_tokens", type=int, default=100_000)
    parser.add_argument("--block_tokens", type=int, default=256)
    parser.add_argument("--num_queries", type=int, default=500)
    parser.add_argument("--num_records", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument(
        "--split_disjoint_templates",
        action="store_true",
        help="Use phrasing families that are disjoint across train/dev/test.",
    )
    parser.add_argument(
        "--heldout_template_set",
        choices=["v2", "v3", "v4", "v5", "v6"],
        default="v2",
        help="Choose the test-only phrasing family when split-disjoint templates are enabled.",
    )
    return parser.parse_args()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def entity_name(index: int) -> str:
    first = SYLLABLES[index % len(SYLLABLES)].title()
    second = SYLLABLES[(index * 7 + 11) % len(SYLLABLES)].title()
    suffix = TOKENS[(index * 13 + 3) % len(TOKENS)].title()
    return f"{first}{second} {suffix}"


def person_name(index: int) -> str:
    first = SYLLABLES[(index * 5 + 2) % len(SYLLABLES)].title()
    second = SYLLABLES[(index * 11 + 7) % len(SYLLABLES)].title()
    family = TOKENS[(index // len(SYLLABLES)) % len(TOKENS)].title()
    return f"{first} {second} {family}"


def value_phrase(index: int) -> str:
    return f"{COLORS[index % len(COLORS)]} {TOKENS[(index // len(COLORS)) % len(TOKENS)]}"


def decoy_value(index: int) -> str:
    return f"provisional {COLORS[(index * 7 + 3) % len(COLORS)]}"


def alias_code(index: int) -> str:
    return f"{TOKENS[(index * 3 + 5) % len(TOKENS)].title()}-{index:04d}"


def split_for_query(query_id: int) -> str:
    remainder = query_id % 5
    if remainder == 0:
        return "test"
    if remainder == 1:
        return "dev"
    return "train"


def filler_sentence(record_id: int, block_id: int, sentence_id: int) -> str:
    subjects = [
        "The archive committee",
        "A regional logistics team",
        "The maintenance office",
        "An independent review panel",
        "The coastal planning group",
        "A rotating documentation crew",
        "The historical records unit",
        "A technical standards board",
    ]
    verbs = [
        "catalogued routine correspondence about",
        "summarized procedural notes concerning",
        "revised a neutral memorandum about",
        "stored background observations regarding",
        "checked ordinary schedules associated with",
        "compiled nonbinding comments about",
        "reviewed administrative material concerning",
        "indexed a general report about",
    ]
    objects = [
        "seasonal transport planning",
        "building ventilation inspections",
        "public meeting timetables",
        "equipment storage practices",
        "staff training calendars",
        "routine shoreline measurements",
        "library preservation procedures",
        "municipal garden maintenance",
        "warehouse lighting surveys",
        "regional weather summaries",
    ]
    index = record_id * 10_000 + block_id * 97 + sentence_id * 17
    subject = subjects[index % len(subjects)]
    verb = verbs[(index // 3) % len(verbs)]
    obj = objects[(index // 7) % len(objects)]
    marker = f"memo {record_id + 1}-{block_id + 1}-{sentence_id + 1}"
    return f"{subject} {verb} {obj} under {marker}; the note carried no operational decision."


def choose_block(
    rng: random.Random,
    block_loads: list[int],
    forbidden: set[int],
) -> int:
    candidates = [index for index in range(len(block_loads)) if index not in forbidden]
    minimum = min(block_loads[index] for index in candidates)
    light = [index for index in candidates if block_loads[index] <= minimum + 80]
    return rng.choice(light)


def lexical_terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def make_task_payload(query_id: int, task: str) -> dict[str, Any]:
    entity = entity_name(query_id)
    answer = value_phrase(query_id)
    variant = query_id % 3
    if task == "lexical":
        evidence_templates = [
            "The registry states that the clearance code for {entity} is {answer}.",
            "According to the final ledger, {entity} has clearance code {answer}.",
            "The clearance code assigned to {entity} is recorded as {answer}.",
        ]
        question_templates = [
            "What is the clearance code for {entity}?",
            "Which clearance code is assigned to {entity}?",
            "State the recorded clearance code of {entity}.",
        ]
        evidence = [evidence_templates[variant].format(entity=entity, answer=answer)]
        decoys = [
            f"A scheduling note mentions {entity}, but it contains no clearance code or authorization value."
        ]
    elif task == "semantic_paraphrase":
        answer = person_name(query_id + 1_000)
        evidence_templates = [
            "Operational authority over {entity} was entrusted to {answer} after the annual review.",
            "The council placed {entity} under the stewardship of {answer} for the coming cycle.",
            "Responsibility for directing {entity} rests with {answer}, as confirmed by the board.",
        ]
        question_templates = [
            "Who is responsible for supervising {entity}?",
            "Who oversees the operations of {entity}?",
            "Name the person charged with managing {entity}.",
        ]
        evidence = [evidence_templates[variant].format(entity=entity, answer=answer)]
        decoys = [
            f"A review asked who oversees {entity}, but the draft intentionally omitted the supervisor's name.",
            f"The inspection calendar for {entity} discusses management procedures without identifying the responsible person.",
        ]
    elif task == "hard_negative":
        evidence_templates = [
            "Following ratification, {entity} received authorization at the {answer} tier.",
            "The binding decision placed {entity} in the {answer} authorization category.",
            "Final approval established {answer} as the operative tier for {entity}.",
        ]
        question_templates = [
            "What clearance tier is assigned to {entity}?",
            "Which authorization tier currently applies to {entity}?",
            "Identify the final clearance category for {entity}.",
        ]
        evidence = [evidence_templates[variant].format(entity=entity, answer=answer)]
        decoys = [
            (
                f"A rejected draft claimed that the clearance tier for {entity} should be "
                f"{decoy_value(query_id * 5 + item)}, but that proposal never took effect."
            )
            for item in range(4)
        ]
    elif task == "multihop":
        code = alias_code(query_id)
        evidence_templates = [
            (
                "Dispatch records identify {code} as the operational alias used for {entity}.",
                "The navigation beacon mounted on {entity} emits the color sequence {answer}.",
            ),
            (
                "Within the routing directory, {code} refers specifically to the vessel {entity}.",
                "Engineers confirmed that {entity}'s navigation beacon displays {answer}.",
            ),
            (
                "The call sign {code} is assigned to {entity} in the current dispatch ledger.",
                "A calibrated beacon aboard {entity} produces the signal color {answer}.",
            ),
        ]
        question_templates = [
            "What color is emitted by the navigation beacon on the vessel known as {code}?",
            "Which beacon color belongs to the craft using call sign {code}?",
            "Give the navigation signal color of the vessel identified by {code}.",
        ]
        first, second = evidence_templates[variant]
        evidence = [
            first.format(code=code, entity=entity),
            second.format(entity=entity, answer=answer),
        ]
        decoys = [
            f"The dispatch alias {code} appears in a training example that does not describe any beacon color.",
            f"A maintenance request for {entity} concerns the beacon housing but omits its emitted color.",
        ]
    else:
        raise ValueError(f"unknown task: {task}")
    question = question_templates[variant].format(entity=entity, code=alias_code(query_id))
    return {
        "entity": entity,
        "answer": answer,
        "question": question,
        "evidence": evidence,
        "decoys": decoys,
    }


def make_split_disjoint_payload(query_id: int, task: str, split: str) -> dict[str, Any]:
    if split == "train":
        return make_task_payload(query_id, task)
    entity = entity_name(query_id)
    answer = value_phrase(query_id)
    code = alias_code(query_id)

    if task == "lexical" and split == "dev":
        evidence = [f"A ratified access entry lists {entity} with credential phrase {answer}."]
        question = f"Retrieve the credential phrase recorded for {entity}."
        decoys = [
            f"An obsolete worksheet names {entity}; auditors voided it before any credential was issued."
        ]
    elif task == "lexical" and split == "test":
        evidence = [f"For {entity}, the approved access pass reads {answer}."]
        question = f"What wording appears on the approved access pass for {entity}?"
        decoys = [
            f"A preliminary form for {entity} was withdrawn and therefore supplies no usable pass wording."
        ]
    elif task == "semantic_paraphrase" and split == "dev":
        answer = person_name(query_id + 1_000)
        evidence = [f"Board minutes confirm that {answer} holds executive custody of {entity}."]
        question = f"Under whose direction does {entity} operate?"
        decoys = [
            f"A discussion paper about leadership at {entity} remained unresolved and named nobody."
        ]
    elif task == "semantic_paraphrase" and split == "test":
        answer = person_name(query_id + 1_000)
        evidence = [
            f"The final governance notice names {answer} as the accountable operator for {entity}."
        ]
        question = f"Who has operational accountability for {entity}?"
        decoys = [
            f"An unsigned governance outline concerning {entity} left the operator field blank."
        ]
    elif task == "hard_negative" and split == "dev":
        evidence = [
            f"After all appeals closed, the governing authorization for {entity} became {answer}."
        ]
        question = f"Which authorization level for {entity} survived the appeal process?"
        decoys = [
            (
                f"A superseded recommendation listed {decoy_value(query_id * 5 + item)} "
                f"for {entity}; the later resolution annulled that recommendation."
            )
            for item in range(4)
        ]
    elif task == "hard_negative" and split == "test":
        evidence = [f"The enforceable access class now attached to {entity} is {answer}."]
        question = f"What is the enforceable access class for {entity}?"
        decoys = [
            (
                f"An earlier consultation floated {decoy_value(query_id * 5 + item)} for "
                f"{entity}, but it expired without approval."
            )
            for item in range(4)
        ]
    elif task == "multihop" and split == "dev":
        evidence = [
            f"The maritime roster cross-references {code} to {entity}.",
            f"On {entity}, the locator lamp shines {answer}.",
        ]
        question = f"What hue does the locator lamp on the craft catalogued as {code} display?"
        decoys = [
            f"A mock drill printed {code} on a card with no real craft or lamp details.",
            f"A decommissioned repair ticket for {entity} discussed a lamp bracket, not its hue.",
        ]
    elif task == "multihop" and split == "test":
        evidence = [
            f"Harbor control uses {code} when communicating with {entity}.",
            f"The active wayfinding light carried by {entity} shows {answer}.",
        ]
        question = f"Report the active wayfinding light shown by the craft addressed as {code}."
        decoys = [
            f"The retired label {code} occurs in a simulation whose light field is empty.",
            f"A canceled equipment order for {entity} mentioned a light but specified no display.",
        ]
    else:
        raise ValueError(f"unsupported split-disjoint task: {task}/{split}")
    return {
        "entity": entity,
        "answer": answer,
        "question": question,
        "evidence": evidence,
        "decoys": decoys,
    }


def make_challenge_test_payload(query_id: int, task: str) -> dict[str, Any]:
    entity = entity_name(query_id)
    answer = value_phrase(query_id)
    code = alias_code(query_id)
    if task == "lexical":
        evidence = [f"The operative entry on {entity}'s authorization badge is {answer}."]
        question = f"Read the operative badge entry associated with {entity}."
        decoys = [
            f"A discarded badge mock-up for {entity} lacks any approved operative entry."
        ]
    elif task == "semantic_paraphrase":
        answer = person_name(query_id + 1_000)
        evidence = [
            f"{answer} is identified in the signed mandate as the party answerable for {entity}."
        ]
        question = f"To whom is {entity} ultimately answerable?"
        decoys = [
            f"A circulated but unexecuted mandate for {entity} left accountability undecided."
        ]
    elif task == "hard_negative":
        evidence = [f"Only {answer} carries legal force as {entity}'s permission grade."]
        question = f"Which permission grade for {entity} has legal force?"
        decoys = [
            (
                f"Meeting minutes considered {decoy_value(query_id * 5 + item)} for {entity}; "
                "the motion lapsed and acquired no force."
            )
            for item in range(4)
        ]
    elif task == "multihop":
        evidence = [
            f"Traffic coordinators address {entity} with radio designator {code}.",
            f"The guidance flare on {entity} currently burns {answer}.",
        ]
        question = f"What does the guidance flare display on the unit reached via {code}?"
        decoys = [
            f"An archived exercise uses {code} as a fictional label with no deployed unit.",
            f"An expired requisition for {entity} mentions a flare casing without its display.",
        ]
    else:
        raise ValueError(f"unknown challenge task: {task}")
    return {
        "entity": entity,
        "answer": answer,
        "question": question,
        "evidence": evidence,
        "decoys": decoys,
    }


def make_blind_test_payload(query_id: int, task: str) -> dict[str, Any]:
    entity = entity_name(query_id)
    answer = value_phrase(query_id)
    code = alias_code(query_id)
    if task == "lexical":
        evidence = [f"The authorization badge issued to {entity} bears the text {answer}."]
        question = f"What text is borne by the issued badge for {entity}?"
        decoys = [
            f"An illustrative design study for {entity} never became an authorized badge record."
        ]
    elif task == "semantic_paraphrase":
        answer = person_name(query_id + 1_000)
        evidence = [f"A sealed directive makes {answer} answerable for the conduct of {entity}."]
        question = f"Who must account for how {entity} is conducted?"
        decoys = [
            f"A memorandum contemplated oversight for {entity} but remained without signatures or effect."
        ]
    elif task == "hard_negative":
        evidence = [f"The permission grade in force for {entity} is {answer}."]
        question = f"Which permission grade is in force for {entity}?"
        decoys = [
            (
                f"A committee discussed {decoy_value(query_id * 5 + item)} for {entity} in a "
                "scenario that ceased before ratification."
            )
            for item in range(4)
        ]
    elif task == "multihop":
        evidence = [
            f"Control-room operators use {code} to reach {entity}.",
            f"The orientation lamp fitted to {entity} radiates {answer}.",
        ]
        question = f"What does the orientation lamp radiate on the unit reached as {code}?"
        decoys = [
            f"A tabletop drill borrowed {code} for an imaginary unit.",
            f"A procurement sketch for {entity} references a lamp but withholds its emitted hue.",
        ]
    else:
        raise ValueError(f"unknown blind task: {task}")
    return {
        "entity": entity,
        "answer": answer,
        "question": question,
        "evidence": evidence,
        "decoys": decoys,
    }


def make_blind_test_payload_v5(query_id: int, task: str) -> dict[str, Any]:
    if task != "multihop":
        return make_blind_test_payload(query_id, task)
    entity = entity_name(query_id)
    answer = value_phrase(query_id)
    code = alias_code(query_id)
    return {
        "entity": entity,
        "answer": answer,
        "question": (
            f"State the display of the signal marker for the unit paired with {code}."
        ),
        "evidence": [
            f"The communications ledger pairs {code} with the unit {entity}.",
            f"The signal marker assigned to {entity} displays {answer}.",
        ],
        "decoys": [
            (
                f"A planning memo uses {code} as a placeholder; no operational unit was "
                "ever linked to it."
            ),
            (
                f"A design note for {entity} discusses a marker without recording any "
                "approved display."
            ),
        ],
    }


def make_blind_test_payload_v6(query_id: int, task: str) -> dict[str, Any]:
    if task != "multihop":
        return make_blind_test_payload(query_id, task)
    entity = entity_name(query_id)
    answer = value_phrase(query_id)
    code = alias_code(query_id)
    return {
        "entity": entity,
        "answer": answer,
        "question": (
            f"Which pattern appears on the compliance pennant of the asset hidden "
            f"behind routing token {code}?"
        ),
        "evidence": [
            (
                f"The fleet allocation sheet lists {entity} as the physical asset "
                f"behind routing token {code}."
            ),
            f"A compliance pennant attached to {entity} carries the pattern {answer}.",
        ],
        "decoys": [
            (
                f"A hypothetical exercise printed routing token {code}, but the token "
                "was not assigned to any physical asset."
            ),
            (
                f"A retired pennant order for {entity} contains no approved pattern "
                "and has no operational effect."
            ),
        ],
    }


def main() -> None:
    args = parse_args()
    if args.seq_tokens < args.block_tokens or args.num_records <= 0:
        raise ValueError("invalid corpus size")
    if args.num_queries < len(TASKS) or args.num_queries % len(TASKS) != 0:
        raise ValueError(f"num_queries must be divisible by {len(TASKS)}")
    block_count = args.seq_tokens // args.block_tokens
    if block_count < args.num_records * 40:
        raise ValueError("each synthetic record must contain at least 40 blocks")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    rng = random.Random(args.seed)

    base_blocks = block_count // args.num_records
    remainder = block_count % args.num_records
    record_block_counts = [
        base_blocks + int(record_id < remainder) for record_id in range(args.num_records)
    ]
    snippets: list[list[list[dict[str, Any]]]] = [
        [[] for _ in range(count)] for count in record_block_counts
    ]
    block_loads: list[list[int]] = [[0 for _ in range(count)] for count in record_block_counts]
    provisional_queries: list[dict[str, Any]] = []
    queries_per_task = args.num_queries // len(TASKS)

    for query_id in range(args.num_queries):
        task = TASKS[query_id // queries_per_task]
        record_id = query_id % args.num_records
        split = split_for_query(query_id)
        if args.split_disjoint_templates and split == "test" and args.heldout_template_set == "v6":
            payload = make_blind_test_payload_v6(query_id, task)
        elif args.split_disjoint_templates and split == "test" and args.heldout_template_set == "v5":
            payload = make_blind_test_payload_v5(query_id, task)
        elif args.split_disjoint_templates and split == "test" and args.heldout_template_set == "v4":
            payload = make_blind_test_payload(query_id, task)
        elif args.split_disjoint_templates and split == "test" and args.heldout_template_set == "v3":
            payload = make_challenge_test_payload(query_id, task)
        elif args.split_disjoint_templates:
            payload = make_split_disjoint_payload(query_id, task, split)
        else:
            payload = make_task_payload(query_id, task)
        occupied: set[int] = set()
        gold_local: list[int] = []
        negative_local: list[int] = []
        evidence_texts: list[str] = []
        for evidence_index, text in enumerate(payload["evidence"]):
            local_block = choose_block(rng, block_loads[record_id], occupied)
            occupied.add(local_block)
            block_loads[record_id][local_block] += len(text)
            snippets[record_id][local_block].append(
                {
                    "query_id": query_id,
                    "role": f"evidence_{evidence_index + 1}",
                    "text": text,
                    "must_contain": payload["entity"] if evidence_index == 0 else payload["answer"],
                }
            )
            gold_local.append(local_block)
            evidence_texts.append(text)
        for decoy_index, text in enumerate(payload["decoys"]):
            local_block = choose_block(rng, block_loads[record_id], occupied)
            occupied.add(local_block)
            block_loads[record_id][local_block] += len(text)
            snippets[record_id][local_block].append(
                {
                    "query_id": query_id,
                    "role": f"hard_negative_{decoy_index + 1}",
                    "text": text,
                    "must_contain": (
                        alias_code(query_id)
                        if task == "multihop" and alias_code(query_id) in text
                        else payload["entity"]
                    ),
                }
            )
            negative_local.append(local_block)
        question_terms = lexical_terms(payload["question"])
        evidence_terms = lexical_terms(" ".join(evidence_texts))
        lexical_jaccard = len(question_terms & evidence_terms) / max(
            1, len(question_terms | evidence_terms)
        )
        provisional_queries.append(
            {
                "query_id": query_id,
                "task_type": task,
                "dataset": f"synthetic_{task}",
                "split": split,
                "record_id": record_id,
                "question": payload["question"],
                "answers": [payload["answer"]],
                "entity": payload["entity"],
                "gold_local_blocks": gold_local,
                "hard_negative_local_blocks": negative_local,
                "evidence_texts": evidence_texts,
                "question_evidence_lexical_jaccard": lexical_jaccard,
            }
        )

    blocks: list[np.ndarray] = []
    block_rows: list[dict[str, Any]] = []
    decoded_blocks: list[str] = []
    block_start = 0
    record_rows: list[dict[str, Any]] = []
    for record_id, local_count in enumerate(record_block_counts):
        uid = hashlib.sha256(f"controlled-synthetic-{args.seed}-{record_id}".encode()).hexdigest()
        for local_block in range(local_count):
            prefix = " ".join(
                filler_sentence(record_id, local_block, sentence_id) for sentence_id in range(2)
            )
            entries = list(snippets[record_id][local_block])
            rng.shuffle(entries)
            core = prefix + " " + " ".join(entry["text"] for entry in entries)
            core_ids = tokenizer(core, add_special_tokens=False)["input_ids"]
            if len(core_ids) > args.block_tokens - 24:
                raise RuntimeError(
                    f"record {record_id} block {local_block} has {len(core_ids)} core tokens"
                )
            text = core
            sentence_id = 2
            ids = core_ids
            while len(ids) < args.block_tokens:
                text += " " + filler_sentence(record_id, local_block, sentence_id)
                sentence_id += 1
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            ids_array = np.asarray(ids[: args.block_tokens], dtype=np.int32)
            decoded = tokenizer.decode(ids_array.tolist(), skip_special_tokens=True)
            for entry in entries:
                if str(entry["must_contain"]).casefold() not in decoded.casefold():
                    raise RuntimeError(
                        f"truncated synthetic fact in record {record_id} block {local_block}: {entry}"
                    )
            global_block = block_start + local_block
            blocks.append(ids_array)
            decoded_blocks.append(decoded)
            block_rows.append(
                {
                    "block_id": global_block,
                    "dataset": "controlled_synthetic",
                    "record_uid": uid,
                    "source_index": record_id,
                    "block_in_record": local_block,
                    "token_start_in_record": local_block * args.block_tokens,
                    "token_end_in_record": (local_block + 1) * args.block_tokens,
                    "synthetic_query_ids": sorted({int(item["query_id"]) for item in entries}),
                    "synthetic_roles": [str(item["role"]) for item in entries],
                }
            )
        record_rows.append(
            {
                "dataset": "controlled_synthetic",
                "record_uid": uid,
                "source_file": "generated:prepare_synthetic_controlled_corpus.py",
                "source_index": record_id,
                "block_start": block_start,
                "block_count": local_count,
                "source_token_count": local_count * args.block_tokens,
                "written_token_count": local_count * args.block_tokens,
            }
        )
        block_start += local_count

    query_rows: list[dict[str, Any]] = []
    answer_missing = 0
    answer_in_negative = 0
    for query in provisional_queries:
        record = record_rows[int(query["record_id"])]
        start = int(record["block_start"])
        gold_ids = [start + int(item) for item in query.pop("gold_local_blocks")]
        negative_ids = [
            start + int(item) for item in query.pop("hard_negative_local_blocks")
        ]
        answer = str(query["answers"][0]).casefold()
        if not any(answer in decoded_blocks[block_id].casefold() for block_id in gold_ids):
            answer_missing += 1
        if any(answer in decoded_blocks[block_id].casefold() for block_id in negative_ids):
            answer_in_negative += 1
        query_rows.append(
            {
                **query,
                "record_uid": record["record_uid"],
                "source_file": record["source_file"],
                "source_index": record["source_index"],
                "block_start": record["block_start"],
                "block_count": record["block_count"],
                "source_token_count": record["source_token_count"],
                "written_token_count": record["written_token_count"],
                "gold_block_ids": gold_ids,
                "hard_negative_block_ids": negative_ids,
            }
        )

    block_array = np.stack(blocks)
    np.save(output_dir / "blocks.npy", block_array)
    write_jsonl(output_dir / "blocks.jsonl", block_rows)
    write_jsonl(output_dir / "records.jsonl", record_rows)
    write_jsonl(output_dir / "queries.jsonl", query_rows)

    task_counts = Counter(str(row["task_type"]) for row in query_rows)
    split_counts = Counter(str(row["split"]) for row in query_rows)
    evidence_counts = Counter(len(row["gold_block_ids"]) for row in query_rows)
    summary = {
        "source": "controlled semi-synthetic text; real model Q/K must be profiled separately",
        "model_tokenizer": args.model_name_or_path,
        "contains_synthetic_text": True,
        "contains_synthetic_vectors": False,
        "seed": args.seed,
        "split_disjoint_templates": args.split_disjoint_templates,
        "heldout_template_set": args.heldout_template_set,
        "requested_seq_tokens": args.seq_tokens,
        "actual_seq_tokens": int(block_array.size),
        "block_tokens": args.block_tokens,
        "num_blocks": int(block_array.shape[0]),
        "num_records": len(record_rows),
        "record_block_counts": record_block_counts,
        "num_queries": len(query_rows),
        "task_counts": dict(task_counts),
        "split_counts": dict(split_counts),
        "gold_blocks_per_query": {str(key): value for key, value in sorted(evidence_counts.items())},
        "mean_question_evidence_lexical_jaccard_by_task": {
            task: float(
                np.mean(
                    [
                        row["question_evidence_lexical_jaccard"]
                        for row in query_rows
                        if row["task_type"] == task
                    ]
                )
            )
            for task in TASKS
        },
        "audit": {
            "queries_missing_answer_in_gold_blocks": answer_missing,
            "queries_with_answer_leak_in_declared_hard_negatives": answer_in_negative,
            "all_records_exceed_39_blocks": all(
                int(record["block_count"]) > 39 for record in record_rows
            ),
        },
        "gold_definition": "generator-declared evidence blocks; multihop queries require both evidence blocks",
        "blocks_path": str(output_dir / "blocks.npy"),
        "blocks_metadata_path": str(output_dir / "blocks.jsonl"),
        "records_metadata_path": str(output_dir / "records.jsonl"),
        "queries_path": str(output_dir / "queries.jsonl"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
