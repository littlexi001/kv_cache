from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_qwen8b_paper_benchmarks import Config, score_blocks, split_ruler_input  # noqa: E402
from run_static_summary_ppl_speed import resolve_dtype  # noqa: E402


def make_config(args: argparse.Namespace) -> Config:
    return Config(
        output_dir="",
        model_name_or_path=args.model_name_or_path,
        adapter_path="",
        longbench_data_dir="",
        ruler_data_dir=args.ruler_data_dir,
        longbench_tasks=(),
        ruler_tasks=(args.task,),
        ruler_context_lengths=(args.context_length,),
        methods=(),
        max_examples_per_task=0,
        case_ids=tuple(args.case_ids.split(",")),
        block_tokens=args.block_tokens,
        recent_tokens=args.recent_tokens,
        max_input_tokens=24000,
        summary10_words=10,
        summary100_words=100,
        summary1000_words=900,
        max_new_tokens_exact=48,
        max_new_tokens_summary=120,
        dtype="float16",
        attn_implementation="sdpa",
        device_map="auto",
        cuda_visible_devices="",
        router_path="",
        seed=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ruler_data_dir", default="/home/fdong/ymluo/external/KVCache-Factory/data/RULER")
    parser.add_argument("--model_name_or_path", default="/home/fdong/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218")
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--case_ids", required=True)
    parser.add_argument("--block_tokens", type=int, default=256)
    parser.add_argument("--recent_tokens", type=int, default=512)
    parser.add_argument("--top_n", type=int, default=16)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    path = Path(args.ruler_data_dir) / str(args.context_length) / f"{args.task}.jsonl"
    wanted = set(args.case_ids.split(","))
    config = make_config(args)
    for idx, line in enumerate(path.open("r", encoding="utf-8")):
        row = json.loads(line)
        case_id = str(row.get("index", idx))
        if case_id not in wanted:
            continue
        memory, query = split_ruler_input(str(row["input"]))
        ids = tokenizer(memory, add_special_tokens=False)["input_ids"]
        old_cut = max(0, len(ids) - args.recent_tokens)
        old_memory = tokenizer.decode(ids[:old_cut], skip_special_tokens=True)
        answers = [str(item) for item in row.get("outputs", [])]
        print(f"\ncase={case_id} task={args.task} length={args.context_length} answers={answers}")
        print(f"query={query.strip()[:500]}")
        for needle in answers + ["1234567890", "magic number", "special magic number"]:
            pos = str(row["input"]).find(needle)
            print(f"find {needle!r} pos={pos}")
            if pos >= 0:
                excerpt = str(row["input"])[max(0, pos - 160) : pos + 240].replace("\n", " ")
                print(f"  {excerpt}")
        print("top scored old blocks:")
        for score, block_idx, block in score_blocks(tokenizer, old_memory, query, config)[: args.top_n]:
            has_answer = any(ans in block for ans in answers)
            has_decoy = "1234567890" in block
            nums = re.findall(r"[A-Za-z]*\d+[A-Za-z0-9-]*", block)
            excerpt = block[:220].replace("\n", " ")
            print(
                f"  block={block_idx:03d} score={score:03d} answer={int(has_answer)} "
                f"decoy={int(has_decoy)} nums={nums[:6]} text={excerpt}"
            )


if __name__ == "__main__":
    main()
