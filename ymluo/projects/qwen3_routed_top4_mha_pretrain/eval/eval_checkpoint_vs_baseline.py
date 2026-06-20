from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from train_routed_top4_qwen import RoutedQwenConfig, RoutedQwenForCausalLM  # noqa: E402


def load_auto_tokenizer() -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("transformers with AutoTokenizer is required.") from exc
    return AutoTokenizer


def load_auto_model_for_causal_lm() -> Any:
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("transformers with AutoModelForCausalLM is required for the official baseline.") from exc
    return AutoModelForCausalLM


@dataclass
class MultipleChoiceExample:
    task: str
    idx: int
    prompt: str
    choices: list[str]
    answer: int
    example_id: str


class CausalLMScorer:
    def __init__(self, name: str, model: torch.nn.Module, tokenizer: Any, device: torch.device, dtype: torch.dtype) -> None:
        self.name = name
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=self.device.type, dtype=self.dtype, enabled=self.device.type == "cuda"):
            output = self.model(input_ids)
            if isinstance(output, dict):
                return output["logits"]
            return output.logits

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    @torch.no_grad()
    def continuation_score(
        self,
        prompt: str,
        continuation: str,
        max_seq_len: int,
        length_normalize: bool,
    ) -> dict[str, float | int]:
        prompt_ids = self.encode(prompt)
        continuation_ids = self.encode(continuation)
        if not continuation_ids:
            return {"score": float("-inf"), "sum_logprob": float("-inf"), "tokens": 0}
        full_ids = prompt_ids + continuation_ids
        if len(full_ids) > max_seq_len:
            overflow = len(full_ids) - max_seq_len
            prompt_ids = prompt_ids[overflow:] if overflow < len(prompt_ids) else []
            full_ids = prompt_ids + continuation_ids
        if len(prompt_ids) == 0:
            bos_id = self.tokenizer.bos_token_id
            if bos_id is None:
                bos_id = self.tokenizer.eos_token_id
            if bos_id is None:
                raise ValueError("Cannot score empty prompt because tokenizer has no BOS or EOS token.")
            full_ids = [int(bos_id)] + full_ids
            prompt_len = 1
        else:
            prompt_len = len(prompt_ids)
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.device)
        logits = self.logits(input_ids)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        start = prompt_len
        token_logprobs = []
        for pos in range(start, len(full_ids)):
            target = int(full_ids[pos])
            token_logprobs.append(float(log_probs[0, pos - 1, target].detach().cpu()))
        sum_logprob = float(sum(token_logprobs))
        score = sum_logprob / max(1, len(token_logprobs)) if length_normalize else sum_logprob
        return {"score": score, "sum_logprob": sum_logprob, "tokens": len(token_logprobs)}


def dtype_from_name(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_routed_scorer(checkpoint_dir: Path, tokenizer_path: Path, device: torch.device, dtype: torch.dtype) -> CausalLMScorer:
    if checkpoint_dir.is_file():
        checkpoint_dir = Path(checkpoint_dir.read_text(encoding="utf-8").strip())
    config_path = checkpoint_dir / "routed_qwen_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing routed config: {config_path}")
    config = RoutedQwenConfig(**json.loads(config_path.read_text(encoding="utf-8")))
    model = RoutedQwenForCausalLM(config, gradient_checkpointing=False)
    state = torch.load(checkpoint_dir / "model.pt", map_location="cpu")
    model.load_state_dict(state)
    model.to(device=device, dtype=dtype)
    AutoTokenizer = load_auto_tokenizer()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return CausalLMScorer("routed_checkpoint", model, tokenizer, device, dtype)


def load_baseline_scorer(model_path: Path, device: torch.device, dtype: torch.dtype) -> CausalLMScorer:
    AutoTokenizer = load_auto_tokenizer()
    AutoModelForCausalLM = load_auto_model_for_causal_lm()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype)
    model.to(device)
    return CausalLMScorer("official_qwen3_0p6b", model, tokenizer, device, dtype)


def answer_to_index(answer: Any, choices: list[str]) -> int:
    if isinstance(answer, int):
        return answer
    if isinstance(answer, str):
        stripped = answer.strip()
        if stripped.isdigit():
            return int(stripped)
        upper = stripped.upper()
        if len(upper) == 1 and "A" <= upper <= "Z":
            return ord(upper) - ord("A")
        for idx, choice in enumerate(choices):
            if stripped == choice:
                return idx
    raise ValueError(f"Cannot convert answer={answer!r} to choice index.")


def load_mc_examples(path: Path, limit: int) -> list[MultipleChoiceExample]:
    task = path.stem
    rows: list[MultipleChoiceExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            choices = raw.get("choices", raw.get("options", raw.get("endings")))
            prompt = raw.get("prompt", raw.get("query", raw.get("question")))
            answer = raw.get("answer", raw.get("label", raw.get("gold", raw.get("answer_idx"))))
            if prompt is None or choices is None or answer is None:
                raise ValueError(f"{path} row must contain prompt, choices, answer: {raw}")
            choices = [str(choice) for choice in choices]
            rows.append(
                MultipleChoiceExample(
                    task=task,
                    idx=len(rows),
                    prompt=str(prompt),
                    choices=choices,
                    answer=answer_to_index(answer, choices),
                    example_id=str(raw.get("id", len(rows))),
                )
            )
            if limit > 0 and len(rows) >= limit:
                break
    return rows


def evaluate_multiple_choice(
    scorer: CausalLMScorer,
    examples: list[MultipleChoiceExample],
    max_seq_len: int,
    choice_prefix: str,
    length_normalize: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    correct = 0
    details = []
    started = time.time()
    for example in examples:
        scored_choices = []
        for choice in example.choices:
            scored = scorer.continuation_score(
                example.prompt,
                choice_prefix + choice,
                max_seq_len=max_seq_len,
                length_normalize=length_normalize,
            )
            scored_choices.append(scored)
        scores = [float(item["score"]) for item in scored_choices]
        pred = max(range(len(scores)), key=lambda idx: scores[idx])
        is_correct = pred == example.answer
        correct += int(is_correct)
        details.append(
            {
                "model": scorer.name,
                "task": example.task,
                "idx": example.idx,
                "id": example.example_id,
                "prediction": pred,
                "answer": example.answer,
                "correct": is_correct,
                "scores": scores,
                "choice_tokens": [int(item["tokens"]) for item in scored_choices],
            }
        )
    summary = {
        "model": scorer.name,
        "task": examples[0].task if examples else "",
        "examples": len(examples),
        "correct": correct,
        "accuracy": correct / max(1, len(examples)),
        "seconds": time.time() - started,
    }
    return summary, details


def discover_text_paths(text_path: Path, glob: str) -> list[Path]:
    if text_path.is_file():
        return [text_path]
    return sorted(path for path in text_path.rglob(glob) if path.is_file())


def read_limited_text(paths: list[Path], max_chars: int) -> str:
    parts = []
    total = 0
    for path in paths:
        if max_chars > 0 and total >= max_chars:
            break
        limit = max_chars - total if max_chars > 0 else -1
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            text = handle.read(limit if limit > 0 else -1)
        parts.append(text)
        total += len(text)
    return "\n".join(parts)


@torch.no_grad()
def evaluate_text_ppl(
    scorer: CausalLMScorer,
    text_path: Path,
    text_glob: str,
    max_chars: int,
    seq_len: int,
    max_batches: int,
) -> dict[str, Any]:
    paths = discover_text_paths(text_path, text_glob)
    if not paths:
        raise FileNotFoundError(f"No text files found for {text_path} with glob={text_glob}")
    text = read_limited_text(paths, max_chars)
    token_ids = scorer.encode(text)
    if len(token_ids) < 2:
        raise ValueError("Validation text produced fewer than two tokens.")
    nll_sum = 0.0
    token_count = 0
    batches = 0
    started = time.time()
    for start in range(0, len(token_ids) - 1, seq_len):
        if max_batches > 0 and batches >= max_batches:
            break
        chunk = token_ids[start : start + seq_len + 1]
        if len(chunk) < 2:
            continue
        input_ids = torch.tensor([chunk[:-1]], dtype=torch.long, device=scorer.device)
        labels = torch.tensor([chunk[1:]], dtype=torch.long, device=scorer.device)
        logits = scorer.logits(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1), reduction="sum")
        nll_sum += float(loss.detach().cpu())
        token_count += int(labels.numel())
        batches += 1
    ce = nll_sum / max(1, token_count)
    return {
        "model": scorer.name,
        "task": "text_ppl",
        "paths": [str(path) for path in paths[:20]],
        "path_count": len(paths),
        "chars": len(text),
        "tokens": token_count,
        "batches": batches,
        "ce": ce,
        "ppl": math.exp(min(ce, 50.0)),
        "seconds": time.time() - started,
    }


def parse_task_args(values: list[str]) -> list[tuple[str, Path]]:
    tasks = []
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
            tasks.append((name.strip(), Path(path.strip())))
        else:
            path = Path(value.strip())
            tasks.append((path.stem, path))
    return tasks


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--baseline_model_path", default="/mnt/workspace/Qwen3-0.6B")
    parser.add_argument("--tokenizer_path", default="/mnt/workspace/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mc_task", action="append", default=[], help="Either path.jsonl or name=path.jsonl")
    parser.add_argument("--mc_limit", type=int, default=200)
    parser.add_argument("--choice_prefix", default=" ")
    parser.add_argument("--no_length_normalize", action="store_true")
    parser.add_argument("--eval_text_path", default="")
    parser.add_argument("--eval_text_glob", default="*.txt")
    parser.add_argument("--eval_text_max_chars", type=int, default=1_000_000)
    parser.add_argument("--eval_seq_len", type=int, default=2048)
    parser.add_argument("--eval_text_max_batches", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--skip_routed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    dtype = dtype_from_name(args.dtype, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "eval_args.json", vars(args))

    scorers: list[CausalLMScorer] = []
    if not args.skip_routed:
        scorers.append(load_routed_scorer(Path(args.checkpoint_dir), Path(args.tokenizer_path), device, dtype))
    if not args.skip_baseline:
        scorers.append(load_baseline_scorer(Path(args.baseline_model_path), device, dtype))

    summaries = []
    details_path = output_dir / "multiple_choice_details.jsonl"
    if details_path.exists():
        details_path.unlink()

    mc_tasks = parse_task_args(args.mc_task)
    for task_name, task_path in mc_tasks:
        examples = load_mc_examples(task_path, args.mc_limit)
        for example in examples:
            example.task = task_name
        for scorer in scorers:
            summary, details = evaluate_multiple_choice(
                scorer,
                examples,
                max_seq_len=args.eval_seq_len,
                choice_prefix=args.choice_prefix,
                length_normalize=not args.no_length_normalize,
            )
            summaries.append(summary)
            append_jsonl(details_path, details)
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    if args.eval_text_path:
        for scorer in scorers:
            summary = evaluate_text_ppl(
                scorer,
                Path(args.eval_text_path),
                args.eval_text_glob,
                args.eval_text_max_chars,
                args.eval_seq_len,
                args.eval_text_max_batches,
            )
            summaries.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)

    write_json(output_dir / "summary.json", summaries)
    print(f"wrote summary: {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
