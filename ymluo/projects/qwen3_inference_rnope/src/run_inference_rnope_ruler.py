from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve()
PROJECTS = HERE.parents[2]
RULER_SRC = PROJECTS / "qwen3_ruler32k_rope_method" / "src"
LONG_SRC = PROJECTS / "qwen3_longbench_rope_method_exploration" / "src"
for directory in (RULER_SRC, LONG_SRC):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import run_ruler32k_rope_sparse as ruler  # noqa: E402
import run_longbench_rope_sparse as core  # noqa: E402


PRIMARY_VARIANTS = (
    "native_rope",
    "nope_every4_offset3",
    "nope_every4_offset0",
    "nope_alternating_odd",
)
ALL_VARIANTS = set(PRIMARY_VARIANTS) | {"native_replay"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference-only RNoPE on Qwen3-8B RULER-32K.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--examples-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--variants", default=",".join(PRIMARY_VARIANTS))
    parser.add_argument("--tasks", default="")
    parser.add_argument("--target-length", type=int, default=32768)
    parser.add_argument("--max-samples-per-task", type=int, default=2)
    parser.add_argument("--max-new-tokens-cap", type=int, default=128)
    parser.add_argument("--prefill-chunk-size", type=int, default=256)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--original-max-position-embeddings", type=int, default=40960)
    parser.add_argument("--global-max-position", type=int, default=40960)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def layer_indices(variant: str, num_layers: int) -> tuple[int, ...]:
    if variant in {"native_rope", "native_replay"}:
        return ()
    if variant == "nope_every4_offset3":
        return tuple(range(3, num_layers, 4))
    if variant == "nope_every4_offset0":
        return tuple(range(0, num_layers, 4))
    if variant == "nope_alternating_odd":
        return tuple(range(1, num_layers, 2))
    raise ValueError(f"unknown variant: {variant}")


class InferenceRNoPE:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.num_layers = int(model.config.num_hidden_layers)
        self.active_layers: frozenset[int] = frozenset()
        self._patch()

    def _patch(self) -> None:
        found = 0
        controller = self
        for module in self.model.modules():
            if module.__class__.__name__ != "Qwen3Attention":
                continue
            original = module.forward
            module._inference_rnope_original_forward = original

            def wrapped_forward(
                this: Any,
                hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: torch.Tensor | None,
                past_key_value: Any = None,
                cache_position: torch.Tensor | None = None,
                _original: Any = original,
                **kwargs: Any,
            ) -> Any:
                if int(this.layer_idx) in controller.active_layers:
                    cos, sin = position_embeddings
                    position_embeddings = (torch.ones_like(cos), torch.zeros_like(sin))
                return _original(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    cache_position=cache_position,
                    **kwargs,
                )

            module.forward = types.MethodType(wrapped_forward, module)
            found += 1
        if found != self.num_layers:
            raise RuntimeError(f"patched {found} Qwen3Attention modules, expected {self.num_layers}")

    @contextlib.contextmanager
    def activate(self, variant: str) -> Iterable[None]:
        previous = self.active_layers
        self.active_layers = frozenset(layer_indices(variant, self.num_layers))
        try:
            yield
        finally:
            self.active_layers = previous


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def finite_logits(logits: torch.Tensor) -> bool:
    return bool(torch.isfinite(logits).all().item())


def first_answer_stats(tokenizer: Any, logits: torch.Tensor, answer: str) -> tuple[float, int]:
    candidates = ruler.first_answer_token_candidates(tokenizer, answer)
    log_probs = F.log_softmax(logits.float(), dim=-1)[0]
    nll = min(-float(log_probs[token_id].item()) for token_id in candidates)
    correct = int(int(logits.argmax(dim=-1).item()) in set(candidates))
    return nll, correct


@torch.inference_mode()
def run_last(model: Any, prompt: torch.Tensor, cache: Any, prefix_length: int) -> tuple[torch.Tensor, Any, float]:
    core.cache_runner.synchronize()
    started = time.perf_counter()
    output = core.cache_runner.forward_with_cache(
        model,
        prompt[:, -1:].to(core.cache_runner.input_device(model)),
        cache,
        prefix_length,
    )
    core.cache_runner.synchronize()
    return output.logits[:, -1, :].float(), output.past_key_values, time.perf_counter() - started


@torch.inference_mode()
def greedy_generate(
    model: Any,
    tokenizer: Any,
    first_logits: torch.Tensor,
    cache: Any,
    prompt_length: int,
    max_new_tokens: int,
) -> tuple[str, list[int], float]:
    started = time.perf_counter()
    logits = first_logits
    generated: list[int] = []
    stops = core.eos_ids(tokenizer)
    past_length = prompt_length
    for step in range(max_new_tokens):
        token_id = int(logits.argmax(dim=-1).item())
        if token_id in stops:
            break
        generated.append(token_id)
        if step + 1 >= max_new_tokens:
            break
        token = torch.tensor([[token_id]], dtype=torch.long, device=core.cache_runner.input_device(model))
        output = core.cache_runner.forward_with_cache(model, token, cache, past_length)
        cache = output.past_key_values
        logits = output.logits[:, -1, :].float()
        past_length += 1
    return tokenizer.decode(generated, skip_special_tokens=True).strip(), generated, time.perf_counter() - started


@torch.inference_mode()
def gold_answer_nll(
    model: Any,
    tokenizer: Any,
    prompt: torch.Tensor,
    cache: Any,
    prefix_length: int,
    answer: str,
) -> tuple[float, int, float]:
    candidates = core.answer_candidates(tokenizer, answer)
    started = time.perf_counter()
    first_logits, cache, _ = run_last(model, prompt, cache, prefix_length)
    if all(len(ids) == 1 for ids in candidates):
        losses = [-float(F.log_softmax(first_logits, dim=-1)[0, ids[0]].item()) for ids in candidates]
        return min(losses), 1, time.perf_counter() - started
    ids = candidates[0]
    logits = first_logits
    losses: list[float] = []
    past_length = int(prompt.shape[1])
    for index, token_id in enumerate(ids):
        losses.append(-float(F.log_softmax(logits, dim=-1)[0, token_id].item()))
        if index + 1 < len(ids):
            token = torch.tensor([[token_id]], dtype=torch.long, device=core.cache_runner.input_device(model))
            output = core.cache_runner.forward_with_cache(model, token, cache, past_length)
            cache = output.past_key_values
            logits = output.logits[:, -1, :].float()
            past_length += 1
    return sum(losses) / len(losses), len(ids), time.perf_counter() - started


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = sorted(set(variants) - ALL_VARIANTS)
    if not variants or unknown:
        raise ValueError(f"unknown or empty variants: {unknown}")
    examples = ruler.select_examples(ruler.read_jsonl(args.examples_jsonl), args)
    if not examples:
        raise RuntimeError("no examples selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    load_args = argparse.Namespace(
        model_name_or_path=args.model_name_or_path,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        original_max_position_embeddings=args.original_max_position_embeddings,
        global_max_position=args.global_max_position,
        load_in_4bit=bool(args.load_in_4bit),
    )
    model, tokenizer = core.local_global.load_model(load_args)
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
        model.eval()
    intervention = InferenceRNoPE(model)
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "resolved_variants": variants,
        "num_layers": intervention.num_layers,
        "variant_layers": {name: list(layer_indices(name, intervention.num_layers)) for name in variants},
        "selected_examples": [asdict(example) | {"context": "<omitted>"} for example in examples],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "weights_frozen": True,
        "intervention": "identity RoPE (cos=1, sin=0) on selected layers",
    }
    write_json(args.output_dir / "config.json", config)

    rows_path = args.output_dir / "rows.jsonl"
    existing = read_jsonl(rows_path) if rows_path.exists() else []
    completed = {(str(row["sample_id"]), str(row["variant"])) for row in existing}

    for sample_index, example in enumerate(examples, start=1):
        task_name = ruler.base_task(example.task)
        prompt, _, _ = ruler.make_prompt(tokenizer, example)
        prompt_tokens = int(prompt.shape[-1])
        prefix_length = prompt_tokens - 1
        native_logits: torch.Tensor | None = None
        print(f"[{sample_index}/{len(examples)}] {example.sample_id} tokens={prompt_tokens}", flush=True)

        for variant in variants:
            if (example.sample_id, variant) in completed:
                continue
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            with intervention.activate(variant):
                started = time.perf_counter()
                legacy, prefill_seconds = core.cache_runner.prefill_sequence(
                    model, prompt[:, :-1], args.prefill_chunk_size
                )
                cache = core.cache_runner.cache_from_legacy(legacy)
                del legacy
                logits, cache, query_seconds = run_last(model, prompt, cache, prefix_length)
                if not finite_logits(logits):
                    raise RuntimeError(f"non-finite logits: {example.sample_id}/{variant}")
                replay_error = None
                if variant == "native_rope":
                    native_logits = logits.detach().cpu()
                elif variant == "native_replay" and native_logits is not None:
                    replay_error = float((logits.detach().cpu() - native_logits).abs().max().item())

                first_nll, first_correct = first_answer_stats(tokenizer, logits, example.answers[0])
                max_new = min(int(example.max_new_tokens), int(args.max_new_tokens_cap))
                prediction, generated_ids, generation_seconds = greedy_generate(
                    model, tokenizer, logits, cache, prompt_tokens, max_new
                )
                core.rope_repair.reset_dynamic_cache(cache, prefix_length)
                gold_nll, gold_tokens, nll_seconds = gold_answer_nll(
                    model, tokenizer, prompt, cache, prefix_length, example.answers[0]
                )
                core.rope_repair.reset_dynamic_cache(cache, prefix_length)
                official_score = ruler.public.score_prediction(
                    example.metric,
                    prediction,
                    example.answers,
                    example.all_classes,
                    task=task_name,
                )
                elapsed = time.perf_counter() - started

            row = {
                "sample_id": example.sample_id,
                "task": task_name,
                "requested_length": int(example.length),
                "prompt_tokens": prompt_tokens,
                "variant": variant,
                "nope_layers": list(layer_indices(variant, intervention.num_layers)),
                "metric": example.metric,
                "answers": example.answers,
                "prediction": prediction,
                "generated_token_ids": generated_ids,
                "generated_tokens": len(generated_ids),
                "official_score": official_score,
                "answer_coverage": ruler.official_answer_coverage(prediction, example.answers),
                "first_answer_next_token_nll": first_nll,
                "first_answer_next_token_ppl": math.exp(min(first_nll, 30.0)),
                "first_answer_next_token_correct": first_correct,
                "gold_answer_mean_nll": gold_nll,
                "gold_answer_ppl": math.exp(min(gold_nll, 30.0)),
                "gold_answer_tokens": gold_tokens,
                "prefill_seconds": prefill_seconds,
                "query_seconds": query_seconds,
                "generation_seconds": generation_seconds,
                "nll_seconds": nll_seconds,
                "elapsed_seconds": elapsed,
                "native_replay_max_logit_error": replay_error,
                "finite_logits": True,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / (1024**3),
            }
            append_jsonl(rows_path, [row])
            completed.add((example.sample_id, variant))
            print(
                f"  {variant}: score={official_score:.3f} first_nll={first_nll:.3f} "
                f"gold_nll={gold_nll:.3f} elapsed={elapsed:.1f}s pred={prediction[:80]!r}",
                flush=True,
            )
            del cache
            core.local_global.clear_allocator()
        del prompt

    rows = read_jsonl(rows_path)
    write_json(args.output_dir / "status.json", {"complete": True, "rows": len(rows)})
    (args.output_dir / "done.txt").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    main()
