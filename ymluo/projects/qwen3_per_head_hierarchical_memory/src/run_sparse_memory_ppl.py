from __future__ import annotations

import argparse
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "src"))

from run_per_head_hierarchical_memory import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    POLICIES,
    PerHeadHierarchicalMemory,
    RetrieverBank,
    functional_method_weights,
    model_forward,
    pick_input_device,
    prefill_cache,
    promotion_slots_from_atlas,
    read_atlas,
    resolve_dtype,
    str2bool,
    write_csv,
)


_ACTIVE_CONTROLLER: "SparseMemoryController | None" = None
_ORIGINAL_EAGER: Any | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject per-head hierarchical L0 masks and measure exact logical-sparse PPL."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--text_path", required=True)
    parser.add_argument("--atlas_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prefill_tokens", type=int, default=16_384)
    parser.add_argument("--train_queries", type=int, default=64)
    parser.add_argument("--test_queries", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--prefill_chunk_size", type=int, default=128)
    parser.add_argument("--l0_capacity", type=int, default=500)
    parser.add_argument("--l0_recent_tokens", type=int, default=448)
    parser.add_argument(
        "--promotion_policy",
        choices=("uniform", "confidence_gated"),
        default="uniform",
    )
    parser.add_argument("--medium_promotion_slots", type=int, default=20)
    parser.add_argument(
        "--promotion_categories",
        default="semantic_evidence,lexical_copy,structural_anchor",
        help="Comma-separated conservative head functions allowed to promote old tokens.",
    )
    parser.add_argument("--l1_capacity", type=int, default=4096)
    parser.add_argument("--l2_block_size", type=int, default=64)
    parser.add_argument("--l2_block_budget", type=int, default=64)
    parser.add_argument("--sink_tokens", type=int, default=4)
    parser.add_argument("--query_window", type=int, default=32)
    parser.add_argument("--repeat_max_n", type=int, default=4)
    parser.add_argument("--l1_retention_bonus", type=float, default=0.05)
    parser.add_argument("--l0_retention_bonus", type=float, default=0.05)
    parser.add_argument("--function_temperature", type=float, default=1.0)
    parser.add_argument("--random_seed", type=int, default=20260716)
    parser.add_argument("--max_chars", type=int, default=8_000_000)
    parser.add_argument(
        "--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--add_special_tokens", type=str2bool, default=False)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument(
        "--policies",
        default="full_attention,sink_recent_500,flat_function_500,hier_function_500",
    )
    return parser.parse_args()


def _sparse_eager_attention_forward(
    module: torch.nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = float(getattr(module, "scaling", 1.0 / math.sqrt(query_states.shape[-1])))
    if key_states.shape[1] != query_states.shape[1]:
        repeat_groups = query_states.shape[1] // key_states.shape[1]
        key_states = key_states.repeat_interleave(repeat_groups, dim=1)
        value_states = value_states.repeat_interleave(repeat_groups, dim=1)
    scores = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : scores.shape[-1]]
    if _ACTIVE_CONTROLLER is None:
        attention_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
    else:
        attention_weights = _ACTIVE_CONTROLLER.sparse_attention(module, scores).to(
            query_states.dtype
        )
    if dropout and module.training:
        attention_weights = F.dropout(attention_weights, p=dropout, training=True)
    attention_output = torch.matmul(attention_weights, value_states)
    attention_output = attention_output.transpose(1, 2).contiguous()
    return attention_output, attention_weights


def install_sparse_attention_patch() -> None:
    global _ORIGINAL_EAGER
    import transformers.models.qwen3.modeling_qwen3 as modeling_qwen3

    if _ORIGINAL_EAGER is None:
        _ORIGINAL_EAGER = modeling_qwen3.eager_attention_forward
    modeling_qwen3.eager_attention_forward = _sparse_eager_attention_forward
    if hasattr(modeling_qwen3, "ALL_ATTENTION_FUNCTIONS"):
        modeling_qwen3.ALL_ATTENTION_FUNCTIONS["eager"] = _sparse_eager_attention_forward


@contextmanager
def activate_controller(controller: "SparseMemoryController | None"):
    global _ACTIVE_CONTROLLER
    previous = _ACTIVE_CONTROLLER
    _ACTIVE_CONTROLLER = controller
    try:
        yield
    finally:
        _ACTIVE_CONTROLLER = previous


class SparseMemoryController:
    def __init__(
        self,
        memory: PerHeadHierarchicalMemory,
        *,
        policy: str,
        layer_count: int,
        head_count: int,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"unsupported sparse policy: {policy}")
        self.memory = memory
        self.policy = policy
        self.layer_count = layer_count
        self.head_count = head_count
        self.max_allowed_history = 0
        self.min_allowed_history = 10**9

    def allowed_mask(
        self,
        *,
        layer: int,
        query_count: int,
        key_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        past_tokens = key_count - query_count
        allowed = torch.zeros(
            (self.head_count, query_count, key_count), dtype=torch.bool, device=device
        )
        flat_start = layer * self.head_count
        flat_end = flat_start + self.head_count
        for query_index in range(query_count):
            current = past_tokens + query_index
            selected = self.memory.selections(current)[self.policy][flat_start:flat_end].to(device)
            allowed[:, query_index].scatter_(1, selected, True)
            # Always preserve the current decode chunk up through the causal query token.
            allowed[:, query_index, past_tokens : current + 1] = True
            history_count = selected.shape[1]
            self.max_allowed_history = max(self.max_allowed_history, history_count)
            self.min_allowed_history = min(self.min_allowed_history, history_count)
        return allowed

    @torch.inference_mode()
    def sparse_attention(
        self, module: torch.nn.Module, scores: torch.Tensor
    ) -> torch.Tensor:
        if scores.shape[0] != 1:
            raise ValueError("sparse-memory PPL evaluator requires batch size 1")
        layer = int(getattr(module, "layer_idx", 0))
        _, heads, query_count, key_count = scores.shape
        if heads != self.head_count:
            raise ValueError(f"expected {self.head_count} heads, got {heads}")
        allowed = self.allowed_mask(
            layer=layer,
            query_count=query_count,
            key_count=key_count,
            device=scores.device,
        )
        masked = scores.masked_fill(~allowed.unsqueeze(0), -torch.inf)
        weights = F.softmax(masked, dim=-1, dtype=torch.float32)
        if layer == self.layer_count - 1 and query_count:
            self.memory.discard_before(key_count)
        return weights


@torch.inference_mode()
def evaluate_sparse_phase(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    past: Any,
    previous_logits: torch.Tensor,
    *,
    start: int,
    end: int,
    chunk_size: int,
    input_device: torch.device,
    controller: SparseMemoryController | None,
    phase: str,
    log_every: int,
) -> tuple[Any, torch.Tensor, float, int]:
    total_nll = 0.0
    token_count = 0
    chunk_count = math.ceil((end - start) / chunk_size)
    for chunk_index, chunk_start in enumerate(range(start, end, chunk_size), start=1):
        chunk_end = min(end, chunk_start + chunk_size)
        chunk = input_ids[:, chunk_start:chunk_end].to(input_device)
        kwargs: dict[str, Any] = {
            "input_ids": chunk,
            "use_cache": True,
            "return_dict": True,
            "output_attentions": False,
            "cache_position": torch.arange(chunk_start, chunk_end, device=input_device),
        }
        if past is not None:
            kwargs["past_key_values"] = past
        with activate_controller(controller):
            outputs = model_forward(model, kwargs)
        logits = outputs.logits
        shifted = torch.cat([previous_logits.unsqueeze(1), logits[:, :-1]], dim=1)
        loss = F.cross_entropy(
            shifted.reshape(-1, shifted.shape[-1]).float(), chunk.reshape(-1), reduction="sum"
        )
        total_nll += float(loss)
        token_count += int(chunk.numel())
        previous_logits = logits[:, -1].detach()
        past = outputs.past_key_values
        if log_every > 0 and (chunk_index % log_every == 0 or chunk_index == chunk_count):
            print(
                f"phase={phase} chunk={chunk_index}/{chunk_count} tokens={chunk_start}:{chunk_end}",
                flush=True,
            )
    return past, previous_logits, total_nll, token_count


def make_memory(
    external_retriever: RetrieverBank,
    weights: torch.Tensor,
    promotion_slots: torch.Tensor,
    args: argparse.Namespace,
) -> PerHeadHierarchicalMemory:
    return PerHeadHierarchicalMemory(
        external_retriever,
        weights,
        l0_capacity=args.l0_capacity,
        l0_recent_tokens=args.l0_recent_tokens,
        l1_capacity=args.l1_capacity,
        l2_block_size=args.l2_block_size,
        l2_block_budget=args.l2_block_budget,
        sink_tokens=args.sink_tokens,
        l1_retention_bonus=args.l1_retention_bonus,
        l0_retention_bonus=args.l0_retention_bonus,
        promotion_slots=promotion_slots,
    )


def main() -> None:
    args = parse_args()
    requested_policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    invalid = [item for item in requested_policies if item != "full_attention" and item not in POLICIES]
    if invalid:
        raise ValueError(f"unknown policies: {invalid}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.random_seed)
    started = time.perf_counter()

    total_tokens = args.prefill_tokens + args.train_queries + args.test_queries
    with Path(args.text_path).open("r", encoding="utf-8", errors="ignore") as handle:
        text = handle.read(args.max_chars) if args.max_chars > 0 else handle.read()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    token_ids = tokenizer(
        text,
        add_special_tokens=args.add_special_tokens,
        truncation=True,
        max_length=total_tokens,
    )["input_ids"]
    if len(token_ids) < total_tokens:
        raise ValueError(f"tokenized text has {len(token_ids)} tokens; need {total_tokens}")
    input_ids = torch.tensor(token_ids, dtype=torch.long).view(1, -1)

    requested_device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": resolve_dtype(args.dtype, requested_device),
    }
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    if args.device_map.lower() == "none":
        model = model.to(requested_device)
    model.eval()
    model.config.use_cache = True
    input_device = pick_input_device(model, requested_device)
    install_sparse_attention_patch()

    with torch.inference_mode():
        embeddings = model.get_input_embeddings()(input_ids.to(input_device))[0].float().cpu()
    decoded = [
        tokenizer.decode([token_id], clean_up_tokenization_spaces=False) for token_id in token_ids
    ]
    external_retriever = RetrieverBank(
        input_ids[0],
        embeddings,
        decoded,
        ratio=0.02,
        query_window=args.query_window,
        block_size=args.l2_block_size,
        repeat_max_n=args.repeat_max_n,
        sink_tokens=args.sink_tokens,
        hybrid_position_fraction=0.5,
        random_seed=args.random_seed,
    )
    atlas_rows = read_atlas(Path(args.atlas_csv))
    weights, _ = functional_method_weights(atlas_rows, temperature=args.function_temperature)
    promotion_slots = promotion_slots_from_atlas(
        atlas_rows,
        l0_capacity=args.l0_capacity,
        l0_recent_tokens=args.l0_recent_tokens,
        sink_tokens=args.sink_tokens,
        policy=args.promotion_policy,
        medium_slots=args.medium_promotion_slots,
        active_categories=args.promotion_categories.split(","),
    )
    layer_count = int(model.config.num_hidden_layers)
    head_count = int(model.config.num_attention_heads)
    train_start = args.prefill_tokens
    train_end = train_start + args.train_queries
    test_end = train_end + args.test_queries

    rows: list[dict[str, Any]] = []
    for policy in requested_policies:
        print(f"policy={policy} prefill starting", flush=True)
        with activate_controller(None):
            past, previous_logits = prefill_cache(
                model,
                input_ids,
                args.prefill_tokens,
                args.prefill_chunk_size,
                input_device,
            )
        controller = None
        memory = None
        if policy != "full_attention":
            memory = make_memory(external_retriever, weights, promotion_slots, args)
            controller = SparseMemoryController(
                memory,
                policy=policy,
                layer_count=layer_count,
                head_count=head_count,
            )
        past, previous_logits, train_nll, train_count = evaluate_sparse_phase(
            model,
            input_ids,
            past,
            previous_logits,
            start=train_start,
            end=train_end,
            chunk_size=args.chunk_size,
            input_device=input_device,
            controller=controller,
            phase=f"{policy}:train",
            log_every=args.log_every,
        )
        _, _, test_nll, test_count = evaluate_sparse_phase(
            model,
            input_ids,
            past,
            previous_logits,
            start=train_end,
            end=test_end,
            chunk_size=args.chunk_size,
            input_device=input_device,
            controller=controller,
            phase=f"{policy}:test",
            log_every=args.log_every,
        )
        rows.append(
            {
                "policy": policy,
                "train_tokens": train_count,
                "train_mean_nll": train_nll / max(1, train_count),
                "train_ppl": math.exp(train_nll / max(1, train_count)),
                "test_tokens": test_count,
                "test_mean_nll": test_nll / max(1, test_count),
                "test_ppl": math.exp(test_nll / max(1, test_count)),
                "min_l0_history_tokens": (
                    controller.min_allowed_history if controller is not None else ""
                ),
                "max_l0_history_tokens": (
                    controller.max_allowed_history if controller is not None else ""
                ),
                "memory_query_updates": len(memory.diagnostic_rows) if memory is not None else 0,
            }
        )
        del past, previous_logits
        if input_device.type == "cuda":
            torch.cuda.empty_cache()

    full = next((row for row in rows if row["policy"] == "full_attention"), None)
    for row in rows:
        if full is None:
            row["test_delta_nll_vs_full"] = ""
            row["test_ppl_ratio_vs_full"] = ""
        else:
            row["test_delta_nll_vs_full"] = float(row["test_mean_nll"]) - float(
                full["test_mean_nll"]
            )
            row["test_ppl_ratio_vs_full"] = float(row["test_ppl"]) / float(full["test_ppl"])
    write_csv(output_dir / "sparse_ppl_results.csv", rows)
    summary = {
        "args": vars(args),
        "results": rows,
        "invariants": {
            "configured_l0_capacity": args.l0_capacity,
            "all_sparse_runs_respect_l0_cap": all(
                int(row["max_l0_history_tokens"]) <= args.l0_capacity
                for row in rows
                if row["policy"] != "full_attention"
            ),
            "heads_with_promotions": int((promotion_slots > 0).sum()),
        },
        "runtime_seconds": time.perf_counter() - started,
        "interpretation": (
            "Exact logical sparse-attention forward over the measured query tokens. "
            "The implementation still stores the full physical KV cache, so this validates "
            "quality rather than wall-clock speed or memory savings."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
