from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import re
import socket
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


FILLER_PARAGRAPHS = [
    "清晨的街道很安静，路边的商店陆续开门，行人拿着纸袋经过广场。",
    "会议记录提到项目进展、设备检查、天气变化和后续安排，没有涉及个人资料。",
    "图书馆新整理了一批资料，管理员按照编号把书放回对应的架子。",
    "下午的训练安排包括热身、分组练习、复盘讨论和器材归还。",
    "城市公园里的长椅刚刷过漆，志愿者在入口处分发路线图。",
    "研究小组记录了温度、湿度、样本编号和观察时间，方便后续核对。",
    "展览大厅里摆放着照片、地图和手写说明，参观者按顺序慢慢浏览。",
    "晚间广播提醒居民注意出行安全，并说明明天部分道路会临时维护。",
    "仓库清单包含纸箱、标签、工具、备用电池和几卷封箱胶带。",
    "老师要求学生整理笔记，重点关注论证结构、例子来源和表达是否清楚。",
    "车站大厅的电子屏滚动显示班次、检票口、到达时间和换乘提示。",
    "办公室里有人在打印材料，有人在检查表格，还有人在调整投影设备。",
    "社区活动安排了手工课程、健康讲座、问卷填写和场地清洁。",
    "远处的山坡被薄雾遮住，河面上偶尔有小船缓慢经过。",
    "技术文档描述了接口格式、错误处理、日志保存和版本兼容策略。",
    "餐厅菜单更新了几道家常菜，服务员把新价目表贴在入口旁边。",
    "博物馆工作人员检查灯光、温度和展柜锁扣，确保展品保存稳定。",
    "邮件正文主要说明时间安排、参会人员、材料提交方式和反馈期限。",
    "操场旁的告示牌提醒大家保持秩序，活动结束后带走随身物品。",
    "实验台上摆着量杯、记录本、手套和干净的玻璃容器。",
    "日报总结了订单数量、物流状态、库存变化和客服处理进度。",
    "旅行手册介绍了交通路线、开放时间、票务规则和附近餐饮。",
    "培训材料强调流程一致性、信息核验、异常记录和复查机制。",
    "小区公告说明电梯保养、绿化修剪、停车登记和公共区域清扫。",
    "新闻简报讨论了市场变化、公共服务、教育活动和文化项目。",
]

ENGLISH_FILLER_PARAGRAPHS = [
    "The morning street was quiet, and several small shops opened while commuters crossed the square.",
    "The meeting notes described project updates, equipment checks, weather changes, and next steps.",
    "The library staff organized a new set of materials and returned each book to its numbered shelf.",
    "The afternoon training plan included warmups, group practice, review discussions, and equipment return.",
    "Fresh paint covered the benches near the city park, and volunteers handed out route maps at the entrance.",
    "The research team recorded temperature, humidity, sample labels, and observation times for later review.",
    "The exhibition hall displayed photographs, maps, and handwritten notes for visitors to inspect in order.",
    "The evening broadcast reminded residents about travel safety and temporary road maintenance tomorrow.",
    "The warehouse list included boxes, labels, tools, spare batteries, and several rolls of packing tape.",
    "The instructor asked the group to organize notes and check structure, examples, and clarity of expression.",
    "The station display showed departure times, platform numbers, arrival notices, and transfer guidance.",
    "In the office, one person printed documents, another checked forms, and another adjusted the projector.",
    "The community event included a craft class, a health lecture, a questionnaire, and cleanup work.",
    "A thin mist covered the distant hillside, and a small boat moved slowly across the river.",
    "The technical guide described interface formats, error handling, log storage, and version compatibility.",
    "The restaurant updated several ordinary dishes and posted a new price sheet near the entrance.",
    "Museum staff checked lighting, temperature, and display case locks to keep the exhibits stable.",
    "The email explained the schedule, attendee list, material submission process, and feedback deadline.",
    "A notice beside the field reminded everyone to keep order and take personal items after the event.",
    "The laboratory table held measuring cups, notebooks, gloves, and clean glass containers.",
    "The daily report summarized order counts, shipping status, inventory changes, and service progress.",
    "The travel booklet introduced transit routes, opening hours, ticket rules, and nearby restaurants.",
    "The training document emphasized process consistency, information checks, exception logs, and review steps.",
    "The neighborhood notice mentioned elevator maintenance, garden trimming, parking records, and shared cleaning.",
    "The brief report discussed market changes, public services, education events, and cultural programs.",
]

TEXT_CONFIGS = {
    "zh": {
        "needle": "小明今年是九岁。",
        "question": "小明今年是几岁？",
        "answer": "九岁",
        "suffix": "\n\n请只根据上文回答。如果上文没有相关信息，回答“无法确定”。\n问题：小明今年是几岁？\n答案：",
    },
    "en": {
        "needle": "Xiaoming is nine years old this year.",
        "question": "How old is Xiaoming this year?",
        "answer": "nine years old",
        "suffix": '\n\nAnswer using only the text above. If the text above does not contain relevant information, answer "unknown".\nQuestion: How old is Xiaoming this year?\nAnswer:',
    },
}

FILLER_BY_LANG = {
    "zh": FILLER_PARAGRAPHS,
    "en": ENGLISH_FILLER_PARAGRAPHS,
}


@dataclass(frozen=True)
class Case:
    case_id: str
    needle_prompt_lang: str
    filler_lang: str
    target_length: int
    depth_percent: float
    seed: int
    rope_factor: float
    max_position_embeddings: int
    evidence_start: int
    evidence_end: int
    prompt_tokens: int
    suffix_tokens: int


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def resolve_dtype(name: str) -> torch.dtype:
    if name == "auto":
        return torch.float16 if torch.cuda.is_available() else torch.float32
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def rope_factor_for_length(target_length: int) -> float:
    if target_length <= 32768:
        return 1.0
    if target_length <= 65536:
        return 2.0
    if target_length <= 131072:
        return 4.0
    return 8.0


def build_filler_ids(tokenizer: Any, target_length: int, seed: int, filler_lang: str) -> list[int]:
    rng = random.Random(seed)
    ids: list[int] = []
    templates = list(FILLER_BY_LANG[filler_lang])
    while len(ids) < target_length + 512:
        rng.shuffle(templates)
        text = "\n".join(templates) + "\n"
        ids.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
    return ids[:target_length]


def build_case_ids(
    tokenizer: Any,
    target_length: int,
    depth_percent: float,
    seed: int,
    max_new_tokens: int,
    needle_prompt_lang: str,
    filler_lang: str,
) -> tuple[Case, torch.Tensor, tuple[int, int]]:
    text_config = TEXT_CONFIGS[needle_prompt_lang]
    context_ids = build_filler_ids(tokenizer, target_length, seed, filler_lang)
    needle_ids = tokenizer(text_config["needle"], add_special_tokens=False)["input_ids"]
    if len(needle_ids) >= target_length:
        raise ValueError("needle is longer than target context")
    start = int(round((target_length - len(needle_ids)) * depth_percent / 100.0))
    context_ids[start : start + len(needle_ids)] = needle_ids
    suffix_ids = tokenizer(text_config["suffix"], add_special_tokens=False)["input_ids"]
    prompt_ids = context_ids + suffix_ids
    factor = rope_factor_for_length(target_length)
    if factor <= 1.0:
        max_pos = max(40960, len(prompt_ids) + max_new_tokens + 8)
    else:
        max_pos = max(int(target_length + len(suffix_ids) + max_new_tokens + 8), int(32768 * factor))
    case = Case(
        case_id=f"{needle_prompt_lang}prompt_{filler_lang}filler_len{target_length}_depth{int(depth_percent)}_seed{seed}",
        needle_prompt_lang=needle_prompt_lang,
        filler_lang=filler_lang,
        target_length=target_length,
        depth_percent=depth_percent,
        seed=seed,
        rope_factor=factor,
        max_position_embeddings=max_pos,
        evidence_start=start,
        evidence_end=start + len(needle_ids),
        prompt_tokens=len(prompt_ids),
        suffix_tokens=len(suffix_ids),
    )
    return case, torch.tensor(prompt_ids, dtype=torch.long).view(1, -1), (start, start + len(needle_ids))


def load_model_and_tokenizer(args: argparse.Namespace, max_case_position: int, max_factor: float) -> tuple[Any, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if max_factor > 1.0:
        config.max_position_embeddings = max_case_position
        config.rope_scaling = {
            "type": "yarn",
            "factor": float(max_factor),
            "original_max_position_embeddings": int(args.original_max_position_embeddings),
        }
    elif max_case_position > int(getattr(config, "max_position_embeddings", 0)):
        config.max_position_embeddings = max_case_position

    load_kwargs: dict[str, Any] = {
        "config": config,
        "trust_remote_code": True,
        "torch_dtype": resolve_dtype(args.dtype),
    }
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    if args.attn_implementation.lower() != "auto":
        load_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **load_kwargs)
    if args.device_map.lower() == "none":
        model = model.to(args.device if torch.cuda.is_available() else "cpu")
    model.eval()
    model.config.use_cache = True
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def legacy_cache(cache: Any) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if hasattr(cache, "to_legacy_cache"):
        return tuple(cache.to_legacy_cache())
    return tuple(cache)


def cache_from_legacy(legacy: tuple[tuple[torch.Tensor, torch.Tensor], ...]) -> Any:
    try:
        from transformers.cache_utils import DynamicCache

        return DynamicCache.from_legacy_cache(legacy)
    except Exception:
        return legacy


def forward_with_cache(
    model: Any,
    input_ids: torch.Tensor,
    past_key_values: Any | None,
    past_len: int,
    *,
    output_attentions: bool = False,
) -> Any:
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "use_cache": True,
        "output_attentions": output_attentions,
        "return_dict": True,
    }
    if past_key_values is not None:
        q_len = int(input_ids.shape[1])
        device = input_ids.device
        kwargs["past_key_values"] = past_key_values
        kwargs["attention_mask"] = torch.ones((1, past_len + q_len), dtype=torch.long, device=device)
        kwargs["position_ids"] = torch.arange(past_len, past_len + q_len, device=device).view(1, -1)
        kwargs["cache_position"] = torch.arange(past_len, past_len + q_len, device=device)
    return model(**kwargs)


def prefill_sequence(model: Any, prompt_prefix: torch.Tensor, chunk_size: int) -> tuple[tuple[tuple[torch.Tensor, torch.Tensor], ...], float]:
    device = input_device(model)
    ids = prompt_prefix.to(device)
    past = None
    past_len = 0
    synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, int(ids.shape[1]), chunk_size):
            chunk = ids[:, start : start + chunk_size]
            out = forward_with_cache(model, chunk, past, past_len)
            past = out.past_key_values
            past_len += int(chunk.shape[1])
    synchronize()
    return legacy_cache(past), time.perf_counter() - started


def score_answer(
    model: Any,
    tokenizer: Any,
    base_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    last_prompt_id: torch.Tensor,
    prompt_len_minus_one: int,
    answer_text: str,
) -> dict[str, Any]:
    device = input_device(model)
    answer_ids = tokenizer(answer_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    total_nll = 0.0
    token_count = int(answer_ids.shape[1])
    with torch.inference_mode():
        out = forward_with_cache(
            model,
            last_prompt_id.to(device),
            cache_from_legacy(base_cache),
            prompt_len_minus_one,
        )
        logits = out.logits[:, -1, :]
        cache = out.past_key_values
        past_len = prompt_len_minus_one + 1
        for idx in range(token_count):
            target = answer_ids[:, idx]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            total_nll -= float(log_probs.gather(1, target.view(1, 1)).item())
            out = forward_with_cache(model, target.view(1, 1), cache, past_len)
            cache = out.past_key_values
            logits = out.logits[:, -1, :]
            past_len += 1
    mean_nll = total_nll / max(1, token_count)
    return {
        "answer_token_count": token_count,
        "answer_nll": mean_nll,
        "answer_ppl": math.exp(mean_nll) if mean_nll < 50 else float("inf"),
    }


def normalize_answer(text: str, answer_lang: str) -> str:
    first = re.split(r"\n|答案[:：]|问题[:：]|Answer[:：]|Question[:：]", text, maxsplit=1)[0]
    compact = re.sub(r"\s+", "", first)
    if answer_lang == "zh":
        if "九岁" in compact or "9岁" in compact or "九歲" in compact:
            return "correct"
        if any(item in compact for item in ("无法确定", "不能确定", "不知道", "没有提到", "未提到")):
            return "miss"
    else:
        lower = first.lower()
        if "nine" in lower or re.search(r"\b9\b", lower):
            return "correct"
        if any(item in lower for item in ("unknown", "cannot determine", "not mentioned", "not contain", "do not know")):
            return "miss"
    return "wrong"


def generate_answer(
    model: Any,
    tokenizer: Any,
    base_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    last_prompt_id: torch.Tensor,
    prompt_len_minus_one: int,
    max_new_tokens: int,
    answer_lang: str,
) -> dict[str, Any]:
    device = input_device(model)
    generated: list[int] = []
    synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        out = forward_with_cache(
            model,
            last_prompt_id.to(device),
            cache_from_legacy(base_cache),
            prompt_len_minus_one,
        )
        logits = out.logits[:, -1, :]
        cache = out.past_key_values
        past_len = prompt_len_minus_one + 1
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        for _ in range(max_new_tokens):
            token_id = int(next_token.item())
            if tokenizer.eos_token_id is not None and token_id == int(tokenizer.eos_token_id):
                break
            generated.append(token_id)
            out = forward_with_cache(model, next_token.to(device), cache, past_len)
            cache = out.past_key_values
            logits = out.logits[:, -1, :]
            past_len += 1
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
    synchronize()
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    cls = normalize_answer(generated_text, answer_lang)
    return {
        "generated_text": generated_text.replace("\n", "\\n"),
        "answer_class": cls,
        "correct": int(cls == "correct"),
        "miss": int(cls == "miss"),
        "wrong": int(cls == "wrong"),
        "generation_seconds": time.perf_counter() - started,
    }


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope_to_q(q: torch.Tensor, position_embeddings: Any) -> torch.Tensor:
    if position_embeddings is None:
        return q
    cos, sin = position_embeddings
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(1).to(device=q.device, dtype=q.dtype)
    sin = sin.unsqueeze(1).to(device=q.device, dtype=q.dtype)
    return (q * cos) + (rotate_half(q) * sin)


def stable_mass_from_logits(logits: torch.Tensor, start: int, end: int) -> float:
    logits_f = logits.float()
    max_value = logits_f.max()
    denom = torch.exp(logits_f - max_value).sum()
    numer = torch.exp(logits_f[start:end] - max_value).sum()
    if float(denom.item()) == 0.0:
        return 0.0
    return float((numer / denom).item())


def attention_mass(
    model: Any,
    base_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
    last_prompt_id: torch.Tensor,
    prompt_len_minus_one: int,
    evidence_span: tuple[int, int],
    page_tokens: int,
) -> dict[str, Any]:
    device = input_device(model)
    start, end = evidence_span
    try:
        layers = list(getattr(getattr(model, "model", None), "layers", []))
        if not layers:
            raise RuntimeError("cannot locate model.model.layers")
        captured_q: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(layer_idx: int):
            def hook(module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
                hidden_states = kwargs.get("hidden_states")
                if hidden_states is None and args:
                    hidden_states = args[0]
                position_embeddings = kwargs.get("position_embeddings")
                if position_embeddings is None and len(args) >= 2:
                    position_embeddings = args[1]
                if hidden_states is None:
                    return
                q = module.q_proj(hidden_states)
                batch, q_len, _ = q.shape
                head_dim = int(getattr(module, "head_dim"))
                num_heads = int(q.shape[-1] // head_dim)
                q = q.view(batch, q_len, num_heads, head_dim).transpose(1, 2)
                q = apply_rope_to_q(q, position_embeddings)
                captured_q[layer_idx] = q[:, :, -1, :].detach()

            return hook

        for idx, layer in enumerate(layers):
            handles.append(layer.self_attn.register_forward_pre_hook(make_hook(idx), with_kwargs=True))
        synchronize()
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                out = forward_with_cache(
                    model,
                    last_prompt_id.to(device),
                    cache_from_legacy(base_cache),
                    prompt_len_minus_one,
                )
        finally:
            for handle in handles:
                handle.remove()
        synchronize()
        cache = legacy_cache(out.past_key_values)
        if not captured_q:
            raise RuntimeError("failed to capture query states")

        masses: list[float] = []
        top_values: list[float] = []
        kv_len = int(cache[-1][0].shape[2])
        page_scores_accum: list[float] | None = None
        last_values_tensor = None
        for layer_idx, layer in enumerate(layers):
            q = captured_q[layer_idx][0]
            key = cache[layer_idx][0][0]
            num_heads = int(q.shape[0])
            kv_heads = int(key.shape[0])
            groups = max(1, num_heads // kv_heads)
            scale = float(getattr(layer.self_attn, "scaling", q.shape[-1] ** -0.5))
            layer_values: list[float] = []
            for head_idx in range(num_heads):
                kv_idx = min(kv_heads - 1, head_idx // groups)
                logits = torch.matmul(key[kv_idx].float(), q[head_idx].float()) * scale
                mass_value = stable_mass_from_logits(logits, start, end)
                masses.append(mass_value)
                layer_values.append(mass_value)
                if page_scores_accum is None:
                    page_scores_accum = [0.0 for _ in range((kv_len + page_tokens - 1) // page_tokens)]
                max_value = logits.max()
                probs = torch.exp(logits - max_value)
                probs = probs / probs.sum().clamp_min(1e-30)
                for page_idx, page_start in enumerate(range(0, kv_len, page_tokens)):
                    page_end = min(kv_len, page_start + page_tokens)
                    page_scores_accum[page_idx] += float(probs[page_start:page_end].sum().item())
            if layer_idx == len(layers) - 1:
                last_values_tensor = torch.tensor(layer_values, dtype=torch.float32)
        if last_values_tensor is None:
            last_values_tensor = torch.tensor([], dtype=torch.float32)
        flat = sorted(masses, reverse=True)
        top_values = flat[:5]
        evidence_page = start // page_tokens
        page_scores = page_scores_accum or []
        ranked_pages = sorted(range(len(page_scores)), key=lambda idx: page_scores[idx], reverse=True)
        evidence_rank = ranked_pages.index(evidence_page) + 1 if evidence_page in ranked_pages else -1
        mass_mean = sum(masses) / max(1, len(masses))
        expected_uniform = max(1, end - start) / max(1, kv_len)
        return {
            "attention_available": 1,
            "attention_seconds": time.perf_counter() - started,
            "mass_mean_all_layers_heads": mass_mean,
            "mass_last_layer_mean_heads": float(last_values_tensor.mean().item()) if last_values_tensor.numel() else 0.0,
            "mass_top_head": max(masses) if masses else 0.0,
            "mass_top5_heads_mean": sum(top_values) / max(1, len(top_values)),
            "normalized_mass": mass_mean / expected_uniform if expected_uniform > 0 else 0.0,
            "evidence_page": evidence_page,
            "evidence_rank_by_page_mass": evidence_rank,
            "page_count": len(page_scores),
        }
    except Exception as exc:
        return {
            "attention_available": 0,
            "attention_error": repr(exc),
            "attention_seconds": 0.0,
            "mass_mean_all_layers_heads": "",
            "mass_last_layer_mean_heads": "",
            "mass_top_head": "",
            "mass_top5_heads_mean": "",
            "normalized_mass": "",
            "evidence_page": start // page_tokens,
            "evidence_rank_by_page_mass": "",
            "page_count": "",
        }


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


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault((int(row["target_length"]), float(row["depth_percent"])), []).append(row)
    summary: list[dict[str, Any]] = []
    for (length, depth), bucket in sorted(buckets.items()):
        n = len(bucket)
        accuracy = sum(int(row["correct"]) for row in bucket) / max(1, n)
        miss_rate = sum(int(row["miss"]) for row in bucket) / max(1, n)
        wrong_rate = sum(int(row["wrong"]) for row in bucket) / max(1, n)
        ppl_values = [float(row["answer_ppl"]) for row in bucket if str(row["answer_ppl"]) not in {"inf", "nan"}]
        mass_values = [
            float(row["mass_mean_all_layers_heads"])
            for row in bucket
            if str(row.get("mass_mean_all_layers_heads", "")) not in {"", "nan"}
        ]
        summary.append(
            {
                "target_length": length,
                "depth_percent": depth,
                "cases": n,
                "accuracy": f"{accuracy:.6f}",
                "miss_rate": f"{miss_rate:.6f}",
                "wrong_rate": f"{wrong_rate:.6f}",
                "mean_answer_ppl": "" if not ppl_values else f"{sum(ppl_values) / len(ppl_values):.6f}",
                "mean_evidence_mass": "" if not mass_values else f"{sum(mass_values) / len(mass_values):.8f}",
            }
        )
    return summary


def write_markdown_summary(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Long Needle Age Summary",
        "",
        "| length | depth | cases | accuracy | miss | wrong | answer PPL | evidence mass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['target_length']} | {row['depth_percent']} | {row['cases']} | "
            f"{row['accuracy']} | {row['miss_rate']} | {row['wrong_rate']} | "
            f"{row['mean_answer_ppl']} | {row['mean_evidence_mass']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3 long needle age retrieval experiment.")
    parser.add_argument("--model_name_or_path", default="/home/fdong/hrj/prove/Qwen3-0.6B")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lengths", default="8192,16384")
    parser.add_argument("--depths", default="50")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--attn_implementation", default="sdpa")
    parser.add_argument("--prefill_chunk_size", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--page_tokens", type=int, default=1024)
    parser.add_argument("--compute_attention", type=str2bool, default=True)
    parser.add_argument("--original_max_position_embeddings", type=int, default=32768)
    parser.add_argument("--needle_prompt_lang", choices=["zh", "en"], default="zh")
    parser.add_argument("--filler_lang", choices=["zh", "en"], default="zh")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[4]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lengths = parse_csv_ints(args.lengths)
    depths = parse_csv_floats(args.depths)
    seeds = parse_csv_ints(args.seeds)
    factors = {rope_factor_for_length(length) for length in lengths}
    non_native_factors = {factor for factor in factors if factor > 1.0}
    if len(non_native_factors) > 1:
        raise ValueError(
            "Do not mix different non-native YaRN factors in one process. "
            f"Requested lengths={lengths} require factors={sorted(non_native_factors)}."
        )
    max_factor = max(factors)
    max_case_position = max(
        max(length + 256, int(32768 * rope_factor_for_length(length))) for length in lengths
    )

    model, tokenizer = load_model_and_tokenizer(args, max_case_position, max_factor)
    device = input_device(model)

    env = {
        "git_commit": git_commit(repo_root),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "model_name_or_path": args.model_name_or_path,
        "effective_config": {
            "max_position_embeddings": getattr(model.config, "max_position_embeddings", None),
            "rope_scaling": getattr(model.config, "rope_scaling", None),
            "rope_theta": getattr(model.config, "rope_theta", None),
            "attn_implementation": getattr(model.config, "_attn_implementation", None),
        },
        "args": vars(args),
    }
    if torch.cuda.is_available():
        env["gpu_name"] = torch.cuda.get_device_name(device)
        env["gpu_count_visible"] = torch.cuda.device_count()
    (output_dir / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for length in lengths:
        for depth in depths:
            for seed in seeds:
                case, prompt_ids, evidence_span = build_case_ids(
                    tokenizer,
                    target_length=length,
                    depth_percent=depth,
                    seed=seed,
                    max_new_tokens=args.max_new_tokens,
                    needle_prompt_lang=args.needle_prompt_lang,
                    filler_lang=args.filler_lang,
                )
                print(f"=== {case.case_id} prompt_tokens={case.prompt_tokens} rope_factor={case.rope_factor} ===", flush=True)
                prompt_prefix = prompt_ids[:, :-1]
                last_prompt_id = prompt_ids[:, -1:]
                base_cache, prefill_seconds = prefill_sequence(model, prompt_prefix, args.prefill_chunk_size)
                answer_text = TEXT_CONFIGS[args.needle_prompt_lang]["answer"]
                ppl = score_answer(model, tokenizer, base_cache, last_prompt_id, case.prompt_tokens - 1, answer_text)
                gen = generate_answer(
                    model,
                    tokenizer,
                    base_cache,
                    last_prompt_id,
                    case.prompt_tokens - 1,
                    args.max_new_tokens,
                    args.needle_prompt_lang,
                )
                if args.compute_attention:
                    attn = attention_mass(
                        model,
                        base_cache,
                        last_prompt_id,
                        case.prompt_tokens - 1,
                        evidence_span,
                        args.page_tokens,
                    )
                else:
                    attn = {"attention_available": 0, "attention_seconds": 0.0}
                row = {
                    **asdict(case),
                    "prefill_seconds": f"{prefill_seconds:.6f}",
                    **ppl,
                    **gen,
                    **attn,
                }
                rows.append(row)
                case_rows.append(asdict(case))
                write_csv(output_dir / "generation_results.csv", rows)
                write_csv(output_dir / "answer_ppl.csv", rows)
                write_csv(output_dir / "evidence_attention_mass.csv", rows)
                summary_rows = summarize(rows)
                write_csv(output_dir / "summary_by_length.csv", summary_rows)
                write_markdown_summary(output_dir / "summary_by_length.md", summary_rows)
                (output_dir / "cases.jsonl").write_text(
                    "\n".join(json.dumps(item, ensure_ascii=False) for item in case_rows) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"done {case.case_id}: class={gen['answer_class']} ppl={ppl['answer_ppl']:.4f} "
                    f"mass={attn.get('mass_mean_all_layers_heads', '')}",
                    flush=True,
                )
                del base_cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    print(f"outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
