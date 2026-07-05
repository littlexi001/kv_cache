from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


THEMES = ["betrayal", "redemption", "survival", "ambition", "loyalty", "exile", "discovery", "justice"]
SETTINGS = ["harbor", "citadel", "forest", "desert", "archive", "island", "frontier", "monastery"]
CONFLICTS = ["invasion", "trial", "plague", "succession", "sabotage", "siege", "expedition", "uprising"]
OUTCOMES = ["reconciled", "escaped", "defeated", "restored", "hidden", "exposed", "rescued", "banished"]
CHARACTERS = [
    "Arin",
    "Bel",
    "Cato",
    "Dina",
    "Eron",
    "Faye",
    "Galen",
    "Hale",
    "Iris",
    "Juno",
    "Kara",
    "Lio",
    "Mira",
    "Nox",
    "Oren",
    "Pia",
]
ROLES = ["captain", "scout", "healer", "rival", "mentor", "scribe", "guard", "witness"]
ACTIONS = ["warns", "betrays", "rescues", "hides", "decodes", "guards", "negotiates", "repairs"]
RESULTS = ["victory", "delay", "capture", "escape", "alliance", "exile", "reversal", "truce"]

NONE_ROLE = 0
TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+")
MEMORY_LEVELS = ("summary10", "summary100", "summary1000", "raw")


@dataclass(frozen=True)
class Config:
    output_dir: str
    train_books: int
    test_books: int
    blocks_per_book: int
    event_slots: int
    active_characters: int
    max_vocab: int
    hidden_dim: int
    epochs: int
    batch_size: int
    lr: float
    seed: int
    raw_block_tokens: int
    summary10_tokens: int
    summary100_tokens: int
    summary1000_tokens: int
    examples: int


@dataclass(frozen=True)
class Block:
    book_id: int
    block_id: int
    theme: int
    setting: int
    conflict: int
    outcome: int
    roles: tuple[int, ...]
    event_actions: tuple[int, ...]
    event_results: tuple[int, ...]
    event_codes: tuple[str, ...]
    raw_text: str


@dataclass(frozen=True)
class Task:
    task_id: str
    kind: str
    book_id: int
    block_id: int | None
    answer: str
    character_id: int | None = None
    event_slot: int | None = None


@dataclass(frozen=True)
class BlockMemory:
    block: Block
    theme: int
    setting: int
    conflict: int
    outcome: int
    roles: tuple[int, ...]
    event_actions: tuple[int, ...]
    event_results: tuple[int, ...]
    is_gold: bool

    def summary10_text(self) -> str:
        return (
            f"theme={THEMES[self.theme]} setting={SETTINGS[self.setting]} "
            f"conflict={CONFLICTS[self.conflict]}"
        )

    def summary100_text(self) -> str:
        active = [
            f"{CHARACTERS[idx]}:{ROLES[role - 1]}"
            for idx, role in enumerate(self.roles)
            if role != NONE_ROLE
        ]
        return f"{self.summary10_text()} outcome={OUTCOMES[self.outcome]} roles={' '.join(active)}"

    def summary1000_text(self) -> str:
        events = []
        for slot, (action, result) in enumerate(zip(self.event_actions, self.event_results)):
            events.append(f"E{slot:02d}:{ACTIONS[action]}->{RESULTS[result]}")
        return f"{self.summary100_text()} events={' '.join(events)}"


@dataclass
class TrialResult:
    method: str
    task_id: str
    kind: str
    answer: str
    prediction: str
    correct: bool
    token_cost: int
    raw_cost: int
    memory_level: str


class SummarizerNet(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, event_slots: int) -> None:
        super().__init__()
        self.event_slots = event_slots
        self.global_encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.role_encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.event_encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.theme = nn.Linear(hidden_dim, len(THEMES))
        self.setting = nn.Linear(hidden_dim, len(SETTINGS))
        self.conflict = nn.Linear(hidden_dim, len(CONFLICTS))
        self.outcome = nn.Linear(hidden_dim, len(OUTCOMES))
        self.roles = nn.Linear(hidden_dim, len(ROLES) + 1)
        self.event_actions = nn.Linear(hidden_dim, len(ACTIONS))
        self.event_results = nn.Linear(hidden_dim, len(RESULTS))

    def forward(self, global_x: torch.Tensor, role_x: torch.Tensor, event_x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.global_encoder(global_x)
        batch_size, character_count, vocab_size = role_x.shape
        _, event_count, _ = event_x.shape
        role_h = self.role_encoder(role_x.reshape(batch_size * character_count, vocab_size))
        event_h = self.event_encoder(event_x.reshape(batch_size * event_count, vocab_size))
        return {
            "theme": self.theme(h),
            "setting": self.setting(h),
            "conflict": self.conflict(h),
            "outcome": self.outcome(h),
            "roles": self.roles(role_h).view(batch_size, character_count, len(ROLES) + 1),
            "event_actions": self.event_actions(event_h).view(batch_size, event_count, len(ACTIONS)),
            "event_results": self.event_results(event_h).view(batch_size, event_count, len(RESULTS)),
        }


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Learned hierarchical summary memory synthetic experiment.")
    parser.add_argument("--output_dir", default="ymluo/projects/learned_hierarchical_summary_memory/outputs/default")
    parser.add_argument("--train_books", type=int, default=160)
    parser.add_argument("--test_books", type=int, default=40)
    parser.add_argument("--blocks_per_book", type=int, default=10)
    parser.add_argument("--event_slots", type=int, default=16)
    parser.add_argument("--active_characters", type=int, default=4)
    parser.add_argument("--max_vocab", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=160)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=2026070302)
    parser.add_argument("--raw_block_tokens", type=int, default=10_000)
    parser.add_argument("--summary10_tokens", type=int, default=10)
    parser.add_argument("--summary100_tokens", type=int, default=100)
    parser.add_argument("--summary1000_tokens", type=int, default=1_000)
    parser.add_argument("--examples", type=int, default=32)
    return Config(**vars(parser.parse_args()))


def words(text: str) -> list[str]:
    return [item.lower() for item in TOKEN_RE.findall(text)]


def build_block(rng: random.Random, book_id: int, block_pos: int, event_slots: int, active_characters: int) -> Block:
    theme = rng.randrange(len(THEMES))
    setting = rng.randrange(len(SETTINGS))
    conflict = rng.randrange(len(CONFLICTS))
    outcome = rng.randrange(len(OUTCOMES))

    roles = [NONE_ROLE] * len(CHARACTERS)
    active = rng.sample(range(len(CHARACTERS)), active_characters)
    for character_id in active:
        roles[character_id] = 1 + rng.randrange(len(ROLES))

    event_actions = tuple(rng.randrange(len(ACTIONS)) for _ in range(event_slots))
    event_results = tuple(rng.randrange(len(RESULTS)) for _ in range(event_slots))
    event_codes = tuple(f"CODE-{book_id:03d}-{block_pos:03d}-{slot:02d}-{rng.randrange(100000, 999999)}" for slot in range(event_slots))

    lines = [
        (
            f"MEMORY10 THEME_{THEMES[theme]} SETTING_{SETTINGS[setting]} "
            f"CONFLICT_{CONFLICTS[conflict]}."
        ),
        (
            f"Book {book_id} block {block_pos}. The chapter theme is {THEMES[theme]}. "
            f"The setting is {SETTINGS[setting]}. The central conflict is {CONFLICTS[conflict]}. "
            f"The final outcome is {OUTCOMES[outcome]}."
        ),
        f"MEMORY100 OUTCOME_{OUTCOMES[outcome]}.",
    ]
    for character_id in active:
        role_name = ROLES[roles[character_id] - 1]
        lines.append(f"MEMORY100 CHARACTER_{CHARACTERS[character_id]} ROLE_{role_name}.")
        lines.append(f"Character {CHARACTERS[character_id]} has role {role_name} in this block.")
    for slot, (action, result, code) in enumerate(zip(event_actions, event_results, event_codes)):
        actor = CHARACTERS[rng.choice(active)]
        lines.append(f"MEMORY1000 EVENT_{slot:02d}_ACTION_{ACTIONS[action]} EVENT_{slot:02d}_RESULT_{RESULTS[result]}.")
        lines.append(
            f"Event E{slot:02d}: {actor} {ACTIONS[action]} the plan, causing {RESULTS[result]}. "
            f"The exact verification code is {code}."
        )
    for noise_id in range(24):
        lines.append(
            f"Background note {noise_id}: archive dust candle road market weather page margin "
            f"{rng.choice(THEMES)} {rng.choice(SETTINGS)} filler."
        )
    rng.shuffle(lines)
    return Block(
        book_id=book_id,
        block_id=book_id * 1000 + block_pos,
        theme=theme,
        setting=setting,
        conflict=conflict,
        outcome=outcome,
        roles=tuple(roles),
        event_actions=event_actions,
        event_results=event_results,
        event_codes=event_codes,
        raw_text="\n".join(lines),
    )


def build_books(config: Config) -> dict[int, list[Block]]:
    rng = random.Random(config.seed)
    books: dict[int, list[Block]] = {}
    for book_id in range(config.train_books + config.test_books):
        books[book_id] = [
            build_block(rng, book_id, block_pos, config.event_slots, config.active_characters)
            for block_pos in range(config.blocks_per_book)
        ]
    return books


def flatten_books(books: dict[int, list[Block]], book_ids: list[int]) -> list[Block]:
    return [block for book_id in book_ids for block in books[book_id]]


def build_vocab(blocks: list[Block], max_vocab: int) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for block in blocks:
        counts.update(words(block.raw_text))
    most_common = counts.most_common(max_vocab)
    return {word: idx for idx, (word, _) in enumerate(most_common)}


def vectorize_blocks(blocks: list[Block], vocab: dict[str, int]) -> torch.Tensor:
    x = torch.zeros((len(blocks), len(vocab)), dtype=torch.float32)
    for row, block in enumerate(blocks):
        counts = Counter(words(block.raw_text))
        total = 0.0
        for word, count in counts.items():
            idx = vocab.get(word)
            if idx is not None:
                value = math.log1p(float(count))
                x[row, idx] = value
                total += value * value
        norm = math.sqrt(total)
        if norm > 0:
            x[row] /= norm
    return x


def evidence_lines(text: str, needles: list[str]) -> str:
    lowered_needles = [needle.lower() for needle in needles]
    selected = []
    for line in text.splitlines():
        lowered = line.lower()
        if any(needle in lowered for needle in lowered_needles):
            selected.append(line)
    return "\n".join(selected)


def role_segment(block: Block, character_id: int) -> str:
    name = CHARACTERS[character_id]
    segment = evidence_lines(block.raw_text, [f"CHARACTER_{name}", f"Character {name}"])
    return segment if segment else f"CHARACTER_{name} ABSENT"


def event_segment(block: Block, slot: int) -> str:
    segment = evidence_lines(block.raw_text, [f"EVENT_{slot:02d}_", f"Event E{slot:02d}"])
    return segment if segment else f"EVENT_{slot:02d} ABSENT"


def vectorize_texts(texts: list[str], vocab: dict[str, int]) -> torch.Tensor:
    x = torch.zeros((len(texts), len(vocab)), dtype=torch.float32)
    for row, text in enumerate(texts):
        counts = Counter(words(text))
        total = 0.0
        for word, count in counts.items():
            idx = vocab.get(word)
            if idx is not None:
                value = math.log1p(float(count))
                x[row, idx] = value
                total += value * value
        norm = math.sqrt(total)
        if norm > 0:
            x[row] /= norm
    return x


def vectorize_role_segments(blocks: list[Block], vocab: dict[str, int]) -> torch.Tensor:
    texts = [role_segment(block, character_id) for block in blocks for character_id in range(len(CHARACTERS))]
    x = vectorize_texts(texts, vocab)
    return x.view(len(blocks), len(CHARACTERS), len(vocab))


def vectorize_event_segments(blocks: list[Block], vocab: dict[str, int], event_slots: int) -> torch.Tensor:
    texts = [event_segment(block, slot) for block in blocks for slot in range(event_slots)]
    x = vectorize_texts(texts, vocab)
    return x.view(len(blocks), event_slots, len(vocab))


def targets_for_blocks(blocks: list[Block]) -> dict[str, torch.Tensor]:
    return {
        "theme": torch.tensor([block.theme for block in blocks], dtype=torch.long),
        "setting": torch.tensor([block.setting for block in blocks], dtype=torch.long),
        "conflict": torch.tensor([block.conflict for block in blocks], dtype=torch.long),
        "outcome": torch.tensor([block.outcome for block in blocks], dtype=torch.long),
        "roles": torch.tensor([block.roles for block in blocks], dtype=torch.long),
        "event_actions": torch.tensor([block.event_actions for block in blocks], dtype=torch.long),
        "event_results": torch.tensor([block.event_results for block in blocks], dtype=torch.long),
    }


def model_loss(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> torch.Tensor:
    loss = (
        F.cross_entropy(outputs["theme"], targets["theme"])
        + F.cross_entropy(outputs["setting"], targets["setting"])
        + F.cross_entropy(outputs["conflict"], targets["conflict"])
        + F.cross_entropy(outputs["outcome"], targets["outcome"])
    )
    role_weights = torch.ones(len(ROLES) + 1, device=outputs["roles"].device)
    role_weights[NONE_ROLE] = 0.25
    role_loss = F.cross_entropy(
        outputs["roles"].reshape(-1, len(ROLES) + 1),
        targets["roles"].reshape(-1),
        weight=role_weights,
    )
    action_loss = F.cross_entropy(
        outputs["event_actions"].reshape(-1, len(ACTIONS)),
        targets["event_actions"].reshape(-1),
    )
    result_loss = F.cross_entropy(
        outputs["event_results"].reshape(-1, len(RESULTS)),
        targets["event_results"].reshape(-1),
    )
    return loss + 3.0 * role_loss + 4.0 * action_loss + 4.0 * result_loss


def train_model(
    config: Config,
    train_global_x: torch.Tensor,
    train_role_x: torch.Tensor,
    train_event_x: torch.Tensor,
    train_targets: dict[str, torch.Tensor],
) -> tuple[SummarizerNet, list[dict[str, float]]]:
    torch.manual_seed(config.seed)
    model = SummarizerNet(train_global_x.shape[1], config.hidden_dim, config.event_slots)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    rows = train_global_x.shape[0]
    generator = torch.Generator().manual_seed(config.seed)
    for epoch in range(1, config.epochs + 1):
        perm = torch.randperm(rows, generator=generator)
        total_loss = 0.0
        total_seen = 0
        model.train()
        for start in range(0, rows, config.batch_size):
            idx = perm[start : start + config.batch_size]
            batch_global_x = train_global_x.index_select(0, idx)
            batch_role_x = train_role_x.index_select(0, idx)
            batch_event_x = train_event_x.index_select(0, idx)
            batch_targets = {key: value.index_select(0, idx) for key, value in train_targets.items()}
            optimizer.zero_grad(set_to_none=True)
            loss = model_loss(model(batch_global_x, batch_role_x, batch_event_x), batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch_global_x.shape[0]
            total_seen += batch_global_x.shape[0]
        history.append({"epoch": float(epoch), "loss": total_loss / max(1, total_seen)})
    return model, history


@torch.inference_mode()
def predict_memories(
    model: SummarizerNet,
    blocks: list[Block],
    global_x: torch.Tensor,
    role_x: torch.Tensor,
    event_x: torch.Tensor,
) -> dict[int, BlockMemory]:
    model.eval()
    out = model(global_x, role_x, event_x)
    theme = out["theme"].argmax(dim=-1).tolist()
    setting = out["setting"].argmax(dim=-1).tolist()
    conflict = out["conflict"].argmax(dim=-1).tolist()
    outcome = out["outcome"].argmax(dim=-1).tolist()
    roles = out["roles"].argmax(dim=-1).tolist()
    actions = out["event_actions"].argmax(dim=-1).tolist()
    results = out["event_results"].argmax(dim=-1).tolist()
    memories: dict[int, BlockMemory] = {}
    for idx, block in enumerate(blocks):
        memories[block.block_id] = BlockMemory(
            block=block,
            theme=int(theme[idx]),
            setting=int(setting[idx]),
            conflict=int(conflict[idx]),
            outcome=int(outcome[idx]),
            roles=tuple(int(item) for item in roles[idx]),
            event_actions=tuple(int(item) for item in actions[idx]),
            event_results=tuple(int(item) for item in results[idx]),
            is_gold=False,
        )
    return memories


def gold_memories(blocks: list[Block]) -> dict[int, BlockMemory]:
    return {
        block.block_id: BlockMemory(
            block=block,
            theme=block.theme,
            setting=block.setting,
            conflict=block.conflict,
            outcome=block.outcome,
            roles=block.roles,
            event_actions=block.event_actions,
            event_results=block.event_results,
            is_gold=True,
        )
        for block in blocks
    }


def component_metrics(blocks: list[Block], memories: dict[int, BlockMemory]) -> dict[str, float]:
    total = len(blocks)
    active_role_total = 0
    active_role_correct = 0
    action_correct = 0
    result_correct = 0
    event_total = 0
    metrics = {
        "theme_acc": 0.0,
        "setting_acc": 0.0,
        "conflict_acc": 0.0,
        "outcome_acc": 0.0,
    }
    for block in blocks:
        memory = memories[block.block_id]
        metrics["theme_acc"] += float(memory.theme == block.theme)
        metrics["setting_acc"] += float(memory.setting == block.setting)
        metrics["conflict_acc"] += float(memory.conflict == block.conflict)
        metrics["outcome_acc"] += float(memory.outcome == block.outcome)
        for char_id, role in enumerate(block.roles):
            if role != NONE_ROLE:
                active_role_total += 1
                active_role_correct += int(memory.roles[char_id] == role)
        for slot in range(len(block.event_actions)):
            event_total += 1
            action_correct += int(memory.event_actions[slot] == block.event_actions[slot])
            result_correct += int(memory.event_results[slot] == block.event_results[slot])
    for key in list(metrics):
        metrics[key] /= max(1, total)
    metrics["active_role_acc"] = active_role_correct / max(1, active_role_total)
    metrics["event_action_acc"] = action_correct / max(1, event_total)
    metrics["event_result_acc"] = result_correct / max(1, event_total)
    return metrics


def make_tasks(config: Config, books: dict[int, list[Block]], test_book_ids: list[int]) -> list[Task]:
    rng = random.Random(config.seed + 991)
    tasks: list[Task] = []
    for book_id in test_book_ids:
        blocks = books[book_id]
        theme_counts = Counter(block.theme for block in blocks)
        dominant_theme = theme_counts.most_common(1)[0][0]
        tasks.append(Task(f"book-theme-{book_id}", "book_theme", book_id, None, THEMES[dominant_theme]))
        for local_idx in range(2):
            block = rng.choice(blocks)
            tasks.append(Task(f"block-theme-{book_id}-{local_idx}", "block_theme", book_id, block.block_id, THEMES[block.theme]))
            tasks.append(Task(f"block-conflict-{book_id}-{local_idx}", "block_conflict", book_id, block.block_id, CONFLICTS[block.conflict]))
            tasks.append(Task(f"block-outcome-{book_id}-{local_idx}", "block_outcome", book_id, block.block_id, OUTCOMES[block.outcome]))
            active_chars = [idx for idx, role in enumerate(block.roles) if role != NONE_ROLE]
            character_id = rng.choice(active_chars)
            tasks.append(
                Task(
                    f"role-{book_id}-{local_idx}",
                    "character_role",
                    book_id,
                    block.block_id,
                    ROLES[block.roles[character_id] - 1],
                    character_id=character_id,
                )
            )
            slot = rng.randrange(config.event_slots)
            tasks.append(
                Task(
                    f"event-action-{book_id}-{local_idx}",
                    "event_action",
                    book_id,
                    block.block_id,
                    ACTIONS[block.event_actions[slot]],
                    event_slot=slot,
                )
            )
            slot = rng.randrange(config.event_slots)
            tasks.append(
                Task(
                    f"event-result-{book_id}-{local_idx}",
                    "event_result",
                    book_id,
                    block.block_id,
                    RESULTS[block.event_results[slot]],
                    event_slot=slot,
                )
            )
            slot = rng.randrange(config.event_slots)
            tasks.append(
                Task(
                    f"exact-code-{book_id}-{local_idx}",
                    "exact_code",
                    book_id,
                    block.block_id,
                    block.event_codes[slot],
                    event_slot=slot,
                )
            )
    rng.shuffle(tasks)
    return tasks


def answer_from_memory(task: Task, memory: BlockMemory, level: str) -> str:
    if level in {"summary10", "summary100", "summary1000"}:
        if task.kind == "block_theme":
            return THEMES[memory.theme]
        if task.kind == "block_conflict":
            return CONFLICTS[memory.conflict]
    if level in {"summary100", "summary1000"}:
        if task.kind == "block_outcome":
            return OUTCOMES[memory.outcome]
        if task.kind == "character_role" and task.character_id is not None:
            role = memory.roles[task.character_id]
            return ROLES[role - 1] if role != NONE_ROLE else ""
    if level == "summary1000":
        if task.kind == "event_action" and task.event_slot is not None:
            return ACTIONS[memory.event_actions[task.event_slot]]
        if task.kind == "event_result" and task.event_slot is not None:
            return RESULTS[memory.event_results[task.event_slot]]
    return ""


def answer_book_theme(task: Task, books: dict[int, list[Block]], memories: dict[int, BlockMemory]) -> str:
    themes = [memories[block.block_id].theme for block in books[task.book_id]]
    return THEMES[Counter(themes).most_common(1)[0][0]]


def raw_answer(task: Task, books: dict[int, list[Block]]) -> str:
    if task.kind == "book_theme":
        themes = [block.theme for block in books[task.book_id]]
        return THEMES[Counter(themes).most_common(1)[0][0]]
    block = next(block for block in books[task.book_id] if block.block_id == task.block_id)
    if task.kind == "block_theme":
        return THEMES[block.theme]
    if task.kind == "block_conflict":
        return CONFLICTS[block.conflict]
    if task.kind == "block_outcome":
        return OUTCOMES[block.outcome]
    if task.kind == "character_role" and task.character_id is not None:
        return ROLES[block.roles[task.character_id] - 1]
    if task.kind == "event_action" and task.event_slot is not None:
        return ACTIONS[block.event_actions[task.event_slot]]
    if task.kind == "event_result" and task.event_slot is not None:
        return RESULTS[block.event_results[task.event_slot]]
    if task.kind == "exact_code" and task.event_slot is not None:
        return block.event_codes[task.event_slot]
    return ""


def raw_cost(task: Task, config: Config) -> int:
    if task.kind == "book_theme":
        return config.blocks_per_book * config.raw_block_tokens
    return config.raw_block_tokens


def memory_cost(task: Task, config: Config, level: str) -> int:
    costs = {
        "summary10": config.summary10_tokens,
        "summary100": config.summary100_tokens,
        "summary1000": config.summary1000_tokens,
        "raw": config.raw_block_tokens,
    }
    base = costs[level]
    if task.kind == "book_theme" and level != "raw":
        return config.blocks_per_book * base
    return base


def adaptive_level(task: Task, allow_raw: bool) -> str:
    if task.kind in {"book_theme", "block_theme", "block_conflict"}:
        return "summary10"
    if task.kind in {"block_outcome", "character_role"}:
        return "summary100"
    if task.kind in {"event_action", "event_result"}:
        return "summary1000"
    if task.kind == "exact_code" and allow_raw:
        return "raw"
    return "summary1000"


def evaluate_method(
    method: str,
    tasks: list[Task],
    books: dict[int, list[Block]],
    memories: dict[int, BlockMemory],
    config: Config,
    fixed_level: str | None = None,
    allow_raw: bool = False,
) -> list[TrialResult]:
    rows: list[TrialResult] = []
    for task in tasks:
        if method == "full_raw":
            prediction = raw_answer(task, books)
            level = "raw"
            cost = raw_cost(task, config)
        else:
            level = fixed_level or adaptive_level(task, allow_raw)
            if level == "raw":
                prediction = raw_answer(task, books)
            elif task.kind == "book_theme":
                prediction = answer_book_theme(task, books, memories)
            elif task.block_id is not None:
                prediction = answer_from_memory(task, memories[task.block_id], level)
            else:
                prediction = ""
            cost = memory_cost(task, config, level)
        rows.append(
            TrialResult(
                method=method,
                task_id=task.task_id,
                kind=task.kind,
                answer=task.answer,
                prediction=prediction,
                correct=prediction == task.answer,
                token_cost=cost,
                raw_cost=raw_cost(task, config),
                memory_level=level,
            )
        )
    return rows


def summarize(rows: list[TrialResult]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[TrialResult]] = defaultdict(list)
    grouped_kind: dict[tuple[str, str], list[TrialResult]] = defaultdict(list)
    for row in rows:
        grouped[row.method].append(row)
        grouped_kind[(row.method, row.kind)].append(row)

    def one(method: str, items: list[TrialResult]) -> dict[str, Any]:
        total = len(items)
        avg_cost = sum(item.token_cost for item in items) / max(1, total)
        avg_raw = sum(item.raw_cost for item in items) / max(1, total)
        return {
            "method": method,
            "tasks": total,
            "accuracy": sum(item.correct for item in items) / max(1, total),
            "avg_token_cost": avg_cost,
            "avg_raw_cost": avg_raw,
            "cost_ratio_vs_raw": avg_cost / max(1.0, avg_raw),
        }

    summary = [one(method, items) for method, items in sorted(grouped.items())]
    by_kind = []
    for (method, kind), items in sorted(grouped_kind.items()):
        row = one(method, items)
        row["kind"] = kind
        by_kind.append(row)
    return summary, by_kind


def route_mix(rows: list[TrialResult]) -> list[dict[str, Any]]:
    grouped: dict[str, list[TrialResult]] = defaultdict(list)
    for row in rows:
        grouped[row.method].append(row)

    mixes: list[dict[str, Any]] = []
    for method, items in sorted(grouped.items()):
        total = len(items)
        counts = Counter(row.memory_level for row in items)
        mix: dict[str, Any] = {"method": method, "tasks": total}
        for level in MEMORY_LEVELS:
            name = "full_attention" if level == "raw" else level
            mix[f"{name}_count"] = counts[level]
            mix[f"{name}_ratio"] = counts[level] / max(1, total)
        mixes.append(mix)
    return mixes


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config = parse_args()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    books = build_books(config)
    train_book_ids = list(range(config.train_books))
    test_book_ids = list(range(config.train_books, config.train_books + config.test_books))
    train_blocks = flatten_books(books, train_book_ids)
    test_blocks = flatten_books(books, test_book_ids)

    vocab = build_vocab(train_blocks, config.max_vocab)
    train_global_x = vectorize_blocks(train_blocks, vocab)
    test_global_x = vectorize_blocks(test_blocks, vocab)
    train_role_x = vectorize_role_segments(train_blocks, vocab)
    test_role_x = vectorize_role_segments(test_blocks, vocab)
    train_event_x = vectorize_event_segments(train_blocks, vocab, config.event_slots)
    test_event_x = vectorize_event_segments(test_blocks, vocab, config.event_slots)
    train_targets = targets_for_blocks(train_blocks)

    model, train_history = train_model(config, train_global_x, train_role_x, train_event_x, train_targets)
    learned_memories = predict_memories(model, test_blocks, test_global_x, test_role_x, test_event_x)
    gold_memory = gold_memories(test_blocks)
    tasks = make_tasks(config, books, test_book_ids)

    all_rows: list[TrialResult] = []
    all_rows.extend(evaluate_method("full_raw", tasks, books, gold_memory, config))
    all_rows.extend(evaluate_method("gold_adaptive_no_raw", tasks, books, gold_memory, config, allow_raw=False))
    all_rows.extend(evaluate_method("gold_adaptive_with_raw", tasks, books, gold_memory, config, allow_raw=True))
    all_rows.extend(evaluate_method("learned_summary10_only", tasks, books, learned_memories, config, fixed_level="summary10"))
    all_rows.extend(evaluate_method("learned_summary100_only", tasks, books, learned_memories, config, fixed_level="summary100"))
    all_rows.extend(evaluate_method("learned_summary1000_only", tasks, books, learned_memories, config, fixed_level="summary1000"))
    all_rows.extend(evaluate_method("learned_adaptive_no_raw", tasks, books, learned_memories, config, allow_raw=False))
    all_rows.extend(evaluate_method("learned_adaptive_with_raw", tasks, books, learned_memories, config, allow_raw=True))

    summary, by_kind = summarize(all_rows)
    mixes = route_mix(all_rows)
    write_csv(output_dir / "summary.csv", summary)
    write_csv(output_dir / "by_kind.csv", by_kind)
    write_csv(output_dir / "route_mix.csv", mixes)
    write_csv(output_dir / "trials.csv", [asdict(row) for row in all_rows])

    metrics = {
        "config": asdict(config),
        "train_blocks": len(train_blocks),
        "test_blocks": len(test_blocks),
        "vocab_size": len(vocab),
        "summarizer_component_metrics": component_metrics(test_blocks, learned_memories),
        "summary": summary,
        "by_kind": by_kind,
        "route_mix": mixes,
        "train_history_tail": train_history[-10:],
    }
    (output_dir / "summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    examples = [row for row in all_rows if row.method == "learned_adaptive_with_raw"][: config.examples]
    with (output_dir / "examples.jsonl").open("w", encoding="utf-8") as handle:
        for row in examples:
            payload = asdict(row)
            if row.memory_level != "raw" and row.task_id:
                task = next(task for task in tasks if task.task_id == row.task_id)
                if task.block_id is not None:
                    memory = learned_memories[task.block_id]
                    if row.memory_level == "summary10":
                        payload["memory_text"] = memory.summary10_text()
                    elif row.memory_level == "summary100":
                        payload["memory_text"] = memory.summary100_text()
                    elif row.memory_level == "summary1000":
                        payload["memory_text"] = memory.summary1000_text()
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    print("method,tasks,accuracy,avg_token_cost,cost_ratio_vs_raw")
    for row in summary:
        print(
            f"{row['method']},{row['tasks']},{row['accuracy']:.4f},"
            f"{row['avg_token_cost']:.1f},{row['cost_ratio_vs_raw']:.4f}"
        )
    print(f"wrote outputs to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
