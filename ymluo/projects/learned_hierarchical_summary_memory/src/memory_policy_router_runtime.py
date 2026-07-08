from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import math
import re

import torch
import torch.nn as nn


class TinyMemoryRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class RouterPrediction:
    raw_action: str
    action: str
    confidence: float
    probabilities: dict[str, float]


class MemoryPolicyRouter:
    """Tiny inference-time router for task-adaptive memory policy selection."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        conservative_generation_upgrade: str = "summary100",
        checkpoint: dict[str, Any] | None = None,
    ) -> None:
        if checkpoint is None:
            checkpoint = load_checkpoint(checkpoint_path, device)
        self.device = torch.device(device)
        self.label_names: list[str] = list(checkpoint["label_names"])
        self.feature_names: list[str] = list(checkpoint["feature_names"])
        self.mean: list[float] = [float(value) for value in checkpoint["mean"]]
        self.std: list[float] = [float(value) for value in checkpoint["std"]]
        self.conservative_generation_upgrade = conservative_generation_upgrade
        self.model = TinyMemoryRouter(
            input_dim=int(checkpoint["input_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            output_dim=len(self.label_names),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()

    def _normalize(self, features: list[float]) -> torch.Tensor:
        if len(features) != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features, got {len(features)}")
        normalized = [
            (float(value) - self.mean[idx]) / max(self.std[idx], 1e-6)
            for idx, value in enumerate(features)
        ]
        return torch.tensor(normalized, dtype=torch.float32, device=self.device).view(1, -1)

    def predict(self, features: list[float], task_family: str = "") -> RouterPrediction:
        with torch.inference_mode():
            logits = self.model(self._normalize(features))
            probs_tensor = torch.softmax(logits, dim=-1).view(-1).cpu()
        probs = {label: float(probs_tensor[idx]) for idx, label in enumerate(self.label_names)}
        best_idx = int(probs_tensor.argmax())
        raw_action = self.label_names[best_idx]
        action = raw_action
        if (
            task_family == "generation"
            and raw_action == "summary10"
            and self.conservative_generation_upgrade
        ):
            action = self.conservative_generation_upgrade
        return RouterPrediction(
            raw_action=raw_action,
            action=action,
            confidence=float(probs_tensor[best_idx]),
            probabilities=probs,
        )


def load_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> dict[str, Any]:
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def parse_block_action(action: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"recent_plus_b(\d+)_span_top(\d+)_b0_a0", action)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def action_estimate_features(action: str, case_features: list[float]) -> list[float]:
    prefix_tokens = max(1.0, float(case_features[12]))
    older_tokens = max(0.0, float(case_features[13]))
    recent_tokens = max(0.0, float(case_features[14]))
    if action == "full_raw":
        selected = prefix_tokens
        return [
            1.0,
            older_tokens,
            max(1.0, math.ceil(older_tokens / 512.0)),
            math.log2(max(2.0, older_tokens)),
            math.log2(max(2.0, older_tokens / 512.0)),
            selected,
            1.0,
            1.0,
            1.0,
        ]
    parsed = parse_block_action(action)
    if parsed is None:
        return [0.0] * 9
    block_tokens, top_k = parsed
    old_blocks = max(1.0, math.ceil(older_tokens / max(1, block_tokens)))
    selected_old = min(older_tokens, float(block_tokens * min(top_k, old_blocks)))
    selected = recent_tokens + selected_old
    return [
        0.0,
        float(block_tokens),
        float(top_k),
        math.log2(float(block_tokens)),
        math.log2(float(max(1, top_k))),
        selected,
        min(1.0, selected / prefix_tokens),
        min(1.0, selected_old / max(1.0, older_tokens)),
        min(1.0, float(block_tokens) / max(1.0, older_tokens)),
    ]


class BlocksizeRiskRouter:
    """Block-size router with a learned action head plus learned danger gating."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        checkpoint: dict[str, Any] | None = None,
        **_: Any,
    ) -> None:
        if checkpoint is None:
            checkpoint = load_checkpoint(checkpoint_path, device)
        self.device = torch.device(device)
        self.label_names: list[str] = list(checkpoint["label_names"])
        self.candidate_actions: list[str] = list(checkpoint.get("candidate_actions", self.label_names))
        self.feature_names: list[str] = list(checkpoint["feature_names"])
        self.action_feature_names: list[str] = list(checkpoint.get("action_feature_names", []))
        self.mean: list[float] = [float(value) for value in checkpoint["mean"]]
        self.std: list[float] = [float(value) for value in checkpoint["std"]]
        self.danger_mean: list[float] = [float(value) for value in checkpoint["danger_mean"]]
        self.danger_std: list[float] = [float(value) for value in checkpoint["danger_std"]]
        self.risk_threshold = float(checkpoint.get("risk_threshold", 0.35))
        hidden_dim = int(checkpoint["hidden_dim"])
        self.action_model = TinyMemoryRouter(
            input_dim=int(checkpoint.get("action_input_dim", checkpoint["input_dim"])),
            hidden_dim=hidden_dim,
            output_dim=len(self.label_names),
        ).to(self.device)
        self.action_model.load_state_dict(checkpoint["action_state_dict"])
        self.action_model.eval()
        self.danger_model = TinyMemoryRouter(
            input_dim=int(checkpoint["danger_input_dim"]),
            hidden_dim=hidden_dim,
            output_dim=1,
        ).to(self.device)
        self.danger_model.load_state_dict(checkpoint["danger_state_dict"])
        self.danger_model.eval()

    def _normalize_case(self, features: list[float]) -> torch.Tensor:
        if len(features) != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features, got {len(features)}")
        normalized = [
            (float(value) - self.mean[idx]) / max(self.std[idx], 1e-6)
            for idx, value in enumerate(features)
        ]
        return torch.tensor(normalized, dtype=torch.float32, device=self.device).view(1, -1)

    def _normalize_action(self, values: list[float]) -> torch.Tensor:
        normalized = [
            (float(value) - self.danger_mean[idx]) / max(self.danger_std[idx], 1e-6)
            for idx, value in enumerate(values)
        ]
        return torch.tensor(normalized, dtype=torch.float32, device=self.device).view(1, -1)

    def danger_prob(self, features: list[float], action: str) -> float:
        values = list(features) + action_estimate_features(action, features)
        with torch.inference_mode():
            return float(torch.sigmoid(self.danger_model(self._normalize_action(values))).item())

    def _ordered_candidates(self, features: list[float]) -> list[str]:
        return sorted(
            self.candidate_actions,
            key=lambda action: (
                action_estimate_features(action, features)[6],
                action_estimate_features(action, features)[1],
                action_estimate_features(action, features)[2],
                action,
            ),
        )

    def predict(self, features: list[float], task_family: str = "") -> RouterPrediction:
        with torch.inference_mode():
            logits = self.action_model(self._normalize_case(features))
            probs_tensor = torch.softmax(logits, dim=-1).view(-1).cpu()
        probs = {label: float(probs_tensor[idx]) for idx, label in enumerate(self.label_names)}
        best_idx = int(probs_tensor.argmax())
        raw_action = self.label_names[best_idx]
        action = raw_action
        raw_risk = self.danger_prob(features, raw_action)
        if raw_risk >= self.risk_threshold:
            raw_ratio = action_estimate_features(raw_action, features)[6]
            for candidate in self._ordered_candidates(features):
                if action_estimate_features(candidate, features)[6] + 1e-12 < raw_ratio:
                    continue
                if self.danger_prob(features, candidate) < self.risk_threshold:
                    action = candidate
                    break
            else:
                action = "full_raw"
        return RouterPrediction(
            raw_action=raw_action,
            action=action,
            confidence=float(probs_tensor[best_idx]),
            probabilities=probs,
        )


def load_router(checkpoint_path: str | Path, **kwargs: Any) -> MemoryPolicyRouter:
    device = str(kwargs.get("device", "cpu"))
    checkpoint = load_checkpoint(checkpoint_path, device)
    if checkpoint.get("router_kind") == "blocksize_risk_router":
        return BlocksizeRiskRouter(checkpoint_path, checkpoint=checkpoint, **kwargs)
    return MemoryPolicyRouter(checkpoint_path, checkpoint=checkpoint, **kwargs)
