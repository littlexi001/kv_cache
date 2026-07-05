from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    ) -> None:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)
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


def load_router(checkpoint_path: str | Path, **kwargs: Any) -> MemoryPolicyRouter:
    return MemoryPolicyRouter(checkpoint_path, **kwargs)
