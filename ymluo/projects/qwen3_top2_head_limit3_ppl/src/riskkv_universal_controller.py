from __future__ import annotations

from dataclasses import dataclass
from typing import Collection


@dataclass(frozen=True)
class UniversalKVAction:
    name: str
    budget_tokens: int
    sink_tokens: int
    recent_tokens: int
    semantic_tokens: int
    echo_tokens: int
    rebuild_cache: bool
    reason: str


@dataclass(frozen=True)
class UniversalControllerConfig:
    base_budget_tokens: int = 2048
    sink_tokens: int = 32
    base_recent_tokens: int = 1536
    expanded_budget_tokens: int = 2816
    echo_match_tokens: int = 8

    @property
    def semantic_tokens(self) -> int:
        return self.base_budget_tokens - self.sink_tokens - self.base_recent_tokens

    @property
    def expanded_recent_tokens(self) -> int:
        return self.expanded_budget_tokens - self.sink_tokens


def match_is_covered(
    match_start: int | None,
    match_tokens: int,
    keep_indices: Collection[int],
) -> bool:
    if match_start is None:
        return False
    keep = keep_indices if isinstance(keep_indices, set) else set(keep_indices)
    return all(match_start + offset in keep for offset in range(match_tokens))


class UniversalKVController:
    """Causal policy for free continuation with semantic, local, and echo memory."""

    def __init__(self, config: UniversalControllerConfig | None = None) -> None:
        self.config = config or UniversalControllerConfig()
        if self.config.semantic_tokens < 0:
            raise ValueError("base budget is smaller than sink + recent allocation")

    def base_action(self, reason: str = "no_remote_repeat") -> UniversalKVAction:
        cfg = self.config
        return UniversalKVAction(
            name="local_semantic_2k",
            budget_tokens=cfg.base_budget_tokens,
            sink_tokens=cfg.sink_tokens,
            recent_tokens=cfg.base_recent_tokens,
            semantic_tokens=cfg.semantic_tokens,
            echo_tokens=0,
            rebuild_cache=False,
            reason=reason,
        )

    def expanded_recent_action(self) -> UniversalKVAction:
        cfg = self.config
        return UniversalKVAction(
            name="expanded_recent_2p8k",
            budget_tokens=cfg.expanded_budget_tokens,
            sink_tokens=cfg.sink_tokens,
            recent_tokens=cfg.expanded_recent_tokens,
            semantic_tokens=0,
            echo_tokens=0,
            rebuild_cache=True,
            reason="independent_continuity_risk_signal",
        )

    def choose_action(
        self,
        match_start: int | None,
        remote_length: int,
        base_keep_indices: Collection[int],
        stable_match: bool = True,
        continuity_risk: bool = False,
    ) -> UniversalKVAction:
        cfg = self.config
        if match_start is None:
            if continuity_risk:
                return self.expanded_recent_action()
            return self.base_action()
        if not stable_match:
            return self.base_action("unstable_remote_repeat")
        if match_is_covered(match_start, cfg.echo_match_tokens, base_keep_indices):
            return self.base_action("repeat_already_in_local_or_semantic_memory")

        return UniversalKVAction(
            name="recurrence_echo_2k",
            budget_tokens=cfg.base_budget_tokens,
            sink_tokens=cfg.sink_tokens,
            recent_tokens=cfg.base_recent_tokens,
            semantic_tokens=0,
            echo_tokens=cfg.semantic_tokens,
            rebuild_cache=True,
            reason="confirmed_repeat_requires_content_addressed_memory",
        )
