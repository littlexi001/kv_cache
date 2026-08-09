from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CountCappedSupport:
    history_tokens: int
    final_token_count: int
    final_fraction: float
    candidate_fraction: float


def count_capped_support(
    history_tokens: int,
    *,
    base_final_fraction: float = 0.02,
    final_token_cap: int = 1280,
    candidate_multiplier: float = 4.0,
    min_candidate_fraction: float = 0.03,
    max_candidate_fraction: float = 0.06,
) -> CountCappedSupport:
    if history_tokens <= 0:
        raise ValueError("history_tokens must be positive")
    if not 0.0 < base_final_fraction <= 1.0:
        raise ValueError("base_final_fraction must be in (0, 1]")
    if final_token_cap <= 0:
        raise ValueError("final_token_cap must be positive")
    final_token_count = min(
        round(base_final_fraction * history_tokens),
        final_token_cap,
    )
    final_token_count = max(1, final_token_count)
    final_fraction = final_token_count / history_tokens
    candidate_fraction = max(
        min_candidate_fraction,
        min(max_candidate_fraction, candidate_multiplier * final_fraction),
    )
    if candidate_fraction <= final_fraction:
        raise ValueError("candidate fraction must exceed final fraction")
    return CountCappedSupport(
        history_tokens=history_tokens,
        final_token_count=final_token_count,
        final_fraction=final_fraction,
        candidate_fraction=candidate_fraction,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("history_tokens", type=int, nargs="+")
    args = parser.parse_args()
    print(
        json.dumps(
            [asdict(count_capped_support(length)) for length in args.history_tokens],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
