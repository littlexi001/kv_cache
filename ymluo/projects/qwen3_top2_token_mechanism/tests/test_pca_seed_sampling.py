from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate_pca_coselection_hybrid import select_seeds  # noqa: E402


def test_seed_sampling_is_unique_and_reproducible() -> None:
    scores = torch.linspace(-2.0, 2.0, 100)
    ranked = torch.argsort(scores, descending=True)
    first_generator = torch.Generator().manual_seed(17)
    second_generator = torch.Generator().manual_seed(17)

    first = select_seeds(
        "score_t1",
        scores,
        ranked,
        12,
        generator=first_generator,
        hybrid_top_fraction=0.7,
        exploration_fraction=0.2,
        band_count=40,
    )
    second = select_seeds(
        "score_t1",
        scores,
        ranked,
        12,
        generator=second_generator,
        hybrid_top_fraction=0.7,
        exploration_fraction=0.2,
        band_count=40,
    )

    assert torch.equal(first, second)
    assert torch.unique(first).numel() == 12


def test_hybrid_band_keeps_deterministic_top_seeds() -> None:
    scores = torch.linspace(-2.0, 2.0, 100)
    ranked = torch.argsort(scores, descending=True)
    generator = torch.Generator().manual_seed(23)
    seeds = select_seeds(
        "hybrid_band",
        scores,
        ranked,
        10,
        generator=generator,
        hybrid_top_fraction=0.7,
        exploration_fraction=0.2,
        band_count=40,
    )

    assert set(ranked[:7].tolist()).issubset(seeds.tolist())
    assert torch.unique(seeds).numel() == 10


def test_top_plus_band_keeps_all_top_seeds_and_adds_exploration() -> None:
    scores = torch.linspace(-2.0, 2.0, 100)
    ranked = torch.argsort(scores, descending=True)
    generator = torch.Generator().manual_seed(29)
    seeds = select_seeds(
        "top_plus_band",
        scores,
        ranked,
        10,
        generator=generator,
        hybrid_top_fraction=0.7,
        exploration_fraction=0.2,
        band_count=40,
    )

    assert set(ranked[:10].tolist()).issubset(seeds.tolist())
    assert torch.unique(seeds).numel() == 12


def test_top_plus_next_is_deterministic_larger_top_prefix() -> None:
    scores = torch.linspace(-2.0, 2.0, 100)
    ranked = torch.argsort(scores, descending=True)
    seeds = select_seeds(
        "top_plus_next",
        scores,
        ranked,
        10,
        generator=torch.Generator().manual_seed(31),
        hybrid_top_fraction=0.7,
        exploration_fraction=0.2,
        band_count=40,
    )

    assert torch.equal(seeds, ranked[:12])
