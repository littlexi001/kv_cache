from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_gqa_envelope_retrieval import (
    certified_prefix_size,
    group_ball_upper_bound,
)


def test_group_ball_is_an_upper_bound_for_every_query() -> None:
    key = torch.tensor([[1.0, 0.0], [0.0, 2.0], [-1.0, 1.0]])
    query = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.2], [0.7, 0.8]]), dim=-1
    )
    upper, _ = group_ball_upper_bound(key, query)
    exact = query @ key.T

    assert torch.all(upper.unsqueeze(0) + 1.0e-6 >= exact)


def test_certified_prefix_contains_every_head_topk() -> None:
    key = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    query = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=-1
    )
    upper, _ = group_ball_upper_bound(key, query)
    exact = query @ key.T
    prefix = certified_prefix_size(upper, exact, top_count=1)
    selected = torch.topk(upper, k=prefix).indices

    for head in range(query.shape[0]):
        target = int(torch.argmax(exact[head]).item())
        assert target in selected.tolist()
