import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_fier_qksieve_retrieval_fair_20260728 import (
    key_pca_basis,
    second_moment,
    sensitivity_heterogeneity,
)
from run_head_top2_targeted_ppl_20260714 import (
    _deterministic_random_orthogonal_basis,
)


def test_sensitivity_am_gm_is_one_for_identical_bands():
    identity = torch.eye(16)
    values = torch.cat([identity] * 8, dim=-1).reshape(1, 1, 16, 128)

    sensitivities, ratio = sensitivity_heterogeneity(values, values)

    assert sensitivities.shape == (1, 1, 8)
    torch.testing.assert_close(
        sensitivities,
        sensitivities[..., :1].expand_as(sensitivities),
    )
    torch.testing.assert_close(ratio, torch.ones_like(ratio))


def test_sensitivity_am_gm_grows_with_band_heterogeneity():
    identity = torch.eye(16)
    keys = torch.cat([identity] * 8, dim=-1).reshape(1, 1, 16, 128)
    queries = keys.clone()
    queries[..., :16] *= 4.0

    sensitivities, ratio = sensitivity_heterogeneity(keys, queries)

    assert sensitivities[..., 0].item() > sensitivities[..., 1].item()
    assert ratio.item() > 1.0


def test_key_pca_basis_is_orthogonal():
    generator = torch.Generator().manual_seed(101)
    keys = torch.randn(2, 3, 64, 128, generator=generator)
    basis = key_pca_basis(second_moment(keys))
    identity = torch.eye(128).expand(2, 3, 128, 128)

    torch.testing.assert_close(
        basis.transpose(-1, -2) @ basis,
        identity,
        atol=2.0e-5,
        rtol=2.0e-5,
    )


def test_key_only_allocator_obeys_query_condition_number_bound():
    query_moment = torch.diag(torch.tensor([4.0, 1.0]))
    candidates = (
        torch.diag(torch.tensor([0.1, 2.0])),
        torch.diag(torch.tensor([1.0, 0.2])),
    )
    key_choice = min(candidates, key=lambda error: float(torch.trace(error)))
    qk_choice = min(
        candidates,
        key=lambda error: float(torch.trace(query_moment @ error)),
    )
    realized_ratio = float(
        torch.trace(query_moment @ key_choice)
        / torch.trace(query_moment @ qk_choice)
    )

    assert key_choice is not qk_choice
    assert realized_ratio <= 4.0


def test_random_rotation_equalizes_fixed_band_energy_in_expectation():
    covariance = torch.diag(torch.linspace(0.1, 3.0, 16))
    energies = []
    for draw in range(512):
        basis = _deterministic_random_orthogonal_basis(
            1,
            1,
            16,
            draw,
            torch.device("cpu"),
            torch.float32,
        )[0, 0]
        projector = basis[:, :4] @ basis[:, :4].T
        energies.append(torch.trace(covariance @ projector))
    observed = torch.stack(energies).mean()
    expected = covariance.trace() * 4.0 / 16.0

    torch.testing.assert_close(observed, expected, rtol=0.05, atol=0.0)
