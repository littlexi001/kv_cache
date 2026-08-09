from __future__ import annotations

from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[4]
PROJECT = Path(__file__).resolve().parents[1]
PAPER = REPO / "ymluo" / "papers" / "countcap_iclr2027"


def test_full_score_regret_keeps_both_cross_band_residuals() -> None:
    source = (
        PAPER / "sections" / "appendix_analysis_statements.tex"
    ).read_text(encoding="utf-8")

    assert "2\\varepsilon_{\\rm sel}\n +|\\Gamma(\\widehat b)|" in source
    assert "+|\\Gamma(b_{\\rm full}^\\star)|" in source


def test_finite_sample_claims_keep_independence_and_production_caveats() -> None:
    source = (
        PAPER / "sections" / "appendix_analysis_statements.tex"
    ).read_text(encoding="utf-8")

    assert "exact enumeration gives\n$M=13{,}817$" in source
    assert "they are not independent" in source
    assert "transform\nis frozen before independent allocation samples" in source
    assert "the theorem is not presented as an unconditional guarantee" in source


def test_chinese_theory_defines_matrix_bernstein_parameters() -> None:
    source = (
        PROJECT / "docs" / "20260728_qksieve_theory_complete_zh.md"
    ).read_text(encoding="utf-8")

    assert r"\|X_j\|_{\mathrm{op}}\le L" in source
    assert r"\|\mathbb E[X_j^2]\|_{\mathrm{op}}\le v" in source
    assert "单样本矩阵方差代理" in source


def test_paper_exposes_paired_query_key_dependence_residual() -> None:
    statements = (
        PAPER / "sections" / "appendix_analysis_statements.tex"
    ).read_text(encoding="utf-8")
    proofs = (
        PAPER / "sections" / "appendix.tex"
    ).read_text(encoding="utf-8")
    main = (
        PAPER / "sections" / "04_analysis.tex"
    ).read_text(encoding="utf-8")

    assert r"H=\mathbb E[(k\otimes q)(k\otimes q)^\top]" in statements
    assert r"\label{eq:paired-qk-dependence}" in statements
    assert "second moments alone cannot certify rank optimality" in statements
    assert "Paired-dependence residual" in proofs
    assert r"\cref{eq:paired-qk-dependence}" in main


def test_paired_dependence_bound_matches_empirical_fourth_moment() -> None:
    rng = np.random.default_rng(20260728)
    sample_count, dimension = 256, 4
    query = rng.normal(size=(sample_count, dimension))
    key = query @ rng.normal(size=(dimension, dimension))
    key += 0.2 * rng.normal(size=key.shape)
    bilinear = rng.normal(size=(dimension, dimension)) * 0.2
    residual_map = np.eye(dimension) - bilinear

    paired_loss = np.mean(
        np.square(np.einsum("ni,ij,nj->n", query, residual_map, key))
    )
    query_moment = query.T @ query / sample_count
    key_moment = key.T @ key / sample_count
    independent_loss = np.trace(
        query_moment @ residual_map @ key_moment @ residual_map.T
    )
    paired_features = np.stack(
        [np.kron(key[index], query[index]) for index in range(sample_count)]
    )
    fourth_moment = paired_features.T @ paired_features / sample_count
    independent_fourth = np.kron(key_moment, query_moment)
    dependence = fourth_moment - independent_fourth
    vectorized_residual = residual_map.reshape(-1, order="F")

    quadratic_gap = (
        vectorized_residual @ dependence @ vectorized_residual
    )
    bound = (
        np.linalg.norm(dependence, ord=2)
        * np.square(np.linalg.norm(residual_map, ord="fro"))
    )

    np.testing.assert_allclose(
        paired_loss - independent_loss,
        quadratic_gap,
        rtol=1e-10,
        atol=1e-10,
    )
    assert abs(quadratic_gap) <= bound + 1e-10
