from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import run_controlled_public_kv_benchmark_v1 as runner  # noqa: E402


def test_infer_demonstration_units_from_role_markers() -> None:
    pages = [
        runner.Page(0, "prefix", 0, 4),
        runner.Page(1, "{'role': 'user', 'content': 'first'}", 4, 8),
        runner.Page(2, "first evidence", 8, 12),
        runner.Page(3, "{'role': 'assistant', 'content': 'A'}", 12, 16),
        runner.Page(4, '{"role": "user", "content": "second"}', 16, 20),
        runner.Page(5, "second evidence", 20, 24),
    ]

    assert runner.infer_demonstration_unit_ids(pages) == [None, 0, 0, 0, 1, 1]


def test_demonstration_closure_adds_tail_and_local_pages() -> None:
    pages = [runner.Page(i, f"page {i}", i * 4, (i + 1) * 4) for i in range(8)]
    pages[0].text = "{'role': 'user'}"
    pages[5].text = "{'role': 'user'}"
    bundle = runner.PromptBundle(
        input_ids=None,
        prefix_token_count=0,
        context_token_start=0,
        query_start=32,
        suffix_token_count=0,
        page_spans={i: (i * 4, (i + 1) * 4) for i in range(8)},
    )
    # Construct only the fields read by add_demonstration_closure_pages.
    stub = type(
        "Stub",
        (),
        {
            "budget_tokens": 32,
            "ours_demonstration_closure_budget_fraction": 0.75,
            "ours_demonstration_closure_tail_pages": 2,
            "ours_demonstration_closure_radius_pages": 1,
        },
    )()
    keep = set(range(8, 12))
    unit_ids = runner.infer_demonstration_unit_ids(pages)

    added, used, selected, unit_id = runner.add_demonstration_closure_pages(
        keep, bundle, pages, pages[2], 24, stub, 0, unit_ids, [pages[2]]
    )

    assert unit_id == 0
    assert added == used == 12
    assert {page.page_id for page in selected} == {1, 3, 4}
    assert set(range(4, 8)) <= keep
    assert set(range(12, 20)) <= keep
