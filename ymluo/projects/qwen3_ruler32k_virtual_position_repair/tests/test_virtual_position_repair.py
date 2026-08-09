from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT.parent
RULE_SRC = PROJECTS / "qwen3_local_rule_failure_boundary" / "src"
if str(RULE_SRC) not in sys.path:
    sys.path.insert(0, str(RULE_SRC))

import run_local_global_rope_probe_8b as method  # noqa: E402
import run_rope_retrieval_repair_8b as rope  # noqa: E402


def test_remote_positions_pack_before_native_recent_window() -> None:
    selected = torch.tensor([[0, 100, 20, 997, 998, 999]])
    remote = torch.tensor([[False, True, True, False, False, False]])
    virtual = method.local_global_virtual_positions(
        selected, remote, query_position=999, local_window=2
    )
    assert virtual.tolist() == [[0.0, 996.0, 995.0, 997.0, 998.0, 999.0]]


def test_alpha_zero_rephase_equals_native_selected_scores() -> None:
    generator = torch.Generator().manual_seed(7)
    query = torch.randn((1, 2, 1, 8), generator=generator)
    keys = torch.randn((1, 2, 10, 8), generator=generator)
    selected = torch.tensor([[0, 3, 8, 9], [0, 5, 8, 9]])
    remote = torch.tensor(
        [[False, True, False, False], [False, True, False, False]]
    )
    inv_freq = torch.tensor([1.0, 0.1, 0.01, 0.001])
    repaired, _, effective = method.rephase_selected_scores(
        query,
        keys,
        selected,
        remote,
        query_position=9,
        local_window=1,
        alpha=0.0,
        inv_freq=inv_freq,
        scaling=1.0,
        attention_mask=None,
    )
    gathered = keys.gather(
        2,
        selected.view(1, 2, 4, 1).expand(1, 2, 4, 8),
    )
    native = torch.matmul(query, gathered.transpose(2, 3))[0, :, 0, :]
    torch.testing.assert_close(repaired, native, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(effective, selected.float())


def test_delta_rotation_matches_direct_rotation_at_virtual_position() -> None:
    generator = torch.Generator().manual_seed(11)
    pre = torch.randn((1, 1, 2, 8), generator=generator)
    old = torch.tensor([[100.0, 1200.0]])
    new = torch.tensor([[900.0, 1300.5]])
    inv_freq = torch.tensor([1.0, 0.1, 0.01, 0.001])

    old_cos, old_sin = rope.rope_angles(old.flatten(), inv_freq, 8, pre.dtype)
    old_post = pre * old_cos.view(1, 1, 2, 8) + rope.rotate_half(pre) * old_sin.view(1, 1, 2, 8)
    moved = rope.apply_rope_delta(old_post, old, new, inv_freq)

    new_cos, new_sin = rope.rope_angles(new.flatten(), inv_freq, 8, pre.dtype)
    direct = pre * new_cos.view(1, 1, 2, 8) + rope.rotate_half(pre) * new_sin.view(1, 1, 2, 8)
    torch.testing.assert_close(moved, direct, atol=2e-4, rtol=2e-4)
