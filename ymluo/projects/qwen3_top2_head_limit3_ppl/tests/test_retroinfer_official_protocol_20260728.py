from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


PROJECT = Path(__file__).resolve().parents[1]
SRC = PROJECT / "src"
PAPER = PROJECT.parents[1] / "papers" / "countcap_iclr2027"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import audit_retroinfer_official_checkout_20260728 as audit  # noqa: E402
import analyze_retroinfer_aligned_longbench_20260728 as analysis  # noqa: E402
import run_retroinfer_aligned_longbench_20260728 as runner  # noqa: E402


def _write_checkout(root: Path) -> None:
    for relative in audit.REQUIRED_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")
    (root / "README.md").write_text(
        "# RetroInfer\nRetrievalAttention\n",
        encoding="utf-8",
    )
    (root / "config/Llama-3.1-8B-Instruct.json").write_text(
        json.dumps(
            {
                "RetroInfer": {
                    "retrieval_budget": 0.018,
                    "estimation_budget": 0.232,
                    "cache_ratio": 0.05,
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "benchmark/longbench/pred.py").write_text(
        "load_dataset('THUDM/LongBench', task)\nignore_eos=True\n",
        encoding="utf-8",
    )


def test_retroinfer_checkout_audit_records_identity_and_native_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_checkout(tmp_path)

    def fake_git(_checkout: Path, *args: str) -> str:
        if args == ("rev-parse", "HEAD"):
            return audit.OFFICIAL_COMMIT
        if args[:2] == ("status", "--short"):
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(audit, "_git_output", fake_git)
    report = audit.audit_checkout(tmp_path)

    assert report["complete"]
    assert report["system_identity"] == "RetroInfer"
    assert report["not_system_identity"].startswith("original RetrievalAttention")
    assert report["checks"]["official_longbench_ignores_eos"]
    assert report["dependency_pins_for_reproduction"][
        "starmys_flash_attention_weighted"
    ] == audit.WEIGHTED_FLASH_ATTN_COMMIT


def test_prepare_script_pins_source_and_does_not_install_packages() -> None:
    source = (
        PROJECT
        / "scripts"
        / "prepare_retroinfer_official_20260728.sh"
    ).read_text(encoding="utf-8")

    assert audit.OFFICIAL_COMMIT in source
    assert "audit_retroinfer_official_checkout_20260728.py" in source
    assert "pip install" not in source
    assert "CUDA_VISIBLE_DEVICES" not in source


def test_protocol_does_not_mislabel_retroinfer_as_retrievalattention() -> None:
    protocol = (
        PROJECT
        / "docs"
        / "20260728_qksieve_retroinfer_official_protocol_zh.md"
    ).read_text(encoding="utf-8")

    assert "RetrievalAttention (paper-reported)" in protocol
    assert "RetroInfer (official system)" in protocol
    assert "不能直接计算 speedup" in protocol
    assert "launch_retroinfer_aligned_longbench_5gpu_20260728.sh" in protocol
    assert "cache prepare、CUDA graph capture" in protocol

    experiments = (
        PAPER / "sections" / "05_experiments.tex"
    ).read_text(encoding="utf-8")
    system_diagnostics = (
        PAPER / "sections" / "appendix_system_diagnostics.tex"
    ).read_text(encoding="utf-8")
    experiment_surface = experiments + system_diagnostics
    related = (
        PAPER / "sections" / "02_related_work.tex"
    ).read_text(encoding="utf-8")
    assert "RetrievalAttention (paper reported)" in experiment_surface
    assert "RetroInfer official system" in experiment_surface
    assert "not a" in experiment_surface
    assert "runnable artifact of original RetrievalAttention" in experiment_surface
    assert "implements RetroInfer rather than" in related


class _FakeTokenizer:
    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return ",".join(str(token_id) for token_id in token_ids)


class _FakeCache:
    def __init__(self) -> None:
        self.capture_count = 0

    def capture_cuda_graph(self) -> None:
        self.capture_count += 1


class _FakeModel:
    def __init__(self, sampled_ids: list[int]) -> None:
        self.layers = [type("Layer", (), {"device": torch.device("cpu")})()]
        self.max_length = 128
        self.sampled_ids = iter(sampled_ids)
        self.kv_cache = _FakeCache()
        self.decode_calls = 0
        self.move_calls = 0

    def init_kv_cache(self, valid_start, attention_config) -> None:
        assert valid_start.tolist() == [0]
        assert attention_config == {"RetroInfer": {"test": True}}

    def prefill_forward(self, *, inputs_ids: torch.Tensor) -> torch.Tensor:
        assert inputs_ids.shape == (1, 4)
        return torch.zeros(1, 1, 8)

    def decode_forward(self, *, inputs_ids: torch.Tensor) -> torch.Tensor:
        assert inputs_ids.shape == (1, 1)
        self.decode_calls += 1
        return torch.zeros(1, 1, 8)

    def sampling(self, logits: torch.Tensor, *, do_sample: bool) -> torch.Tensor:
        assert not do_sample
        return torch.tensor([[next(self.sampled_ids)]])

    def move(self) -> None:
        self.move_calls += 1


def test_aligned_generation_excludes_stop_token_and_captures_native_graph() -> None:
    model = _FakeModel([3, 4, 7, 5])
    result = runner.generate_aligned(
        model,
        _FakeTokenizer(),
        torch.tensor([[1, 2, 3, 4]]),
        attention_type="RetroInfer",
        attention_config={"RetroInfer": {"test": True}},
        max_new_tokens=6,
        stop_token_ids={7},
    )

    assert result["generated_ids"] == [3, 4]
    assert result["prediction"] == "3,4"
    assert result["decode_steps"] == 2
    assert model.decode_calls == 2
    assert model.move_calls == 1
    assert model.kv_cache.capture_count == 1


def test_aligned_generation_stops_on_first_prefill_token() -> None:
    model = _FakeModel([7, 5])
    result = runner.generate_aligned(
        model,
        _FakeTokenizer(),
        torch.tensor([[1, 2, 3, 4]]),
        attention_type="Full_Flash_Attn",
        attention_config={"RetroInfer": {"test": True}},
        max_new_tokens=6,
        stop_token_ids={7},
    )

    assert result["generated_ids"] == []
    assert result["decode_steps"] == 0
    assert model.decode_calls == 0
    assert model.kv_cache.capture_count == 0


def _aligned_row(task: str, method: str) -> dict[str, str]:
    retro = method == runner.RETROINFER_METHOD
    return {
        "task": task,
        "sample_id": "0",
        "method": method,
        "protocol": "qksieve_aligned_longbench_v1",
        "prompt_sha256": "prompt-hash",
        "prompt_tokens": "7500",
        "prompt_truncation_mode": "official_middle",
        "prompt_wrapper": "llama3",
        "stop_token_ids": "[1, 2, 3]",
        "max_new_tokens": "32",
        "official_repository_commit": audit.OFFICIAL_COMMIT,
        "model_name_or_path": "llama",
        "dtype": "float16",
        "score": "0.99" if retro else "1.0",
        "decode_seconds": "0.1" if retro else "0.2",
        "decode_steps": "2",
        "total_seconds": "1.0" if retro else "2.0",
        "cache_init_seconds": "0.1",
        "cache_prepare_seconds": "0.2",
        "graph_capture_seconds": "0.3" if retro else "0.0",
        "gpu_peak_allocated_bytes": "10",
        "gpu_peak_reserved_bytes": "20",
        "cpu_peak_rss_bytes": "30",
        "retrieval_budget": "0.018" if retro else "1.0",
        "estimation_budget": "0.232" if retro else "0.0",
        "cache_ratio": "0.05" if retro else "1.0",
    }


def test_aligned_summary_requires_strict_pairs_and_reports_system_speed() -> None:
    rows = [
        _aligned_row(f"task{task_index}", method)
        for task_index in range(16)
        for method in (runner.FULL_METHOD, runner.RETROINFER_METHOD)
    ]

    report = analysis.analyze(rows, expected_pairs=16)

    assert report["quality_retention"] == 0.99
    assert report["decode_speedup"] == 2.0
    assert report["request_total_speedup"] == 2.0
    assert report["native_operating_point"]["retrieval_budget"] == 0.018
    assert "not an original RetrievalAttention reproduction" in report[
        "claim_boundary"
    ]

    broken = [dict(row) for row in rows]
    broken[-1]["prompt_sha256"] = "different"
    try:
        analysis.analyze(broken, expected_pairs=16)
    except ValueError as error:
        assert "prompt_sha256" in str(error)
    else:
        raise AssertionError("mismatched RetroInfer prompt was accepted")


def test_aligned_launcher_preserves_gpu_and_pairing_guards() -> None:
    source = (
        PROJECT
        / "scripts"
        / "launch_retroinfer_aligned_longbench_5gpu_20260728.sh"
    ).read_text(encoding="utf-8")

    assert '[[ ! "$gpu" =~ ^[0-5]$ ]]' in source
    assert "declare -A seen_gpus=()" in source
    assert "retroinfer_stack_full_flash,retroinfer_official_aligned" in source
    assert "assert len(rows) == 2" in source
    assert "analyze_retroinfer_aligned_longbench_20260728.py" in source
