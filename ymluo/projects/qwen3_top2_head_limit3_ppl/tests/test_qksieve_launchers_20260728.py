from __future__ import annotations

from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT / "scripts"


def _read(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


def test_cuda_arch_is_overridable_for_submission_launchers() -> None:
    launchers = sorted(SCRIPTS.glob("*20260728*.sh"))
    hardcoded = [
        path.name
        for path in launchers
        if "export TORCH_CUDA_ARCH_LIST=8.6" in path.read_text(
            encoding="utf-8"
        )
    ]

    assert not hardcoded
    for name in (
        "launch_qksieve_fulltopk_longbench_5gpu_20260728.sh",
        "launch_qksieve_fulltopk_ruler_6gpu_20260728.sh",
        "launch_qksieve_frozen_samepath_length_6gpu_20260728.sh",
        "launch_qksieve_fier_packed_longbench_5gpu_20260728.sh",
        "run_qksieve_qfused_correctness_20260728.sh",
    ):
        assert (
            'TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"'
            in _read(name)
        )


def test_qfused_launcher_uses_full_validation_matrix_and_gpu_guard() -> None:
    source = _read("run_qksieve_qfused_correctness_20260728.sh")

    assert '[[ ! "$GPU" =~ ^[0-5]$ ]]' in source
    assert "validate_qksieve_qfused_matrix_20260728.py" in source
    assert '--group_counts "${QKSIEVE_GROUP_COUNTS:-4,8}"' in source
    assert '--dtypes "${QKSIEVE_DTYPES:-float16,bfloat16}"' in source
    assert '--output "$OUT_ROOT/validation_matrix.json"' in source


def test_qfused_longbench_smoke_requires_matrix_and_execution_diagnostics() -> None:
    source = _read("run_qksieve_qfused_longbench_smoke_20260728.sh")

    assert '[[ ! "$GPU" =~ ^[0-5]$ ]]' in source
    assert 'report.get("all_passed") is not True' in source
    assert "--collect_attention_stats" in source
    assert "qksieve_fullprompt_auto_plain_fulltopk" in source
    assert "qksieve_fullprompt_auto_plain_qfused_fulltopk" in source
    assert "analyze_qksieve_qfused_longbench_smoke_20260728.py" in source


def test_parallel_launchers_reject_out_of_range_and_duplicate_gpus() -> None:
    for name in (
        "launch_qksieve_fulltopk_longbench_5gpu_20260728.sh",
        "launch_qksieve_fier_packed_longbench_5gpu_20260728.sh",
        "launch_qksieve_public_selectors_longbench_5gpu_20260728.sh",
        "launch_qksieve_free_generation_drift_6gpu_20260728.sh",
        "launch_qksieve_teacher_forced_drift_6gpu_20260728.sh",
    ):
        source = _read(name)
        assert '[[ ! "$gpu" =~ ^[0-5]$ ]]' in source
        assert "declare -A seen_gpus=()" in source
        assert '"${seen_gpus[$gpu]+x}"' in source


def test_public_selector_launcher_is_quality_only_and_strictly_paired() -> None:
    source = _read(
        "launch_qksieve_public_selectors_longbench_5gpu_20260728.sh"
    )

    assert "quest_p16_fullprompt_matchedbudget" in source
    assert "rabitqcache_rtn1_fullprompt_matchedbudget" in source
    assert "sparq_r32_selector_fullprompt_matchedbudget" in source
    assert "sparq_r32_formula_fullprompt_matchedbudget" in source
    assert "assert len(rows) == 6" in source
    assert 'shard_count="${#gpus[@]}"' in source
    assert '--num_shards "$shard_count"' in source
    assert "analyze_qksieve_public_selectors_longbench_20260728.py" in source
    assert '--expected_pairs "$EXPECTED_PAIRS"' in source
    assert "--collect_attention_stats" not in source


def test_targeted_ruler_launcher_uses_only_six_allowed_gpus() -> None:
    source = _read("launch_qksieve_targeted_ruler_6gpu_20260801.sh")

    assert '[[ ! "$gpu" =~ ^[0-5]$ ]]' in source
    assert "declare -A seen_gpus=()" in source
    assert '"${#gpus[@]}" -ne 6' in source
    assert "8192,16384,32768 2" in source
    assert "65536 1" in source
    assert "8192:2,16384:2,32768:2,65536:1" in source
    assert "--device_map \"$device_map\"" in source


def test_binarypc_launcher_is_matched_quality_only_and_gpu_guarded() -> None:
    source = _read(
        "launch_qksieve_binarypc_longbench_m5_6gpu_20260801.sh"
    )

    assert '[[ ! "$gpu" =~ ^[0-5]$ ]]' in source
    assert "declare -A seen_gpus=()" in source
    assert "binarypc_offline64_fullprompt_matchedbudget" in source
    assert "--binarypc_projection_path \"$PROJECTION\"" in source
    assert "--reference_root \"$REFERENCE_ROOT\"" in source
    assert "analyze_qksieve_binarypc_longbench_20260801.py" in source
    assert "--collect_attention_stats" not in source


def test_frozen_launchers_keep_the_same_method_and_budget_contract() -> None:
    longbench = _read(
        "launch_qksieve_fulltopk_longbench_5gpu_20260728.sh"
    )
    ruler = _read("launch_qksieve_fulltopk_ruler_6gpu_20260728.sh")
    samepath = _read(
        "launch_qksieve_frozen_samepath_length_6gpu_20260728.sh"
    )

    method = "qksieve_fullprompt_auto_plain_fulltopk"
    score_mode = (
        "pca_hierarchical_autoqmsetotal15z_qkmetric_packed_fulltopk"
    )
    assert method in longbench and method in ruler
    for source in (longbench, ruler):
        assert "pca_hierarchical_autoqmsetotal15z_" in source
        assert "qkmetric_packed_fulltopk" in source
    assert score_mode in samepath
    assert "--direct_min_tokens 256" in samepath
    assert "--direct_max_tokens 1280" in samepath
    assert "--direct_fraction 0.06" in samepath
    assert "--candidate_overfetch 1.0" in samepath
    assert "--protect_recent_tokens 0" in samepath
