from pathlib import Path
import sys


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from countcap_cost_gate_20260723 import choose_countcap_path


def row(tokens, dense, qk, key, qk_quality=0.98, key_quality=0.98):
    def model(pair):
        return {"fixed_seconds": pair[0], "step_seconds": pair[1]}

    return {
        "mean_prompt_tokens": tokens,
        "full_decode_cost_model": model(dense),
        "methods": {
            "countcap_fullprompt": {
                "quality_retention": qk_quality,
                "decode_cost_model": model(qk),
            },
            "countcap_fullprompt_keypca": {
                "quality_retention": key_quality,
                "decode_cost_model": model(key),
            },
        },
    }


def test_short_context_stays_dense_when_sparse_step_is_slower():
    profile = {"lengths": [row(8192, (0.0, 0.04), (1.0, 0.06), (0.6, 0.05))]}
    decision = choose_countcap_path(profile, 8192, 256)
    assert decision["selected_path"] == "full_kv"


def test_long_generation_amortizes_build_and_selects_fastest_path():
    profile = {"lengths": [row(65536, (0.0, 0.20), (2.0, 0.07), (1.0, 0.08))]}
    decision = choose_countcap_path(profile, 65536, 100)
    assert decision["selected_path"] == "countcap_fullprompt_keypca"
    assert decision["predicted_decode_speedup"] > 1.03


def test_short_generation_does_not_amortize_index_build():
    profile = {"lengths": [row(65536, (0.0, 0.20), (2.0, 0.07), (1.0, 0.08))]}
    decision = choose_countcap_path(profile, 65536, 4)
    assert decision["selected_path"] == "full_kv"


def test_quality_floor_blocks_fast_unsafe_variant():
    profile = {
        "lengths": [
            row(
                65536,
                (0.0, 0.20),
                (2.0, 0.07),
                (1.0, 0.05),
                qk_quality=0.98,
                key_quality=0.80,
            )
        ]
    }
    decision = choose_countcap_path(profile, 65536, 100)
    assert decision["selected_path"] == "countcap_fullprompt"


def test_interpolation_uses_conservative_quality_envelope():
    profile = {
        "lengths": [
            row(8192, (0.0, 0.05), (1.0, 0.04), (1.0, 0.04), qk_quality=0.94),
            row(32768, (0.0, 0.15), (1.0, 0.04), (1.0, 0.04), qk_quality=0.99),
        ]
    }
    decision = choose_countcap_path(profile, 16384, 100)
    assert decision["candidates"]["countcap_fullprompt"]["quality_retention"] == 0.94
    assert not decision["candidates"]["countcap_fullprompt"]["eligible"]


def test_profile_may_contain_only_measured_sparse_path():
    profile_row = row(
        32768,
        (0.0, 0.12),
        (2.0, 0.07),
        (1.0, 0.05),
    )
    del profile_row["methods"]["countcap_fullprompt"]
    decision = choose_countcap_path(
        {"lengths": [profile_row]},
        32768,
        100,
    )

    assert decision["selected_path"] == "countcap_fullprompt_keypca"
    assert not decision["candidates"]["countcap_fullprompt"]["available"]
