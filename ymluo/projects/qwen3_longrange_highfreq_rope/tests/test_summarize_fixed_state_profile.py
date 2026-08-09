import importlib.util
from pathlib import Path


PATH = Path(__file__).parents[1] / "src" / "summarize_fixed_state_profile.py"
SPEC = importlib.util.spec_from_file_location("summarize_fixed_state_profile", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(sample: str, mass: float, delta: float) -> dict:
    return {
        "sample_id": sample,
        "high_phase_contribution_mean": -0.2,
        "evidence_mass_delta_nope_minus_native": delta,
        "evidence_rank_delta_nope_minus_native": -1,
        "pre_rope_cosine_max": 0.2,
        "q_pair_energy_l1_uniform": 1.0,
        "q_high8_energy_mass": 0.04,
        "native_evidence_mass": mass,
        "nope_high_evidence_mass": mass + delta,
    }


def test_top_native_mass_is_selected_per_sample() -> None:
    rows = [row("a", value, 0.1) for value in range(10)] + [row("b", value, 0.1) for value in range(10)]
    selected = MODULE.top_native_mass(rows, 0.2)
    assert len(selected) == 4
    assert sorted(item["native_evidence_mass"] for item in selected) == [8, 8, 9, 9]
