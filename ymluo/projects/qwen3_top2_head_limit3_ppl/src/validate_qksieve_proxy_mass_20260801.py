from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import qksieve_mass_cuda_20260801 as mass_cuda
import run_direct_countcap_denseprompt_ppl_20260725 as direct
import run_head_top2_targeted_ppl_20260714 as runner


PROXY_MASS_MODE = (
    "pca_hierarchical_autokeytotal15z_qkmetric_qfused_gqa4_wmma_"
    "kappend_proxymass_unbiased_packed_direct"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    state: dict[str, object] = {}
    runner._configure_packed_qmse_state(state, PROXY_MASS_MODE)
    assert PROXY_MASS_MODE in runner._QKSIEVE_FAST_RUNTIME_MODES
    assert state["packed_qmse_proxy_mass_correction"]
    assert not state["packed_qmse_sampled_mass_correction"]
    assert PROXY_MASS_MODE in direct.PACKED_PREFILL_QUERY_SCORE_MODES
    assert runner._resolve_packed_qmse_sample_count(8192, 4096, 0.01) == 2048
    assert runner._resolve_packed_qmse_sample_count(4096, 256, 0.01) == 1600

    payload: dict[str, object] = {
        "score_mode_contract": "passed",
        "prefill_query_capture_contract": "passed",
        "sample_cap_contract": "passed",
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        torch.manual_seed(20260801)
        maximum_error: dict[str, float] = {}
        for dtype in (torch.float16, torch.bfloat16):
            sparse = torch.randn(
                2, 1, 32, 128, dtype=dtype, device="cuda"
            )
            value_mean = torch.randn(
                2, 8, 128, dtype=torch.float32, device="cuda"
            )
            mass = torch.rand(2, 32, dtype=torch.float32, device="cuda")
            actual = mass_cuda.mean_value_blend(
                sparse, value_mean, mass
            )
            reference = (
                mass.unsqueeze(1).unsqueeze(-1) * sparse.float()
                + (1.0 - mass).unsqueeze(1).unsqueeze(-1)
                * value_mean.repeat_interleave(4, dim=1).unsqueeze(1)
            ).to(dtype).contiguous()
            error = float(
                (actual.float() - reference.float()).abs().max().item()
            )
            tolerance = 2.0e-3 if dtype == torch.float16 else 3.2e-2
            assert error <= tolerance, (dtype, error, tolerance)
            maximum_error[str(dtype)] = error
        payload["fused_blend_max_abs_error"] = maximum_error
        payload["fused_blend_contract"] = "passed"

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
