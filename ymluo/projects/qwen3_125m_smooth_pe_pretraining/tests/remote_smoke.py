from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train_synthetic_pe import ModelConfig, QwenStyleLM, make_batch, rope_pair_scales


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the remote smoke test")
    device = torch.device("cuda", 0)
    config = ModelConfig()
    torch.manual_seed(20260807)
    model = QwenStyleLM(config, "smooth_layer_frequency").to(device)
    tokens, loss_weights, _, digest = make_batch(1, 128, config.vocab_size, 1701)
    tokens = tokens.to(device)
    loss_weights = loss_weights.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss, selected = model(tokens, loss_weights)
    loss.backward()
    scales = {
        variant: {
            "l0_f0": float(rope_pair_scales(config, variant, 0, device)[0].item()),
            "l11_f0": float(rope_pair_scales(config, variant, 11, device)[0].item()),
            "l11_f40": float(rope_pair_scales(config, variant, 11, device)[40].item()),
        }
        for variant in ("native", "deep_highfreq_drop", "slow_rope", "smooth_layer_frequency")
    }
    print(
        {
            "torch": torch.__version__,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "loss": float(loss.item()),
            "selected_labels": selected,
            "data_hash": digest,
            "max_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            "scales": scales,
        }
    )


if __name__ == "__main__":
    main()
