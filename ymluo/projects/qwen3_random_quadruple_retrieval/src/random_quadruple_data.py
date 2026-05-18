from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True)
class QuadrupleTableSpec:
    token_min: int = 1
    token_max: int = 1000
    quadruple_len: int = 4
    num_quadruples: int = 100_000
    seed: int = 20_260_518

    def validate(self) -> None:
        if self.token_min < 0:
            raise ValueError("token_min must be >= 0.")
        if self.token_max < self.token_min:
            raise ValueError("token_max must be >= token_min.")
        if self.quadruple_len < 1:
            raise ValueError("quadruple_len must be >= 1.")
        if self.num_quadruples < 1:
            raise ValueError("num_quadruples must be >= 1.")

    def metadata(self) -> dict[str, Any]:
        return {
            "token_min": self.token_min,
            "token_max": self.token_max,
            "quadruple_len": self.quadruple_len,
            "num_quadruples": self.num_quadruples,
            "seed": self.seed,
            "format": "torch.save({'metadata': dict, 'quadruples': LongTensor[num_quadruples, quadruple_len]})",
        }


def generate_quadruple_table(spec: QuadrupleTableSpec) -> torch.Tensor:
    spec.validate()
    generator = torch.Generator(device="cpu").manual_seed(spec.seed)
    return torch.randint(
        low=spec.token_min,
        high=spec.token_max + 1,
        size=(spec.num_quadruples, spec.quadruple_len),
        generator=generator,
        dtype=torch.long,
        device="cpu",
    )


def save_quadruple_table(path: Path, spec: QuadrupleTableSpec, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Quadruple table already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    table = generate_quadruple_table(spec)
    payload = {
        "metadata": spec.metadata(),
        "quadruples": table,
    }
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    return path


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_quadruple_table(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = _torch_load(Path(path))
    if isinstance(payload, torch.Tensor):
        return payload.long().contiguous(), {}
    if not isinstance(payload, dict) or "quadruples" not in payload:
        raise ValueError(f"Unsupported quadruple table file format: {path}")

    table = payload["quadruples"]
    if not isinstance(table, torch.Tensor):
        raise ValueError(f"`quadruples` must be a tensor in {path}")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return table.long().contiguous(), metadata


def validate_quadruple_table(
    table: torch.Tensor,
    spec: QuadrupleTableSpec,
    metadata: dict[str, Any] | None = None,
) -> None:
    spec.validate()
    if table.ndim != 2:
        raise ValueError(f"Quadruple table must be 2D, got shape={tuple(table.shape)}.")
    expected_shape = (spec.num_quadruples, spec.quadruple_len)
    if tuple(table.shape) != expected_shape:
        raise ValueError(
            f"Quadruple table shape mismatch: expected {expected_shape}, got {tuple(table.shape)}."
        )
    if table.numel() == 0:
        raise ValueError("Quadruple table is empty.")

    min_id = int(table.min().item())
    max_id = int(table.max().item())
    if min_id < spec.token_min or max_id > spec.token_max:
        raise ValueError(
            "Quadruple table token ids are out of range: "
            f"expected [{spec.token_min}, {spec.token_max}], got [{min_id}, {max_id}]."
        )

    if metadata:
        for key in ("token_min", "token_max", "quadruple_len", "num_quadruples"):
            if key in metadata and int(metadata[key]) != int(getattr(spec, key)):
                raise ValueError(
                    f"Quadruple table metadata mismatch for {key}: "
                    f"expected {getattr(spec, key)}, got {metadata[key]}."
                )


def ensure_quadruple_file(path: Path, spec: QuadrupleTableSpec, overwrite: bool = False) -> Path:
    path = Path(path)
    if overwrite or not path.exists():
        save_quadruple_table(path, spec, overwrite=True)
    table, metadata = load_quadruple_table(path)
    validate_quadruple_table(table, spec, metadata)
    return path
