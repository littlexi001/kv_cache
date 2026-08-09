from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import torch


HERE = Path(__file__).resolve()
PROJECTS = HERE.parents[2]
RULER_SRC = PROJECTS / "qwen3_ruler32k_rope_method" / "src"
if str(RULER_SRC) not in sys.path:
    sys.path.insert(0, str(RULER_SRC))

import run_ruler32k_rope_sparse as runner  # noqa: E402


REPHASE_VARIANTS = {
    "local_global_rephase02",
    "local_global_rephase05",
    "local_global_rephase10",
    "local_global_rephase15",
    "local_global_rephase25",
    "local_global_rephase50",
    "local_global_rephase75",
    "local_global_rephase100",
}


class SelectionAuditedController(runner.core.AuditedController):
    """Add a first-query support digest without storing all selections."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._selection_hasher = hashlib.sha256()

    def record(
        self,
        positions: torch.Tensor,
        weights: torch.Tensor,
        key_count: int,
        remote_mask: torch.Tensor | None,
    ) -> None:
        if self.collect_metrics:
            array = positions.detach().to(device="cpu", dtype=torch.int64).numpy()
            self._selection_hasher.update(str(tuple(array.shape)).encode("ascii"))
            self._selection_hasher.update(array.tobytes())
        super().record(positions, weights, key_count, remote_mask)

    def audit_summary(self) -> dict[str, Any]:
        return {
            **super().audit_summary(),
            "selection_sha256": self._selection_hasher.hexdigest(),
        }


def main() -> None:
    runner.ALLOWED_VARIANTS.update(REPHASE_VARIANTS)
    runner.core.AuditedController = SelectionAuditedController
    runner.main()


if __name__ == "__main__":
    main()
