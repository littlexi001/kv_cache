from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "src" / "select_test_specs.py"


def test_selection_rejects_mean_preserving_task_swap(tmp_path: Path) -> None:
    rows = [
        {
            "variant": "native_rope",
            "spec": {"name": "native_rope", "atoms": []},
            "min_seed_official_delta": 0.0,
            "official_degraded": 0,
            "mean_gold_nll_improvement": 0.0,
        },
        {
            "variant": "l25_g3_f46_a0",
            "spec": {"name": "l25_g3_f46_a0", "atoms": [{}]},
            "min_seed_official_delta": 0.0,
            "official_degraded": 1,
            "mean_gold_nll_improvement": 0.3,
        },
        {
            "variant": "stable",
            "spec": {"name": "stable", "atoms": [{}]},
            "min_seed_official_delta": 0.0,
            "official_degraded": 0,
            "mean_gold_nll_improvement": 0.1,
        },
    ]
    source = tmp_path / "summary.json"
    output = tmp_path / "specs.json"
    source.write_text(json.dumps(rows), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--validation-summary",
            str(source),
            "--output",
            str(output),
            "--limit",
            "3",
        ],
        check=True,
    )
    names = [spec["name"] for spec in json.loads(output.read_text(encoding="utf-8"))["specs"]]
    assert names == ["native_rope", "stable", "l25_g3_f46_a0"]
