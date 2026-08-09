from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


GRID_PATTERN = re.compile(r"^late_l(?P<layer>\d+)_f00_(?P<end>\d+)_delete$")


def parse_grid_cell(name: str) -> tuple[int, int] | None:
    match = GRID_PATTERN.match(name)
    if match is None:
        return None
    return int(match.group("layer")), int(match.group("end")) + 1


def markdown_matrix(
    title: str,
    cells: dict[tuple[int, int], float],
    layers: list[int],
    widths: list[int],
    scale: float,
) -> list[str]:
    lines = [title, "", "| Layer start \\ width | " + " | ".join(map(str, widths)) + " |"]
    lines.append("|---:|" + "---:|" * len(widths))
    for layer in layers:
        values = [f"{cells[(layer, width)] * scale:+.3f}" for width in widths]
        lines.append(f"| L{layer} | " + " | ".join(values) + " |")
    return lines


def analyze(summary: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {str(row["variant"]): row for row in summary}
    official: dict[tuple[int, int], float] = {}
    nll: dict[tuple[int, int], float] = {}
    for name, row in by_name.items():
        cell = parse_grid_cell(name)
        if cell is None:
            continue
        official[cell] = float(row["paired_official_delta"])
        nll[cell] = float(row["mean_nll_improvement"])
    layers = sorted({layer for layer, _ in official})
    widths = sorted({width for _, width in official})
    if len(official) != len(layers) * len(widths):
        raise RuntimeError("incomplete layer-by-width grid")
    best_official = max(official, key=lambda cell: (official[cell], nll[cell]))
    best_nll = max(nll, key=lambda cell: (nll[cell], official[cell]))
    return {
        "layers": layers,
        "widths": widths,
        "official_delta": {f"L{l}_W{w}": value for (l, w), value in official.items()},
        "nll_improvement": {f"L{l}_W{w}": value for (l, w), value in nll.items()},
        "best_official_cell": {
            "layer_start": best_official[0],
            "width": best_official[1],
            "official_delta": official[best_official],
            "nll_improvement": nll[best_official],
        },
        "best_nll_cell": {
            "layer_start": best_nll[0],
            "width": best_nll[1],
            "official_delta": official[best_nll],
            "nll_improvement": nll[best_nll],
        },
        "non_grid": [row for row in summary if parse_grid_cell(str(row["variant"])) is None],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    result = analyze(summary)
    (args.run_dir / "grid_analysis.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    official = {
        tuple(map(int, key.removeprefix("L").replace("_W", " ").split())): value
        for key, value in result["official_delta"].items()
    }
    nll = {
        tuple(map(int, key.removeprefix("L").replace("_W", " ").split())): value
        for key, value in result["nll_improvement"].items()
    }
    lines = ["# Late spectral grid", ""]
    lines.extend(
        markdown_matrix(
            "## Paired official-score change (percentage points)",
            official,
            result["layers"],
            result["widths"],
            100.0,
        )
    )
    lines.append("")
    lines.extend(
        markdown_matrix(
            "## Mean Gold-NLL improvement",
            nll,
            result["layers"],
            result["widths"],
            1.0,
        )
    )
    (args.run_dir / "grid_analysis.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
