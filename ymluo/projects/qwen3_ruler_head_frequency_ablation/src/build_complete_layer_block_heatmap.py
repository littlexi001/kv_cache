from __future__ import annotations

import argparse
import json
from pathlib import Path


def compact_cells(payload: dict) -> list[dict]:
    return [
        {
            "b": cell["layer_block"],
            "g": cell["head_group"],
            "q": cell["query_heads"],
            "f": cell["frequency_band"],
            "m": round(cell["mean_official_delta_pp"], 8),
            "bl": cell["best_layer"],
            "bv": round(cell["best_official_delta_pp"], 8),
            "wl": cell["worst_layer"],
            "wv": round(cell["worst_official_delta_pp"], 8),
            "n": round(cell["mean_gold_nll_improvement"], 8),
        }
        for cell in payload["cells"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-output", type=Path)
    parser.add_argument("--preview-stylesheet", type=Path)
    parser.add_argument("--preview-kit", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.data.read_text(encoding="utf-8"))
    template = args.template.read_text(encoding="utf-8")
    marker = "__DATA__"
    if template.count(marker) != 1:
        raise ValueError("template must contain exactly one __DATA__ marker")
    data = json.dumps(compact_cells(payload), ensure_ascii=False, separators=(",", ":"))
    rendered = template.replace(marker, data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output} ({len(rendered)} characters)")

    preview_arguments = (
        args.preview_output,
        args.preview_stylesheet,
        args.preview_kit,
    )
    if any(preview_arguments):
        if not all(preview_arguments):
            raise ValueError(
                "--preview-output, --preview-stylesheet, and --preview-kit must be supplied together"
            )
        stylesheet = args.preview_stylesheet.read_text(encoding="utf-8")
        kit = args.preview_kit.read_text(encoding="utf-8")
        marker = "<!--__INLINE_VISUALIZATION_FRAGMENT__-->"
        if kit.count(marker) != 1:
            raise ValueError("preview kit must contain exactly one fragment marker")
        body = kit.replace(marker, rendered)
        preview = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>完整 RoPE Head × Frequency 扫描</title>
<style>{stylesheet}
html>body{{padding:1rem}}</style>
</head>
<body>
{body}
</body>
</html>
"""
        args.preview_output.parent.mkdir(parents=True, exist_ok=True)
        args.preview_output.write_text(preview, encoding="utf-8")
        print(f"wrote {args.preview_output} ({len(preview)} characters)")


if __name__ == "__main__":
    main()
