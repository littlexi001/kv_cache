from __future__ import annotations

import argparse
import json
from pathlib import Path


def compact(payload: dict) -> list[dict]:
    return [
        {
            "l": cell["layer"],
            "g": cell["head_group"],
            "q": cell["query_heads"],
            "f": cell["frequency_band"],
            "d": round(cell["official_delta_pp"], 8),
            "s": round(cell["official_score"], 8),
            "n": round(cell["gold_nll_improvement"], 8),
            "p": round(cell["gold_ppl"], 8),
            "r": round(cell["gold_ppl_relative_change_percent"], 8),
            "i": cell["score_improved_samples"],
            "e": cell["score_degraded_samples"],
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
    rendered = template.replace(
        marker,
        json.dumps(compact(payload), ensure_ascii=False, separators=(",", ":")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output} ({len(rendered)} characters)")

    preview_args = (args.preview_output, args.preview_stylesheet, args.preview_kit)
    if any(preview_args):
        if not all(preview_args):
            raise ValueError("all preview arguments must be supplied together")
        css = args.preview_stylesheet.read_text(encoding="utf-8")
        kit = args.preview_kit.read_text(encoding="utf-8")
        fragment_marker = "<!--__INLINE_VISUALIZATION_FRAGMENT__-->"
        if kit.count(fragment_marker) != 1:
            raise ValueError("preview kit must contain exactly one fragment marker")
        body = kit.replace(fragment_marker, rendered)
        preview = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>逐层 RoPE 扫描</title><style>{css}\nhtml>body{{padding:1rem}}</style></head><body>{body}</body></html>'''
        args.preview_output.parent.mkdir(parents=True, exist_ok=True)
        args.preview_output.write_text(preview, encoding="utf-8")
        print(f"wrote {args.preview_output} ({len(preview)} characters)")


if __name__ == "__main__":
    main()
