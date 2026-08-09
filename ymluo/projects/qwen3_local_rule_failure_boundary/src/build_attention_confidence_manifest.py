from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any


ANALYSIS_ONLY_ATTENTION_KEYS = (
    "head_role_logit_mean",
    "head_role_logit_max",
    "head_role_best_rank",
    "head_role_cosine_mean",
    "head_role_cosine_max",
    "head_role_key_norm_mean",
    "head_query_norm",
    "head_max_logit",
    "head_logsumexp",
    "head_top2pct_kept_mass",
    "head_top2pct_role_mass",
)


def compact_for_browser(value: Any, significant_digits: int) -> Any:
    """Keep chart fidelity while avoiding ten noisy float digits in static JSON."""

    if isinstance(value, float):
        return float(f"{value:.{significant_digits}g}")
    if isinstance(value, list):
        return [compact_for_browser(item, significant_digits) for item in value]
    if isinstance(value, dict):
        return {
            key: compact_for_browser(item, significant_digits)
            for key, item in value.items()
        }
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the lazy-loaded dashboard manifest.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--site_data_dir",
        default="",
        help="Optional compact browser bundle: manifest.json plus one gzip-compressed file per length.",
    )
    parser.add_argument("--browser_float_digits", type=int, default=6)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    data_dir = output_dir / "data"
    summaries: list[dict[str, Any]] = []
    sample: dict[str, Any] | None = None
    for path in data_dir.glob("length_*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        sample = sample or payload
        summaries.append(
            {
                "length": int(payload["target_context_tokens"]),
                "body_tokens": int(payload["body_tokens"]),
                "prompt_tokens": int(payload["prompt_tokens"]),
                "gold_ppl": float(payload["answer"]["gold_ppl"]),
                "gold_mean_nll": float(payload["answer"]["gold_mean_nll"]),
                "overall_entropy": float(payload["attention"]["overall_entropy"]),
                "overall_effective_tokens": float(payload["attention"]["overall_effective_tokens"]),
                "overall_role_mass": payload["attention"]["overall_role_mass"],
                "prefill_seconds": float(payload["timing"]["prefill_seconds"]),
                "file": f"data/{path.name}",
            }
        )
    summaries.sort(key=lambda row: row["length"])
    if not summaries or sample is None:
        raise SystemExit(f"no length_*.json files found under {data_dir}")
    manifest = {
        "schema_version": 1,
        "title": "Qwen3-8B · Clean two-hop attention confidence sweep",
        "model": sample["model"],
        "model_config": sample["model_config"],
        "condition": sample["condition"],
        "code_mode": sample.get("code_mode", "legacy"),
        "placement": sample["placement"],
        "gold_codes": sample["gold_codes"],
        "role_order": sample["attention"]["role_order"],
        "max_top": sample["attention"]["max_top"],
        "length_step": 500,
        "completed_lengths": [row["length"] for row in summaries],
        "summaries": summaries,
    }
    destination = output_dir / "manifest.json"
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination} with {len(summaries)} lengths")
    if args.site_data_dir:
        site_data_dir = Path(args.site_data_dir)
        site_data_dir.mkdir(parents=True, exist_ok=True)
        compact_summaries: list[dict[str, Any]] = []
        for summary in summaries:
            source = output_dir / summary["file"]
            compressed_name = source.name + ".gz"
            compressed_path = site_data_dir / compressed_name
            browser_payload = compact_for_browser(
                json.loads(source.read_text(encoding="utf-8")),
                args.browser_float_digits,
            )
            # These dense per-head tensors are retained in the raw experiment
            # JSON and CSV reports.  The dashboard never reads them; excluding
            # them keeps 257 lazy-loaded browser files responsive on localhost.
            browser_attention = browser_payload.get("attention", {})
            for key in ANALYSIS_ONLY_ATTENTION_KEYS:
                browser_attention.pop(key, None)
            with gzip.open(compressed_path, "wt", encoding="utf-8", compresslevel=9) as output_handle:
                json.dump(
                    browser_payload,
                    output_handle,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            compact_summaries.append(
                {**summary, "file": f"data/{compressed_name}"}
            )
        compact_manifest = {
            **manifest,
            "summaries": compact_summaries,
            "compression": "gzip",
            "browser_float_significant_digits": args.browser_float_digits,
            "analysis_only_attention_keys_excluded": list(ANALYSIS_ONLY_ATTENTION_KEYS),
        }
        (site_data_dir / "manifest.json").write_text(
            json.dumps(compact_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote compact browser data to {site_data_dir}")


if __name__ == "__main__":
    main()
