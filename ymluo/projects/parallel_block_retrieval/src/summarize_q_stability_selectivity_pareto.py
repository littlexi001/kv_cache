from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolidate generated-Q smoothing/probe retrieval summaries."
    )
    parser.add_argument("--summary", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path_text in args.summary:
        path = Path(path_text)
        payload = json.loads(path.read_text(encoding="utf-8"))
        transition = payload["transition"]
        trajectory = payload["trajectory"]
        rows.append(
            {
                "method": payload.get("q_field", "native_generation_q"),
                "selected_q_consecutive_cosine": transition["selected_q_cosine_mean"],
                "head_top16_consecutive_jaccard": transition[
                    "head_top16_jaccard_mean"
                ],
                "rrf39_consecutive_jaccard": transition["rrf39_jaccard_mean"],
                "hop2_rrf39_ever_recall": trajectory[
                    "q_rrf_hop2_ever_hit39_rate"
                ],
                "hop2_anyhead_top16_ever_recall": trajectory[
                    "q_anyhead_hop2_ever_hit16_rate"
                ],
                "world_size": payload["world_size"],
                "qk_scan_wall_seconds": payload["qk_scan_wall_seconds"],
            }
        )
    output = {
        "source": "stability-selectivity Pareto over real generated Q and real 10M K",
        "contains_synthetic_vectors": False,
        "rows": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
