import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_md", required=True)
    args = parser.parse_args()

    root = Path(args.root)
    rows = []
    for path in sorted(root.glob("gpus*/summary.json")):
        with path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        gpu_count = int(path.parent.name.replace("gpus", ""))
        for row in summary["method_results"]:
            rows.append(
                {
                    "gpus": gpu_count,
                    "method": row["method"],
                    "elapsed_s": row["elapsed_s_mean"],
                    "elapsed_s_std": row["elapsed_s_std"],
                    "queries_per_s": row["queries_per_s"],
                    "blocks_per_s": row["blocks_per_s"],
                    "all_support_recall": row["all_support_recall"],
                    "any_support_recall": row["any_support_recall"],
                    "answer_proxy_acc": row["answer_proxy_acc"],
                    "clean_answer_proxy_acc": row.get("clean_answer_proxy_acc", ""),
                    "risk_trigger_rate": row.get("risk_trigger_rate", ""),
                }
            )

    by_method = {}
    for row in rows:
        by_method.setdefault(row["method"], {})[row["gpus"]] = row

    for method, gpu_rows in by_method.items():
        base = gpu_rows.get(1)
        if not base:
            continue
        base_time = float(base["elapsed_s"])
        for row in gpu_rows.values():
            row["speedup_vs_1gpu"] = base_time / float(row["elapsed_s"])
            row["parallel_efficiency"] = row["speedup_vs_1gpu"] / int(row["gpus"])

    fieldnames = [
        "gpus",
        "method",
        "elapsed_s",
        "elapsed_s_std",
        "queries_per_s",
        "blocks_per_s",
        "speedup_vs_1gpu",
        "parallel_efficiency",
        "all_support_recall",
        "any_support_recall",
        "answer_proxy_acc",
        "clean_answer_proxy_acc",
        "risk_trigger_rate",
    ]
    with Path(args.out_csv).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r["method"], r["gpus"])):
            writer.writerow(row)

    lines = []
    lines.append("# Parallel retrieval scaling summary")
    lines.append("")
    lines.append("| method | gpus | elapsed s | speedup | efficiency | all support recall | clean proxy | blocks/s |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in sorted(rows, key=lambda r: (r["method"], r["gpus"])):
        lines.append(
            "| {method} | {gpus} | {elapsed_s:.4f} | {speedup_vs_1gpu:.2f} | "
            "{parallel_efficiency:.2f} | {all_support_recall:.4f} | "
            "{clean_answer_proxy_acc:.4f} | {blocks_per_s:.2f} |".format(
                **{
                    **row,
                    "elapsed_s": float(row["elapsed_s"]),
                    "speedup_vs_1gpu": float(row.get("speedup_vs_1gpu", 1.0)),
                    "parallel_efficiency": float(row.get("parallel_efficiency", 1.0)),
                    "all_support_recall": float(row["all_support_recall"]),
                    "clean_answer_proxy_acc": float(row.get("clean_answer_proxy_acc") or 0.0),
                    "blocks_per_s": float(row["blocks_per_s"]),
                }
            )
        )
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
