from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def normalize_answer(text: str, answer_lang: str) -> str:
    text = text.replace("\\n", "\n")
    first = re.split(r"\n|答案[:：]|问题[:：]|Answer[:：]|Question[:：]", text, maxsplit=1)[0]
    compact = re.sub(r"\s+", "", first)
    if answer_lang == "zh":
        if "九岁" in compact or "9岁" in compact or "九歲" in compact:
            return "correct"
        if any(item in compact for item in ("无法确定", "不能确定", "不知道", "没有提到", "未提到")):
            return "miss"
    else:
        lower = first.lower()
        if "nine" in lower or re.search(r"\b9\b", lower):
            return "correct"
        if any(item in lower for item in ("unknown", "cannot determine", "not mentioned", "not contain", "do not know")):
            return "miss"
    return "wrong"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    buckets: dict[tuple[int, float], list[dict[str, str]]] = {}
    for row in rows:
        buckets.setdefault((int(row["target_length"]), float(row["depth_percent"])), []).append(row)
    summary: list[dict[str, object]] = []
    for (length, depth), bucket in sorted(buckets.items()):
        count = len(bucket)
        accuracy = sum(int(row["correct"]) for row in bucket) / max(1, count)
        miss = sum(int(row["miss"]) for row in bucket) / max(1, count)
        wrong = sum(int(row["wrong"]) for row in bucket) / max(1, count)
        ppl = [float(row["answer_ppl"]) for row in bucket if row.get("answer_ppl")]
        mass = [
            float(row["mass_mean_all_layers_heads"])
            for row in bucket
            if row.get("mass_mean_all_layers_heads") not in {"", None}
        ]
        summary.append(
            {
                "target_length": length,
                "depth_percent": depth,
                "cases": count,
                "accuracy": f"{accuracy:.6f}",
                "miss_rate": f"{miss:.6f}",
                "wrong_rate": f"{wrong:.6f}",
                "mean_answer_ppl": "" if not ppl else f"{sum(ppl) / len(ppl):.6f}",
                "mean_evidence_mass": "" if not mass else f"{sum(mass) / len(mass):.8f}",
            }
        )
    return summary


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Long Needle Age Summary",
        "",
        "| length | depth | cases | accuracy | miss | wrong | answer PPL | evidence mass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['target_length']} | {row['depth_percent']} | {row['cases']} | "
            f"{row['accuracy']} | {row['miss_rate']} | {row['wrong_rate']} | "
            f"{row['mean_answer_ppl']} | {row['mean_evidence_mass']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reclassify(output_dir: Path) -> None:
    rows = read_csv(output_dir / "generation_results.csv")
    for row in rows:
        answer_lang = row.get("needle_prompt_lang", "zh")
        answer_class = normalize_answer(row.get("generated_text", ""), answer_lang)
        row["answer_class"] = answer_class
        row["correct"] = str(int(answer_class == "correct"))
        row["miss"] = str(int(answer_class == "miss"))
        row["wrong"] = str(int(answer_class == "wrong"))
    for filename in ["generation_results.csv", "answer_ppl.csv", "evidence_attention_mass.csv"]:
        write_csv(output_dir / filename, rows)
    summary = summarize(rows)
    write_csv(output_dir / "summary_by_length.csv", summary)
    write_markdown(output_dir / "summary_by_length.md", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reclassify existing generated_text fields and rebuild summaries.")
    parser.add_argument("output_dirs", nargs="+")
    args = parser.parse_args()
    for item in args.output_dirs:
        output_dir = Path(item)
        reclassify(output_dir)
        print(f"updated {output_dir}")


if __name__ == "__main__":
    main()
