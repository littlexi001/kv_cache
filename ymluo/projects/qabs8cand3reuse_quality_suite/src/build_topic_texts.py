from __future__ import annotations

import argparse
from pathlib import Path


TOPICS = {
    "science": [
        "A field notebook from a climate observatory describes how ocean temperature, cloud cover, and aerosol density interact across several seasons.",
        "The researcher compares satellite measurements with buoy records, then explains why small calibration errors can accumulate into large model differences.",
        "Each section introduces a hypothesis, reports the measured uncertainty, and revises the conclusion when later observations contradict the first explanation.",
    ],
    "finance": [
        "A portfolio memo reviews cash flow, duration risk, inventory turnover, and foreign exchange exposure for a manufacturer with customers in three regions.",
        "The analyst separates recurring operating profit from one-time gains, then checks whether debt covenants would still hold after a demand shock.",
        "Management argues that margin expansion came from logistics efficiency, but the footnotes show that pricing power and deferred maintenance also contributed.",
    ],
    "software": [
        "The incident report traces a latency regression from an API gateway through a queue consumer, a cache invalidation path, and a database migration.",
        "Engineers reproduce the bug with a deterministic fixture, add counters around the hot path, and remove a retry loop that amplified partial failures.",
        "The final patch keeps the public interface stable while changing ownership of the connection pool and tightening timeout propagation between services.",
    ],
    "history": [
        "The chapter compares two port cities that grew through trade, migration, shipbuilding, and changes in imperial tax policy during the nineteenth century.",
        "Local archives show that merchants adapted faster than officials, creating informal credit networks before formal banking institutions reached the harbor.",
        "The narrative alternates between political decisions, household letters, and cargo ledgers to show how global events changed ordinary routines.",
    ],
    "literature": [
        "The novel opens with a quiet dinner where every polite remark hides an older disappointment, and the narrator notices what the guests refuse to say.",
        "A recurring image of rain on the station roof links memory, departure, and the fragile promises that characters make when they cannot speak directly.",
        "The critic argues that the style depends less on plot surprise than on rhythm, withheld judgment, and the gradual return of a forgotten phrase.",
    ],
    "mixed_qa": [
        "Question: Why did the laboratory repeat the trial? Answer: The first measurement conflicted with the control sample, so the team needed a cleaner estimate.",
        "Question: What caused the service outage? Answer: A stale cache entry triggered repeated retries, which saturated the worker queue during peak traffic.",
        "Question: How did the expedition finance repairs? Answer: The captain sold part of the cargo early and used the proceeds to replace damaged rigging.",
    ],
}


def build_text(topic: str, target_repetitions: int) -> str:
    paragraphs = TOPICS[topic]
    blocks: list[str] = []
    for index in range(target_repetitions):
        blocks.append(f"Section {index + 1}: {topic.replace('_', ' ').title()}\n")
        for paragraph in paragraphs:
            blocks.append(paragraph)
        blocks.append(
            "The passage deliberately keeps a coherent local topic while changing details across repetitions, "
            "so sparse attention modes are compared on consistent but non-identical context."
        )
        blocks.append("")
    return "\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create controlled topic text files for qabs8cand3reuse quality tests.")
    parser.add_argument("--output_dir", default="ymluo/projects/qabs8cand3reuse_quality_suite/data/topic_texts")
    parser.add_argument("--target_repetitions", type=int, default=260)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for topic in TOPICS:
        path = output_dir / f"{topic}.txt"
        if path.exists() and not args.overwrite:
            continue
        path.write_text(build_text(topic, args.target_repetitions), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()

