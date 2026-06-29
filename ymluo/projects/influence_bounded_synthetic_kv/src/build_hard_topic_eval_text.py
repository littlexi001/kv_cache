from __future__ import annotations

import argparse
import random
from pathlib import Path


TOPICS = [
    "implant firmware audit",
    "cross-border tax dispute",
    "quantum control calibration",
    "satellite fault isolation",
    "rare disease case review",
    "battery warranty litigation",
    "database recovery drill",
    "protein assay deviation",
    "grid congestion auction",
    "compiler miscompile report",
]


FIELDS = [
    "phase_offset",
    "risk_weight",
    "liability_cap",
    "enzyme_rate",
    "queue_depth",
    "thermal_slope",
    "jurisdiction_code",
    "packet_loss",
    "confidence_floor",
    "stability_margin",
]


def token(rng: random.Random, prefix: str, width: int = 6) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return prefix + "-" + "".join(rng.choice(alphabet) for _ in range(width))


def build_case(rng: random.Random, idx: int) -> str:
    topic = TOPICS[idx % len(TOPICS)]
    local_fields = rng.sample(FIELDS, 5)
    case_id = token(rng, f"C{idx % 997:03d}")
    clause_id = token(rng, "CL", 5)
    sensor_id = token(rng, "S", 7)
    checksum = token(rng, "HX", 8)
    values = [rng.uniform(-70.0, 170.0), rng.uniform(0.0003, 9.7), rng.uniform(12.0, 8800.0)]
    ints = [rng.randint(11, 99999), rng.randint(2000, 8999), rng.randint(1, 57)]
    decision = rng.choice(["hold", "reverse", "escalate", "split", "deny", "accept-with-reserve"])
    if idx % 3 == 0:
        trap = f"The exception is that {local_fields[0]} must be ignored unless {clause_id} appears after {sensor_id}."
    elif idx % 3 == 1:
        trap = f"The reviewer must compare {local_fields[1]} against the second number, not the largest number."
    else:
        trap = f"The final decision changes only if {checksum} and {case_id} share the same suffix class."
    return (
        f"Record {case_id}: topic={topic}. "
        f"Primary memo: {local_fields[0]}={values[0]:.6f}, {local_fields[1]}={values[1]:.6f}, "
        f"{local_fields[2]}={values[2]:.3f}. "
        f"Identifiers: sensor={sensor_id}, clause={clause_id}, checksum={checksum}. "
        f"Counts: event_count={ints[0]}, filing_year={ints[1]}, retry_bucket={ints[2]}. "
        f"Constraint: {trap} "
        f"Counterfactual note: if {local_fields[3]} is replaced by a cached approximation, preserve the next-token decision distribution rather than the literal evidence list. "
        f"Adjudication result={decision}; reviewer_code={token(rng, 'RV', 4)}; audit_link={token(rng, 'AUD', 7)}.\n"
    )


def build_text(paragraphs: int, seed: int) -> str:
    rng = random.Random(seed)
    lines: list[str] = []
    for idx in range(paragraphs):
        lines.append(build_case(rng, idx))
        if idx % 5 == 4:
            refs = ", ".join(token(rng, "REF", 5) for _ in range(6))
            lines.append(
                f"Cross-reference bundle {idx // 5}: use refs [{refs}] only after reconciling the previous five records. "
                f"The bundle deliberately changes topic and identifier style to make public-corpus memorization unlikely.\n"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build harder deterministic topic evaluation text.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--paragraphs", type=int, default=360)
    parser.add_argument("--seed", type=int, default=202606291)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_text(args.paragraphs, args.seed), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
