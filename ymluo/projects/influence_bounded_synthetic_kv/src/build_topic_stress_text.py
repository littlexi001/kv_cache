from __future__ import annotations

import argparse
import random
from pathlib import Path


TOPICS = [
    (
        "silicon photonics",
        "The wafer run studies ring-resonator drift under bursty inference traffic.",
        ["heater trim", "phase noise", "reticle edge", "thermal guardband"],
    ),
    (
        "clinical triage",
        "The hospital board compares overnight queue policies for stroke alerts.",
        ["door-to-needle", "lab turnaround", "bed transfer", "contraindication"],
    ),
    (
        "maritime law",
        "The arbitration memo reviews liability for an autonomous cargo vessel.",
        ["salvage claim", "bill of lading", "port state", "collision rule"],
    ),
    (
        "grid storage",
        "The operator models lithium iron phosphate packs during a heat advisory.",
        ["state of charge", "frequency response", "curtailment", "degradation"],
    ),
    (
        "protein design",
        "The wet lab screens binder variants against a mutated receptor pocket.",
        ["affinity cliff", "glycan shield", "loop closure", "solubility"],
    ),
    (
        "compiler verification",
        "The proof team audits a vectorizing pass for undefined behavior leaks.",
        ["alias set", "SSA phi", "memory fence", "refinement proof"],
    ),
    (
        "climate sensing",
        "The field station reconciles drone lidar with ground humidity probes.",
        ["canopy gap", "soil flux", "radiometer bias", "calibration pole"],
    ),
    (
        "market microstructure",
        "The risk desk replays auction messages around an exchange halt.",
        ["quote fade", "imbalance feed", "latency band", "self trade"],
    ),
]


def code(rng: random.Random, prefix: str) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return prefix + "-" + "".join(rng.choice(alphabet) for _ in range(5))


def build_text(paragraphs: int, seed: int) -> str:
    rng = random.Random(seed)
    lines: list[str] = []
    for idx in range(paragraphs):
        topic, lead, terms = TOPICS[idx % len(TOPICS)]
        rng.shuffle(terms)
        case_id = code(rng, f"T{idx % 97:02d}")
        checksum = code(rng, "CHK")
        value_a = rng.uniform(0.03, 97.0)
        value_b = rng.uniform(100.0, 9900.0)
        value_c = rng.randint(7, 9991)
        decision = rng.choice(["defer", "approve", "escalate", "rollback", "quarantine"])
        owner = rng.choice(["north cell", "delta bench", "red team", "night desk", "field unit"])
        paragraph = (
            f"[{case_id}] Topic: {topic}. {lead} "
            f"The primary variable is {terms[0]}, but the error budget is dominated by {terms[1]}. "
            f"Analysts record value_a={value_a:.5f}, value_b={value_b:.2f}, and event_count={value_c}. "
            f"The interim decision is {decision}; owner={owner}; checksum={checksum}. "
            f"Ablation note: replacing the remote evidence with prototypes must preserve the downstream output, "
            f"not the literal archive. The reviewer asks whether {terms[2]} changes when {terms[3]} is held fixed. "
            f"Final observation {code(rng, 'OBS')} links this case to batch {code(rng, 'B')}, "
            f"which should be difficult to memorize from public corpora.\n"
        )
        lines.append(paragraph)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic multi-topic stress text for PPL tests.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--paragraphs", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260629)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_text(args.paragraphs, args.seed), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
