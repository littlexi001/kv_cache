from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence


OUT_DIR = Path("fdong_seq_compress/data/synthetic_texts")


DOMAIN_CONFIGS = [
    {
        "slug": "biomed_long_range_facts",
        "title": "Specialized Biomedical Trial Registry",
        "entity": "trial",
        "domains": ["oncology", "immunology", "nephrology", "cardiology"],
        "attributes": [
            ("primary endpoint", ["DOR-17", "MFS-29", "PASI-41", "eGFR-63", "CRP-08"]),
            ("enzyme panel", ["CYP2C9", "UGT1A6", "NAT2", "SLC22A8", "ABCB1"]),
            ("adverse-event watchword", ["marginal edema", "delayed neutropenia", "transient rash", "quiet tachycardia"]),
            ("assay platform", ["NanoString-74", "ELISA-Delta", "LCMS-Orion", "FlowCyto-M7"]),
        ],
        "filler": [
            "The monitoring board required every site to preserve the original randomization packet and to report protocol deviations before database lock.",
            "Interim safety narratives were written in a deliberately conservative style because the registry mixed pragmatic and mechanistic cohorts.",
            "The statistical appendix emphasized covariate balance, blinded endpoint adjudication, and a narrow definition of rescue therapy.",
        ],
    },
    {
        "slug": "compiler_long_range_facts",
        "title": "Compiler Runtime Incident Notebook",
        "entity": "module",
        "domains": ["register allocation", "garbage collection", "vector lowering", "link-time optimization"],
        "attributes": [
            ("fault signature", ["SPILL-Delta-72", "GC-Safepoint-19", "VEC-Lane-44", "LTO-Reloc-88", "IR-Null-31"]),
            ("mitigation flag", ["--late-coalesce", "--pin-shadow-stack", "--split-wide-load", "--freeze-outline"]),
            ("trace marker", ["theta-frame", "ravel-block", "indigo-slot", "copper-edge"]),
            ("owner queue", ["runtime-core", "backend-simd", "toolchain-release", "optimizer-nightly"]),
        ],
        "filler": [
            "The compiler team reproduced each failure with deterministic scheduling and a pinned optimization seed before accepting the regression.",
            "Engineers treated every counterexample as a constraint on the lowering pipeline rather than as an isolated crash report.",
            "The nightly dashboard separated front-end parse drift from back-end register pressure so that triage remained mechanically auditable.",
        ],
    },
    {
        "slug": "geology_long_range_facts",
        "title": "Planetary Geology Core Sample Dossier",
        "entity": "sample",
        "domains": ["basalt stratigraphy", "impact glass", "cryovolcanic deposits", "magnetite veins"],
        "attributes": [
            ("isotope ratio code", ["Ar40-K39-118", "Sr87-Sr86-204", "O18-O16-077", "Nd143-Nd144-512"]),
            ("mineral inclusion", ["olivine needle", "zircon rind", "hematite seam", "pyroxene bead"]),
            ("instrument bay", ["Aster-3", "Borealis-9", "Calyx-5", "Dione-2"]),
            ("stratigraphic tag", ["unit lambda", "unit basalt-nine", "unit ceres-blue", "unit polar-ash"]),
        ],
        "filler": [
            "The field report distinguished transport texture from thermal alteration because the cores were collected across several depositional regimes.",
            "Each laboratory note preserved the original chain-of-custody identifier so later analysts could audit contamination hypotheses.",
            "The mineral map was interpreted alongside crater chronology, magnetic anomalies, and volatile retention estimates.",
        ],
    },
]


def make_code(rng: random.Random, prefix: str, idx: int) -> str:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return f"{prefix}-{rng.choice(letters)}{rng.choice(letters)}-{idx:03d}-{rng.randrange(1000, 9999)}"


def make_nonce_value(rng: random.Random, idx: int) -> str:
    letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    words = ["azurite", "kepler", "mistral", "solstice", "quartz", "vector", "umbra", "helix"]
    return (
        f"{rng.choice(letters)}{rng.choice(letters)}{rng.choice(letters)}-"
        f"{rng.randrange(10000, 99999)}-"
        f"{rng.choice(words)}-{idx:03d}-"
        f"{rng.choice(letters)}{rng.randrange(10, 99)}"
    )


def make_facts(config: Dict, rng: random.Random, count: int) -> List[Dict[str, str]]:
    facts = []
    for idx in range(count):
        code = make_code(rng, config["entity"].upper()[:3], idx)
        domain = rng.choice(config["domains"])
        attr_name, values = rng.choice(config["attributes"])
        value = rng.choice(values)
        # Add an arbitrary suffix so this cannot be answered from world knowledge alone.
        value = f"{value} / batch {rng.choice(['amber', 'cobalt', 'silver', 'violet'])}-{rng.randrange(11, 97)}"
        facts.append(
            {
                "code": code,
                "domain": domain,
                "attribute": attr_name,
                "value": value,
            }
        )
    return facts


def make_hard_facts(config: Dict, rng: random.Random, count: int) -> List[Dict[str, str]]:
    facts = []
    hard_attributes = [
        "private calibration key",
        "masked audit token",
        "nonstandard control label",
        "sealed retrieval code",
    ]
    for idx in range(count):
        facts.append(
            {
                "code": make_code(rng, config["entity"].upper()[:3], idx),
                "domain": rng.choice(config["domains"]),
                "attribute": rng.choice(hard_attributes),
                "value": make_nonce_value(rng, idx),
            }
        )
    return facts


def render_filler(config: Dict, rng: random.Random, blocks: int) -> List[str]:
    paragraphs = []
    for block_idx in range(blocks):
        chosen = rng.sample(config["filler"], k=len(config["filler"]))
        paragraphs.append(
            " ".join(chosen)
            + f" The review paragraph number {block_idx:02d} deliberately avoids restating the registry answers, "
            + "so the later audit section must rely on the earlier ledger rather than nearby lexical repetition."
        )
    return paragraphs


def render_text(config: Dict, seed: int, fact_count: int, filler_blocks: int, query_repeats: int) -> str:
    rng = random.Random(seed)
    facts = make_facts(config, rng, fact_count)
    return render_from_facts(config, rng, facts, filler_blocks, query_repeats, hard_mode=False)


def render_hard_text(config: Dict, seed: int, fact_count: int, filler_blocks: int, query_repeats: int) -> str:
    rng = random.Random(seed)
    facts = make_hard_facts(config, rng, fact_count)
    return render_from_facts(config, rng, facts, filler_blocks, query_repeats, hard_mode=True)


def render_from_facts(
    config: Dict,
    rng: random.Random,
    facts: Sequence[Dict[str, str]],
    filler_blocks: int,
    query_repeats: int,
    hard_mode: bool,
) -> str:
    lines: List[str] = []
    lines.append(config["title"])
    lines.append("")
    lines.append(
        "This document is a controlled long-range retrieval benchmark. "
        "The early ledger contains arbitrary identifiers and attribute values. "
        "The later audit asks for those values after a long distractor region."
    )
    lines.append("")
    lines.append("EARLY FACT LEDGER")
    for i, fact in enumerate(facts):
        lines.append(
            f"Ledger item {i:03d}: In the {fact['domain']} record, {fact['code']} has "
            f"{fact['attribute']} = {fact['value']}."
        )

    lines.append("")
    lines.append("DISTRACTOR TECHNICAL REPORT")
    lines.extend(render_filler(config, rng, filler_blocks))

    lines.append("")
    lines.append("LONG-RANGE RETRIEVAL AUDIT")
    query_order = list(range(len(facts)))
    rng.shuffle(query_order)
    for repeat in range(query_repeats):
        rng.shuffle(query_order)
        for qi, fact_idx in enumerate(query_order):
            fact = facts[fact_idx]
            if hard_mode:
                lines.append(
                    f"Audit query {repeat:02d}-{qi:03d}: The dossier asks for only one exact string. "
                    f"Look up {fact['code']} and copy its {fact['attribute']}. Exact answer: {fact['value']}."
                )
            else:
                lines.append(
                    f"Audit query {repeat:02d}-{qi:03d}: For {fact['code']}, what is the recorded "
                    f"{fact['attribute']}? Answer: {fact['value']}."
                )
            if qi % 7 == 0:
                lines.append(
                    "The auditor marks this as a memory lookup rather than a local language modeling exercise."
                )
    lines.append("")
    lines.append("END OF CONTROLLED LONG-RANGE FACT DOCUMENT")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for offset, config in enumerate(DOMAIN_CONFIGS):
        variants = [
            ("", 64, 90, 3, "Long version. Audit starts around 11k tokens for future large-context runs."),
            (
                "_compact",
                24,
                12,
                4,
                "Compact version. Audit starts around 2k tokens so 4k-token full-forward checks include retrieval answers.",
            ),
            (
                "_hard_compact",
                32,
                10,
                3,
                "Hard compact version. Answers are unique random nonce strings, so late audit tokens require exact long-range copying.",
            ),
        ]
        for suffix, fact_count, filler_blocks, query_repeats, design_note in variants:
            if suffix == "_hard_compact":
                text = render_hard_text(
                    config,
                    seed=20260604 + offset + 200,
                    fact_count=fact_count,
                    filler_blocks=filler_blocks,
                    query_repeats=query_repeats,
                )
            else:
                text = render_text(
                    config,
                    seed=20260604 + offset + (100 if suffix else 0),
                    fact_count=fact_count,
                    filler_blocks=filler_blocks,
                    query_repeats=query_repeats,
                )
            path = OUT_DIR / f"{config['slug']}{suffix}.txt"
            path.write_text(text, encoding="utf-8")
            manifest.append(
                {
                    "path": str(path),
                    "title": config["title"],
                    "fact_count": fact_count,
                    "query_repeats": query_repeats,
                    "design": f"Early arbitrary professional-domain fact ledger, distractor report, late retrieval audit. {design_note}",
                }
            )
            print(f"Wrote {path} chars={len(text)}")
    manifest_path = OUT_DIR / "long_range_fact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
