from __future__ import annotations

from pathlib import Path


INTRO = """A research consortium is preparing a long technical report about resilient energy systems for remote communities. The report follows one central project across chemistry, manufacturing, deployment planning, economic modeling, field operations, and policy review. It is deliberately written as a coherent long document rather than as unrelated paragraphs, because the KV-cache geometry experiment should observe how a single growing context accumulates repeated entities, new topics, cross references, and delayed dependencies."""


SECTION_TEMPLATES = [
    {
        "title": "Material Platform",
        "body": [
            "The material team studies a sodium-based cathode, a porous carbon scaffold, and an electrolyte additive that slows surface degradation during storage at elevated temperature.",
            "The cathode chemistry is described with repeated attention to cost, abundance, and compatibility with existing coating equipment.",
            "The carbon scaffold appears several times because it connects microscopic transport, mechanical stability, and long-cycle performance.",
            "The electrolyte additive is treated as a small component with an outsized effect: it changes interface formation, storage behavior, and safety margin.",
        ],
    },
    {
        "title": "Manufacturing Flow",
        "body": [
            "The manufacturing group defines a process that includes slurry preparation, coating, drying, calendaring, cell stacking, electrolyte filling, sealing, formation cycling, and final inspection.",
            "Each stage produces measurements that later sections reuse: coating uniformity, residual solvent, electrode density, impedance, gas generation, and capacity retention.",
            "The process is not merely a list of operations. It is a chain of constraints in which an early moisture error can become a late safety problem.",
            "The report therefore treats manufacturing records as a form of memory that must remain searchable when later deployment decisions are made.",
        ],
    },
    {
        "title": "Laboratory Evaluation",
        "body": [
            "The laboratory protocol tests charge rate, discharge rate, high-temperature storage, low-temperature recovery, calendar aging, cycle aging, impedance growth, and abuse tolerance.",
            "The researchers distinguish local measurements from global conclusions. A single cell may show excellent retention, but the deployment model needs a distribution across many cells.",
            "Some measurements are repeated because they anchor later claims. Capacity retention, impedance growth, and thermal response are used again in the field model.",
            "The document asks the reader to remember which experiment supports which operational claim, and this creates a long-range retrieval requirement.",
        ],
    },
    {
        "title": "Microgrid Deployment",
        "body": [
            "The deployment team studies a coastal town, a desert clinic, an island school, and a mountain relay station. Each site has a different climate, maintenance schedule, and load profile.",
            "The same battery technology is evaluated through different constraints: transportation cost, installation labor, replacement interval, enclosure cooling, emergency reserve, and certification.",
            "A fact introduced in the chemistry section can become relevant again when the deployment model discusses hot climates or limited maintenance visits.",
            "This section introduces many new nouns but also reuses earlier technical entities, which makes it useful for measuring whether the KV geometry expands or revisits an existing subspace.",
        ],
    },
    {
        "title": "Economic Model",
        "body": [
            "The economic model includes capital cost, shipping cost, installation cost, diesel offset, maintenance visits, expected lifetime, replacement schedule, insurance, and financing.",
            "The model is sensitive to cycle life and high-temperature stability, so it repeatedly points back to laboratory results without restating every experimental detail.",
            "The report compares sodium-based storage with lithium iron phosphate storage and diesel-only backup. Each comparison depends on a different subset of evidence.",
            "This creates delayed dependencies: a number measured early in the document becomes important only after the reader reaches a later economic scenario.",
        ],
    },
    {
        "title": "Risk Register",
        "body": [
            "The risk register lists supplier concentration, additive cost, coating variability, gas generation, certification delay, site training, spare part availability, and climate mismatch.",
            "Some risks are technical, some are operational, and some are institutional. The document intentionally mixes these categories because real planning does not preserve clean boundaries.",
            "The report repeatedly asks whether a risk is local to one site or global across the technology platform.",
            "A useful memory representation should keep the stable entities visible while allowing local details to fade unless they are needed by a later question.",
        ],
    },
]


CONNECTORS = [
    "The authors return to this point later when they compare laboratory evidence with field constraints.",
    "This detail is intentionally repeated because it should become a recognizable anchor in a long context.",
    "The section also introduces a minor exception that matters only after several more pages of discussion.",
    "The report uses this example to connect a physical measurement with a planning decision.",
    "The same entity appears under a new role, which is useful for testing whether the representation tracks identity across distance.",
]


def build_section(index: int) -> str:
    template = SECTION_TEMPLATES[index % len(SECTION_TEMPLATES)]
    site = ["coastal town", "desert clinic", "island school", "mountain relay station"][index % 4]
    season = ["summer", "winter", "storm season", "dry season"][index % 4]
    paragraphs = [f"Section {index + 1}: {template['title']}."]
    for repeat in range(3):
        for sentence in template["body"]:
            paragraphs.append(sentence)
        paragraphs.append(CONNECTORS[(index + repeat) % len(CONNECTORS)])
        paragraphs.append(
            f"In scenario {index + 1}.{repeat + 1}, the {site} is evaluated during {season}, "
            f"and the analysis records maintenance interval M{index % 7 + 1}, temperature band T{repeat + 2}, "
            f"and evidence tag E{index + 100}-{repeat + 1} for later cross reference."
        )
        paragraphs.append(
            "The repeated identifiers are not meant to be meaningful by themselves. "
            "They create exact tokens that should remain recoverable even as surrounding prose becomes compressible."
        )
    return "\n".join(paragraphs)


def main() -> None:
    out_path = Path("fdong_seq_compress/data/synthetic_texts/long_english_12000_words.txt")
    sections = [INTRO]
    for idx in range(72):
        sections.append(build_section(idx))
    text = "\n\n".join(sections) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"Approx words: {len(text.split())}")
    print(f"Characters: {len(text)}")


if __name__ == "__main__":
    main()
