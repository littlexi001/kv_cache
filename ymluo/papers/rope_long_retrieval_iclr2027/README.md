# When Position Overrules Evidence -- ICLR 2027 draft

> **Method-audit update (2026-08-01):** the current `SAGE-RoPE` system should
> not be presented as a mature new positional encoding. RoPE-free proposal has
> direct prior art, while the registered suppression-gate, sparse phase-repair,
> query-span, and KVQ-relay probes did not pass their stop rules. The recommended
> paper is now a causal-mechanism study, with the pre-RoPE proposal retained only as a
> diagnostic intervention baseline. A strict no-op-matched BF16 intervention
> now reproduces the local score-to-margin closure at 8K, 16K, and 24K (eight
> seeds each; evidence-coordinate Pearson 0.960/0.973/0.936), while attention
> probability and grid-envelope RoPE suppression show no reliable seed-macro
> gold-minus-conflict mean gap. This strengthens the mechanism pivot, not the method
> claim. See the
> [full verdict](../../projects/qwen3_local_rule_failure_boundary/analysis/rope_method_search_20260801/final_method_verdict_20260801.md).
> The manuscript has now been restructured around this mechanism claim.
> The pre-RoPE proposal remains only in the appendix as a diagnostic intervention baseline.

Protocol scope is explicit: the NF4 layer-0 phase reconstruction, the
position-extended BF16 age trajectory, the native-window BF16 population
intervention, and the held-out NF4 retrieval diagnostic are separate controlled
regimes. They triangulate adjacent links; they are not pooled as one full-path
mediation experiment.

This directory is a self-contained LaTeX project for the paper currently
titled **“When Position Overrules Evidence: Tracing RoPE-Induced Retrieval
Failures Through Transformer Depth.”**

The draft uses the official ICLR 2027 style files downloaded from the
[ICLR 2027 author guide](https://iclr.cc/Conferences/2027/AuthorGuidelines).
The main paper is anonymous by default. `main_author.tex` builds an internal
author version; update its author metadata before circulation.

## Build on this Windows workspace

```powershell
cd ymluo\papers\rope_long_retrieval_iclr2027
.\build.ps1
```

The script uses the bundled Tectonic executable, generates all figures from
the checked-in CSV files, and writes PDFs to:

- `output/pdf/RoPE_Mechanism_ICLR2027_draft_anonymous.pdf`
- `output/pdf/RoPE_Mechanism_ICLR2027_draft_author.pdf`
- `output/pdf/RoPE_Mechanism_ICLR2027_draft_zh.pdf` (Chinese reading edition)

This avoids installing a full TeX Live distribution. The first Tectonic build
may download missing LaTeX packages; later builds are cached and fast.

## Editing map

- `main.tex`: title, abstract, shared macros, mandatory statements.
- `main_zh.tex` and `sections_zh/`: synchronized Chinese reading edition.
- `scripts/check_bilingual_sync.py`: rejects drift in equations, labels,
  citations, measured numbers, environments, and TODO counts.
- `sections/01_introduction.tex`: problem → mechanism → causal evidence story.
- `sections/03_mechanism.tex`: RoPE phase, answer-directed score utility, and cross-layer chain.
- `sections/04_method.tex`: causal interventions and phase-repair identifiability.
- `sections/05_experiments.tex`: no-op-matched BF16 closure and strict limits.
- `sections/appendix.tex`: proofs, audits, diagnostic baseline, and planned work.
- `TODO_EXPERIMENTS.md`: prioritized experiment checklist for a submission-ready
  paper.
- `notes/introduction_outline_zh.md`: the original Chinese introduction outline.

All red `TODO` boxes intentionally identify claims that are not yet supported
by completed experiments. They should be resolved, not hidden, before
submission.
