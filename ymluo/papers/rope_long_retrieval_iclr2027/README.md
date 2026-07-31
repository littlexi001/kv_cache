# SAGE-RoPE -- ICLR 2027 draft

This directory is a self-contained LaTeX project for the paper currently
titled **“Local Position, Global Semantics: Phase-Sensitive Retrieval under
RoPE.”**

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

- `output/pdf/SAGE_RoPE_ICLR2027_draft_anonymous.pdf`
- `output/pdf/SAGE_RoPE_ICLR2027_draft_author.pdf`
- `output/pdf/SAGE_RoPE_ICLR2027_draft_zh.pdf` (Chinese reading edition)

This avoids installing a full TeX Live distribution. The first Tectonic build
may download missing LaTeX packages; later builds are cached and fast.

## Editing map

- `main.tex`: title, abstract, shared macros, mandatory statements.
- `main_zh.tex` and `sections_zh/`: synchronized Chinese reading edition.
- `scripts/check_bilingual_sync.py`: rejects drift in equations, labels,
  citations, measured numbers, environments, and TODO counts.
- `sections/01_introduction.tex`: problem → analysis → method → results story.
- `sections/03_mechanism.tex`: first-layer RoPE derivation and cross-layer chain.
- `sections/04_method.tex`: SAGE-RoPE and the conservative SAGE-Post variant.
- `sections/05_experiments.tex`: verified pilot results and clearly marked TODOs.
- `sections/appendix.tex`: proofs, full tables, protocols, and planned work.
- `TODO_EXPERIMENTS.md`: prioritized experiment checklist for a submission-ready
  paper.
- `notes/introduction_outline_zh.md`: the original Chinese introduction outline.

All red `TODO` boxes intentionally identify claims that are not yet supported
by completed experiments. They should be resolved, not hidden, before
submission.
