# Qwen3-8B LongBench gold-evidence compression pilot

This project tests a narrow question: if a retriever perfectly selects the
human-annotated evidence for a small LongBench HotpotQA subset, how much answer
quality can the reader recover?

The experiment is a **perfect-evidence retrieval diagnostic**, not a mathematical
upper bound on LongBench and not a position-preserving sparse-KV experiment.
Compressing the context also shortens positions and removes softmax competitors.

Primary research records:

- `docs/design.md`
- `docs/experiment_design.md`
- `docs/visualization_results.md`

Main programs:

- `src/run_hotpot_oracle_pilot.py`: alignment, frozen sampling, inference, and
  per-shard artifacts.
- `src/summarize_hotpot_oracle_pilot.py`: paired aggregation and the final report.
- `scripts/run_remote_gpu67.sh`: two-GPU launcher for the remote server.

Closed pilot results are in
`outputs/hotpot_gold_evidence_pilot_20260802_v2/merged/`; the concise Chinese
interpretation is in `docs/visualization_results.md`.
