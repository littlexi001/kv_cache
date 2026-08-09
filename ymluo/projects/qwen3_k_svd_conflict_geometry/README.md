# Qwen3 K-SVD conflict geometry

This project studies how correct and contradictory synthetic reasoning chains are
represented in the key (`K`) geometry of Qwen3-0.6B.  It reuses the exactly paired
four-condition generator from `qwen3_local_rule_failure_boundary`.

The main comparison is the length- and position-matched 8K pair:

- filler + gold chain
- the same filler + gold chain + a contradictory chain

For each layer and KV head, the scripts construct a shared, uncentered covariance
matrix from the two prompts' post-RoPE keys.  Its eigenspace is used to decompose
gold/conflict keys and the answer-position query into principal and tail components.
Ranks 4, 8, 16, 32, and 64 are reported; rank 16 is the primary analysis.

The experiment also pairs the exact sub-tokens of the shared start code at its
occurrence in `VERIFIED RULE T0` and `DECOY RULE X0`.

The completed 64-seed results and Chinese analysis are in
`doc/k_svd_conflict_geometry_results_20260717.md`.  Aggregated CSVs are under
`outputs/k_svd_geometry_64seed_final_20260717/final_results/`; full per-seed rows
remain on the remote server because they are substantially larger.
