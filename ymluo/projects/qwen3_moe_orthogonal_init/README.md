# Qwen3 MoE Orthogonal Init

This project tests whether expert and gate initialization controls final MoE
selectivity.

At model construction time it can orthogonalize:

- router rows, one row per expert;
- expert flattened parameter vectors, using the Frobenius inner product.

For head-level MoE, orthogonalization is applied independently inside each
attention head.

Run:

```bash
bash ymluo/projects/qwen3_moe_orthogonal_init/scripts/run_train.sh
```

Background run:

```bash
bash ymluo/projects/qwen3_moe_orthogonal_init/scripts/nohup_train.sh
```

Useful overrides:

```bash
INIT_CHECKPOINT=ymluo/projects/qwen3_moe_orthogonal_init/outputs/train/moe-orthogonal-init/checkpoints/1000.pth \
ORTHOGONALIZE_GATE=true \
ORTHOGONALIZE_EXPERTS=true \
ORTHOGONAL_INIT_MODE=preserve_norm \
SEED=1 \
bash ymluo/projects/qwen3_moe_orthogonal_init/scripts/run_train.sh
```

Use seed sweeps to test sensitivity:

```bash
for s in 1 2 3 4 5; do
  RUN_NAME=orth-init-seed-${s} SEED=${s} \
  bash ymluo/projects/qwen3_moe_orthogonal_init/scripts/nohup_train.sh
done
```

Eval rows in `metrics.jsonl` report loss, accuracy,
`same_higher_same_expert`, `local_slot_history_mass`, and
`higher_level_history_mass`.
