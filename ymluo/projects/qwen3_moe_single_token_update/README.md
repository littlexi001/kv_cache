# Qwen3 MoE Single Token Update

This project tests the small-batch / one-token-update hypothesis. The model
still receives a full synthetic context, but each optimizer step backpropagates
cross-entropy through one target token only.

Run:

```bash
bash ymluo/projects/qwen3_moe_single_token_update/scripts/run_train.sh
```

Background run:

```bash
bash ymluo/projects/qwen3_moe_single_token_update/scripts/nohup_train.sh
```

Useful overrides:

```bash
INIT_CHECKPOINT=ymluo/projects/qwen3_moe_single_token_update/outputs/train/moe-single-token-update/checkpoints/1000.pth \
BATCH_SIZE=1 \
SINGLE_TOKEN_POSITION=random \
TOTAL_STEPS=20000 \
MOE_HEAD_LEVEL=true \
bash ymluo/projects/qwen3_moe_single_token_update/scripts/run_train.sh
```

Set `SINGLE_TOKEN_POSITION=last` to always train on the final next-token
prediction, or `cycle` to sweep positions deterministically.

Eval rows in `metrics.jsonl` report loss, accuracy,
`same_higher_same_expert`, `local_slot_history_mass`, and
`higher_level_history_mass`.

Eval a saved checkpoint without training:

```bash
CKPT_FILE=ymluo/projects/qwen3_moe_single_token_update/outputs/train/moe-single-token-update/checkpoints/10000.pth \
bash ymluo/projects/qwen3_moe_single_token_update/scripts/run_eval.sh
```
