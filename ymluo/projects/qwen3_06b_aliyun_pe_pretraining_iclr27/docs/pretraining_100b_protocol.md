# Qwen3-0.6B: 16-condition, 100B-token pretraining protocol

## Exact meaning of “pretraining”

The default `INITIALIZATION=from_scratch` reads the architecture config and
tokenizer from `/mnt/workspace/Qwen3-0.6B`, but does **not** load
`model.safetensors`. All model weights are initialized from the same seed. Set
`INITIALIZATION=checkpoint` only when intentionally switching back to continued
pretraining.

## Fixed training contract

| Variable | Value | Meaning |
|---|---:|---|
| Tasks | 16 | one PE condition per independent machine |
| GPUs per task | 8 | one distributed process per GPU |
| Sequence length | 8,192 | tokens in every packed training sequence |
| Micro-batch | 1 sequence/GPU | 8 sequences per micro-step |
| Gradient accumulation | 32 | 32 micro-steps per optimizer update |
| Global batch | 256 sequences | \(1\times8\times32\) |
| Tokens/update | 2,097,152 | \(256\times8192\) |
| Target/task | 100,000,000,000 tokens | 100B per PE condition |
| Optimizer steps | 47,684 | ceiling of target divided by tokens/update |
| Actual tokens | 100,000,595,968 | unavoidable final-step overshoot |
| Peak LR | \(10^{-4}\) | user-specified |
| Warmup | 500 updates | about 1.05% of training |
| Scheduler | cosine | decays to zero at the shared final step |
| AdamW | \(\beta_1=0.9,\beta_2=0.95,\epsilon=10^{-8}\) | fixed for every task |
| Weight decay | 0.1 | fixed for every task |

If the actual GPU count, micro-batch, or accumulation does not produce global
batch 256, the run fails before training. Therefore a missing GPU cannot silently
change the number of tokens seen per checkpoint.

## Checkpoint and evaluation schedule

Every condition checkpoints near 0.1B, 1B, 10B, 25B, 50B, 75B, and 100B
tokens. The controller converts token targets to steps using ceiling division
and saves the exact schedule in `token_schedule.json`.

At each checkpoint the distributed trainer exits cleanly, then the controller
runs:

1. held-out DCLM next-token PPL;
2. controlled single-needle, multi-needle, and variable-tracking probes at
   2K/4K/8K;
3. the configured fixed LongBench subset;
4. per-example gold-answer NLL and generation metrics.

It then resumes from the validated local checkpoint. A candidate passes a
screening checkpoint only if it improves a long-context metric over task 0
(`native_rope`) at matched tokens, has identical completed sample counts, and
keeps DCLM PPL within 2% of native.

## Data contract and limitation

The default manifest deterministically selects 200,000 training text files and
1,024 disjoint validation files from `/mnt/workspace/dclm`. Every task records
the same manifest SHA256; result merging refuses different hashes. The dataset
streams packed text indefinitely. Before publication, inspect corpus token
coverage and increase `TRAIN_FILES` if this sample would repeat too often during
100B tokens. A claimed “100B-token pretraining result” must report both tokens
processed and estimated unique-corpus coverage.

## TensorBoard

Trainer writes event files to:

```text
/mnt/workspace/pe_pretrain_100b_iclr27/<strategy>/tensorboard/
```

`run_pretrain_worker.sh` automatically starts TensorBoard on port 6006. The
visible scalars include training loss, learning rate, gradient norm,
`progress/tokens_seen`, `progress/percent`, held-out PPL, controlled retrieval,
and LongBench checkpoint metrics. Open:

```text
http://<machine-address>:6006
```

`TENSORBOARD_HOST=0.0.0.0` allows remote access. Restrict the instance security
group or use SSH port forwarding if the endpoint should not be public:

```bash
ssh -L 6006:127.0.0.1:6006 user@machine
```

## One task per machine

After installing the same package and `.env` on all instances:

```bash
# Run ID N on machine N, where N is 0..15.
bash scripts/run_pretrain_worker.sh N
```

With passwordless SSH, fill `configs/sixteen_hosts.conf` and dispatch all tasks:

```bash
cp configs/sixteen_hosts.example configs/sixteen_hosts.conf
bash scripts/launch_sixteen_hosts.sh configs/sixteen_hosts.conf
```

Status and stop commands are:

```bash
bash scripts/status_sixteen_hosts.sh configs/sixteen_hosts.conf
bash scripts/stop_sixteen_hosts.sh configs/sixteen_hosts.conf
```

## Claim boundary

Sixteen single-seed 100B runs can rank architectural candidates, but they do not
by themselves establish a publication claim. Advance at most two candidates,
then run multiple seeds and a full RULER/LongBench/short-capability evaluation.
Methods 14 and 15 are new project candidates; their novelty and effectiveness
remain hypotheses until literature review and matched experimental results are
complete.
