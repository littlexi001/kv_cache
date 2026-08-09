# Qwen3-0.6B: 16-way PE pretraining package

This package trains 16 Qwen3-0.6B positional-encoding conditions from the same
random initialization. It uses the Qwen checkpoint directory for architecture
and tokenizer files but, by default, does not load pretrained weights. Each
condition runs independently on one eight-GPU Alibaba Cloud instance.

## Fixed default protocol

- 100B processed tokens per condition;
- global batch 256 sequences;
- 8,192 tokens per sequence;
- micro-batch 1 × 8 GPUs × gradient accumulation 32;
- peak learning rate `1e-4`, 500 warmup steps, cosine decay;
- checkpoints and automatic evaluation near 0.1B, 1B, 10B, 25B, 50B, 75B,
  and 100B tokens;
- TensorBoard logging and a background server on port 6006.

At this batch and sequence length, one optimizer step processes 2,097,152
tokens. The target maps to 47,684 steps and 100,000,595,968 actual tokens.

## Install on every machine

```bash
unzip qwen3_06b_aliyun_pe_pretraining_iclr27.zip
cd qwen3_06b_aliyun_pe_pretraining_iclr27
bash scripts/install.sh
cp configs/experiment.env.example .env
```

Expected inputs:

```text
/mnt/workspace/Qwen3-0.6B
/mnt/workspace/dclm
```

## Launch 16 tasks

Run task ID `N` on machine `N`:

```bash
bash scripts/run_pretrain_worker.sh N  # N=0..15
```

The command uses GPUs `0,1,2,3,4,5,6,7`, starts training with
`nohup + setsid`, and starts TensorBoard. Disconnection does not stop it.
Use `DRY_RUN=1 bash scripts/run_pretrain_worker.sh N` to validate the mapping
and exact protocol without creating a process.

If one controller has passwordless SSH access to all instances:

```bash
cp configs/sixteen_hosts.example configs/sixteen_hosts.conf
# edit the 16 host rows
bash scripts/launch_sixteen_hosts.sh configs/sixteen_hosts.conf
```

Status and stop:

```bash
bash scripts/status_sixteen_hosts.sh configs/sixteen_hosts.conf
bash scripts/stop_sixteen_hosts.sh configs/sixteen_hosts.conf
```

Open `http://<machine>:6006` for that machine's live TensorBoard. Protect the
port with the instance security group or an SSH tunnel.

All new outputs use `/mnt/workspace/pe_pretrain_100b_iclr27`, deliberately
separate from earlier continued-pretraining runs.

## Methods and evidence

`configs/sixteen_machine_plan.json` maps task IDs to strategies. Every method
has a separate specification under `docs/methods/`. The full training contract,
falsification rules, and plot definitions are:

- `docs/pretraining_100b_protocol.md`
- `docs/design.md`
- `docs/experiment_design.md`
- `docs/visualization_results.md`

Each milestone automatically evaluates held-out DCLM PPL, controlled
RULER-style tasks, and a fixed LongBench subset, then resumes from the validated
local checkpoint. Result merging checks both processed-token counts and the
DCLM manifest hash before computing changes versus native RoPE.

To intentionally return to continued pretraining, set
`INITIALIZATION=checkpoint`; do not mix that run with from-scratch results.
