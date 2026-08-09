# Legacy four-machine subset

This is retained for compatibility. The current primary protocol is the
16-machine from-scratch run in `docs/pretraining_100b_protocol.md`; do not mix
old four-machine continued-pretraining outputs with that experiment.

## What kind of training is this?

This is **full-parameter causal-LM continued pretraining** from the released
Qwen3-0.6B checkpoint on DCLM. In this project it is a mid-training experiment:
the model is not initialized from scratch, and the objective is still next-token
language modeling rather than instruction SFT, preference optimization, or RL.

## Fixed four-machine assignment

| Machine ID | Strategy | Role |
|---:|---|---|
| 0 | `native_rope` | matched-token native RoPE control |
| 1 | `deep_highfreq_drop` | remove F0--F11 only in deep layers |
| 2 | `uniform_slow_rope` | use 0.5x RoPE phase rate in all layers |
| 3 | `smooth_layer_frequency` | smoothly vary phase scale by layer and frequency |

The native condition is required: a candidate beating only the untouched base
checkpoint does not show that the PE intervention beats ordinary continued
pretraining. `smooth_remote_warp` is reserved for a second wave after these four
matched runs.

All four machines must use the same `.env`, except that `GPU_LIST` may differ if
their hardware differs. For a valid matched comparison, GPU count, sequence
length, micro-batch, gradient accumulation, milestones, seeds, DCLM path, and
evaluation settings must be identical. The result merger also checks nominal
training-token counts.

## Universal one-command launch

After installing the same package and copying the same `.env` to all four
instances, run exactly one command in each interactive terminal:

```bash
# machine 0
bash scripts/run_four_machine_worker.sh 0

# machine 1
bash scripts/run_four_machine_worker.sh 1

# machine 2
bash scripts/run_four_machine_worker.sh 2

# machine 3
bash scripts/run_four_machine_worker.sh 3
```

The worker defaults to GPUs `0,1,2,3,4,5,6,7`. An explicit override is:

```bash
GPU_LIST=0,1,2,3,4,5,6,7 bash scripts/run_four_machine_worker.sh 2
```

Named wrappers are also provided: `run_machine_0_native.sh`,
`run_machine_1_deep_drop.sh`, `run_machine_2_slow_rope.sh`, and
`run_machine_3_smooth_pe.sh`.

If the controller machine has passwordless SSH access to all instances, copy
`configs/four_hosts.example` to `configs/four_hosts.conf`, edit the four hosts,
then dispatch all four jobs with one local command:

```bash
bash scripts/launch_four_hosts.sh configs/four_hosts.conf
```

The same project must already exist at `REMOTE_PROJECT` on every host. Override
that path when needed:

```bash
REMOTE_PROJECT=/another/path/qwen3_06b_aliyun_pe_pretraining_iclr27 \
  bash scripts/launch_four_hosts.sh configs/four_hosts.conf
```

## Automatic checkpoint evaluation

Each background controller executes the same state machine:

1. validate model, data, and PE configuration;
2. deterministically build the DCLM train/validation manifests;
3. evaluate the untouched base checkpoint;
4. train to steps 50, 200, 500, and 1000;
5. after every checkpoint, release the distributed trainer and evaluate held-out
   DCLM PPL, controlled RULER-style probes, and the configured LongBench subset;
6. resume training from that checkpoint;
7. write summaries and a result-only `tar.gz` bundle.

Re-running the same worker resumes the latest complete local checkpoint and
skips completed evaluations. Network or terminal disconnection does not stop the
`nohup + setsid` background controller.

Check one machine locally with:

```bash
bash scripts/four_machine_status.sh
tail -f /mnt/workspace/pe_pretrain_100b_iclr27/<strategy>/nohup.log
```

After all runs, copy the four archives from
`/mnt/workspace/pe_pretrain_100b_iclr27/<strategy>/bundles/` into one directory and run:

```bash
bash scripts/merge_result_bundles.sh /path/to/bundles /path/to/combined_results
```
