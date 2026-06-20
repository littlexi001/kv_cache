# Downstream Evaluation

## Purpose

Compare a routed-top4 checkpoint against the official Qwen3-0.6B model on the
same downstream examples.

The comparison answers:

```text
How much downstream task ability has the routed checkpoint acquired so far?
```

It does not answer:

```text
Whether the routed architecture is better than official Qwen3 under equal
training compute.
```

The official model is fully pretrained. The routed checkpoint is a partial
pretraining run from random initialization.

## Multiple-Choice Metric

Each example has:

```text
prompt:  context and question
choices: candidate answer strings
answer:  integer index of the correct choice
```

For each choice, the evaluator computes:

```text
score(choice) = mean log p(choice_token | prompt, earlier_choice_tokens)
```

The predicted answer is:

```text
argmax_choice score(choice)
```

The reported metric is:

```text
accuracy = correct_examples / total_examples
```

Length normalization is enabled by default, so a long answer is not punished
only because it has more tokens.

## Prepare Evaluation Data

On the server:

```bash
cd ymluo/projects/qwen3_routed_top4_mha_pretrain
pip install datasets transformers
bash scripts/prepare_downstream_eval_data.sh
```

Defaults:

```text
tasks = piqa, hellaswag, winogrande, arc_easy, arc_challenge, boolq
split = validation
max_examples = 500 per task
```

The converted JSONL files are saved under:

```text
output/eval_data
```

## Run Checkpoint vs Baseline

Evaluate the latest routed run:

```bash
bash scripts/eval_checkpoint_vs_baseline.sh
```

Evaluate a specific checkpoint:

```bash
CHECKPOINT_DIR=/path/to/checkpoint-0008500 bash scripts/eval_checkpoint_vs_baseline.sh
```

Use fewer examples for a quick smoke run:

```bash
MC_LIMIT=50 bash scripts/eval_checkpoint_vs_baseline.sh
```

## Outputs

Each evaluation run writes:

```text
output/downstream_eval_results/<run_name>_<timestamp>/summary.json
output/downstream_eval_results/<run_name>_<timestamp>/multiple_choice_details.jsonl
output/downstream_eval_results/<run_name>_<timestamp>/eval_args.json
```

`summary.json` contains one row per model per task.

`multiple_choice_details.jsonl` contains per-example scores, predictions, and
correct labels for error analysis.

Print a compact comparison table:

```bash
python eval/summarize_eval_results.py \
  output/downstream_eval_results/<run_name>_<timestamp>/summary.json
```

## Optional Text PPL

The same evaluator can also compute CE/PPL on a held-out text file or directory:

```bash
bash scripts/eval_checkpoint_vs_baseline.sh \
  --eval_text_path /mnt/workspace/dclm/global-shard_01_of_10/local-shard_0_of_10/part-00001.txt \
  --eval_text_max_chars 1000000
```

This is useful for checking whether train loss and held-out text loss move
together.

## Claim Boundary

If the routed checkpoint is far behind official Qwen3, that does not falsify the
routed architecture. The training budget is much smaller.

If the routed checkpoint beats random and improves over checkpoints, that
supports the claim that the routed model is learning useful language behavior.
