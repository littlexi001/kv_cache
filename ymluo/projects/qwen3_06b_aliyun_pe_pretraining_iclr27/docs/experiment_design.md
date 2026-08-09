# Experiment and falsification plan

## Research question

At fixed architecture, random initialization, DCLM token order, optimizer,
global batch, and 100B-token budget, which position-to-phase function improves
long-context retrieval over native RoPE without harming ordinary language
modeling?

## Publication-grade setup contract

- Model: Qwen3-0.6B architecture, all weights trained from scratch.
- Objective: causal next-token cross entropy on packed 8,192-token sequences.
- Data: 200,000 deterministically sampled DCLM train files and 1,024 disjoint
  held-out files; seed 1701; manifest SHA256 must match across machines.
- Hardware: one independent eight-GPU machine per condition; 16 conditions.
- Global batch: 256 sequences = micro-batch 1 × 8 GPUs × accumulation 32.
- Tokens/update: 2,097,152; target: 100B/condition; final step: 47,684.
- Optimizer: AdamW, peak LR \(10^{-4}\), betas 0.9/0.95, epsilon \(10^{-8}\),
  weight decay 0.1, gradient clip 1.0.
- Schedule: 500-step warmup then cosine decay to the common final step.
- Model seed: 20260808. Attention implementation and BF16 dtype are fixed.
- PE conditions: task IDs 0–15 in `configs/sixteen_machine_plan.json`; each is
  defined before use in its corresponding `docs/methods/*.md` card.

Known limitation: 100B processed tokens do not automatically mean 100B unique
tokens. Corpus coverage and repetition must be measured before publication.

## Token checkpoints

Evaluate near 0.1B, 1B, 10B, 25B, 50B, 75B, and 100B processed tokens. Ceiling
division maps them to optimizer steps. `token_schedule.json` stores requested
and actual tokens so plots use actual cumulative tokens on the x-axis.

## Metrics

1. **Held-out DCLM PPL:** exponential of mean next-token NLL; lower is better.
2. **Controlled QA F1/exact/contains:** deterministic single-needle,
   multi-needle, and variable-tracking diagnostics at 2K/4K/8K; higher is better.
3. **Controlled gold-answer NLL:** teacher-forced answer confidence; lower is
   better even before greedy output crosses the answer boundary.
4. **LongBench QA F1/exact/contains and gold NLL:** fixed official-data subset
   with package-local prompts; this is not labeled an official leaderboard run.
5. **Training stability:** loss, gradient norm, learning rate, throughput,
   memory, non-finite values, OOMs, and failed samples.
6. **Position robustness for finalists:** full RULER and fixed-content position
   sweep; lower variance across evidence positions is better.

## Screening decision

A condition passes a checkpoint only if:

- its long-context score and/or gold-answer NLL is better than task 0 at the
  same actual token count;
- evaluation sample counts and data-manifest hashes match;
- DCLM PPL is at most 2% worse than task 0;
- training and checkpoint integrity are complete.

It fails this operationalization when matched long metrics are worse or the PPL
guardrail fails. Evidence is insufficient when any evaluation is unavailable,
tokens differ, manifests differ, or only the modified condition is present.

## Compute-saving stop rule

The code does not automatically kill a scientifically weak task because one
early checkpoint may be noisy. A human may stop a condition after 10B only when
all three are true: PPL is more than 5% worse than native, both controlled and
LongBench metrics do not improve, and training instability is not a temporary
warmup effect. Record the stop decision and evidence rather than silently
removing the run.

## Profiling and failure interpretation

- NoPE improves retrieval but hurts PPL: content matching gained at the cost of
  local order; test selective rather than global removal.
- Soft high-frequency beats hard deletion: residual phase is useful; the broad
  prior survives but deletion is falsified.
- Middle/low bands match high-band gains: the mechanism is not specifically
  high-frequency and the conjecture must be revised.
- Synthetic retrieval improves but LongBench does not: the condition learned a
  narrow lookup behavior.
- Gold NLL improves without generation improvement: confidence changed but did
  not cross the output decision boundary.
- Phase-diverse improves mean but not position variance: evidence does not
  support the proposed alias-coverage mechanism.

## After screening

Advance no more than two candidates. Re-run native and finalists with at least
three seeds, full RULER/LongBench, short-context PPL, and short-capability tasks.
Report confidence intervals, total tokens, unique data coverage, wall time,
energy/compute if available, and all stopped or failed conditions.
