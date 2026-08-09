# Frozen local-order control

## Question

Can a long-range RoPE intervention improve remote evidence retrieval without
damaging the pretrained model's nearby order sensitivity?

This is a **frozen-model inference control**. The prefix is encoded by native
Qwen3-8B RoPE, no weights are updated, and only the final query-token attention
is replaced by the tested attention variant.

## Two deliberately separated task families

| Family | Construction | What changes the answer | Primary metrics |
|---|---|---|---|
| `local_order` | A four-word sequence is placed immediately before the question. A paired prompt contains exactly the same words, filler, and question, but swaps the two words after a fixed anchor. | Local order only | Counterfactual pair accuracy, restricted-candidate accuracy, answer margin, Gold PPL |
| `remote_retrieval` | The existing controlled two-hop `VERIFIED RULE` chain is near the prefix and the query is at the end. | Retrieval of remote semantic evidence | Evidence-token recall, two-line complete hit rate, evidence attention mass, Gold PPL, candidate accuracy |

For seed 0, an actual tokenizer audit produced the pair:

```text
coffee, salt, silver, hammer  -> after salt: silver
coffee, salt, hammer, silver  -> after salt: hammer
```

Both decisive sequence tokens were 50--52 tokens from the final query token,
well inside the 128-token local window. The corresponding 8K remote evidence
was 7,991--8,024 tokens away.

## Compared variants

- `full_rope`: native full attention.
- `rope_top2`: exact post-RoPE Top-2% sparse baseline.
- `local_global_postscore`: existing local-window + pre-RoPE retrieval sparse baseline.
- `remote_nope_cal_full`: calibrated position-free remote ablation.
- `phase_coherent_w4k_c4_cal_full`: current frequency-aware phase candidate.

The output summary reports deltas relative to `full_rope` independently for
each task family and length. A useful candidate should satisfy both:

1. negligible local counterfactual-pair/accuracy loss;
2. positive remote recall, evidence-mass, PPL, or accuracy gain.

## Server command (GPU 6--7 only)

After synchronizing the three implementation files to the server:

```bash
cd /home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary
bash scripts/run_phase_rope_local_order_control_gpu67_20260801.sh
```

The launcher assigns `local_order` only to GPU 6 and `remote_retrieval` only to
GPU 7. It does not touch GPUs 0--5. Outputs are written under:

```text
outputs/20260801_phase_rope_local_order_control_gpu67/
```

Each shard contains `rows.jsonl`, `rows.csv`, `summary.csv`, `summary.json`,
`config.json`, `protocol.json`, and `done.txt`.

## Interpretation boundary

Passing this control shows that the **final-query retrofit** preserves a small,
controlled nearby ordering relation. It is not sufficient to claim that a
model trained end to end with the new positional kernel preserves all syntax,
negation, code order, or temporal reasoning. Those require broader downstream
local-order benchmarks after the mechanism screen succeeds.
