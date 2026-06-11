# Qwen3 Attention Value Decomposition

This project studies the output vectors produced by different parts of the
attention distribution.

For each selected layer/head/query token, the script computes three vectors:

```text
full_output   = sum(all attention weights * V)
top90_output  = sum(top attention-mass weights * V)
tail10_output = sum(remaining attention weights * V)
```

`top90` is defined by sorting attention weights from high to low and taking the
smallest prefix whose cumulative attention mass reaches `TOP_MASS=0.90`.
`tail10` is the remaining valid attention mass.

By default, top/tail vectors use the original attention weights and are not
renormalized. Therefore:

```text
top90_output + tail10_output ~= full_output
```

This is useful for analyzing how much of the final attention output is carried
by high-attention tokens versus low-attention tail tokens.

## Outputs

Default output directory:

```text
ymluo/projects/qwen3_attention_value_decomposition/outputs/attention_value_decomposition/
```

Main files:

- `value_decomposition_by_head.csv`
- `ppl_by_attention_value_mode.csv`
- `summary.json`

`value_decomposition_by_head.csv` contains per-layer/per-head means:

- full/top90/tail10 output norms
- conditional top90/tail10 norms after renormalizing each selected part
- top/tail attention mass
- top/tail selected token counts
- cosine similarities between full, top90, and tail10 outputs
- L2 distances
- reconstruction error of `full - top90 - tail10`

`ppl_by_attention_value_mode.csv` evaluates behavior when selected layers/heads
use:

- `full`: normal attention output
- `top90`: only the top attention-mass contribution
- `tail10`: only the tail contribution

By default PPL modes also use unnormalized selected weights. Set
`PPL_RENORMALIZE_SELECTED=true` to test conditional top/tail attention outputs.

## Run

```bash
bash ymluo/projects/qwen3_attention_value_decomposition/scripts/run_analysis.sh
```

Smoke test:

```bash
PREFILL_TOKENS=128 EVAL_TOKENS=64 CHUNK_SIZE=32 LAYERS=0 HEADS=0 \
bash ymluo/projects/qwen3_attention_value_decomposition/scripts/run_analysis.sh
```

Vector analysis only:

```bash
COMPUTE_PPL=false \
bash ymluo/projects/qwen3_attention_value_decomposition/scripts/run_analysis.sh
```

PPL only:

```bash
COMPUTE_VECTOR_STATS=false PPL_MODES=full,top90,tail10 \
bash ymluo/projects/qwen3_attention_value_decomposition/scripts/run_analysis.sh
```

Useful parameters:

```text
PREFILL_TOKENS=5000
EVAL_TOKENS=5000
CHUNK_SIZE=128
LAYERS=all
HEADS=all
TOP_MASS=0.90
PPL_RENORMALIZE_SELECTED=false
```
