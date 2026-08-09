# Visualization and results

## Plot contracts

1. **Quality comparison:** x-axis method; y-axis QA-F1/EM and gold NLL; show
   paired sample points and 95% paired-bootstrap intervals. Answers whether LS
   improves the matched-budget baseline.
2. **Retrieval-to-answer scatter:** x-axis LS minus post2 evidence recall or
   mass; y-axis LS minus post2 gold NLL. Label rescued and harmed samples.
   Answers whether retrieval changes co-occur with answer support.
3. **Per-sample paired deltas:** x-axis frozen sample ID; y-axis NLL/F1 delta;
   color by early/middle/late evidence position. Prevents an aggregate from
   hiding one influential sample.
4. **Runtime and fidelity audit:** method runtime with dense replay logit error,
   prompt-hash mismatch count, alignment failures, and support-budget failures.

Each plot must state units, sample count, aggregation level, and whether lower
or higher is better. The accompanying table is authoritative; plots may not
drop failed or zero-valued samples.

## Current evidence before this run

On a separate 24-seed synthetic two-hop protocol, the frozen LS method improved
matched-budget PPL over exact post-RoPE Top-2% at 16K (5.858 to 3.080) and 32K
(21.636 to 10.840), with higher evidence recall. The advantage was not reliable
at 8K or 64K. These numbers motivate this natural-task test but cannot answer
the LongBench question.

The strict direct phase-repair smoke is a negative result: it did not restore
first-token accuracy, was unstable under wider intervention, and was tens to
hundreds of times slower. The paper-facing candidate is therefore semantic
proposal plus native-RoPE consumption, not virtual-position repair.

## LongBench strict-18 result

The physical GPU 6-7 run completed all 18 frozen samples and six arms (108
rows). Prompt hashes, support budgets, duplicate checks, and evidence alignment
all passed. With eager attention, `native_full` and `full_rope_replay` had zero
maximum logit difference.

| Method | QA-F1 | EM | Gold NLL | Gold PPL | Evidence recall | Evidence mass |
|---|---:|---:|---:|---:|---:|---:|
| Native Full | 65.56 | 55.56 | 1.2206 | 3.389 | -- | -- |
| Exact post-RoPE Top-2% | 60.00 | 50.00 | 1.1386 | 3.122 | 4.76% | 0.598% |
| Final-Query pre-RoPE Top-2% | 48.89 | 38.89 | 1.6526 | 5.220 | 6.45% | 0.607% |
| Local/global native consumer | **71.11** | **61.11** | 1.1541 | 3.171 | 3.24% | 0.490% |
| Local/global blend-25 | **71.11** | **61.11** | 1.1352 | **3.112** | 3.25% | 0.482% |

Relative to exact post-RoPE Top-2%, the frozen native-consumer LS arm gained
11.11 QA-F1 and 11.11 EM points, with two EM rescues and no harms. However,
paired gold NLL changed by (+0.0155), with 95% bootstrap CI
([-0.2324,+0.3697]). The blend arm changed NLL by (-0.0033), with CI
([-0.2688,+0.3633]). Neither likelihood result is stable.

More importantly, LS reduced mean evidence recall by 1.52 percentage points and
evidence mass by 0.108 percentage points relative to exact post-RoPE Top-2%.
The two EM rescues also had lower global evidence recall and mass. Therefore,
the QA improvement cannot currently be attributed to rescuing the annotated
evidence; local/sink preservation, head-specific routing, non-annotated useful
context, or generation-path changes remain alternatives.

**Primary decision: INSUFFICIENT.** There is a promising task-level signal but
not the coherent retrieval-to-answer evidence required for the paper's H4
mechanism claim.

Artifacts:

- `outputs/hotpot_strict18_20260803/merged/summary.csv`
- `outputs/hotpot_strict18_20260803/merged/comparisons.csv`
- `outputs/hotpot_strict18_20260803/merged/paired_ls_vs_post2.csv`
- `outputs/hotpot_strict18_20260803/merged/quality_summary.png`
- `outputs/hotpot_strict18_20260803/merged/retrieval_answer_scatter.png`

## Query-span screening result

The 18-sample first-token screen tested whether a single terminal pre-RoPE
Query was the bottleneck. Question-span token-max selection raised gold
evidence recall from 3.23% to 5.26%, a paired gain of 2.03 percentage points
(95% CI [1.17, 3.02]). But first-token NLL worsened from 3.3319 to 5.6116,
with paired delta (+2.2797) and 95% CI ([+0.3342,+4.7028]). Attention mass
also fell from 0.492% to 0.436%. Block selection was slower, had only 4.36%
recall, 0.195% mass, and 4.9535 NLL.

An additional one-sample screen mixed calibrated token-max semantic scores into
the native consumer. It worsened NLL from 20.3146 (post-score consumption) to
23.5050 (blend) and 23.5667 (monotone boost), so it was not expanded.

**Decision: NO-GO for query-span expansion.** Retrieving more annotated tokens
does not make them useful when their native post-RoPE scores/Values do not write
answer-directed information, while direct score blending is unstable.

Artifacts:

- `outputs/queryspan_strict18_20260803/merged/summary.csv`
- `outputs/queryspan_strict18_20260803/merged/comparisons.csv`
- `outputs/queryspan_strict18_20260803/merged/report.md`
