# Viewer harness record

## Expanded-condition plot contract

- **Plot title:** Answer quality after selecting human-annotated support documents.
- **Question answered:** Does perfect support-document selection outperform the
  full LongBench context and an equally short random context?
- **Metrics:** official QA-F1 and normalized exact match, each in percentage
  points; higher is better.
- **Data:** exhaustive frozen 18-sample strictly aligned Qwen3-8B HotpotQA
  expansion, aggregated across
  samples; random is first averaged across three contexts within sample.
- **x-axis:** Full, Oracle current evidence spans, Oracle support documents, BM25,
  matched random, and Question-only.
- **y-axis:** answer score from 0% to 100%.
- **Legend:** QA-F1 versus exact match.
- **Allowed conclusion:** which supplied context condition produced better
  answers on this frozen sample.
- **Not proved:** a causal split among retrieval, RoPE distance, softmax
  competition, and reader reasoning.

## Confidence plot contract

- **Plot title:** Correct-answer confidence and retained context size.
- **Question answered:** Does the evidence condition lower teacher-forced
  gold-answer perplexity, and how many context tokens does it retain?
- **Metric:** geometric mean gold-answer perplexity on a log-scaled y-axis;
  lower is better. Context-token counts are text annotations, not values sharing
  the y-axis.
- **Data and x-axis:** identical to the answer-quality plot.
- **Allowed conclusion:** whether correct-answer confidence changes with the
  selected context condition.
- **Not proved:** lower perplexity does not by itself establish correct free
  generation or a specific internal mechanism.

## Render audit

Status: **complete**.

- Rendered artifact: `outputs/hotpot_semantic_aligned_18_20260802/merged/oracle_expansion_summary.png`.
- The title reports the frozen `n=18`; Oracle sentence is relabeled Oracle span.
- QA-F1 and EM share a 0--100% axis; PPL uses a log axis and context size appears
  only as text, so unlike units are not visually conflated.
- All labels, legends, annotations, and bars are visible without clipping.
- The plot shows aggregate condition comparisons only; CIs and sensitivity
  results remain in the result document rather than being implied by bar height.
