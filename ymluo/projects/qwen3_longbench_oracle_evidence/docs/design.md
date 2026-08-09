# Design: LongBench gold-evidence compression ceiling

## Research question

For a fixed Qwen3-8B reader and a fixed LongBench HotpotQA question, does
replacing the full context with the human-annotated HotpotQA supporting evidence
recover answer quality?

Let \(S(c)\) be the official LongBench QA-F1 under context condition \(c\). The
main paired estimand is

\[
\Delta_{\mathrm{oracle-doc}}
=
S(\mathrm{oracle\ document})-S(\mathrm{full}).
\]

The falsifiable conjecture is that both of the following are positive on this
frozen pilot set:

\[
S(\mathrm{oracle\ document})-S(\mathrm{full})>0,
\qquad
S(\mathrm{oracle\ document})-S(\mathrm{random\ matched})>0.
\]

The random matched-budget comparison is essential: without it, an improvement
could be caused only by making the prompt shorter.

## Exact evidence definition

The LongBench JSONL does not contain evidence annotations. Each LongBench
HotpotQA row is therefore joined to the original HotpotQA distractor validation
set by a unique normalized question. The join is accepted only when every
original `supporting_facts = [title, sentence_id]` sentence occurs in the
LongBench context after whitespace normalization.

- **Oracle sentence:** every annotated supporting sentence, grouped under its
  original document title and kept in original order.
- **Oracle document:** every complete source document containing at least one
  annotated supporting sentence.
- The answer text is never used to select evidence.

## Algorithm

For each eligible row:

1. Match the LongBench question uniquely to the original HotpotQA example.
2. Resolve every supporting-fact title and sentence index.
3. Reject the row if any supporting sentence is absent from LongBench context.
4. Build the oracle-sentence and oracle-document contexts.
5. Build a random non-support document context with the same number of documents
   and the closest Qwen3-token budget; run three fixed random tie-break seeds.
6. Build a BM25 document context using only the question.
7. Keep the official LongBench instruction and question unchanged in every
   executable QA condition; only `{context}` changes.
8. Greedily decode with Qwen3 thinking disabled and score with official
   LongBench QA-F1 plus normalized exact match.
9. Compute the teacher-forced NLL of the first gold answer as an auxiliary
   within-sample measure.

## Claim boundary

This experiment estimates the reader performance under perfect external
evidence selection and re-encoding. It jointly changes context length, RoPE
positions, the softmax denominator, and surrounding text. It therefore does not
isolate which mechanism caused a full-context error, and an eight-sample pilot
does not estimate all-LongBench performance.

## Iteration 2: strict current-passage alignment

The first iteration required every old HotpotQA supporting sentence to occur
verbatim in the newer LongBench Wikipedia passage and accepted only 8/200 rows.
The rejected set mixes harmless copy edits with substantive version changes:
some passages paraphrase the same fact, while others delete it or change a
number. Exact title equality alone is therefore insufficient for a gold-evidence
claim.

The expansion aligns each old supporting sentence to a span in the uniquely
title-matched **actual LongBench passage**. It is fail-closed:

1. canonical token-subsequence matches pass directly;
2. otherwise, a one- or two-sentence local span must achieve sequence
   similarity, bag-of-token F1, and source-token recall of at least 0.95, with a
   length ratio in \([0.75,1.33]\) and a best-candidate margin of at least 0.05;
3. protected factual anchors---numbers, dates, times, negation, and comparison
   direction---must be preserved exactly;
4. every annotated fact must pass, or the whole example is rejected.

Neither answer strings nor model outputs participate in alignment. The tight
Oracle condition uses the accepted current LongBench spans, and the document
Oracle uses their complete actual LongBench passages. Thus both executable
conditions contain only text available in the Full baseline.

The pre-inference target was 20 examples. The frozen alignment audit found only
18 rows satisfying every gate, so the experiment uses the complete 18-row
eligible cohort; no model output was inspected before this count was fixed.
This is a deliberately conservative aligned subset, not a LongBench-wide
estimate; a larger title-only cohort would answer a different question and is
not pooled into the main result.
