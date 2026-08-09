# Head-Routed Heterogeneous Retrieval: War-and-Peace 4K Pilot

Date: 2026-07-14

## Conclusion

The pilot gives a mixed but useful answer.

1. Head-specific external retrieval is measurable: the train-selected routed
   policy improves held-out oracle Top-2% position recall from `31.5125%` to
   `31.8554%`, and 55 individual heads improve over the position policy.
2. The aggregate gain is only `+0.3428` percentage points.  These retrievers are
   not yet close enough to the oracle to justify calling the selection problem
   solved.
3. Repeat/induction retrieval is the clear positive result.  Several early and
   middle-layer heads gain 5--12 points, and the best three are L2H11
   (`+11.83pp`), L4H13 (`+11.70pp`), and L2H10 (`+11.35pp`).
4. Raw lexical overlap, raw input-embedding similarity, and generic format
   features do not retrieve the remote oracle positions well enough.  This is
   the present bottleneck, not GQA union cost.
5. A shared position scaffold is necessary.  Giving 50% of the same strict 2%
   budget to sink/recent and 50% to a specialist improves both position recall
   and oracle-mass recall over the pure-specialist routed pilot.

This supports the narrow claim that some heads can be matched by a different
external retrieval operator, especially induction-style heads.  It does not yet
support the stronger claim that the current retriever bank can replace oracle
Top-2% attention for PPL.

## Protocol

- Model: Qwen3-0.6B, 28 layers, 16 query heads, 8 KV heads.
- Text: War and Peace.
- Prefix: 4096 tokens.
- Retriever-selection split: 128 train queries followed by 128 held-out test
  queries.
- Oracle teacher: for every layer, query head, and query, select
  `ceil(0.02 * history_length)` historical positions by exact QK score.
- Every external policy returns exactly the same number of historical tokens and
  never reads QK scores.
- Current self token is not part of the historical 2% budget.
- Position roles: first 4 tokens are sink, last 256 history tokens are recent,
  and the remainder is remote.
- Per-head routing is fitted only on the train-query position recall.  All
  headline metrics are from the held-out test queries.
- `semantic` uses frozen Qwen input-embedding block means.  It is a diagnostic
  retriever, not a separately trained semantic encoder.
- Each `hybrid_*` policy uses 50% of the same budget for a position scaffold and
  50% for its specialist.  It does not add budget.

The final result was reproduced on an NVIDIA RTX 3090 server after a local AMD
7900 XTX run.  Routed position recall differed by only `1.34e-5` between the two
runs.  The remote server runtime was 52.24 seconds.

## Held-out aggregate results

| Equal-budget policy | Oracle position recall | Remote position recall | Oracle selected-mass recall |
| --- | ---: | ---: | ---: |
| homogeneous position | 31.5125% | 0.0000% | 83.0825% |
| homogeneous lexical | 21.9636% | 2.6912% | 23.8579% |
| homogeneous semantic | 21.8815% | 2.3184% | 23.2822% |
| homogeneous format | 11.9573% | 3.0563% | 7.4718% |
| homogeneous repeat | 28.1420% | 2.2537% | 27.7493% |
| homogeneous hybrid-repeat | 29.8744% | 1.9895% | 83.3639% |
| head-routed, train-best | **31.8554%** | 1.1221% | 83.1907% |
| head-routed, train-balanced | 31.2890% | **1.9690%** | 81.1593% |
| random | 2.3624% | 1.9379% | 1.1339% |

The balanced router puts equal weight on overall and remote recall.  It nearly
doubles routed remote recall but loses overall recall and mass.  This is a
retriever-quality tradeoff rather than a useful final operating point.

## What the head map says

Train-selected assignments:

| Retrieval policy | Heads |
| --- | ---: |
| position | 385 |
| repeat | 23 |
| hybrid-repeat | 21 |
| hybrid-lexical | 18 |
| hybrid-format | 1 |

No head selected pure lexical, semantic, format, or hybrid-semantic on the train
split.  In a diagnostic that is allowed to choose the best method after seeing
the test split, 373 heads prefer position, 65 prefer a repeat-family method, 9
prefer hybrid-semantic, and 1 prefers hybrid-lexical.  The train-selected method
matches this test-oracle method for 408/448 heads (`91.07%`).

Relative to applying position to every head:

- 55 heads improve;
- 8 heads degrade;
- 385 heads are unchanged because their route remains position.

The strongest gains are concentrated in early and middle layers.  This is
consistent with induction/repetition being a real functional retrieval pattern,
while the later-layer hybrid-lexical assignments are less stable and account for
most negative heads.

## Remote evidence is still the failure point

Remote positions make up `48.16%` of held-out oracle Top-2% events.  Yet the best
homogeneous external rule recalls only `3.06%` of those remote events, and the
train-best routed rule recalls `1.12%`.

Position achieves `83.08%` oracle selected-mass recall while retrieving only
`31.51%` of oracle positions and no remote positions.  This repeats the earlier
finding that token 0 and other high-mass anchors dominate mass statistics.  Mass
recall alone therefore cannot certify functional equivalence; remote position
recall and paired NLL/PPL remain mandatory.

## GQA physical cost

The per-query-head budget is exactly 2%.  For each Qwen3 KV group, physical KV
loading uses the union of its two query-head selections.

- Mean group union: `1.0394x` one-head budget.
- Mean physical history fraction: `2.0896%`.
- 35/224 layer/KV groups expand beyond one-head budget.
- Maximum group expansion: `1.5105x`.

Thus heterogeneous routing is cheap in this pilot because most paired query
heads choose the same position policy.  GQA union is not the current bottleneck.

## Decision and next experiment

Do not proceed directly to a broad downstream PPL claim with the present
retrievers.  The next selector should have three stages while preserving the
strict 2% output budget:

1. A small shared scaffold for sink/recent/mandatory delimiters.
2. A heterogeneous candidate union at roughly 8--10% history from lexical,
   sentence-level semantic, relation/graph, format, and repeat indexes.
3. A shared learned reranker with layer/head embeddings, trained from oracle
   Top-2% ranks and remote-positive weighting, that cuts the candidate union back
   to exactly 2%.

The repeat-family heads should also receive a direct behavioral ablation: apply
repeat or hybrid-repeat only to the 44 train-selected repeat heads, use position
for the remaining heads, and measure paired NLL/PPL.  That is the cheapest next
causal test of the functional-routing hypothesis.

For semantic and format functions, War and Peace is not a sufficient benchmark.
The next training/evaluation split should include multiple documents plus the
controlled conflict/non-conflict evidence data, so semantic, relation, and
format operators each have tasks on which their target evidence is observable.

## Artifacts

- Authoritative remote run: `outputs/pilot_war4k_hybrid_remote_20260714_1933`
- Head route map: `outputs/pilot_war4k_hybrid_remote_20260714_1933/analysis/head_route_map.png`
- Head recall-gain map: `outputs/pilot_war4k_hybrid_remote_20260714_1933/analysis/head_recall_gain_map.png`
- Policy comparison: `outputs/pilot_war4k_hybrid_remote_20260714_1933/analysis/retriever_policy_comparison.png`
- Machine-readable analysis: `outputs/pilot_war4k_hybrid_remote_20260714_1933/analysis/analysis_summary.json`
