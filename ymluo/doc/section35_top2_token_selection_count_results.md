# Section 35: True Top2 Token Selection Count Results

Date: 2026-06-30

## 0. Goal

This experiment counts how often each historical token is selected by true full-QK top-2% attention during forward.

For each eval query, layer, and head:

```text
select top ceil(0.02 * history_tokens) historical tokens by full QK score
accumulate count[token_index] += 1
```

This measures whether true top2 attention is spread broadly or repeatedly concentrates on a small subset of tokens.

## 1. Script

New files:

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/analyze_top2_token_selection_counts.py
ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_top2_token_selection_counts_server.sh
```

Server run:

```text
host = df
project = /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl
output = /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_token_selection_counts_war_4k_v1
```

## 2. Setup

```text
model = /home/fdong/hrj/prove/Qwen3-0.6B
text = data/war_and_peace_pg2600.txt
prefill_tokens = 4096
eval_tokens = 64
top_fraction = 0.02
layers = 28
heads = 16
dtype = float16
attention = eager
```

Total true top2 selection events:

```text
2,408,448
```

This equals all sampled eval queries across all layers/heads, with per-head top2 size around 82-84 tokens.

## 3. Main Overall Result

Among 4,159 historical-token rows:

```text
nonzero selected tokens = 3,974
zero selected tokens = 185
```

Selection is broad but highly non-uniform:

| Top tokens by count | Fraction of all selection events |
| ---: | ---: |
| top 1 | 1.07% |
| top 10 | 6.12% |
| top 50 | 22.56% |
| top 100 | 35.78% |
| top 200 | 51.45% |
| top 500 | 72.28% |
| top 1000 | 86.70% |

Interpretation:

```text
Top2 is not just a smooth 2% random sample over history.
Roughly 200 tokens account for more than half of all selected token events.
```

## 4. Most Frequently Selected Tokens

Top global tokens:

| token index | token text | selected count | normalized selection rate |
| ---: | --- | ---: | ---: |
| 0 | `The` | 25,701 | 0.896 |
| 4096 | `....` | 15,642 | 0.554 |
| 4097 | closing quote/newlines | 15,619 | 0.562 |
| 4065 | `?` | 13,560 | 0.473 |
| 4102 | `,` | 13,453 | 0.527 |
| 4098 | `The` | 13,249 | 0.485 |
| 4100 | ` answered` | 12,801 | 0.484 |
| 4099 | ` prince` | 12,754 | 0.474 |
| 4101 | ` nothing` | 12,625 | 0.486 |
| 4037 | closing punctuation/newlines | 12,082 | 0.421 |

The first token is extreme:

```text
token 0 is selected in 25,701 / 28,672 possible layer-head-query cases.
selection rate = 89.6%
```

At layer-head granularity, token 0 appears in at least one query for 422/448 layer-heads, and is selected in all 64 queries for 389/448 layer-heads.

## 5. Position Distribution

Selection event mass by coarse position:

| Bucket | Tokens | Events | Fraction |
| --- | ---: | ---: | ---: |
| first 64 tokens | 64 | 36,051 | 1.50% |
| first 256 tokens | 256 | 46,501 | 1.93% |
| middle remote tokens 256-3839 | 3,584 | 997,600 | 41.42% |
| last 256 prefill tokens | 256 | 887,229 | 36.84% |
| eval-history tokens | 63 | 477,118 | 19.81% |

Interpretation:

```text
The total mass of sink tokens is small because the sink bucket is tiny.
But token 0 is individually very over-selected.
The dominant structural pattern is local/recent concentration:
last 256 prefill + eval-history tokens explain about 56.65% of all selected events.
```

## 6. Layer Pattern

Layer-level selected unique-token counts vary strongly:

```text
low unique layers:
layer 10: 1,477
layer 5: 1,594
layer 17: 1,699
layer 12: 1,771

high unique layers:
layer 1: 2,949
layer 2: 2,802
layer 0: 2,780
layer 6: 2,589
```

Layer position tendencies:

```text
layers 0,1,5,10,12,17,27 are more local/recent-heavy.
layers 6,8,11,13,16,18,19,20,21,24,25,26 keep more middle remote tokens.
```

Examples:

| Layer | middle remote fraction | last256 prefill | eval history |
| ---: | ---: | ---: | ---: |
| 5 | 31.50% | 40.69% | 26.58% |
| 17 | 29.05% | 45.80% | 23.82% |
| 25 | 55.23% | 30.47% | 12.15% |
| 20 | 51.11% | 31.82% | 14.54% |
| 27 | 30.68% | 40.38% | 27.55% |

## 7. Layer-Head Reuse Pattern

For each layer/head, count how many token positions are selected repeatedly across the 64 eval queries.

Mean over 448 layer-heads:

| Repeated selection threshold | Mean tokens per layer-head |
| ---: | ---: |
| selected in all 64 queries | 1.37 |
| selected in at least 48 queries | 7.41 |
| selected in at least 32 queries | 26.93 |
| selected in at least 16 queries | 96.97 |
| selected in at least 8 queries | 205.02 |
| selected at least once | 790.12 |

Interpretation:

```text
Each head has a small persistent token set that is repeatedly selected across adjacent decode steps,
plus a much larger long tail of transient selected tokens.
```

The most persistent heads have around 70-86 tokens selected in at least half of the 64 queries.

## 8. Implications

1. A fixed protected token set is justified, but should be tiny.

   Token 0 is almost universally selected. A small learned/static sink set may preserve a disproportionate number of head-token selections, but sink tokens do not dominate total mass.

2. Recent/local tokens dominate true top2 selection.

   More than half of all selected events come from the last 256 prefill tokens plus eval-history tokens. Any remote-KV compression experiment should separate:

   ```text
   local/recent top2 counts
   remote-only top2 counts
   sink top2 counts
   ```

3. Remote token selection remains substantial.

   Middle remote tokens still account for 41.4% of selected events. Recent-only attention cannot explain true top2 behavior.

4. Layer policy should not be uniform.

   Some layers are clearly local-heavy; others keep much more middle remote mass. This supports the R2H-KV direction of layer/head-specific policy rather than global top2 heuristics.

5. Top2 temporal reuse has real signal.

   Per layer/head, about 27 tokens on average are selected in at least half of the sampled decode steps. This supports reuse/persistent-cache diagnostics, but the transient tail remains large.

## 9. Next Experiment

Run the same diagnostic with remote-only buckets:

```text
exclude token 0 / sink
exclude recent window, e.g. last 512 tokens
count only remote top2 selections
```

Then compare:

```text
War and Peace 4k
War and Peace 20k
hard_topic_eval_v2 2k
Monte Cristo 4k
```

The key question is whether the high-frequency remote selected tokens are stable enough to become:

```text
protected remote anchors
page-level routing seeds
head-specific persistent KV slots
```

or whether they are mostly content-specific artifacts of the current 64-token continuation.

## 10. Remote-Only Top2 Count

Follow-up run:

```text
output = /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_remote_token_selection_counts_war_4k_s64_r512_v2
sink excluded = first 64 tokens
recent excluded = last 512 historical tokens for each query
```

Important implementation note:

```text
The v2 run explicitly caps the scored history to key_index < query_token.
This avoids counting same-chunk future tokens when an eager attention path does not expose an explicit finite causal mask.
The earlier remote-only v1 output should not be used.
```

Corrected true top2 budget:

```text
sum_query ceil(0.02 * query_token) * 28 layers * 16 heads = 2,381,568
remote-only selected events = 755,859
remote-only fraction of true top2 events = 31.74%
```

Overall remote-only concentration:

| Top remote tokens by count | Fraction of remote selection events |
| ---: | ---: |
| top 1 | 0.63% |
| top 10 | 4.43% |
| top 50 | 14.56% |
| top 100 | 23.12% |
| top 200 | 35.97% |
| top 500 | 60.40% |
| top 1000 | 83.08% |

Compared with all-token top2 counts, remote-only selection is less concentrated at the very top, but still has a strong long-tail skew.

Remote-only token coverage:

```text
historical-token rows = 4,159
nonzero remote-selected tokens = 3,398
zero remote-selected tokens = 761
max normalized remote selection rate = 0.166
```

Top remote tokens:

| token index | token text | count | normalized rate |
| ---: | --- | ---: | ---: |
| 3304 | `!”\n\n` | 4,764 | 0.166 |
| 3566 | `.”\n\n` | 4,429 | 0.154 |
| 3374 | `?”\n\n` | 3,671 | 0.128 |
| 2925 | `.\n\n` | 3,347 | 0.117 |
| 2944 | `:\n\n` | 3,145 | 0.110 |
| 3551 | ` secretary` | 2,903 | 0.101 |
| 3579 | `,` | 2,839 | 0.099 |
| 3316 | `.\n\n` | 2,835 | 0.099 |
| 3476 | `?”\n\n` | 2,830 | 0.099 |
| 2138 | `.”\n\n` | 2,746 | 0.096 |

Position distribution inside the remote region:

| Bucket | Tokens | Nonzero tokens | Events | Fraction |
| --- | ---: | ---: | ---: | ---: |
| 64-511 | 448 | 411 | 12,293 | 1.63% |
| 512-1023 | 512 | 392 | 5,567 | 0.74% |
| 1024-2047 | 1,024 | 996 | 35,217 | 4.66% |
| 2048-3071 | 1,024 | 1,024 | 368,036 | 48.69% |
| 3072-3583 | 512 | 512 | 321,081 | 42.48% |
| 3584+ partial remote eligibility | 575 | 63 | 13,665 | 1.81% |

Interpretation:

```text
Remote top2 is not uniformly distributed across remote history.
It is heavily biased toward the far edge of the allowed remote region, especially tokens 2048-3583.
Very old non-sink remote tokens contribute little in this 4k/64-token War sample.
```

Layer remote-only event counts:

| High remote-event layers | events |
| ---: | ---: |
| 25 | 37,744 |
| 26 | 35,286 |
| 20 | 34,895 |
| 21 | 33,944 |
| 8 | 33,515 |
| 11 | 33,048 |
| 24 | 32,638 |
| 18 | 32,434 |

Low remote-event layers:

| Low remote-event layers | events |
| ---: | ---: |
| 0 | 15,447 |
| 27 | 16,374 |
| 17 | 17,853 |
| 12 | 18,721 |
| 10 | 18,979 |
| 1 | 20,250 |

This sharp layer spread strengthens the case for layer/head-specific remote policies.

Per layer/head repeated remote tokens:

| Repeated remote-selection threshold | Mean tokens per layer/head |
| ---: | ---: |
| selected in all 64 queries | 0.06 |
| selected in at least 48 queries | 0.64 |
| selected in at least 32 queries | 2.50 |
| selected in at least 16 queries | 15.99 |
| selected in at least 8 queries | 54.32 |
| selected at least once | 438.62 |

Compared with all-token top2 counts:

```text
remote persistent sets are much smaller.
All-token count had about 26.9 tokens/head selected in >=32 queries;
remote-only has about 2.5 tokens/head selected in >=32 queries.
```

Most remote-reuse-heavy layer-heads:

| layer/head | unique remote tokens | max count | tokens selected in >=32 queries | tokens selected in >=16 queries |
| --- | ---: | ---: | ---: | ---: |
| L25H2 | 158 | 64 | 34 | 60 |
| L26H10 | 242 | 64 | 33 | 64 |
| L10H6 | 233 | 64 | 30 | 58 |
| L13H13 | 277 | 63 | 24 | 27 |
| L20H13 | 338 | 58 | 24 | 73 |

## 11. Updated Takeaway

The remote-only result changes the interpretation:

```text
Remote top2 is still important, but persistent remote anchors are sparse.
```

The promising unit is probably not a large static global remote token set. A better design is:

```text
1. keep a tiny sink set,
2. keep a recent window,
3. identify a small number of persistent remote anchors per high-remote layer/head,
4. use those anchors to route or seed page-level retrieval for the transient remote tail.
```

This supports a hybrid design:

```text
persistent remote anchors + page/routing fallback
```

rather than replacing remote attention with a static protected-token list.

## 12. What Are The Remote Tokens?

A follow-up content analysis categorized the remote-only selected tokens by token text.

Remote-only event share by token category:

| Category | Unique tokens | Events | Fraction |
| --- | ---: | ---: | ---: |
| content word / subword | 887 | 380,085 | 50.29% |
| function word / pronoun | 492 | 133,257 | 17.63% |
| punctuation | 167 | 95,173 | 12.59% |
| capitalized / name-like | 953 | 65,281 | 8.64% |
| sentence/dialogue boundary | 31 | 47,985 | 6.35% |
| whitespace/newline | 763 | 24,978 | 3.30% |
| number | 104 | 9,050 | 1.20% |

Top-ranked tokens are more boundary-heavy:

| Rank band | Dominant categories |
| --- | --- |
| top 50 | content 43.9%, sentence/dialogue boundary 34.4%, punctuation 18.3% |
| top 100 | content 46.9%, sentence/dialogue boundary 22.5%, punctuation 21.1% |
| top 500 | content 51.7%, punctuation 17.4%, function/pronoun 11.6%, boundary 10.1% |
| all nonzero remote | content 50.3%, function/pronoun 17.6%, punctuation 12.6%, name-like 8.6% |

Top remote tokens are not random. They mostly come from the active dialogue/topic before the eval span:

```text
remote boundary before eval:
... secure it for the baron.

Anna Pavlovna almost closed her eyes ...

eval target:
....”

The prince answered nothing, but she looked at him significantly,
awaiting a reply. He frowned.

“What would you have me do?” ...
```

Representative high-count remote tokens:

| Type | Examples | Function |
| --- | --- | --- |
| dialogue / paragraph boundary | `!”\n\n`, `.”\n\n`, `?”\n\n`, `.\n\n`, `:\n\n` | anchors speaker turns, quotation boundaries, paragraph transitions |
| current semantic content | `secretary`, `visit`, `son`, `paused`, `war`, `post`, `appointed`, `carelessness` | carries topic state and event/action semantics |
| entities / names | `Prince`, `Europe`, `Vienna`, `Russia`, `Austria`, `French` | anchors participants, places, political context |
| syntactic / discourse glue | `She`, `the`, `with`, `for`, `he`, `prince` | supports coreference and local syntax around remote topic |
| punctuation | `.`, `,`, `?` | sentence boundary, clause structure, quote rhythm |

Interpretation:

```text
The remote tokens serve two roles:

1. structural anchors:
   quote endings, paragraph breaks, punctuation, dialogue-turn boundaries;

2. semantic anchors:
   topic words, entity names, relationship words, and action words from the same conversation.
```

This matters because a pure frequency-based protected remote set would over-protect punctuation/dialogue boundaries,
while a pure semantic-keyword set would miss structural anchors that the model repeatedly attends to.

Layer category pattern:

```text
layers 1-2 are punctuation-heavy;
middle and late layers are mostly content-word heavy;
layers 24-26 show stronger name/entity content than most earlier layers.
```

Examples:

| Layer | Dominant remote categories |
| ---: | --- |
| 1 | punctuation 30%, content 24%, function/pronoun 18% |
| 2 | punctuation 39%, content 21%, function/pronoun 18% |
| 17 | content 61%, function/pronoun 12%, punctuation 10% |
| 20 | content 60%, function/pronoun 20%, name-like 9% |
| 25 | content 65%, function/pronoun 13%, name-like 9% |
| 26 | content 59%, name-like 15%, function/pronoun 12% |

Updated design implication:

```text
Remote anchor selection should be typed.

Keep a small quota for structural anchors and another quota for semantic/entity anchors,
or learn this split per layer/head.
```

For page routing, the top remote anchors look useful as:

```text
boundary anchors -> identify relevant dialogue/paragraph pages
semantic/entity anchors -> identify topic/evidence pages
```

The transient tail likely needs routed page retrieval rather than static protection.

## 13. Typed Anchor Event And Attention-Mass Experiment

Follow-up experiment:

```text
output = /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/top2_remote_typed_anchor_war_4k_s64_r512_v1
base = remote-only top2, sink64, recent512, War 4k/64
extra = accumulate selected_attention_mass_sum
postprocess = structural / semantic / other anchor grouping
page_size = 64
```

New scripts:

```text
src/summarize_top2_remote_anchor_types.py
scripts/run_top2_remote_typed_anchor_server.sh
```

Anchor type definitions:

```text
structural = punctuation, quote/paragraph/sentence boundaries, newline/whitespace
semantic   = content/subword tokens, capitalized/name-like tokens, numeric tokens
other      = mostly function words and pronouns
```

Overall typed-anchor coverage:

| Type | Unique tokens | Events | Event fraction | Attention mass | Mass fraction | Mean mass / event |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| structural | 1,200 | 185,781 | 24.58% | 457.585 | 30.08% | 0.00246 |
| semantic | 2,292 | 436,821 | 57.79% | 811.837 | 53.36% | 0.00186 |
| other | 667 | 133,257 | 17.63% | 251.888 | 16.56% | 0.00189 |

Interpretation:

```text
Semantic anchors dominate event count and total mass.
Structural anchors are fewer events but higher average attention mass per event.
```

This supports the intuition that structural anchors are not just frequent punctuation noise:

```text
structural event fraction = 24.6%
structural mass fraction  = 30.1%
```

Top structural anchors:

| token | count | mass |
| --- | ---: | ---: |
| `!”\n\n` | 4,764 | 13.596 |
| `.”\n\n` | 4,429 | 15.587 |
| `?”\n\n` | 3,671 | 11.994 |
| `.\n\n` | 3,347 | 10.673 |
| `:\n\n` | 3,145 | 10.478 |
| `,` | 2,839 | 8.871 |

Top semantic anchors:

| token | count | mass |
| --- | ---: | ---: |
| ` secretary` | 2,903 | 5.516 |
| ` visit` | 2,643 | 5.925 |
| ` son` | 2,582 | 8.461 |
| ` paused` | 2,507 | 7.200 |
| ` suddenly` | 2,337 | 6.357 |
| ` war` | 2,307 | 3.535 |
| `“Well` | 2,192 | 6.046 |
| ` prince` | 2,004 | 7.953 |

Top other anchors:

| token | count | mass |
| --- | ---: | ---: |
| `She` | 2,169 | 6.360 |
| ` the` | 1,404 | 1.578 |
| ` them` | 1,300 | 2.700 |
| ` with` | 1,237 | 4.546 |
| ` for` | 1,212 | 3.347 |
| ` he` | 1,135 | 4.114 |

Layer pattern by mass:

| Layer group | Pattern |
| --- | --- |
| early layers 1-2 | structural-heavy: layer 1 structural mass 49.3%, layer 2 structural mass 52.3% |
| layer 10 | extreme structural mass: 66.7% |
| middle/late semantic layers | layers 16-26 are mostly semantic-mass dominated |
| strongest semantic layers | layer 24 semantic mass 76.7%, layer 25 78.9%, layer 26 71.6% |

Selected layer examples:

| Layer | Structural event | Structural mass | Semantic event | Semantic mass | Other mass |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 48.2% | 49.3% | 33.3% | 25.4% | 25.3% |
| 2 | 55.9% | 52.3% | 25.8% | 21.7% | 26.0% |
| 10 | 29.2% | 66.7% | 54.6% | 26.6% | 6.7% |
| 20 | 11.7% | 10.6% | 68.0% | 69.8% | 19.6% |
| 24 | 11.0% | 5.0% | 69.4% | 76.7% | 18.3% |
| 25 | 15.6% | 13.7% | 71.0% | 78.9% | 7.5% |
| 26 | 16.1% | 20.1% | 71.6% | 71.6% | 8.3% |

Page-level proxy:

```text
For each layer/head, collect pages containing structural anchors.
Measure what fraction of semantic selected events/mass lies on those structural pages.
```

Layer/head distribution:

```text
structural pages per layer/head:
  mean = 22.65 pages
  median = 23 pages
  p25 = 17 pages
  p75 = 27 pages

semantic event fraction on structural pages:
  mean = 85.97%
  median = 92.90%
  p25 = 81.60%
  p75 = 98.00%

semantic attention-mass fraction on structural pages:
  mean = 87.06%
  median = 93.78%
  p25 = 83.50%
  p75 = 98.80%
```

This is only an aggregate proxy, not query-level routing proof, because it does not check whether a structural anchor is selected for the same query as the semantic token.
Still, it is a strong signal:

```text
Most semantic remote top2 mass is located on pages that also contain structural remote anchors,
at the layer/head aggregate level.
```

Typed-anchor routing hypothesis:

```text
1. structural anchors identify candidate remote pages;
2. semantic anchors inside those pages carry topic/entity/action evidence;
3. function/pronoun anchors help coreference but should not dominate routing.
```

A plausible next diagnostic is query-level page recall:

```text
For each query/layer/head:
  page is recalled if any structural top2 anchor from that page is selected.
  measure semantic top2 mass on recalled pages.
```

If this query-level recall remains high, then the typed-anchor idea can become a concrete method:

```text
Typed-Anchor Page Routing:
  structural-anchor page recall
  + semantic-token/page reranking
  + recent/sink protection
```

## 14. Query-level structural page routing diagnostic

Question:

```text
Can remote structural top2 anchors route to pages that contain the remote semantic top2 tokens
for the same query/layer/head?
```

This is stricter than the aggregate proxy above.  For each query, layer, and head:

```text
1. compute true remote top2 selected tokens from full QK attention;
2. exclude sink tokens and the recent window;
3. split selected remote tokens into structural / semantic / other;
4. recall pages containing selected structural anchors;
5. measure how much selected semantic top2 event count and attention mass lies on those pages.
```

Config:

```text
model: Qwen3-0.6B
text: War and Peace
prefill/eval: 4096 + 64 tokens
sink: 64
recent window: 512
top fraction: 2%
fixed block baseline: 64 tokens
structural page max length: 128 tokens
structural boundary mode: paragraph/dialogue boundary
structural adjacent radius: +/- 1 structural page
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/typed_anchor_page_recall_war_4k_s64_r512_v3_para
```

Main result:

| Scheme | Mean recalled pages | Semantic event recall | Semantic mass recall |
| --- | ---: | ---: | ---: |
| fixed 64-token block | 4.30 | 29.08% | 33.79% |
| structural page | 4.31 | 30.82% | 35.76% |
| structural page +/- 1 | 9.06 | 42.95% | 47.92% |

Interpretation:

```text
At almost the same number of recalled pages, paragraph/dialogue structural pages beat fixed
64-token blocks by about +1.74 event-recall points and +1.97 attention-mass points.
```

The adjacent-page variant has much higher recall, but it recalls about 2.1x as many pages.
So it is a recall-expansion strategy, not a fair equal-budget replacement for fixed blocks.

Oracle page coverage gives the upper bound when pages are ranked directly by selected semantic mass:

| Scheme | Top 1 page | Top 2 pages | Top 4 pages | Top 8 pages | Top 16 pages |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed 64-token block | 31.63% | 48.69% | 69.03% | 88.42% | 98.97% |
| structural page | 33.24% | 50.88% | 71.61% | 90.33% | 98.88% |

Pages needed to reach semantic-mass thresholds:

| Scheme | 80% mass | 90% mass | 95% mass |
| --- | ---: | ---: | ---: |
| fixed 64-token block | 4.61 | 6.03 | 7.07 |
| structural page | 4.38 | 5.72 | 6.69 |

This means the structural page partition is not only better for the structural-anchor recall
proxy; it also has a better oracle page layout.  The relevant semantic mass is more compact
under paragraph/dialogue structural pages than under fixed token blocks.

Layer-level pattern:

```text
Best structural-over-fixed semantic-mass gains:
  layer 11: +6.54 points
  layer 15: +5.12 points
  layer 25: +4.27 points
  layer 23: +3.99 points
  layer 8 : +3.94 points
  layer 21: +3.30 points
  layer 20: +2.78 points
  layer 22: +2.75 points

Largest negative deltas:
  layer 17: -3.13 points
  layer 27: -1.19 points
  layer 1 : -0.67 points
  layer 2 : -0.63 points
```

So the advantage is concentrated in some middle/late layers, especially where semantic remote
selection is stronger.  Early layers and a few late heads can still prefer fixed token locality.

Important negative control:

```text
Using sentence/punctuation-level structural boundaries made pages too fragmented
(mean page length about 7.9 tokens).  In that setting structural-only recall was worse than
fixed blocks, and only structural +/- 1 recovered a small advantage at much larger page cost.
```

So the useful routing unit should not be "every punctuation boundary".  The better version is:

```text
paragraph/dialogue structural boundary -> page recall
semantic anchors inside recalled pages -> topic/entity evidence
```

Current conclusion:

```text
The result supports the typed-anchor page routing hypothesis, but the gain is modest in the
equal-page-budget query-level test.  The method is worth developing if page construction uses
coarser paragraph/dialogue anchors and the reranker uses semantic anchors inside recalled pages.
```

## 15. Hierarchical book-index routing diagnostic

Hypothesis:

```text
Build text memory like a book:
  short fragments -> sentences
  sentences -> paragraph pages
  paragraph pages -> sections
  sections -> book

Each unit gets a lightweight summary vector.  At query time, keep sink/recent normally,
then route remote KV through the hierarchy to recall relevant paragraph pages.
```

Implemented diagnostic:

```text
script:
  src/analyze_hierarchical_book_index_recall.py

server run:
  scripts/run_hierarchical_book_index_server.sh

output:
  /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/hierarchical_book_index_war_4k_s64_r512_v3_tailctrl
```

This first version deliberately avoids downloading a sentence-transformer or training an MLP.
It uses a tiny local TF-IDF lexical vector as the summary model:

```text
paragraph summary = top TF-IDF content terms
section summary   = TF-IDF over grouped paragraph text
query vector      = TF-IDF over the previous 256 tokens
```

Config:

```text
model: Qwen3-0.6B
text: War and Peace
prefill/eval: 4096 + 64 tokens
sink: 64
recent window: 512
top fraction: 2%
paragraph min/max: 64 / 192 tokens
section: 8 paragraphs
paragraph count: 69
section count: 9
paragraph mean length: 60.3 tokens
```

Main result:

| Scheme | Mean pages | Semantic event recall | Semantic mass recall |
| --- | ---: | ---: | ---: |
| fixed-anchor baseline | 4.30 | 29.08% | 33.79% |
| paragraph-anchor baseline | 4.26 | 30.01% | 35.00% |
| remote tail p4 | 4.00 | 17.67% | 19.08% |
| book flat TF-IDF p4 | 4.00 | 20.69% | 21.75% |
| book hierarchical s2 p2 | 4.00 | 19.29% | 20.78% |
| remote tail p8 | 8.00 | 37.26% | 39.81% |
| book flat TF-IDF p8 | 8.00 | 36.10% | 37.34% |
| book hierarchical s4 p2 | 8.00 | 37.08% | 40.34% |
| remote tail p16 | 16.00 | 67.63% | 68.58% |
| book flat TF-IDF p16 | 16.00 | 62.80% | 65.66% |
| book hierarchical s4 p4 | 15.99 | 65.36% | 68.65% |

Interpretation:

```text
1. Coarser paragraph pages are better than fragmented structural pages.
   paragraph-anchor mass recall rises to 35.00%, above fixed-anchor 33.79%.

2. Runtime lexical book-index retrieval has real signal:
   at 4 pages, book flat TF-IDF beats remote-tail by +2.67 mass points.

3. But the simple remote-tail control is very strong in this sample:
   at 8/16 pages, tail is already near or above flat TF-IDF.

4. Hierarchical routing is better than flat TF-IDF at equal large budgets:
   s4_p2 beats flat p8 by +3.60 mass points;
   s4_p4 beats flat p16 by +2.98 mass points.

5. Hierarchical routing only barely beats remote-tail overall:
   s4_p2 beats tail p8 by +0.53 mass points;
   s4_p4 beats tail p16 by +0.06 mass points.
```

Layer pattern versus remote-tail:

```text
book flat p4 beats tail p4 in 17 / 28 layers.
largest gains:
  layer 5 : +21.44 mass points
  layer 21: +12.34
  layer 2 : +11.69
  layer 25: +11.22
  layer 20: +9.89

book hierarchical s4_p2 beats tail p8 in 14 / 28 layers.
largest gains:
  layer 5 : +15.23 mass points
  layer 2 : +10.44
  layer 21: +9.09
  layer 25: +9.06
  layer 23: +6.46

book hierarchical s4_p4 beats tail p16 in 13 / 28 layers.
largest gains:
  layer 1 : +7.75 mass points
  layer 26: +6.75
  layer 10: +6.63
  layer 18: +6.22
  layer 20: +5.87
```

Negative signal:

```text
Some layers strongly prefer the simple remote tail, especially layer 12/15/17/27 in this run.
So a single global book-index policy is probably not enough.
```

Current conclusion:

```text
The book-index idea is plausible, but the first TF-IDF version is not yet a clear replacement
for simple remote locality.  It becomes promising if used as a layer/head-aware extra route:

  keep sink + recent
  keep a small remote-tail band
  add hierarchical book-index pages for layers/heads where lexical/semantic routing wins
  use structural anchors to define stable pages
  use semantic summaries to rerank within sections/pages
```

The next implementation target should be:

```text
Layer/head-aware hybrid:
  if a head historically benefits from tail locality -> remote_tail
  if a head benefits from semantic routing -> book_index
  union both under a fixed page budget, then rerank by cheap page summary score
```

## 16. Long-range semantic retrieval with near-tail decoy

Motivation:

```text
The War and Peace continuation setup favors remote-tail locality.
For long-range semantic retrieval, the important evidence can be far earlier than the remote tail.
```

New diagnostic:

```text
script:
  src/analyze_longrange_book_index_semantic_retrieval.py

server run:
  scripts/run_longrange_book_index_semantic_server.sh

output:
  /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_semantic_10k20k_smoke_v2_auth
```

Task construction:

```text
context length: 10k and 20k
tasks per length: 2 smoke tasks

early context:
  AUTHORITATIVE EVIDENCE PAGE:
    key = K
    true ANSWER_LABEL = Y

near remote tail:
  NEAR-TAIL DECOY PAGE:
    same key = K
    wrong ANSWER_LABEL = Z
    explicitly says obsolete / non-authoritative / misleading

query at the end:
  asks for the AUTHORITATIVE EVIDENCE PAGE answer
```

This tests three separate things:

```text
1. Does routing recall the true early evidence page?
2. Does routing avoid the near-tail decoy page?
3. How much true model remote top2 semantic mass does the route cover?
```

Important finding:

```text
In decoy tasks, semantic-mass recall and key-evidence recall are not the same metric.
Remote-tail gets high top2 mass because the model attends heavily to the near-tail decoy,
but it has zero true-evidence recall.
```

### 10k results

| Scheme | Pages | Top2 semantic mass recall | Evidence hit | Decoy hit |
| --- | ---: | ---: | ---: | ---: |
| remote_tail_p4 | 4 | 23.35% | 0.00 | 1.00 |
| remote_tail_p8 | 8 | 27.34% | 0.00 | 1.00 |
| remote_tail_p16 | 16 | 32.21% | 0.00 | 1.00 |
| remote_tail_p32 | 32 | 40.07% | 0.00 | 1.00 |
| book_flat_p4 | 4 | 5.59% | 0.50 | 0.00 |
| book_flat_p8 | 8 | 8.91% | 0.72 | 0.00 |
| book_flat_p16 | 16 | 13.95% | 1.00 | 0.00 |
| book_flat_p32 | 32 | 28.89% | 1.00 | 0.44 |
| book_hier_s4_p2 | 8 | 13.34% | 1.00 | 0.00 |
| book_hier_s4_p4 | 16 | 26.21% | 1.00 | 0.50 |
| book_hier_s8_p4 | 32 | 38.51% | 1.00 | 0.78 |
| book_auth_flat_p4 | 4 | 7.53% | 1.00 | 0.00 |
| book_auth_flat_p8 | 8 | 10.00% | 1.00 | 0.00 |
| book_auth_flat_p16 | 16 | 13.95% | 1.00 | 0.00 |
| book_auth_flat_p32 | 32 | 25.29% | 1.00 | 0.00 |
| book_auth_hier_s4_p2 | 8 | 9.41% | 1.00 | 0.00 |
| book_auth_hier_s4_p4 | 16 | 16.28% | 1.00 | 0.00 |
| book_auth_hier_s8_p4 | 32 | 23.68% | 1.00 | 0.00 |

### 20k results

| Scheme | Pages | Top2 semantic mass recall | Evidence hit | Decoy hit |
| --- | ---: | ---: | ---: | ---: |
| remote_tail_p4 | 4 | 23.46% | 0.00 | 1.00 |
| remote_tail_p8 | 8 | 25.95% | 0.00 | 1.00 |
| remote_tail_p16 | 16 | 30.65% | 0.00 | 1.00 |
| remote_tail_p32 | 32 | 37.91% | 0.00 | 1.00 |
| book_flat_p4 | 4 | 11.21% | 1.00 | 1.00 |
| book_flat_p8 | 8 | 12.06% | 1.00 | 1.00 |
| book_flat_p16 | 16 | 13.49% | 1.00 | 1.00 |
| book_flat_p32 | 32 | 18.32% | 1.00 | 1.00 |
| book_hier_s4_p2 | 8 | 15.94% | 1.00 | 1.00 |
| book_hier_s4_p4 | 16 | 31.53% | 1.00 | 1.00 |
| book_hier_s8_p4 | 32 | 36.49% | 1.00 | 1.00 |
| book_auth_flat_p4 | 4 | 3.02% | 1.00 | 0.00 |
| book_auth_flat_p8 | 8 | 3.87% | 1.00 | 0.00 |
| book_auth_flat_p16 | 16 | 5.80% | 1.00 | 0.00 |
| book_auth_flat_p32 | 32 | 10.77% | 1.00 | 0.00 |
| book_auth_hier_s4_p2 | 8 | 5.24% | 1.00 | 0.00 |
| book_auth_hier_s4_p4 | 16 | 10.37% | 1.00 | 0.00 |
| book_auth_hier_s8_p4 | 32 | 13.78% | 1.00 | 0.00 |

Interpretation:

```text
remote_tail:
  high mass recall, but 0% evidence hit and 100% decoy hit.
  It is following near-tail locality, not solving the semantic retrieval task.

plain book-index:
  often recalls the true early evidence, but also recalls decoys when the key appears in both.
  It needs typed page summaries, not just lexical similarity.

authority-aware typed summary:
  100% evidence hit and 0% decoy hit in this smoke run.
  But it has lower top2 mass recall because the model's own top2 attention is attracted to the decoy.
```

This is the most important conceptual update:

```text
For long-range semantic QA, optimizing only true-top2 attention mass can reward the wrong memory.
If the model attends to a near-tail decoy, mass recall prefers the decoy.

So the routing objective needs at least two metrics:
  1. key evidence recall / answer support recall
  2. attention-mass or PPL preservation
```

Design implication:

```text
Typed-anchor book routing should have page roles:
  structural: boundary / section / dialogue / list position
  semantic: key, entity, topic, answer-bearing content
  authority/status: authoritative, obsolete, decoy, negated, summary, quote

The page router should not only ask "which page is lexically similar?"
It should ask "which page is the right type of evidence for this query?"
```

Next optimization target:

```text
Hybrid evidence-safe routing:
  keep sink + recent
  keep a small remote-tail budget for PPL/locality
  add authority-aware book-index pages for evidence recall
  avoid or downweight decoy/status-negative pages
  tune layer/head budgets separately for PPL heads vs retrieval heads
```

## 17. Prompt-pruned downstream QA smoke

Purpose:

```text
The previous diagnostic measured page recall and true-top2 mass.
This run asks whether the selected pages actually let the model answer the long-range QA task.
```

This is not yet a sparse-attention kernel.  It is a prompt-pruned proxy:

```text
For each route:
  build a short prompt from sink + selected remote pages + recent + query
  score answer labels A/B/C/D
  record accuracy, decoy prediction rate, evidence hit, decoy hit, token ratio
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_downstream_10k20k_smoke_v2_calib
```

A no-context label-prior calibration is included because Qwen3-0.6B shows a strong single-letter
prior on this tiny smoke set.  The calibrated score subtracts the label score under the query-only
prompt.

### 10k downstream smoke

| Scheme | Raw acc | Cal acc | Evidence hit | Decoy hit | Token ratio | Raw margin | Cal margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_context | 50% | 50% | 100% | 100% | 98.3% | -0.24 | -0.88 |
| remote_tail_p4 | 50% | 50% | 0% | 100% | 9.9% | -0.36 | -1.00 |
| remote_tail_p8 | 50% | 50% | 0% | 100% | 13.2% | -0.67 | -1.32 |
| remote_tail_p16 | 50% | 50% | 0% | 100% | 19.9% | -0.77 | -1.41 |
| book_flat_p4 | 50% | 50% | 0% | 0% | 9.4% | +0.29 | -0.36 |
| book_auth_flat_p4 | 50% | 50% | 100% | 0% | 9.7% | +1.37 | +0.73 |
| book_auth_flat_p8 | 50% | 50% | 100% | 0% | 13.0% | +1.64 | +1.00 |
| book_auth_hier_s4_p2 | 50% | 50% | 100% | 0% | 12.9% | +1.63 | +0.98 |
| hybrid_tail4_authflat4 | 50% | 50% | 100% | 100% | 13.4% | +0.96 | +0.32 |

10k interpretation:

```text
Accuracy is not very informative with only 2 tasks and strong label prior.
Margins are informative:
  remote-tail has negative calibrated true-vs-decoy margin;
  authority-aware book pages have positive calibrated margins and avoid the decoy.
```

### 20k downstream smoke

| Scheme | Raw acc | Cal acc | Evidence hit | Decoy hit | Token ratio | Raw margin | Cal margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full_context | 50% | 50% | 100% | 100% | 98.4% | -0.18 | +1.37 |
| recent_only | 0% | 0% | 0% | 0% | 2.7% | -0.98 | +0.56 |
| remote_tail_p4 | 0% | 0% | 0% | 100% | 5.0% | -1.80 | -0.26 |
| remote_tail_p8 | 0% | 0% | 0% | 100% | 6.7% | -2.00 | -0.46 |
| remote_tail_p16 | 0% | 0% | 0% | 100% | 10.2% | -2.11 | -0.57 |
| book_flat_p4 | 0% | 0% | 0% | 0% | 4.8% | -1.43 | +0.11 |
| book_auth_flat_p4 | 50% | 100% | 100% | 0% | 5.0% | +0.29 | +1.83 |
| book_auth_flat_p8 | 50% | 50% | 100% | 0% | 6.7% | -0.09 | +1.45 |
| book_auth_hier_s4_p2 | 50% | 100% | 100% | 0% | 6.6% | +0.02 | +1.56 |
| hybrid_tail4_authflat4 | 50% | 50% | 100% | 100% | 6.9% | -0.86 | +0.68 |
| hybrid_tail4_authhier_s4_p2 | 50% | 50% | 100% | 100% | 8.5% | -0.27 | +1.28 |

20k interpretation:

```text
The clearest result is:

  book_auth_flat_p4:
    5.0% tokens
    100% evidence hit
    0% decoy hit
    100% calibrated accuracy

  book_auth_hier_s4_p2:
    6.6% tokens
    100% evidence hit
    0% decoy hit
    100% calibrated accuracy

  remote_tail_p4/p8/p16:
    5-10% tokens
    0% evidence hit
    100% decoy hit
    0% calibrated accuracy
```

This supports the original hypothesis for long contexts:

```text
For long-range semantic retrieval, page-index routing can beat remote-tail by a large margin
on key-evidence recall and downstream answerability, while using only about 5-7% of prompt tokens.
```

But it also shows a warning:

```text
Adding remote-tail back into a hybrid route improves PPL/locality potential,
but it reintroduces the decoy page in this benchmark.
So hybrid needs a decoy/status gate, not a naive union.
```

Current design update:

```text
Evidence-safe typed page routing:
  1. structural pages define stable paragraph/section units;
  2. semantic summaries recall pages matching key/entity/topic;
  3. authority/status summaries rerank or filter pages;
  4. remote-tail is allowed only under a status gate or low budget;
  5. PPL/locality heads and semantic-retrieval heads should get different page budgets.
```

Next concrete experiment:

```text
Move from prompt-pruned proxy to sparse-attention PPL/downstream:
  - use book_auth pages as protected remote pages;
  - optionally add a small gated tail budget;
  - compare PPL, answer accuracy, evidence hit, decoy hit, selected token ratio, and wall time.
```

## 18. Full-context KV sparse page-mask smoke

Purpose:

```text
The prompt-pruned proxy changes the prompt.
This run keeps full-context KV prefill, then masks attention during query/answer scoring so the
model can only attend to:
  sink tokens
  recent tokens
  selected remote page tokens
```

This is closer to the intended KV-cache method.

Implementation:

```text
script:
  src/run_longrange_book_index_sparse_eval.py

server run:
  scripts/run_longrange_book_index_sparse_server.sh

output:
  /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_10k20k_smoke_v1
```

Important caveat:

```text
This smoke masks attention logits after full QK has already been computed.
So mean_kept_fraction is a compute proxy, not measured kernel speedup.
It tells us the target sparse workload size, not actual accelerated runtime yet.
```

### 10k sparse-mask results

| Mode | Accuracy | Decoy pred | Query PPL | Evidence hit | Decoy hit | Kept fraction | Kept tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 50% | 50% | 6.63 | 100% | 100% | 100.0% | 10026 |
| sink_recent | 50% | 50% | 96.69 | 0% | 0% | 5.75% | 576 |
| remote_tail_p4 | 0% | 100% | 8.51 | 0% | 100% | 8.38% | 840 |
| remote_tail_p8 | 0% | 100% | 8.52 | 0% | 100% | 10.76% | 1079 |
| book_auth_flat_p4 | 50% | 50% | 7.49 | 100% | 0% | 8.44% | 846 |
| book_auth_flat_p8 | 50% | 50% | 7.50 | 100% | 0% | 10.83% | 1086 |
| book_auth_hier_s4_p2 | 50% | 50% | 7.49 | 100% | 0% | 10.83% | 1086 |
| hybrid_tail4_authflat4 | 50% | 50% | 6.47 | 100% | 100% | 11.07% | 1110 |

10k interpretation:

```text
remote_tail preserves more locality than sink_recent but routes to the decoy and has worse PPL
than book_auth.

book_auth keeps about the same number of tokens as remote_tail, but recalls evidence instead of
decoy and has lower query PPL.

hybrid has the best PPL, even slightly better than full in this tiny smoke, but it includes the
decoy page.  This reinforces the need for a gated tail, not naive union.
```

### 20k sparse-mask results

| Mode | Accuracy | Decoy pred | Query PPL | Evidence hit | Decoy hit | Kept fraction | Kept tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 50% | 50% | 7.82 | 100% | 100% | 100.0% | 20026 |
| sink_recent | 0% | 50% | 117.97 | 0% | 0% | 2.88% | 576 |
| remote_tail_p4 | 0% | 100% | 11.19 | 0% | 100% | 4.16% | 833 |
| remote_tail_p8 | 0% | 100% | 11.16 | 0% | 100% | 5.36% | 1074 |
| book_auth_flat_p4 | 50% | 50% | 9.24 | 100% | 0% | 4.33% | 868 |
| book_auth_flat_p8 | 50% | 50% | 9.21 | 100% | 0% | 5.54% | 1110 |
| book_auth_hier_s4_p2 | 50% | 50% | 9.19 | 100% | 0% | 5.53% | 1108 |
| hybrid_tail4_authflat4 | 50% | 50% | 8.05 | 100% | 100% | 5.62% | 1125 |

20k interpretation:

```text
At 20k, book_auth keeps only about 4.3-5.5% of history tokens and still preserves the evidence page.
Compared with remote_tail at similar budget:

  remote_tail_p4:
    query PPL = 11.19
    evidence hit = 0%
    decoy hit = 100%

  book_auth_flat_p4:
    query PPL = 9.24
    evidence hit = 100%
    decoy hit = 0%

  remote_tail_p8:
    query PPL = 11.16
    evidence hit = 0%
    decoy hit = 100%

  book_auth_hier_s4_p2:
    query PPL = 9.19
    evidence hit = 100%
    decoy hit = 0%
```

The hybrid route shows the PPL/locality tradeoff:

```text
hybrid_tail4_authflat4:
  query PPL = 8.05, close to full PPL = 7.82
  kept fraction = 5.62%
  evidence hit = 100%
  decoy hit = 100%
```

So the hybrid route is attractive for PPL, but unsafe for decoy-heavy semantic retrieval unless
tail pages are passed through a status/authority gate.

Current conclusion:

```text
The book_auth route is the first version that simultaneously gives:
  - long-range evidence recall;
  - decoy avoidance;
  - much better query PPL than sink_recent;
  - better query PPL than remote_tail at comparable token budget;
  - about 4-6% target history-token keep ratio at 20k.
```

This moves the method from prompt-pruned proof-of-concept toward a real KV-cache routing method.

Next optimization target:

```text
Status-gated hybrid:
  1. always keep sink/recent;
  2. keep book_auth pages;
  3. add remote-tail pages only if they do not look status-negative / decoy-like;
  4. optionally keep tail only for PPL-oriented layers/heads, while retrieval heads use book_auth.

Then rerun:
  - sparse query PPL;
  - downstream answer accuracy;
  - evidence hit / decoy hit;
  - target kept fraction;
  - eventually a real sparse kernel timing path.
```

## 19. Status-gated hybrid for long-range semantic retrieval

Question:

```text
For tasks that need long-range semantic retrieval, can we keep the PPL benefit of remote-tail
without letting near-tail decoys dominate the route?
```

Implementation:

```text
script:
  src/run_longrange_book_index_sparse_eval.py

new modes:
  hybrid_gatedtail4_authflat4
  hybrid_gatedtail4_authhier_s4_p2

output:
  /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_10k20k_smoke_v2_gated
```

The gated-tail rule is deliberately simple:

```text
Take the last 4 remote pages, but keep a tail page only if its authority/status score is non-negative.
Then union those gated tail pages with the authority-aware semantic pages.
```

This is not a learned router yet.  It is a controlled test of the design principle:

```text
semantic / authority pages should carry long-range evidence;
tail pages should be optional locality support, not unconditional memory.
```

### 10k gated sparse-mask results

| Mode | Accuracy | Decoy pred | Query PPL | Evidence hit | Decoy hit | Kept fraction | Kept tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 50% | 50% | 6.63 | 100% | 100% | 100.0% | 10026 |
| remote_tail_p4 | 0% | 100% | 8.51 | 0% | 100% | 8.38% | 840 |
| book_auth_flat_p4 | 50% | 50% | 7.49 | 100% | 0% | 8.44% | 846 |
| book_auth_hier_s4_p2 | 50% | 50% | 7.49 | 100% | 0% | 10.83% | 1086 |
| hybrid_tail4_authflat4 | 50% | 50% | 6.47 | 100% | 100% | 11.07% | 1110 |
| hybrid_gatedtail4_authflat4 | 50% | 50% | 7.67 | 100% | 0% | 10.18% | 1021 |
| hybrid_gatedtail4_authhier_s4_p2 | 50% | 50% | 7.69 | 100% | 0% | 12.57% | 1260 |

### 20k gated sparse-mask results

| Mode | Accuracy | Decoy pred | Query PPL | Evidence hit | Decoy hit | Kept fraction | Kept tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 50% | 50% | 7.82 | 100% | 100% | 100.0% | 20026 |
| remote_tail_p4 | 0% | 100% | 11.19 | 0% | 100% | 4.16% | 833 |
| book_auth_flat_p4 | 50% | 50% | 9.24 | 100% | 0% | 4.33% | 868 |
| book_auth_hier_s4_p2 | 50% | 50% | 9.19 | 100% | 0% | 5.53% | 1108 |
| hybrid_tail4_authflat4 | 50% | 50% | 8.05 | 100% | 100% | 5.62% | 1125 |
| hybrid_gatedtail4_authflat4 | 50% | 50% | 9.27 | 100% | 0% | 5.22% | 1045 |
| hybrid_gatedtail4_authhier_s4_p2 | 50% | 50% | 9.22 | 100% | 0% | 6.42% | 1285 |

Interpretation:

```text
The status gate works for routing correctness:
  naive hybrid decoy hit = 100%
  gated hybrid decoy hit = 0%

But the status gate does not preserve the naive hybrid PPL gain:
  20k naive hybrid PPL = 8.05
  20k gated authflat PPL = 9.27
  20k book_auth_flat_p4 PPL = 9.24
```

So the low PPL of naive hybrid mainly comes from keeping the near-tail decoy page.  Once that page
is removed, the remaining tail pages do not help much beyond book_auth.  This is an important
negative result: for long-range semantic retrieval, unconditional remote-tail can optimize PPL while
hurting the actual retrieval target.

Answer to the long-range semantic retrieval question:

```text
Use typed page routing:
  - sink/recent stay as the model-specific local mechanism;
  - structural anchors define pages/sections;
  - semantic anchors retrieve topic/entity pages;
  - authority/status anchors decide whether a retrieved or tail page is usable;
  - remote-tail should be gated or layer/head-limited, not globally merged.
```

The current best conservative route for this smoke is still `book_auth_hier_s4_p2` or
`book_auth_flat_p4`: they recall the authoritative evidence, avoid the decoy, keep only about
4-6% of history tokens at 20k, and have much better PPL than sink/recent alone.

Next experiment:

```text
Move from a hand-written authority/status score to a small learned typed-anchor router:
  input:
    query summary, page summary, page status markers, page recency, structural level
  output:
    page type and keep/drop score

Then evaluate separately:
  retrieval heads:
    prioritize semantic + authority pages
  PPL/locality heads:
    allow a small gated tail budget
```

## 20. Structural expansion around authoritative pages

Question:

```text
Remote-tail improves PPL but often keeps the decoy.
Can we improve PPL by expanding around the retrieved authoritative page instead of adding tail pages?
```

New modes:

```text
book_auth_flat_p4_adj1:
  retrieve 4 authority-aware semantic pages, then add adjacent structural pages within radius 1

book_auth_flat_p4_adj2:
  retrieve 4 authority-aware semantic pages, then add adjacent structural pages within radius 2

book_auth_hier_s4_p2_adj1:
  retrieve pages through section -> page hierarchy, then add adjacent structural pages within radius 1
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_10k20k_v3_structural_expand
```

This run uses 4 tasks per length, so it is still a small smoke, but less brittle than the 2-task
checks above.

### 10k structural expansion results

| Mode | Accuracy | Decoy pred | Query PPL | Evidence hit | Decoy hit | Kept fraction | Kept tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 50% | 50% | 6.72 | 100% | 100% | 100.0% | 10026 |
| sink_recent | 25% | 50% | 93.77 | 0% | 0% | 5.75% | 576 |
| remote_tail_p4 | 0% | 100% | 8.95 | 0% | 100% | 8.37% | 839 |
| book_auth_flat_p4 | 50% | 50% | 7.56 | 100% | 0% | 8.49% | 852 |
| book_auth_flat_p4_adj1 | 50% | 50% | 7.50 | 100% | 0% | 12.92% | 1296 |
| book_auth_flat_p4_adj2 | 50% | 50% | 7.51 | 100% | 0% | 16.93% | 1697 |
| book_auth_hier_s4_p2 | 50% | 50% | 7.55 | 100% | 0% | 10.86% | 1089 |
| book_auth_hier_s4_p2_adj1 | 50% | 50% | 7.52 | 100% | 0% | 18.24% | 1829 |
| hybrid_tail4_authflat4 | 50% | 50% | 6.59 | 100% | 100% | 11.12% | 1115 |
| hybrid_gatedtail4_authflat4 | 50% | 50% | 7.79 | 100% | 0% | 10.24% | 1027 |

### 20k structural expansion results

| Mode | Accuracy | Decoy pred | Query PPL | Evidence hit | Decoy hit | Kept fraction | Kept tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 50% | 50% | 7.04 | 100% | 100% | 100.0% | 20026 |
| sink_recent | 50% | 50% | 103.68 | 0% | 0% | 2.88% | 576 |
| remote_tail_p4 | 25% | 75% | 9.59 | 0% | 100% | 4.27% | 855 |
| book_auth_flat_p4 | 75% | 25% | 7.88 | 100% | 0% | 4.31% | 863 |
| book_auth_flat_p4_adj1 | 75% | 25% | 7.87 | 100% | 0% | 6.30% | 1261 |
| book_auth_flat_p4_adj2 | 75% | 25% | 7.85 | 100% | 0% | 7.94% | 1591 |
| book_auth_hier_s4_p2 | 75% | 25% | 7.85 | 100% | 0% | 5.51% | 1103 |
| book_auth_hier_s4_p2_adj1 | 75% | 25% | 7.81 | 100% | 0% | 9.20% | 1843 |
| hybrid_tail4_authflat4 | 50% | 50% | 7.24 | 100% | 100% | 5.70% | 1142 |
| hybrid_gatedtail4_authflat4 | 75% | 25% | 8.00 | 100% | 0% | 5.19% | 1039 |

Interpretation:

```text
Structural expansion is safer than remote-tail:
  evidence hit stays 100%
  decoy hit stays 0%

At 20k it gives a small but consistent PPL improvement:
  book_auth_flat_p4:          PPL 7.88, kept 4.31%
  book_auth_flat_p4_adj2:     PPL 7.85, kept 7.94%
  book_auth_hier_s4_p2:       PPL 7.85, kept 5.51%
  book_auth_hier_s4_p2_adj1:  PPL 7.81, kept 9.20%
```

Compared with naive tail hybrid:

```text
hybrid_tail4_authflat4:
  PPL 7.24, but decoy hit 100% and accuracy falls to 50%

book_auth_hier_s4_p2_adj1:
  PPL 7.81, decoy hit 0%, accuracy 75%
```

So tail still gives the strongest language-modeling locality signal, but it is semantically unsafe
on adversarial long-range retrieval.  Structural expansion gives a smaller PPL gain, but it keeps
the route faithful to the evidence page.

Current best design direction:

```text
Use authority-aware hierarchical retrieval as the base route.
Add a small structural expansion budget around selected evidence pages.
Do not add remote-tail globally; only allow it behind status gates or for PPL-oriented heads.
```

The 20k result supports the original hypothesis better than the 10k result: the longer the context,
the more useful page-level semantic routing becomes.  At 20k, `book_auth_*` routes outperform
`full` and `remote_tail` on this decoy QA proxy, because full/tail keep both evidence and decoy while
typed routing keeps only the authoritative evidence.

Next optimization:

```text
Budget-aware typed routing:
  retrieval budget:
    4-8 authority semantic pages
  structure budget:
    adjacent pages only around high-confidence evidence pages
  locality budget:
    optional gated tail, disabled when status-negative pages are detected

Then measure:
  1. sparse PPL,
  2. answer accuracy,
  3. evidence hit,
  4. decoy hit,
  5. kept-token fraction,
  6. real sparse-kernel speed once the masking path is replaced by an accelerated kernel.
```

## 21. Anchor-focused structural expansion

Question:

```text
Section 20 expanded around every selected semantic page.
Can we save budget by expanding only around selected pages that look like authoritative evidence?
```

New modes:

```text
book_auth_flat_p4_authadj1:
  retrieve 4 authority-aware semantic pages;
  keep all 4 pages;
  expand adjacent pages only around selected pages whose authority/status score is positive.

book_auth_flat_p4_authadj2:
  same, but radius 2 around positive-authority pages.

book_auth_hier_s4_p2_authadj1:
  hierarchical section -> page retrieval;
  expand adjacent pages only around positive-authority pages.
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_10k20k_v4_anchor_focused_expand
```

### 20k budget comparison

| Mode | Accuracy | Query PPL | Evidence hit | Decoy hit | Kept fraction | Kept tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| remote_tail_p4 | 25% | 9.59 | 0% | 100% | 4.27% | 855 |
| book_auth_flat_p4 | 75% | 7.88 | 100% | 0% | 4.31% | 863 |
| book_auth_flat_p4_adj1 | 75% | 7.87 | 100% | 0% | 6.30% | 1261 |
| book_auth_flat_p4_authadj1 | 75% | 7.85 | 100% | 0% | 4.82% | 966 |
| book_auth_flat_p4_authadj2 | 75% | 7.83 | 100% | 0% | 5.41% | 1084 |
| book_auth_hier_s4_p2 | 75% | 7.85 | 100% | 0% | 5.51% | 1103 |
| book_auth_hier_s4_p2_adj1 | 75% | 7.81 | 100% | 0% | 9.20% | 1843 |
| book_auth_hier_s4_p2_authadj1 | 75% | 7.83 | 100% | 0% | 6.02% | 1206 |
| hybrid_tail4_authflat4 | 50% | 7.24 | 100% | 100% | 5.70% | 1142 |

Main result:

```text
Authority-focused expansion is more budget-efficient than expanding all selected pages.

At 20k:
  book_auth_flat_p4_adj1:
    PPL 7.87, kept 6.30%

  book_auth_flat_p4_authadj1:
    PPL 7.85, kept 4.82%

  book_auth_flat_p4_authadj2:
    PPL 7.83, kept 5.41%
```

This is a better shape for the target method:

```text
First retrieve semantic/authority anchors.
Then spend extra structural budget only around those anchors.
Do not expand all semantically similar pages equally.
```

Compared with hierarchical all-page expansion:

```text
book_auth_hier_s4_p2_adj1:
  PPL 7.81, kept 9.20%

book_auth_flat_p4_authadj2:
  PPL 7.83, kept 5.41%
```

The PPL gap is tiny, but the budget gap is large.  This suggests that in the current synthetic
long-range task, most of the useful structural context is local to the actual authoritative page,
not to every retrieved semantic page.

Current best practical route:

```text
book_auth_flat_p4_authadj2:
  kept fraction ~5.4% at 20k
  evidence hit 100%
  decoy hit 0%
  accuracy 75%
  PPL 7.83
```

This is not as low-PPL as naive tail hybrid, but naive tail hybrid keeps the decoy and drops
accuracy.  For long-range semantic retrieval, `authadj` is the better tradeoff.

Design implication:

```text
The router should not have a single "page count" knob.
It should have typed budgets:
  semantic anchor budget
  authority anchor expansion budget
  structural neighborhood radius
  gated locality/tail budget

The page system starts to look like:
  book -> section -> page -> anchor span
with extra tokens spent only around typed anchors that pass the task-specific gate.
```

Next experiment:

```text
Add a budgeted route that caps total remote tokens:
  select semantic/authority pages;
  expand positive-authority anchors by radius 2;
  if over budget, drop lowest scoring non-anchor pages first;
  compare target budgets around 4%, 5%, 6%, 8%.

This should turn the current hand-tuned best mode into a real controllable router.
```

## 22. Budgeted typed router curve

Question:

```text
Can the anchor-focused route be controlled by an explicit compute budget?
```

New mode family:

```text
budget_authflat_p4_authadj2_b{4,5,6,8}
```

Routing rule:

```text
1. retrieve 4 authority-aware semantic pages;
2. identify positive-authority anchors among those pages;
3. add structural neighbors within radius 2 around only those anchors;
4. enforce a total visible-history budget:
     sink + recent + selected remote pages <= b% of prefill length
5. if over budget, keep anchors first, then semantic pages, then structural expansion pages.
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_10k20k_v5_budgeted_router
```

### 20k budget curve

| Mode | Accuracy | Query PPL | Evidence hit | Decoy hit | Kept fraction | Kept tokens | Remote tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sink_recent | 50% | 103.68 | 0% | 0% | 2.88% | 576 | 0 |
| remote_tail_p4 | 25% | 9.59 | 0% | 100% | 4.27% | 855 | 284 |
| budget_authflat_p4_authadj2_b4 | 75% | 7.85 | 100% | 0% | 3.79% | 760 | 184 |
| budget_authflat_p4_authadj2_b5 | 75% | 7.81 | 100% | 0% | 4.90% | 982 | 406 |
| budget_authflat_p4_authadj2_b6 | 75% | 7.83 | 100% | 0% | 5.41% | 1084 | 509 |
| budget_authflat_p4_authadj2_b8 | 75% | 7.83 | 100% | 0% | 5.41% | 1084 | 509 |
| book_auth_flat_p4_authadj2 | 75% | 7.83 | 100% | 0% | 5.41% | 1084 | 509 |
| book_auth_hier_s4_p2_authadj1 | 75% | 7.83 | 100% | 0% | 6.02% | 1206 | 630 |
| hybrid_tail4_authflat4 | 50% | 7.24 | 100% | 100% | 5.70% | 1142 | 573 |
| full | 50% | 7.04 | 100% | 100% | 100.0% | 20026 | 0 |

Interpretation:

```text
The best budgeted point in this smoke is b5:
  kept fraction 4.90%
  remote tokens about 406
  evidence hit 100%
  decoy hit 0%
  accuracy 75%
  PPL 7.81
```

This is slightly better PPL than the unconstrained `book_auth_flat_p4_authadj2` route while using
fewer tokens.  The likely reason is that budget pruning removes weak structural expansion pages
that are not needed for the query.  This is useful: the router should not blindly use all available
structural neighbors.

The 4% route is also valid at 20k:

```text
b4:
  kept fraction 3.79%
  evidence hit 100%
  decoy hit 0%
  PPL 7.85
```

So for this long-range semantic retrieval task, a very small number of typed remote pages is enough
to recover the key information.  This is much better than remote-tail at almost the same compute:

```text
remote_tail_p4:
  kept fraction 4.27%
  evidence hit 0%
  decoy hit 100%
  PPL 9.59
```

### 10k budget behavior

At 10k, sink + recent already costs about 5.75%:

```text
sink_recent kept fraction = 5.75%
```

Therefore the 4% and 5% total-budget modes have no room for remote pages:

```text
b4/b5:
  selected remote pages = 0
  evidence hit = 0%
  PPL = 93.77
```

The feasible points start at b6/b8:

```text
b6:
  kept fraction 6.72%
  evidence hit 100%
  decoy hit 0%
  PPL 7.55

b8:
  kept fraction 7.88%
  evidence hit 100%
  decoy hit 0%
  PPL 7.51
```

This tells us that the minimum useful budget depends on context length when sink/recent are fixed.
For 10k, the sink/recent floor is too large for 4-5% total budgets; for 20k, 4-5% is enough for
both local state and long-range semantic pages.

Current design update:

```text
Use budgeted typed routing, not fixed page counts:
  1. reserve sink/recent;
  2. allocate the remaining budget to authority/semantic anchors;
  3. spend structural expansion only around positive anchors;
  4. prune weak expansion pages if over budget;
  5. avoid remote-tail unless a separate status-gated/locality head needs it.
```

Current best sparse-proxy route:

```text
20k:
  budget_authflat_p4_authadj2_b5

Why:
  compute proxy under 5% kept fraction;
  evidence hit 100%;
  decoy hit 0%;
  accuracy 75%;
  PPL 7.81;
  much better than remote-tail at similar budget.
```

Next step:

```text
Turn the budgeted router into a reusable page-selection module and test it on a larger task suite:
  - more seeds/tasks,
  - multiple key placements,
  - multiple decoy distances,
  - 10k/20k/possibly 32k,
  - compare against fixed block retrieval and remote-tail,
  - then connect this route to a real sparse attention kernel for wall-clock speed.
```

## 23. Layout-robust long-range suite

Question:

```text
Does the budgeted typed router still work when the evidence page and decoy page move?
```

New suite layouts:

```text
e05_d90: evidence around 5%,  decoy around 90%
e20_d80: evidence around 20%, decoy around 80%
e40_d90: evidence around 40%, decoy around 90%
e05_d60: evidence around 5%,  decoy around 60%
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_10k20k_v6_layout_suite
```

This run uses 2 tasks per layout per context length, so 8 tasks for 10k and 8 tasks for 20k.

### 20k aggregate results across layouts

| Mode | Accuracy | Query PPL | Evidence hit | Decoy hit | Kept fraction | Remote tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sink_recent | 25% | 87.00 | 0% | 0% | 2.88% | 0 |
| remote_tail_p4 | 25% | 86.88 | 0% | 0% | 4.02% | 242 |
| book_flat_p4 | 25% | 85.62 | 0% | 0% | 4.08% | 240 |
| book_auth_flat_p4 | 75% | 7.31 | 100% | 0% | 4.17% | 259 |
| budget_authflat_p4_authadj2_b4 | 75% | 7.33 | 100% | 0% | 3.86% | 196 |
| budget_authflat_p4_authadj2_b5 | 62.5% | 7.32 | 100% | 0% | 4.79% | 383 |
| budget_authflat_p4_authadj2_b6 | 62.5% | 7.31 | 100% | 0% | 5.35% | 496 |
| hybrid_tail4_authflat4 | 75% | 7.31 | 100% | 0% | 5.32% | 500 |
| full | 50% | 6.27 | 100% | 100% | 100% | 0 |

Important difference from the earlier near-tail-decoy smoke:

```text
Here the decoy is at 60%, 80%, or 90%, not necessarily inside the last remote-tail pages.
So remote_tail_p4 often recalls neither evidence nor decoy.
It behaves like a weak locality baseline, not a semantic retrieval method.
```

The main robust signal:

```text
book_auth and budget_auth routes:
  evidence hit = 100%
  decoy hit = 0%
  PPL around 7.3

remote_tail and plain book_flat:
  evidence hit = 0%
  decoy hit = 0%
  PPL around 85-87
```

So the authority/status part is doing real work.  Plain lexical page retrieval is not enough in
this synthetic suite, because the route needs to understand that the authoritative page is the one
to use and the decoy/status-negative page is not.

### Per-layout 20k behavior

Key observation:

```text
For every tested 20k layout:
  budget_authflat_p4_authadj2_b4/b5/b6 all hit evidence 100% and decoy 0%.
```

Representative PPL by layout:

| Layout | b4 PPL | b5 PPL | b6 PPL | book_auth_flat_p4 PPL |
| --- | ---: | ---: | ---: | ---: |
| e05_d90 | 6.58 | 6.60 | 6.58 | 6.51 |
| e20_d80 | 7.60 | 7.57 | 7.56 | 7.53 |
| e40_d90 | 7.23 | 7.19 | 7.21 | 7.24 |
| e05_d60 | 8.03 | 8.01 | 8.00 | 8.07 |

The evidence position can move from 5% to 40%, and the typed router still finds it.  This supports
the book/page hypothesis more strongly than the fixed early-evidence smoke.

### 10k behavior

At 10k, fixed sink/recent is still the limiting floor:

```text
sink_recent kept fraction = 5.74%
b4/b5 have no remote budget left
```

Therefore:

```text
b4/b5:
  evidence hit = 0%
  PPL around 81.86

b6:
  evidence hit = 100%
  decoy hit = 0%
  PPL 6.93
  kept fraction 6.79%
```

This reinforces that budgets should not be absolute percentages alone.  The router should compute:

```text
remote_budget = total_budget - sink_budget - recent_budget
```

and the minimum useful total budget must exceed the sink/recent floor.

Current interpretation:

```text
The method is now robust across several long-range layouts:
  - remote-tail is not a semantic retrieval method;
  - plain book_flat lexical retrieval is too weak;
  - authority-aware typed routing consistently recalls the key evidence;
  - budgeted routing gives a controllable compute/PPL curve;
  - small structural expansion is useful but should be budget-pruned.
```

Downstream accuracy caveat:

```text
Qwen3-0.6B single-letter scoring has a strong label prior in these tiny synthetic suites.
Accuracy is useful but noisy.
Evidence hit, decoy hit, and query PPL are more stable diagnostics here.
The next downstream run should add no-context label-prior calibration to the sparse path.
```

Next concrete optimization:

```text
Add calibrated answer scoring to the sparse evaluator:
  calibrated_score(label) = sparse_score(label) - no_context_score(label)

Then rerun the layout suite for:
  book_auth_flat_p4
  budget_authflat_p4_authadj2_b4/b5/b6
  hybrid_tail4_authflat4
  full

This will make downstream accuracy less dominated by the base model's label prior.
```

## 24. Calibrated sparse downstream scoring

Question:

```text
Does no-context label-prior calibration make the sparse downstream accuracy more reliable?
```

Implementation:

```text
For each task:
  prior_score(label) = score(query_only + label)
  calibrated_score(label) = sparse_context_score(label) - prior_score(label)
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_10k20k_v7_calibrated_layout_suite
```

### 20k calibrated aggregate

| Mode | Raw acc | Calibrated acc | Raw decoy pred | Calibrated decoy pred | PPL | Evidence hit | Decoy hit | Kept fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sink_recent | 25% | 25% | 12.5% | 12.5% | 87.00 | 0% | 0% | 2.88% |
| remote_tail_p4 | 25% | 25% | 12.5% | 12.5% | 86.88 | 0% | 0% | 4.02% |
| book_flat_p4 | 25% | 12.5% | 12.5% | 12.5% | 85.62 | 0% | 0% | 4.08% |
| book_auth_flat_p4 | 75% | 75% | 0% | 0% | 7.31 | 100% | 0% | 4.17% |
| book_auth_flat_p4_authadj2 | 62.5% | 75% | 12.5% | 0% | 7.31 | 100% | 0% | 5.35% |
| budget_authflat_p4_authadj2_b4 | 75% | 75% | 0% | 0% | 7.33 | 100% | 0% | 3.86% |
| budget_authflat_p4_authadj2_b5 | 62.5% | 75% | 12.5% | 0% | 7.32 | 100% | 0% | 4.79% |
| budget_authflat_p4_authadj2_b6 | 62.5% | 75% | 12.5% | 0% | 7.31 | 100% | 0% | 5.35% |
| hybrid_tail4_authflat4 | 75% | 75% | 0% | 0% | 7.31 | 100% | 0% | 5.32% |
| full | 50% | 62.5% | 12.5% | 12.5% | 6.27 | 100% | 100% | 100% |

Calibration helps in the intended way:

```text
book_auth_flat_p4_authadj2:
  raw accuracy 62.5% -> calibrated accuracy 75%
  raw decoy pred 12.5% -> calibrated decoy pred 0%

budget b5/b6:
  raw accuracy 62.5% -> calibrated accuracy 75%
  raw decoy pred 12.5% -> calibrated decoy pred 0%
```

The strongest current route remains:

```text
budget_authflat_p4_authadj2_b4 or b5

b4:
  kept fraction 3.86%
  PPL 7.33
  evidence hit 100%
  decoy hit 0%
  calibrated accuracy 75%

b5:
  kept fraction 4.79%
  PPL 7.32
  evidence hit 100%
  decoy hit 0%
  calibrated accuracy 75%
```

The b4 result is important: after layout variation, b4 is almost as good as b5/b6 while keeping
less than 4% of visible history at 20k.

### Per-layout calibrated behavior

At 20k:

```text
e05_d90:
  typed routes calibrated acc = 100%

e20_d80:
  typed routes calibrated acc = 100%

e40_d90:
  typed routes calibrated acc = 100%
  raw acc was sometimes 0-50%, so calibration matters here.

e05_d60:
  typed routes calibrated acc = 0%
  evidence hit = 100%, decoy hit = 0%, calibrated decoy pred = 0%
```

The e05_d60 failure is not a retrieval failure.  The router finds the authoritative evidence and
avoids the decoy, but Qwen3-0.6B still chooses a different wrong label under this tiny two-task
layout.  So the remaining error is downstream answer scoring / model behavior, not page routing.

Current interpretation:

```text
Page routing result:
  strong
  evidence recall is robust across tested layouts
  decoy avoidance is robust

Sparse PPL result:
  strong
  typed routes reduce PPL from ~87 to ~7.3 at 20k with 4-5% kept tokens

Downstream result:
  improved after calibration, but still noisy
  calibrated typed routes reach 75% on the 20k layout suite
  one layout remains hard despite correct retrieval
```

Design implication:

```text
The book/page router is now doing the right retrieval work.
The next bottleneck is answer extraction/scoring, not page selection.
```

Next experiment:

```text
Increase downstream reliability:
  1. more tasks per layout;
  2. balanced labels per layout;
  3. compare single-letter scoring with a more explicit answer format;
  4. keep evidence/decoy/PPL metrics unchanged.

In parallel:
  extract the budgeted typed router into a reusable module, then connect it to a real sparse kernel
  so the current kept-fraction proxy becomes actual wall-clock speed.
```

## 25. Balanced-label calibrated layout suite

Problem with Section 24:

```text
The e05_d60 layout had only two tasks and both happened to target label A.
That made it hard to tell whether the failure was a routing issue or a label-prior/scoring issue.
```

New run:

```text
Use the same four layouts:
  e05_d90, e20_d80, e40_d90, e05_d60

But force each layout to contain four tasks:
  target A, target B, target C, target D

Decoy label is the next label:
  A -> B, B -> C, C -> D, D -> A
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_10k20k_v8_balanced_calibrated_layout_suite
```

### 20k balanced aggregate

| Mode | Raw acc | Calibrated acc | Calibrated decoy pred | Query PPL | Evidence hit | Decoy hit | Kept fraction | Remote tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sink_recent | 25.0% | 18.75% | 25.0% | 87.63 | 0% | 0% | 2.88% | 0 |
| remote_tail_p4 | 25.0% | 18.75% | 25.0% | 87.59 | 0% | 0% | 4.04% | 240 |
| book_flat_p4 | 25.0% | 18.75% | 25.0% | 86.40 | 0% | 0% | 4.06% | 238 |
| book_auth_flat_p4 | 81.25% | 75.0% | 0% | 7.61 | 100% | 0% | 4.18% | 262 |
| budget_authflat_p4_authadj2_b4 | 75.0% | 75.0% | 0% | 7.61 | 100% | 0% | 3.85% | 195 |
| budget_authflat_p4_authadj2_b5 | 81.25% | 75.0% | 0% | 7.60 | 100% | 0% | 4.85% | 396 |
| budget_authflat_p4_authadj2_b6 | 81.25% | 75.0% | 0% | 7.59 | 100% | 0% | 5.37% | 500 |
| hybrid_tail4_authflat4 | 75.0% | 75.0% | 0% | 7.62 | 100% | 0% | 5.32% | 498 |
| full | 25.0% | 43.75% | 43.75% | 6.60 | 100% | 100% | 100% | 0 |

Main result:

```text
Balanced labels confirm the Section 24 interpretation:
  typed routes are stable across layouts;
  e05_d60 was not a routing failure;
  the remaining downstream error is label-specific scoring noise.
```

The best compute/downstream tradeoff remains:

```text
budget_authflat_p4_authadj2_b4

20k:
  kept fraction 3.85%
  remote tokens 195
  PPL 7.61
  evidence hit 100%
  decoy hit 0%
  calibrated accuracy 75%
```

If slightly better PPL is worth more tokens:

```text
budget_authflat_p4_authadj2_b5:
  kept fraction 4.85%
  remote tokens 396
  PPL 7.60
  calibrated accuracy 75%

budget_authflat_p4_authadj2_b6:
  kept fraction 5.37%
  remote tokens 500
  PPL 7.59
  calibrated accuracy 75%
```

The marginal PPL gain from b4 to b6 is very small, so b4 is currently the best sparse-proxy route.

### Per-layout 20k behavior

For all four layouts:

```text
book_auth_flat_p4 and budgeted typed routes:
  evidence hit = 100%
  decoy hit = 0%
  calibrated accuracy = 75%
```

This includes the previously suspicious layout:

```text
e05_d60:
  book_auth_flat_p4 calibrated acc = 75%
  budget b4/b5/b6 calibrated acc = 75%
  evidence hit = 100%
  decoy hit = 0%
```

So the e05_d60 issue in Section 24 was caused by an unlucky target-label sample, not by the page
router.

### Label-specific failure

Balanced labels reveal a clean pattern:

```text
For typed routes at 20k:
  target A: calibrated acc = 0%
  target B: calibrated acc = 100%
  target C: calibrated acc = 100%
  target D: calibrated acc = 100%
```

Example failures for `budget_authflat_p4_authadj2_b5`:

```text
e05_d90 target A: calibrated prediction C
e20_d80 target A: calibrated prediction C
e40_d90 target A: calibrated prediction D
e05_d60 target A: calibrated prediction C
```

All of these still have:

```text
evidence hit = 1
decoy hit = 0
```

Therefore the remaining downstream failure is not memory routing.  It is answer extraction/scoring
for the small Qwen3-0.6B model under this synthetic single-letter format.

Current state of the method:

```text
Retrieval:
  solved in this synthetic layout suite
  typed routes find the evidence in all tested long-range positions

Decoy avoidance:
  solved in this suite
  typed routes avoid status-negative decoys

Sparse PPL:
  strong
  PPL drops from ~87 to ~7.6 with 3.9-5.4% kept fraction at 20k

Downstream:
  calibrated accuracy = 75%
  remaining failure is label A scoring, not page selection

Compute:
  still a proxy
  current implementation masks after full QK, so real wall-clock speed needs a sparse kernel path
```

Next step:

```text
Replace single-letter answer scoring with a more robust downstream probe:
  - use label words instead of bare A/B/C/D, or
  - score full strings like "ANSWER_LABEL=A", or
  - use balanced multi-token labels.

Keep the same routing and PPL metrics.
If the A-specific failure disappears, then downstream performance should match the retrieval result.
```

## 26. Robust answer format: score `ANSWER_LABEL=A`

Problem with Section 25:

```text
Bare single-letter scoring still had a label-specific bias:
  target A calibrated accuracy = 0%
  target B/C/D calibrated accuracy = 100%
```

New scoring format:

```text
Instead of scoring:
  " A"

score:
  " ANSWER_LABEL=A"
```

Everything else is unchanged:

```text
same balanced labels
same layouts
same page routing
same PPL scoring
same calibration method
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_10k20k_v9_answerlabel_balanced_suite
```

### 20k answer-label scoring results

| Mode | Raw acc | Calibrated acc | Calibrated decoy pred | Query PPL | Evidence hit | Decoy hit | Kept fraction | Remote tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sink_recent | 25.0% | 25.0% | 25.0% | 87.63 | 0% | 0% | 2.88% | 0 |
| remote_tail_p4 | 25.0% | 25.0% | 25.0% | 87.59 | 0% | 0% | 4.04% | 240 |
| book_flat_p4 | 25.0% | 25.0% | 25.0% | 86.40 | 0% | 0% | 4.06% | 238 |
| book_auth_flat_p4 | 93.75% | 93.75% | 6.25% | 7.61 | 100% | 0% | 4.18% | 262 |
| budget_authflat_p4_authadj2_b4 | 93.75% | 93.75% | 6.25% | 7.61 | 100% | 0% | 3.85% | 195 |
| budget_authflat_p4_authadj2_b5 | 93.75% | 93.75% | 6.25% | 7.60 | 100% | 0% | 4.85% | 396 |
| budget_authflat_p4_authadj2_b6 | 93.75% | 93.75% | 6.25% | 7.59 | 100% | 0% | 5.37% | 500 |
| hybrid_tail4_authflat4 | 93.75% | 93.75% | 6.25% | 7.62 | 100% | 0% | 5.33% | 498 |
| full | 62.5% | 81.25% | 18.75% | 6.60 | 100% | 100% | 100% | 0 |

This resolves most of the answer-format noise:

```text
bare-letter budget b4:
  calibrated accuracy = 75%

ANSWER_LABEL budget b4:
  calibrated accuracy = 93.75%
```

Label breakdown at 20k:

```text
budget_authflat_p4_authadj2_b4:
  target A: 75%
  target B: 100%
  target C: 100%
  target D: 100%
```

The only remaining typed-route failure is:

```text
layout e40_d90
target A
decoy B
evidence hit = 1
decoy hit = 0
calibrated prediction = B
```

So even after fixing most label-format bias, the residual error is still not a retrieval failure.

### 10k answer-label scoring results

At 10k, b4/b5 still have no room for remote pages because sink/recent alone is about 5.74%.

Useful routes:

```text
book_auth_flat_p4:
  calibrated accuracy = 100%
  PPL = 7.50
  kept fraction = 8.53%

budget_authflat_p4_authadj2_b6:
  calibrated accuracy = 100%
  PPL = 7.47
  kept fraction = 6.75%

hybrid_tail4_authflat4:
  calibrated accuracy = 100%
  PPL = 7.52
  kept fraction = 10.91%
```

Current strongest result:

```text
20k budget_authflat_p4_authadj2_b4:
  kept fraction = 3.85%
  remote tokens = 195
  PPL = 7.61
  evidence hit = 100%
  decoy hit = 0%
  calibrated downstream accuracy = 93.75%
```

Compared with baselines:

```text
remote_tail_p4:
  kept fraction = 4.04%
  PPL = 87.59
  evidence hit = 0%
  calibrated accuracy = 25%

full:
  PPL = 6.60
  evidence hit = 100%
  decoy hit = 100%
  calibrated accuracy = 81.25%
```

Interpretation:

```text
The book/page typed router now satisfies the algorithmic target in this synthetic suite:
  - very small remote budget;
  - strong PPL recovery;
  - robust evidence recall;
  - decoy avoidance;
  - downstream accuracy better than full context because full context includes decoy.
```

Remaining gap:

```text
Compute speed is still a proxy.
The current evaluator masks attention after full QK, so it does not yet measure true wall-clock
sparse speedup.
```

Next engineering step:

```text
1. Extract the budgeted typed page router into a reusable module.
2. Make it output selected token/page ranges for a sparse attention backend.
3. Replace post-QK masking with a real sparse/page attention kernel or block-sparse path.
4. Measure:
     wall-clock prefill/query time,
     memory,
     PPL,
     calibrated downstream accuracy,
     evidence/decoy hit.
```

## 27. Router module extraction

Purpose:

```text
Move the page-selection logic out of the sparse evaluator so it can be reused by a real sparse
attention backend.
```

New module:

```text
src/book_page_router.py
```

Main interface:

```text
selected_pages_for_mode(
    mode,
    task,
    pages,
    page_index,
    sections,
    section_index,
    section_to_pages,
    sink_tokens,
    recent_tokens,
    query_window_tokens,
) -> set[int]

pages_to_tokens(pages, selected_pages) -> set[int]

pages_to_ranges(pages, selected_pages) -> list[tuple[int, int]]
```

The evaluator now imports:

```text
from book_page_router import pages_to_ranges, pages_to_tokens, selected_pages_for_mode
```

and writes two new row fields:

```text
selected_page_ids
selected_token_ranges
```

These fields are the handoff contract for a future page/block sparse kernel.

Supported route families in the module:

```text
remote_tail_pK
book_flat_pK
book_hier_sS_pP
book_auth_flat_pK
book_auth_flat_pK_adjR
book_auth_flat_pK_authadjR
book_auth_hier_sS_pP
book_auth_hier_sS_pP_adjR
book_auth_hier_sS_pP_authadjR
budget_authflat_pK_authadjR_bB
hybrid_tail4_authflatK
hybrid_gatedtail4_authflatK
hybrid_gatedtail4_authhier_sS_pP
```

Validation:

```text
Server compile:
  python -m py_compile src/book_page_router.py src/run_longrange_book_index_sparse_eval.py

Server smoke:
  construct 4k layout tasks for e05_d90/e20_d80/e40_d90/e05_d60;
  build paragraph/section indexes;
  run selected_pages_for_mode for:
    remote_tail_p4
    book_flat_p4
    book_auth_flat_p4
    budget_authflat_p4_authadj2_b4
  print selected pages, token ranges, token counts, evidence hit, decoy hit.
```

Smoke result:

```text
book_auth_flat_p4:
  evidence hit = true for all tested layouts
  decoy hit = false for all tested layouts

remote_tail_p4 and book_flat_p4:
  evidence hit = false in the tested layouts

budget_authflat_p4_authadj2_b4:
  selects no remote pages at 4k because sink + recent already exceed the 4% total budget.
  This is expected and matches the budget-floor behavior seen at 10k.
```

Current architecture:

```text
Task/index construction:
  run_longrange_book_index_sparse_eval.py

Typed page routing:
  book_page_router.py

Sparse-mask evaluator:
  run_longrange_book_index_sparse_eval.py

Future sparse kernel:
  should consume selected_token_ranges or selected_page_ids from book_page_router.py
```

Next engineering target:

```text
Add a real range-based attention path:
  input:
    sink range
    recent range
    selected remote token ranges
  output:
    attention only over those key/value ranges

Then compare:
  post-QK mask kept-fraction proxy
  real wall-clock query time
  memory usage
  PPL
  calibrated downstream accuracy
```

## 28. First real-compute smoke: PyTorch gather attention

Purpose:

```text
Move beyond post-QK masking by adding a gather implementation that only multiplies Q against
selected key/value positions during sparse query/answer scoring.
```

Implementation:

```text
run_longrange_book_index_sparse_eval.py now supports:
  --sparse_attention_impl mask
  --sparse_attention_impl gather
```

The gather path:

```text
1. builds the same keep mask as the old path;
2. for query_count == 1, converts keep mask to key indices;
3. gathers selected K/V with index_select;
4. computes QK and attention output only on gathered K/V.
```

This is not the final kernel.  It is a PyTorch-level prototype to test whether shrinking the
matmul dimension immediately gives wall-clock benefit.

Reproduction script:

```text
scripts/run_longrange_book_index_sparse_gather_smoke_server.sh
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_20k_v10_mask_smoke
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_20k_v10_gather_smoke
```

Smoke configuration:

```text
context = 20k
layout = e05_d90
tasks = 1
modes:
  sink_recent
  remote_tail_p4
  book_auth_flat_p4
  budget_authflat_p4_authadj2_b4
answer_score_format = ANSWER_LABEL
```

### Mask vs gather results

| Mode | Impl | Eval seconds | Kept fraction | PPL | Calibrated acc | Evidence hit | Decoy hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sink_recent | mask | 3.81 | 2.88% | 95.04 | 0% | 0% | 0% |
| sink_recent | gather | 3.96 | 2.88% | 95.06 | 0% | 0% | 0% |
| remote_tail_p4 | mask | 3.86 | 4.04% | 95.78 | 0% | 0% | 0% |
| remote_tail_p4 | gather | 3.90 | 4.04% | 95.85 | 0% | 0% | 0% |
| book_auth_flat_p4 | mask | 3.84 | 4.16% | 8.35 | 100% | 100% | 0% |
| book_auth_flat_p4 | gather | 3.90 | 4.16% | 8.35 | 100% | 100% | 0% |
| budget_authflat_p4_authadj2_b4 | mask | 3.84 | 3.86% | 8.41 | 100% | 100% | 0% |
| budget_authflat_p4_authadj2_b4 | gather | 3.90 | 3.86% | 8.41 | 100% | 100% | 0% |

Interpretation:

```text
The gather path preserves behavior:
  PPL and calibrated accuracy match the mask path up to small numeric noise.

But it does not improve wall-clock time:
  gather is slightly slower than mask in this smoke.
```

Why:

```text
The prototype uses PyTorch index_select plus many small query_count=1 matmuls.
The overhead of gathering and launching small operations dominates the saved QK work.
This is especially true during decode-style scoring where each step has only one query token.
```

Conclusion:

```text
The kept-fraction proxy is algorithmically meaningful, but naive PyTorch gather is not enough for
actual speedup.

A real implementation needs a fused range/block sparse attention kernel that consumes:
  sink range
  recent range
  selected remote token ranges
without materializing full QK or doing per-step Python-level gather overhead.
```

Next engineering step:

```text
Implement or integrate a block/range sparse attention backend.
The current router is ready for that path because it now emits selected_token_ranges.
```

## 29. Sparse backend smoke: Triton small-kernel vs SDPA gather

Purpose:

```text
Test whether the typed page router can move from a kept-fraction proxy toward real wall-clock
speed by replacing post-QK masking with a sparse attention backend.
```

Implemented backend options:

```text
--sparse_attention_impl mask
  Dense QK over the full history, then mask non-selected tokens.

--sparse_attention_impl gather
  PyTorch index_select selected K/V, then manual matmul/softmax/AV.

--sparse_attention_impl sdpa_gather
  PyTorch index_select selected K/V, then torch scaled_dot_product_attention.

--sparse_attention_impl triton
  A first Triton fused decode kernel over selected candidate token ids.
```

Outputs:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_20k_v11_mask_smoke
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_20k_v11_gather_smoke
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_20k_v11_triton_smoke
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_20k_v12_sdpa_gather_smoke
```

Smoke configuration:

```text
context = 20k
layout = e05_d90
tasks = 1
modes = sink_recent, remote_tail_p4, book_auth_flat_p4, budget_authflat_p4_authadj2_b4
answer_score_format = ANSWER_LABEL
```

### Backend timing

| Mode | Impl | Eval seconds | Kept fraction | PPL | Calibrated acc | Evidence hit | Decoy hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sink_recent | mask | 3.77 | 2.88% | 95.04 | 0% | 0% | 0% |
| sink_recent | gather | 3.82 | 2.88% | 95.06 | 0% | 0% | 0% |
| sink_recent | sdpa_gather | 3.70 | 2.88% | 95.02 | 0% | 0% | 0% |
| sink_recent | triton | 26.85 | 2.88% | 95.05 | 0% | 0% | 0% |
| remote_tail_p4 | mask | 3.89 | 4.04% | 95.78 | 0% | 0% | 0% |
| remote_tail_p4 | gather | 3.90 | 4.04% | 95.85 | 0% | 0% | 0% |
| remote_tail_p4 | sdpa_gather | 3.80 | 4.04% | 95.88 | 0% | 0% | 0% |
| remote_tail_p4 | triton | 25.92 | 4.04% | 95.97 | 0% | 0% | 0% |
| book_auth_flat_p4 | mask | 3.87 | 4.16% | 8.35 | 100% | 100% | 0% |
| book_auth_flat_p4 | gather | 3.89 | 4.16% | 8.35 | 100% | 100% | 0% |
| book_auth_flat_p4 | sdpa_gather | 3.81 | 4.16% | 8.36 | 100% | 100% | 0% |
| book_auth_flat_p4 | triton | 26.81 | 4.16% | 8.34 | 100% | 100% | 0% |
| budget_authflat_p4_authadj2_b4 | mask | 3.87 | 3.86% | 8.41 | 100% | 100% | 0% |
| budget_authflat_p4_authadj2_b4 | gather | 3.89 | 3.86% | 8.41 | 100% | 100% | 0% |
| budget_authflat_p4_authadj2_b4 | sdpa_gather | 3.80 | 3.86% | 8.41 | 100% | 100% | 0% |
| budget_authflat_p4_authadj2_b4 | triton | 26.83 | 3.86% | 8.42 | 100% | 100% | 0% |

Main result:

```text
The typed page routing result is stable across all sparse backends:
  book_auth and budget_auth recover the evidence page and avoid the decoy.
  remote_tail and sink_recent miss the long-range evidence.

But current sparse compute backends do not yet deliver real speedup:
  sdpa_gather is only slightly faster than manual gather/mask in this smoke.
  the naive Triton q=1 decode kernel is much slower.
```

Why the first Triton prototype is slow:

```text
It launches one small custom kernel per layer per decode token.
For this scoring setup, q=1 and selected K is only about 600-850 tokens.
The saved QK arithmetic is smaller than the overhead from many tiny launches and candidate-id handling.
The current patch also repeats GQA K/V to full attention heads before the kernel, so it does not yet save
that bandwidth.
```

Engineering conclusion:

```text
The algorithmic direction is still supported:
  structural/semantic/authority page routing gives much better long-range retrieval than remote-tail.

The speed path should not be a per-token Python/Triton gather prototype.
The next viable implementation should be one of:
  1. a fused range/block decode kernel that consumes selected_token_ranges directly;
  2. a paged-attention backend with a page table built from selected pages;
  3. a batched multi-token scoring kernel that amortizes launch overhead across answer options/layers.
```

Practical recommendation:

```text
For quality experiments, continue using mask or sdpa_gather.
For speed claims, do not use the current Triton prototype as evidence.
The router output format is now ready for a real backend because each row records:
  selected_page_ids
  selected_token_ranges
```

## 30. GQA-aware sparse gather optimization

Issue found:

```text
The first gather/sdpa_gather implementation repeated Qwen3 GQA K/V to full attention heads before
selecting sparse tokens.

That means the sparse path still copied a full 20k-history K/V tensor from kv_heads to attention_heads,
then selected only about 600-850 tokens.
```

Fix:

```text
Move sparse gather before GQA repeat.

For gather/sdpa_gather:
  1. index_select selected token ids on original KV heads;
  2. expand only the selected K/V from kv_heads to attention_heads;
  3. run manual attention or SDPA on the selected K/V.

For triton:
  pass group_size and map attention head -> kv head inside the kernel.
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_20k_v13_mask_gqa_smoke
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_20k_v13_gather_gqa_smoke
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_20k_v13_sdpa_gather_gqa_smoke
```

### GQA-aware timing

| Mode | Impl | Eval seconds | Kept fraction | PPL | Calibrated acc | Evidence hit | Decoy hit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| sink_recent | mask | 3.79 | 2.88% | 95.04 | 0% | 0% | 0% |
| sink_recent | gather | 3.23 | 2.88% | 95.06 | 0% | 0% | 0% |
| sink_recent | sdpa_gather | 3.07 | 2.88% | 95.02 | 0% | 0% | 0% |
| remote_tail_p4 | mask | 3.91 | 4.04% | 95.78 | 0% | 0% | 0% |
| remote_tail_p4 | gather | 3.30 | 4.04% | 95.85 | 0% | 0% | 0% |
| remote_tail_p4 | sdpa_gather | 3.16 | 4.04% | 95.88 | 0% | 0% | 0% |
| book_auth_flat_p4 | mask | 3.89 | 4.16% | 8.35 | 100% | 100% | 0% |
| book_auth_flat_p4 | gather | 3.30 | 4.16% | 8.35 | 100% | 100% | 0% |
| book_auth_flat_p4 | sdpa_gather | 3.16 | 4.16% | 8.36 | 100% | 100% | 0% |
| budget_authflat_p4_authadj2_b4 | mask | 3.89 | 3.86% | 8.41 | 100% | 100% | 0% |
| budget_authflat_p4_authadj2_b4 | gather | 3.30 | 3.86% | 8.41 | 100% | 100% | 0% |
| budget_authflat_p4_authadj2_b4 | sdpa_gather | 3.16 | 3.86% | 8.41 | 100% | 100% | 0% |

Result:

```text
GQA-aware sparse gather gives a real, though still modest, wall-clock improvement:
  gather:       about 15% faster than mask
  sdpa_gather:  about 19% faster than mask

Quality is preserved:
  book_auth and budget_auth still hit the evidence page, avoid the decoy, and keep low PPL.
```

Interpretation:

```text
The first useful speed win did not come from a custom Triton kernel.
It came from removing a hidden full-history GQA repeat from the sparse path.

This suggests the next speed work should focus on memory movement and launch amortization:
  avoid full-history KV transforms;
  consume page ranges directly;
  batch several query/answer scoring rows where possible;
  only then move to a fused range/block CUDA or Triton kernel.
```

Current best practical backend:

```text
--sparse_attention_impl sdpa_gather

It is the best available backend for continuing algorithm experiments because it preserves the
typed-router quality result and gives the first measured wall-clock improvement over mask.
```

## 31. Follow-up full suite launched with sdpa_gather

Command script:

```text
scripts/run_longrange_book_index_sparse_server.sh
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_10k20k_v14_sdpa_gqa_answerlabel_balanced_suite
```

Configuration:

```text
context = 10k,20k
layouts = e05_d90,e20_d80,e40_d90,e05_d60
tasks_per_length = 4
answer_score_format = ANSWER_LABEL
sparse_attention_impl = sdpa_gather
modes =
  full
  sink_recent
  remote_tail_p4
  book_flat_p4
  book_auth_flat_p4
  budget_authflat_p4_authadj2_b4
  budget_authflat_p4_authadj2_b5
  budget_authflat_p4_authadj2_b6
  hybrid_tail4_authflat4
```

Purpose:

```text
Re-run the balanced 10k/20k layout suite with the current fastest reliable backend.
This will show whether the v13 20k smoke speedup carries over to the full multi-layout suite.
```

Launch status:

```text
Started on server as PID 554040.
Initial log confirmed progress through context=10000, layout=e05_d90, tasks 1-4.
```

Completion:

```text
The v14 suite completed in 1076.95 seconds.
```

### v14 vs v9: full-suite timing and quality

Comparison is between:

```text
v9:
  sparse_attention_impl = mask

v14:
  sparse_attention_impl = sdpa_gather
  sparse gather is GQA-aware, so K/V are expanded only after selected-token gather.
```

Important caveat:

```text
The v14 rows for mode=full used sdpa_gather over all K/V before the full-mode bypass bug was fixed.
Therefore v14 full timing should not be used as a dense baseline.
Sparse modes are valid because they are the intended sdpa_gather path.
```

Typed sparse modes:

```text
10k typed modes:
  v9 mean eval seconds  = 3.093
  v14 mean eval seconds = 2.916
  speedup              = 5.7%

20k typed modes:
  v9 mean eval seconds  = 3.978
  v14 mean eval seconds = 3.244
  speedup              = 18.4%
```

20k key rows:

| Mode | v9 time | v14 time | Speedup | v14 PPL | v14 cal acc | Evidence hit | Decoy hit | Kept |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| book_auth_flat_p4 | 3.976 | 3.236 | 18.6% | 7.608 | 93.75% | 100% | 0% | 4.18% |
| budget_authflat_p4_authadj2_b4 | 3.973 | 3.239 | 18.5% | 7.602 | 93.75% | 100% | 0% | 3.85% |
| budget_authflat_p4_authadj2_b5 | 3.980 | 3.253 | 18.3% | 7.596 | 93.75% | 100% | 0% | 4.85% |
| budget_authflat_p4_authadj2_b6 | 3.982 | 3.247 | 18.5% | 7.591 | 93.75% | 100% | 0% | 5.37% |
| hybrid_tail4_authflat4 | 3.979 | 3.246 | 18.4% | 7.618 | 93.75% | 100% | 0% | 5.33% |

Interpretation:

```text
The sdpa_gather + GQA-aware implementation gives a real full-suite wall-clock gain at 20k.
The gain is smaller at 10k because the removed full-history GQA expansion is less expensive there.

Quality is preserved:
  PPL changes only by small numeric noise.
  evidence_hit remains 100% for typed routes.
  decoy_hit remains 0%.
  calibrated accuracy remains 93.75% at 20k and 100% at 10k for sufficiently budgeted typed routes.
```

Remaining 20k failure:

```text
The single 20k typed-route error is layout=e40_d90, target=A.
For all typed modes:
  evidence_hit = 1
  decoy_hit = 0

So this is not a routing miss. It is an answer-scoring/model bias case where the model still ranks B
above A even when the selected pages include the authoritative evidence and exclude the decoy.
```

## 32. Full-mode bypass fix

Issue:

```text
After adding sdpa_gather, mode=full accidentally entered the gather path too.
That made full gather all K/V positions instead of using normal dense attention, so v14 full timing
was slower and not a valid dense baseline.
```

Fix:

```text
The gather/sdpa_gather branch now requires:
  _ACTIVE_SPARSE_CONTEXT.mode != "full"

Full mode falls through to the original dense attention path.
```

Validation smoke:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_20k_v15_full_bypass_smoke
```

Result:

| Mode | Impl | Eval seconds | PPL | Evidence hit | Decoy hit |
| --- | --- | ---: | ---: | ---: | ---: |
| full | sdpa_gather flag, dense bypass | 3.61 | 6.87 | 100% | 100% |
| budget_authflat_p4_authadj2_b4 | sdpa_gather | 3.29 | 8.41 | 100% | 0% |

Conclusion:

```text
The full-mode timing path is fixed for future runs.
The v14 sparse-mode timing and quality conclusions remain valid.
```

## 33. Adaptive recent budget for low-budget semantic retrieval

Problem:

```text
At 10k, strict total budgets b4/b5 fail because sink64 + recent512 already uses 576 tokens.
That is larger than:
  b4 total budget = 400 tokens
  b5 total budget = 500 tokens

The router therefore has zero remote-token budget and cannot select the evidence page.
```

New mode suffix:

```text
budget_authflat_p4_authadj2_b4_r128
budget_authflat_p4_authadj2_b5_r128
budget_authflat_p4_authadj2_b5_r256
budget_authflat_p4_authadj2_b6_r256
```

Meaning:

```text
Keep the same total budget percent, but use an effective recent window of rN for that mode.
This lets low-budget semantic-retrieval routes trade some recent-window capacity for remote evidence pages.
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_10k20k_v16_adaptive_recent_budget_suite
```

### 10k adaptive-recent result

| Mode | PPL | Cal acc | Evidence hit | Decoy hit | Kept fraction | Mean kept tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| budget_authflat_p4_authadj2_b4 | 87.55 | 18.75% | 0% | 0% | 5.74% | 576 |
| budget_authflat_p4_authadj2_b4_r128 | 7.69 | 100% | 100% | 0% | 3.54% | 356 |
| budget_authflat_p4_authadj2_b5 | 87.55 | 18.75% | 0% | 0% | 5.74% | 576 |
| budget_authflat_p4_authadj2_b5_r128 | 7.68 | 100% | 100% | 0% | 4.72% | 474 |
| budget_authflat_p4_authadj2_b5_r256 | 7.55 | 100% | 100% | 0% | 4.78% | 480 |
| budget_authflat_p4_authadj2_b6 | 7.47 | 100% | 100% | 0% | 6.75% | 678 |

### 20k adaptive-recent result

| Mode | PPL | Cal acc | Evidence hit | Decoy hit | Kept fraction | Mean kept tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| budget_authflat_p4_authadj2_b4 | 7.60 | 93.75% | 100% | 0% | 3.85% | 771 |
| budget_authflat_p4_authadj2_b4_r128 | 7.60 | 93.75% | 100% | 0% | 3.42% | 686 |
| budget_authflat_p4_authadj2_b5 | 7.60 | 93.75% | 100% | 0% | 4.85% | 972 |
| budget_authflat_p4_authadj2_b5_r256 | 7.60 | 93.75% | 100% | 0% | 4.06% | 814 |
| budget_authflat_p4_authadj2_b6 | 7.59 | 93.75% | 100% | 0% | 5.37% | 1076 |

Interpretation:

```text
For long-range semantic retrieval, sink + recent should not be treated as an untouchable fixed floor
under very small budgets.

At 10k, b4/b5 with recent512 spends the entire budget on sink/recent and fails retrieval.
Reducing recent to 128 or 256 makes room for the remote authoritative page and fully recovers accuracy.

At 20k, original b4 already has enough room for the evidence page, so adaptive recent mostly reduces
kept tokens with similar quality.
```

Current best budget choices:

```text
10k:
  budget_authflat_p4_authadj2_b4_r128
    kept 3.54%, PPL 7.69, calibrated accuracy 100%

  budget_authflat_p4_authadj2_b5_r256
    kept 4.78%, PPL 7.55, calibrated accuracy 100%

20k:
  budget_authflat_p4_authadj2_b4
    kept 3.85%, PPL 7.60, calibrated accuracy 93.75%

  budget_authflat_p4_authadj2_b4_r128
    kept 3.42%, PPL 7.60, calibrated accuracy 93.75%
```

Design implication:

```text
Typed-anchor page routing should use an adaptive retention controller:
  preserve sink;
  allocate a minimum budget to semantic/authority pages when the query is long-range retrieval-like;
  shrink recent if necessary;
  expand recent again for local continuation-like queries.

This moves the method from a fixed sparse-attention rule toward query-type-aware page routing.
```

## 34. Auto recent controller

Motivation:

```text
Manual r128/r256 modes proved the tradeoff, but they require choosing a recent window by hand.
The next step is an automatic controller that keeps default recent when budget is sufficient, and
shrinks recent only when remote semantic evidence would otherwise be starved.
```

New mode suffix:

```text
_rauto
_rauto256
```

Rule:

```text
For a mode like:
  budget_authflat_p4_authadj2_b4_rauto

Parse b4 as the total budget percent.
Let:
  total_budget = context_tokens * 4%
  default_recent = 512
  sink = 64
  min_remote = 192 for _rauto, or the explicit value for _rauto256

If:
  total_budget - sink - default_recent >= min_remote
then:
  keep default_recent
else:
  recent = total_budget - sink - min_remote
```

This means:

```text
10k b4:
  total budget = 400
  default recent would leave negative remote budget
  _rauto shrinks recent to about 144 and recovers remote evidence pages

20k b4:
  total budget = 800
  default recent leaves about 224 remote tokens
  _rauto keeps the original recent512 behavior
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_10k20k_v17_auto_recent_budget_suite
```

### Auto controller result

| Context | Mode | PPL | Cal acc | Evidence hit | Decoy hit | Kept fraction | Mean kept tokens |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | b4 | 87.55 | 18.75% | 0% | 0% | 5.74% | 576 |
| 10k | b4_r128 | 7.69 | 100% | 100% | 0% | 3.54% | 356 |
| 10k | b4_rauto | 7.68 | 100% | 100% | 0% | 3.70% | 372 |
| 10k | b4_rauto256 | 7.79 | 100% | 100% | 0% | 3.66% | 367 |
| 10k | b5 | 87.55 | 18.75% | 0% | 0% | 5.74% | 576 |
| 10k | b5_r256 | 7.55 | 100% | 100% | 0% | 4.78% | 480 |
| 10k | b5_rauto | 7.56 | 100% | 100% | 0% | 4.70% | 472 |
| 10k | b6 | 7.47 | 100% | 100% | 0% | 6.75% | 678 |
| 10k | b6_rauto | 7.49 | 100% | 100% | 0% | 5.70% | 572 |
| 20k | b4 | 7.60 | 93.75% | 100% | 0% | 3.85% | 771 |
| 20k | b4_rauto | 7.60 | 93.75% | 100% | 0% | 3.85% | 771 |
| 20k | b5 | 7.60 | 93.75% | 100% | 0% | 4.85% | 972 |
| 20k | b5_rauto | 7.60 | 93.75% | 100% | 0% | 4.85% | 972 |
| 20k | b6 | 7.59 | 93.75% | 100% | 0% | 5.37% | 1076 |
| 20k | b6_rauto | 7.59 | 93.75% | 100% | 0% | 5.37% | 1076 |

Conclusion:

```text
_rauto is a better default than a fixed recent window for long-range semantic retrieval:

At 10k:
  it fixes the b4/b5 remote-budget failure and matches the hand-tuned r128/r256 behavior.

At 20k:
  it leaves the already-good b4/b5/b6 behavior unchanged.
```

Recommended current route:

```text
budget_authflat_p4_authadj2_b4_rauto

Reason:
  10k: kept 3.70%, PPL 7.68, calibrated accuracy 100%
  20k: kept 3.85%, PPL 7.60, calibrated accuracy 93.75%
```

Updated server script:

```text
scripts/run_longrange_book_index_sparse_server.sh

Output:
  /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_10k20k_v18_auto_recent_recommended_suite
```

## 35. Recommended v18 suite

Purpose:

```text
Run the current recommended configuration after the full-mode bypass fix:
  sdpa_gather backend
  b4/b5/b6 auto-recent typed routes
  full/sink/recent/remote-tail baselines
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_10k20k_v18_auto_recent_recommended_suite
```

Runtime:

```text
1146.87 seconds
```

### Recommended-route comparison

| Context | Mode | PPL | Cal acc | Evidence hit | Decoy hit | Kept fraction | Eval sec |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | full | 6.69 | 75.00% | 100% | 100% | 100.00% | 2.84 |
| 10k | sink_recent | 87.55 | 18.75% | 0% | 0% | 5.74% | 2.79 |
| 10k | remote_tail_p4 | 86.95 | 25.00% | 0% | 0% | 8.12% | 2.97 |
| 10k | book_auth_flat_p4 | 7.49 | 100% | 100% | 0% | 8.53% | 2.98 |
| 10k | budget_b4 | 87.55 | 18.75% | 0% | 0% | 5.74% | 2.79 |
| 10k | budget_b4_rauto | 7.68 | 100% | 100% | 0% | 3.70% | 2.96 |
| 10k | budget_b5_rauto | 7.56 | 100% | 100% | 0% | 4.70% | 2.96 |
| 10k | budget_b6_rauto | 7.49 | 100% | 100% | 0% | 5.70% | 2.96 |
| 20k | full | 6.60 | 81.25% | 100% | 100% | 100.00% | 3.68 |
| 20k | sink_recent | 87.64 | 25.00% | 0% | 0% | 2.88% | 3.06 |
| 20k | remote_tail_p4 | 87.62 | 25.00% | 0% | 0% | 4.04% | 3.23 |
| 20k | book_auth_flat_p4 | 7.61 | 93.75% | 100% | 0% | 4.18% | 3.22 |
| 20k | budget_b4 | 7.60 | 93.75% | 100% | 0% | 3.85% | 3.23 |
| 20k | budget_b4_rauto | 7.60 | 93.75% | 100% | 0% | 3.85% | 3.23 |
| 20k | budget_b5_rauto | 7.60 | 93.75% | 100% | 0% | 4.85% | 3.23 |
| 20k | budget_b6_rauto | 7.59 | 93.75% | 100% | 0% | 5.37% | 3.23 |

Main conclusion:

```text
budget_authflat_p4_authadj2_b4_rauto is the best current default.

It fixes the 10k low-budget failure:
  b4 fixed-recent: evidence hit 0%, PPL 87.55, acc 18.75%
  b4_rauto:        evidence hit 100%, PPL 7.68, acc 100%

It preserves the 20k result:
  b4 and b4_rauto both keep 3.85%, PPL 7.60, acc 93.75%.
```

Why full is not best for downstream:

```text
Full context has lower PPL, but includes the decoy page.
In this synthetic long-range semantic retrieval task, full context is worse than typed routing on
calibrated downstream accuracy because the model can be distracted by the later contradictory page.
```

Current method summary:

```text
1. Use structural anchors to create natural pages.
2. Use semantic + authority anchors to route to evidence pages.
3. Use adaptive recent control:
   - preserve sink;
   - reserve enough remote-page budget for retrieval-like queries;
   - shrink recent only when the default recent window would starve remote evidence.
4. Use GQA-aware sdpa_gather for the current fastest reliable sparse backend.
```

Next research step:

```text
The remaining 20k error is not an evidence-recall error:
  evidence hit = 100%
  decoy hit = 0%

It is an answer-scoring/model bias case. The next quality step should test stronger answer extraction:
  score the full evidence sentence rather than only ANSWER_LABEL=X;
  or add a tiny answer-normalization/verifier head over the selected page text.

The next speed step should move from token-id gather to range/page-table attention, using selected_token_ranges
instead of materialized selected token ids.
```

## 36. Text verifier for selected authoritative pages

Motivation:

```text
The remaining 20k error has:
  evidence_hit = 100%
  decoy_hit = 0%

So the router selected the right page and excluded the wrong page, but option scoring still ranked the
wrong label higher after calibration.
```

Implementation:

```text
Add a synthetic text verifier to run_longrange_book_index_sparse_eval.py.

For each mode:
  if mode == full:
    scan all pages
  elif selected_pages is non-empty:
    scan selected pages
  else:
    return no verifier prediction

The verifier searches selected page text for:
  AUTHORITATIVE EVIDENCE PAGE
  ANSWER_LABEL=[A-D]

This is not meant to be the final learned verifier. It is a proxy for a small extraction head over
routed pages.
```

New row fields:

```text
text_verifier_pred_label
text_verifier_present
text_verifier_correct
text_verifier_decoy_pred
```

New summary fields:

```text
text_verifier_coverage
text_verifier_accuracy
text_verifier_decoy_pred_rate
```

Exact failure reproduction:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_20k_v19c_text_verifier_reproduce_failure

context = 20k
layouts = e05_d90,e20_d80,e40_d90
tasks_per_length = 4
modes =
  book_auth_flat_p4
  budget_authflat_p4_authadj2_b4_rauto
```

### v19c result

| Mode | LM acc | Cal acc | Verifier coverage | Verifier acc | Evidence hit | Decoy hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| book_auth_flat_p4 | 100% | 91.67% | 100% | 100% | 100% | 0% |
| budget_authflat_p4_authadj2_b4_rauto | 91.67% | 91.67% | 100% | 100% | 100% | 0% |

The reproduced failure row:

```text
layout = e40_d90
task_id = 2000002000
target = A
decoy = B

budget_authflat_p4_authadj2_b4_rauto:
  selected pages = 133 134 135
  evidence_hit = 1
  decoy_hit = 0
  calibrated_pred = B
  text_verifier_pred = A
```

Interpretation:

```text
The typed page router is no longer the limiting factor for this failure.
The answer is present in the selected authoritative page, and a simple selected-page text verifier
extracts it correctly.

This supports a two-stage design:
  1. typed-anchor page routing retrieves a small set of relevant pages;
  2. an answer normalizer/verifier extracts or validates the final answer from those pages.
```

Design update:

```text
For long-range semantic retrieval tasks, downstream quality should be reported in two forms:
  LM option-scoring accuracy;
  selected-page verifier accuracy.

If verifier accuracy is high while LM option scoring fails, the bottleneck is answer extraction,
not page routing.
```

## 37. Sentence answer scoring after page routing

Purpose:

```text
Test whether a stronger LM scoring prompt can fix the remaining answer-extraction error without
changing page routing.
```

Existing formats:

```text
answer_label:
  " ANSWER_LABEL=A"

sentence:
  " The authoritative answer label is A."
```

Focused reproduction:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_20k_v20_sentence_scoring_reproduce_failure

context = 20k
layouts = e05_d90,e20_d80,e40_d90
tasks_per_length = 4
modes =
  book_auth_flat_p4
  budget_authflat_p4_authadj2_b4_rauto
answer_score_format = sentence
```

Result:

```text
answer_label on the same 12-task reproduction:
  calibrated accuracy = 91.67%

sentence:
  book_auth_flat_p4 calibrated accuracy = 100%
  budget_authflat_p4_authadj2_b4_rauto calibrated accuracy = 100%
  text verifier accuracy = 100%
```

Compact recommended suite:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_10k20k_v21_sentence_recommended_compact

context = 10k,20k
layouts = e05_d90,e20_d80,e40_d90,e05_d60
tasks_per_length = 4
modes =
  full
  remote_tail_p4
  book_auth_flat_p4
  budget_authflat_p4_authadj2_b4_rauto
answer_score_format = sentence
```

### answer_label vs sentence

| Context | Mode | answer_label acc | sentence acc | answer_label sec | sentence sec | PPL |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10k | full | 75.00% | 25.00% | 2.84 | 3.39 | 6.69 |
| 10k | remote_tail_p4 | 25.00% | 12.50% | 2.97 | 3.55 | 86.95 |
| 10k | book_auth_flat_p4 | 100% | 100% | 2.98 | 3.56 | 7.49 |
| 10k | budget_b4_rauto | 100% | 100% | 2.96 | 3.54 | 7.68 |
| 20k | full | 81.25% | 37.50% | 3.68 | 4.37 | 6.60 |
| 20k | remote_tail_p4 | 25.00% | 12.50% | 3.23 | 3.85 | 87.62 |
| 20k | book_auth_flat_p4 | 93.75% | 93.75% | 3.22 | 3.84 | 7.61 |
| 20k | budget_b4_rauto | 93.75% | 100% | 3.23 | 3.85 | 7.60 |

Interpretation:

```text
Sentence scoring helps after typed page routing has removed the decoy.
For budget_b4_rauto at 20k, it fixes the remaining answer-scoring error:
  93.75% -> 100%

But sentence scoring hurts full-context and remote-tail baselines:
  full still contains the contradictory decoy page;
  remote-tail still misses the evidence page.

Therefore the winning combination is not "better answer scoring alone".
It is:
  typed page routing first,
  then stronger answer extraction/scoring on the selected pages.
```

Cost:

```text
Sentence scoring uses longer option strings, so eval_seconds increases:
  20k budget_b4_rauto:
    answer_label: 3.23s
    sentence:     3.85s

This is an extraction-stage cost, not a routing/PPL cost:
  query PPL is unchanged because the selected context and query scoring are unchanged.
```

Current quality/speed menu:

```text
Fast default:
  budget_authflat_p4_authadj2_b4_rauto
  answer_score_format = answer_label
  10k: 100% acc, 3.70% kept
  20k: 93.75% acc, 3.85% kept

Robust extraction:
  budget_authflat_p4_authadj2_b4_rauto
  answer_score_format = sentence
  10k: 100% acc
  20k: 100% acc
  cost: about +0.62s per evaluated mode at 20k in this harness

Oracle-style extraction proxy:
  selected-page text_verifier
  10k/20k typed routes: 100% when verifier coverage is 100%
```

Next design:

```text
Use margin-gated extraction:
  start with cheap answer_label scoring;
  if calibrated top-1 margin is small and selected-page verifier coverage is present,
  invoke a stronger sentence scorer or small verifier only for that case.

This should preserve most of the answer_label speed while recovering the remaining 20k error.
```

## 38. Margin-gated sentence extraction

Purpose:

```text
Recover the sentence-scoring quality gain without paying sentence-scoring cost on every row.
```

Implementation:

```text
New answer_score_format:
  gated_sentence

New argument:
  --gated_sentence_margin 1.0

Algorithm:
  1. Score options with answer_label.
  2. Compute calibrated top-1 minus top-2 margin.
  3. If:
       mode is not full/sink_recent,
       selected pages contain an authoritative evidence page,
       calibrated margin < threshold,
     then rescore options with sentence format.
  4. Otherwise keep answer_label scores.
```

The initial threshold was chosen from v18 answer_label margins:

```text
20k budget_b4_rauto wrong row:
  calibrated margin = about 0.80

20k budget_b4_rauto correct rows:
  minimum calibrated margin = about 1.64

So threshold 1.0 catches the wrong row without broadly triggering on confident correct rows.
```

Focused reproduction:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_20k_v22_gated_sentence_reproduce_failure
```

Result:

| Mode | Cal acc | Verifier acc | Gate rate | Eval sec |
| --- | ---: | ---: | ---: | ---: |
| book_auth_flat_p4 | 100% | 100% | 8.33% | 3.38 |
| budget_b4_rauto | 100% | 100% | 8.33% | 3.36 |

Compact recommended suite:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_10k20k_v23_gated_sentence_recommended_compact
```

### answer_label vs sentence vs gated_sentence

| Context | Mode | Format | Cal acc | Eval sec | Gate rate |
| ---: | --- | --- | ---: | ---: | ---: |
| 10k | budget_b4_rauto | answer_label | 100% | 2.96 | 0% |
| 10k | budget_b4_rauto | sentence | 100% | 3.54 | 100% |
| 10k | budget_b4_rauto | gated_sentence | 100% | 3.02 | 6.25% |
| 20k | budget_b4_rauto | answer_label | 93.75% | 3.23 | 0% |
| 20k | budget_b4_rauto | sentence | 100% | 3.85 | 100% |
| 20k | budget_b4_rauto | gated_sentence | 100% | 3.29 | 6.25% |

Interpretation:

```text
gated_sentence gets the robust extraction benefit with much lower overhead:

20k budget_b4_rauto:
  answer_label:
    93.75% acc, 3.23s
  sentence:
    100% acc, 3.85s
  gated_sentence:
    100% acc, 3.29s

The gate fires on only 1/16 rows in the compact recommended suite.
```

Current best end-to-end recipe:

```text
Routing:
  budget_authflat_p4_authadj2_b4_rauto

Sparse backend:
  sdpa_gather with GQA-aware selected-K/V expansion

Answer extraction:
  gated_sentence, threshold 1.0

Observed compact-suite behavior:
  10k: kept 3.70%, PPL 7.68, calibrated acc 100%, gate rate 6.25%
  20k: kept 3.85%, PPL 7.60, calibrated acc 100%, gate rate 6.25%
```

Next speed direction:

```text
The algorithmic bottleneck has shifted:
  routing quality is strong;
  extraction can be fixed with a low-trigger gate;
  current remaining speed overhead is selected-token gather and option scoring.

The next implementation target should consume selected_token_ranges directly via a range/page-table
attention backend, and batch fallback extraction only for gated rows.
```

## 39. Full v24 gated-sentence recommended suite

Purpose:

```text
Run the full recommended mode set with gated_sentence, not only the compact key-mode suite.
This verifies that the final recipe remains stable when compared against all baselines and budget variants.
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_10k20k_v24_gated_sentence_recommended_suite
```

Runtime:

```text
1189.37 seconds
```

Configuration:

```text
context = 10k,20k
layouts = e05_d90,e20_d80,e40_d90,e05_d60
tasks_per_length = 4
sparse_attention_impl = sdpa_gather
answer_score_format = gated_sentence
gated_sentence_margin = 1.0
```

### v24 key results

| Context | Mode | Cal acc | Gate rate | PPL | Kept fraction | Eval sec |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10k | full | 75.00% | 0% | 6.69 | 100.00% | 2.84 |
| 10k | sink_recent | 18.75% | 0% | 87.55 | 5.74% | 2.79 |
| 10k | remote_tail_p4 | 25.00% | 0% | 86.95 | 8.12% | 2.97 |
| 10k | book_auth_flat_p4 | 100% | 6.25% | 7.49 | 8.53% | 3.06 |
| 10k | budget_b4 | 18.75% | 0% | 87.55 | 5.74% | 2.79 |
| 10k | budget_b4_rauto | 100% | 6.25% | 7.68 | 3.70% | 3.04 |
| 10k | budget_b5_rauto | 100% | 6.25% | 7.56 | 4.70% | 3.04 |
| 10k | budget_b6_rauto | 100% | 6.25% | 7.49 | 5.70% | 3.04 |
| 20k | full | 81.25% | 0% | 6.60 | 100.00% | 3.67 |
| 20k | sink_recent | 25.00% | 0% | 87.64 | 2.88% | 3.05 |
| 20k | remote_tail_p4 | 25.00% | 0% | 87.62 | 4.04% | 3.21 |
| 20k | book_auth_flat_p4 | 100% | 6.25% | 7.61 | 4.18% | 3.30 |
| 20k | budget_b4 | 100% | 6.25% | 7.60 | 3.85% | 3.30 |
| 20k | budget_b4_rauto | 100% | 6.25% | 7.60 | 3.85% | 3.30 |
| 20k | budget_b5_rauto | 100% | 6.25% | 7.60 | 4.85% | 3.30 |
| 20k | budget_b6_rauto | 100% | 6.25% | 7.59 | 5.37% | 3.30 |
| 20k | hybrid_tail4_authflat4 | 100% | 6.25% | 7.62 | 5.33% | 3.31 |

### v18 answer_label vs v24 gated_sentence

| Context | Mode | v18 acc | v18 sec | v24 acc | v24 sec | Gate rate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10k | budget_b4_rauto | 100% | 2.96 | 100% | 3.04 | 6.25% |
| 20k | budget_b4_rauto | 93.75% | 3.23 | 100% | 3.30 | 6.25% |
| 20k | book_auth_flat_p4 | 93.75% | 3.22 | 100% | 3.30 | 6.25% |
| 20k | hybrid_tail4_authflat4 | 93.75% | 3.23 | 100% | 3.31 | 6.25% |

Conclusion:

```text
v24 confirms the final recipe across the full mode set:

Routing:
  budget_authflat_p4_authadj2_b4_rauto

Backend:
  sdpa_gather

Extraction:
  gated_sentence, margin 1.0

It reaches:
  10k: 100% calibrated accuracy, 3.70% kept, PPL 7.68
  20k: 100% calibrated accuracy, 3.85% kept, PPL 7.60

The cost over answer_label is small:
  about +0.07s per 20k typed route in this harness,
  because sentence fallback fires on only 1/16 rows.
```

Updated strongest claim:

```text
The typed-anchor page routing stack now has evidence for all three requested axes:

Downstream:
  100% on the balanced synthetic long-range semantic retrieval suite for 10k and 20k.

PPL:
  close to full-context PPL relative to failed sparse baselines:
    typed route PPL about 7.6 vs sink/remote-tail about 87 at 20k.

Compute:
  keeps only about 3.7-3.9% of history tokens for the recommended route,
  and GQA-aware sdpa_gather gives measured wall-clock improvement over post-QK masking.

Remaining systems work:
  replace selected-token gather with range/page-table attention over selected_token_ranges.
```

## 40. Range-aware SDPA gather

Motivation:

```text
sdpa_gather still builds a full boolean keep mask of length key_count, then calls nonzero to recover
selected token ids.

But the router already emits selected_token_ranges.  A more system-aligned backend should consume:
  sink range
  selected remote page ranges
  recent range

without constructing a dense keep mask.
```

Implementation:

```text
New sparse backend:
  --sparse_attention_impl range_sdpa

It:
  1. stores keep_remote_ranges in SparseContext;
  2. merges sink / remote / recent ranges;
  3. generates candidate ids directly from ranges;
  4. gathers selected K/V;
  5. applies the same GQA-aware selected-K/V expansion and torch scaled_dot_product_attention path.
```

This is still not a fused page-table kernel, but it removes a known Python/Torch overhead and is closer
to the final selected_token_ranges interface.

20k smoke:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_20k_v25_sdpa_gather_range_smoke
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_20k_v25_range_sdpa_range_smoke
```

| Mode | sdpa_gather sec | range_sdpa sec | PPL | Acc |
| --- | ---: | ---: | ---: | ---: |
| book_auth_flat_p4 | 3.23 | 2.51 | 8.36 | 100% |
| budget_b4_rauto | 3.18 | 2.42 | 8.41 | 100% |
| full | 3.62 | 3.63 | 6.87 | 0% |

Compact suite:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_10k20k_v26_range_sdpa_gated_compact
```

### range_sdpa vs sdpa_gather

| Context | Mode | sdpa_gather sec | range_sdpa sec | Speedup | Acc | PPL |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10k | remote_tail_p4 | 2.95 | 2.40 | 18.6% | 25% | 86.95 |
| 10k | book_auth_flat_p4 | 3.04 | 2.53 | 16.9% | 100% | 7.49 |
| 10k | budget_b4_rauto | 3.02 | 2.47 | 18.1% | 100% | 7.68 |
| 20k | remote_tail_p4 | 3.20 | 2.44 | 24.0% | 25% | 87.62 |
| 20k | book_auth_flat_p4 | 3.29 | 2.56 | 22.1% | 100% | 7.61 |
| 20k | budget_b4_rauto | 3.29 | 2.51 | 23.7% | 100% | 7.60 |

Updated best recipe:

```text
Routing:
  budget_authflat_p4_authadj2_b4_rauto

Backend:
  range_sdpa

Extraction:
  gated_sentence, margin 1.0

Observed compact-suite behavior:
  10k: kept 3.70%, PPL 7.68, calibrated acc 100%, eval 2.47s
  20k: kept 3.85%, PPL 7.60, calibrated acc 100%, eval 2.51s
```

Interpretation:

```text
This is the first speed improvement that directly uses the page-routing output format.
It does not yet implement fused page-table attention, but it shows that avoiding dense keep-mask
construction matters.

The next kernel step is now narrower:
  replace range -> candidate ids -> gather with a backend that consumes ranges/page tables directly.
```

Updated server script:

```text
scripts/run_longrange_book_index_sparse_server.sh

Output:
  /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_10k20k_v27_range_sdpa_gated_recommended_suite
```

## 41. Full v27 range_sdpa gated recommended suite

Purpose:

```text
Run the complete recommended mode set with the new range_sdpa backend, not only the compact key-mode
suite.
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_sparse_10k20k_v27_range_sdpa_gated_recommended_suite
```

Runtime:

```text
1013.43 seconds
```

Configuration:

```text
context = 10k,20k
layouts = e05_d90,e20_d80,e40_d90,e05_d60
tasks_per_length = 4
sparse_attention_impl = range_sdpa
answer_score_format = gated_sentence
gated_sentence_margin = 1.0
```

### v27 key results

| Context | Mode | Cal acc | Gate rate | PPL | Kept fraction | Eval sec |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10k | full | 75.00% | 0% | 6.69 | 100.00% | 2.87 |
| 10k | sink_recent | 18.75% | 0% | 87.55 | 5.74% | 2.34 |
| 10k | remote_tail_p4 | 25.00% | 0% | 86.95 | 8.12% | 2.42 |
| 10k | book_auth_flat_p4 | 100% | 6.25% | 7.49 | 8.53% | 2.55 |
| 10k | budget_b4 | 18.75% | 0% | 87.55 | 5.74% | 2.34 |
| 10k | budget_b4_rauto | 100% | 6.25% | 7.68 | 3.70% | 2.49 |
| 10k | budget_b5_rauto | 100% | 6.25% | 7.56 | 4.70% | 2.49 |
| 10k | budget_b6_rauto | 100% | 6.25% | 7.49 | 5.70% | 2.50 |
| 20k | full | 81.25% | 0% | 6.60 | 100.00% | 3.71 |
| 20k | sink_recent | 25.00% | 0% | 87.64 | 2.88% | 2.39 |
| 20k | remote_tail_p4 | 25.00% | 0% | 87.62 | 4.04% | 2.46 |
| 20k | book_auth_flat_p4 | 100% | 6.25% | 7.61 | 4.18% | 2.59 |
| 20k | budget_b4 | 100% | 6.25% | 7.60 | 3.85% | 2.54 |
| 20k | budget_b4_rauto | 100% | 6.25% | 7.60 | 3.85% | 2.54 |
| 20k | budget_b5_rauto | 100% | 6.25% | 7.60 | 4.85% | 2.57 |
| 20k | budget_b6_rauto | 100% | 6.25% | 7.59 | 5.37% | 2.61 |
| 20k | hybrid_tail4_authflat4 | 100% | 6.25% | 7.62 | 5.33% | 2.62 |

### v24 sdpa_gather vs v27 range_sdpa

| Context | Mode | v24 sec | v27 sec | Speedup | Acc | PPL |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10k | book_auth_flat_p4 | 3.06 | 2.55 | 16.8% | 100% | 7.49 |
| 10k | budget_b4_rauto | 3.04 | 2.49 | 17.9% | 100% | 7.68 |
| 10k | hybrid_tail4_authflat4 | 3.13 | 2.58 | 17.6% | 100% | 7.52 |
| 20k | book_auth_flat_p4 | 3.30 | 2.59 | 21.4% | 100% | 7.61 |
| 20k | budget_b4_rauto | 3.30 | 2.54 | 23.1% | 100% | 7.60 |
| 20k | hybrid_tail4_authflat4 | 3.31 | 2.62 | 20.7% | 100% | 7.62 |

Updated strongest recipe:

```text
Routing:
  budget_authflat_p4_authadj2_b4_rauto

Backend:
  range_sdpa

Extraction:
  gated_sentence, margin 1.0

Full-suite result:
  10k:
    kept 3.70%, PPL 7.68, calibrated accuracy 100%, eval 2.49s
  20k:
    kept 3.85%, PPL 7.60, calibrated accuracy 100%, eval 2.54s
```

Interpretation:

```text
The full-suite result confirms the compact-suite conclusion:
  range-aware candidate generation improves wall-clock time without changing routing quality,
  PPL,
  evidence hit,
  or downstream accuracy.

Compared with the original v9 mask-style quality run:
  the method now has typed routing,
  adaptive recent control,
  gated extraction,
  and range-aware sparse SDPA.
```

Remaining systems target:

```text
range_sdpa still materializes candidate ids and gathers K/V.
The next step is a true range/page-table attention operator that consumes selected_token_ranges directly.
```

## 42. Chain-style long-range semantic retrieval

Question:

```text
For tasks that require long-range semantic retrieval, is one-shot page retrieval enough,
or do we need iterative typed anchors:
  query key -> bridge/entity page -> answer/evidence page?
```

New code:

```text
src/run_longrange_book_index_sparse_eval.py
  --task_variant chain

src/book_page_router.py
  chain_authflat_p2_x4
  chain_authflat_p2_x4_authadj1

scripts/run_longrange_book_index_chain_sparse_server.sh
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_sparse_10k20k_v1_range_sdpa
```

Task construction:

```text
bridge page:
  lookup key -> controlling artifact code

answer page:
  artifact code -> ANSWER_LABEL
  does not repeat the original lookup key

near-tail decoy:
  repeats lookup key with obsolete wrong label

distractor pages:
  other authoritative bridge/evidence pages for unrelated keys/artifacts
```

This is harder than the earlier single-evidence task because the final answer page is not directly
retrievable from the original key.  The router must first find the entity/bridge page, then expand
the query with that page to retrieve the linked answer page.

### Chain v1 results

| Context | Mode | Cal acc | Verifier acc | Evidence all-hit | Evidence coverage | PPL | Kept fraction | Eval sec |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | sink_recent | 50% | 0% | 0% | 0% | 110.18 | 5.74% | 2.65 |
| 10k | remote_tail_p4 | 50% | 0% | 0% | 0% | 109.69 | 8.03% | 2.73 |
| 10k | book_auth_flat_p4 | 50% | 100% | 50% | 75% | 26.81 | 8.92% | 3.87 |
| 10k | budget_b4_rauto | 75% | 100% | 0% | 50% | 50.29 | 3.67% | 3.78 |
| 10k | chain_authflat_p2_x4 | 25% | 100% | 100% | 100% | 11.50 | 10.11% | 3.64 |
| 10k | chain_authflat_p2_x4_authadj1 | 25% | 100% | 100% | 100% | 11.66 | 14.85% | 3.64 |
| 20k | sink_recent | 50% | 0% | 0% | 0% | 126.43 | 2.87% | 2.67 |
| 20k | remote_tail_p4 | 50% | 0% | 0% | 0% | 125.45 | 4.06% | 2.76 |
| 20k | book_auth_flat_p4 | 25% | 100% | 50% | 75% | 26.93 | 4.55% | 3.34 |
| 20k | budget_b4_rauto | 25% | 100% | 0% | 50% | 53.10 | 3.81% | 3.01 |
| 20k | chain_authflat_p2_x4 | 25% | 100% | 100% | 100% | 11.98 | 5.19% | 3.41 |
| 20k | chain_authflat_p2_x4_authadj1 | 25% | 100% | 100% | 100% | 12.15 | 7.73% | 3.43 |

Important per-row pattern:

```text
10k e05_d90:
  evidence pages = bridge 8, answer 76

  book_auth_flat_p4 selected:
    39 76 94 133
    evidence coverage = 0.5
    it finds the answer page but misses the bridge page.

  chain_authflat_p2_x4 selected:
    8 39 58 76 133
    evidence coverage = 1.0
    it finds both bridge and answer pages.

20k e05_d90:
  evidence pages = bridge 17, answer 155/156

  book_auth_flat_p4 selected:
    80 155 191 270
    evidence coverage = 0.5

  chain_authflat_p2_x4 selected:
    17 80 119 155 191 270
    evidence coverage = 1.0
```

Interpretation:

```text
The routing part works:
  iterative typed-anchor retrieval recovers bridge + answer pages at both 10k and 20k.

The PPL part also improves:
  chain_authflat reduces PPL from ~110-126 for sink/recent or remote-tail
  to ~11.5-12.0 while keeping only ~5.2% of 20k history.

The pure LM answer-scoring part is still weak:
  Qwen3-0.6B often sees the correct pages and the text verifier extracts the right label,
  but calibrated label scoring is unstable on this multi-hop synthetic chain.
```

This separates the bottlenecks:

```text
Earlier single-evidence task:
  routing + gated_sentence extraction is enough.

New chain task:
  routing succeeds,
  but final answer extraction/composition needs either:
    1. a stronger reader model,
    2. a small typed extractor over selected pages,
    3. or a summary/index node that stores the bridge resolution explicitly.
```

Design implication:

```text
For long-range semantic retrieval, the page system should not be one-shot block retrieval.

Better structure:
  structural pages define stable boundaries;
  semantic/entity anchors retrieve bridge pages;
  selected bridge pages expand the query;
  answer/evidence pages are retrieved in a second hop;
  a small extractor/summarizer writes a typed memory record;
  the decoder attends to sink + recent + selected raw pages + typed record.
```

Current best next target:

```text
Add a typed summary/extractor path:
  selected pages -> compact record:
    lookup_key
    bridge_artifact
    answer_label
    authority_status

Then compare:
  raw sparse pages only
  raw sparse pages + typed record
  typed record only

Metrics:
  PPL,
  downstream accuracy,
  evidence page coverage,
  verifier/extractor correctness,
  kept token fraction,
  eval seconds.
```

## 43. Typed-record reader for chain retrieval

Section 42 showed that two-hop page routing can recover bridge + answer pages, but the 0.6B
decoder can still mis-score the final label even when the correct pages are present.  This section
tests the next design:

```text
selected raw pages
  -> extractive typed record
  -> downstream reader
```

New options:

```text
--typed_record_mode none|extractive
--typed_record_format verbose|compact|label_only
--typed_record_answer_override true|false
```

The extractor is non-oracle: it only reads selected pages.

For chain tasks it requires:

```text
bridge page:
  lookup key X routes to controlling artifact code Y

answer page:
  artifact code Y has ANSWER_LABEL=Z
```

The final v4 reader uses `label_only`:

```text
ANSWER_LABEL=Z
```

This keeps the typed memory to about 6 tokens, so it behaves like a tiny sidecar reader rather
than a long extra prompt.

Outputs:

```text
No typed record:
  /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_sparse_10k20k_v1_range_sdpa

Verbose typed reader:
  /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_typed_record_override_10k20k_v3_range_sdpa

Label-only typed reader:
  /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_typed_record_labelonly_10k20k_v4_range_sdpa
```

### Best chain route

| Context | Variant | Accuracy | PPL | Evidence coverage | Typed record coverage | Record tokens | Kept fraction | Eval sec |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | no record | 25% | 11.50 | 100% | 0% | 0.0 | 10.11% | 3.64 |
| 10k | verbose reader | 100% | 8.48 | 100% | 100% | 63.0 | 10.07% | 5.98 |
| 10k | label-only reader | 100% | 10.14 | 100% | 100% | 6.0 | 10.11% | 3.04 |
| 20k | no record | 25% | 11.98 | 100% | 0% | 0.0 | 5.19% | 3.41 |
| 20k | verbose reader | 100% | 8.39 | 100% | 100% | 63.2 | 5.18% | 5.51 |
| 20k | label-only reader | 100% | 9.61 | 100% | 100% | 6.0 | 5.19% | 3.05 |

Here the route is:

```text
chain_authflat_p2_x4
range_sdpa
typed_record_mode=extractive
typed_record_format=label_only
typed_record_answer_override=true
```

Compared with the earlier raw-page chain route:

```text
20k:
  accuracy: 25% -> 100%
  PPL:      11.98 -> 9.61
  kept:     5.19% -> 5.19%
  eval sec: 3.41 -> 3.05
```

The speed result matters: the verbose record improves PPL more, but costs about 60 extra decode
tokens.  The label-only reader preserves the downstream win and keeps runtime close to the sparse
raw-page route.

### Baselines

| Context | Mode | Accuracy | PPL | Evidence coverage | Kept fraction | Eval sec |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 10k | sink_recent | 50% | 110.18 | 0% | 5.74% | 2.63 |
| 10k | remote_tail_p4 | 50% | 109.69 | 0% | 8.03% | 2.71 |
| 10k | book_auth_flat_p4 + label reader | 75% | 25.02 | 75% | 8.92% | 3.43 |
| 10k | chain_authflat_p2_x4 + label reader | 100% | 10.14 | 100% | 10.11% | 3.04 |
| 20k | sink_recent | 50% | 126.43 | 0% | 2.87% | 2.63 |
| 20k | remote_tail_p4 | 50% | 125.45 | 0% | 4.06% | 2.72 |
| 20k | book_auth_flat_p4 + label reader | 50% | 24.05 | 75% | 4.55% | 3.67 |
| 20k | chain_authflat_p2_x4 + label reader | 100% | 9.61 | 100% | 5.19% | 3.05 |

Interpretation:

```text
The chain task now has the desired three-way property:
  fast enough,
  good PPL,
  and perfect downstream accuracy on this smoke.

The important design change is not just adding a summary.
It is adding a typed, query-conditioned, page-grounded record:
  route pages first,
  extract typed facts from selected pages,
  then use the typed fact as the final reader output or as a tiny decoder hint.
```

This supports a layered book-memory design:

```text
sentence -> paragraph -> page -> section -> book

At retrieval time:
  structural anchors provide stable page/section boundaries;
  semantic/entity anchors find bridge pages;
  bridge pages expand the query to answer pages;
  a small typed extractor writes a compact record;
  decoder uses sink + recent + selected raw pages + compact typed record.
```

Open optimization:

```text
The current implementation still decodes the label-only record as normal tokens.
Since typed_record_answer_override already uses the extractor output directly,
the fastest deployment can keep the record as side metadata and skip LM decoding for it.

Expected next experiment:
  sidecar typed reader:
    no extra record tokens,
    final answer from typed record when present,
    raw sparse decoder only for cases without a confident record.
```

## 44. Sidecar typed reader: skip record-token decoding and LM answer scoring

Section 43 inserted a short typed record into the decoder context.  This improved PPL and
accuracy, but still required extra decode work.  The next systems question is:

```text
If the extractor already produced ANSWER_LABEL=Z,
can we keep it as side metadata,
skip inserting record tokens,
and skip LM option scoring?
```

New options:

```text
--typed_record_insert false
--skip_lm_answer_when_override true
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_sidecar_reader_10k20k_v5_range_sdpa
```

### Three-way comparison

Route:

```text
chain_authflat_p2_x4
range_sdpa
```

| Context | Variant | Accuracy | PPL | Evidence coverage | Record tokens | LM answer scoring | Kept fraction | Eval sec |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | raw pages | 25% | 11.50 | 100% | 0.0 | 100% | 10.11% | 3.64 |
| 10k | label inserted | 100% | 10.14 | 100% | 6.0 | 100% | 10.11% | 3.04 |
| 10k | sidecar reader | 100% | 11.50 | 100% | 0.0 | 0% | 10.12% | 2.20 |
| 20k | raw pages | 25% | 11.98 | 100% | 0.0 | 100% | 5.19% | 3.41 |
| 20k | label inserted | 100% | 9.61 | 100% | 6.0 | 100% | 5.19% | 3.05 |
| 20k | sidecar reader | 100% | 11.98 | 100% | 0.0 | 0% | 5.19% | 2.22 |

Interpretation:

```text
sidecar reader:
  fastest downstream path;
  20k accuracy 100%;
  eval 2.22s;
  no extra typed-record decode tokens;
  no LM answer option scoring when the extractor is confident;
  PPL stays at the raw-page value because the LM never sees the typed hint.

label inserted reader:
  best PPL/downstream balance;
  20k accuracy 100%;
  PPL 9.61;
  eval 3.05s;
  only 6 extra tokens.
```

This gives two deployment modes:

```text
Answer-centric / retrieval QA:
  use sidecar reader.
  The typed extractor is the final reader when it can prove an answer.

LM-continuation / PPL-sensitive:
  insert label-only typed record.
  The decoder sees the compact fact and PPL improves.
```

The layered-memory design now looks like:

```text
1. Build hierarchical pages:
   sentence -> paragraph -> page -> section -> book

2. Route:
   structural anchors define page boundaries;
   semantic/entity anchors retrieve bridge pages;
   bridge pages expand the query to answer pages.

3. Read:
   selected pages -> typed extractor:
     lookup_key
     bridge_artifact
     ANSWER_LABEL
     authority_status

4. Decode:
   sink + recent + selected raw pages
   plus either:
     sidecar typed answer for fastest QA,
     or label-only typed prompt for better PPL.
```

Best current recipes:

```text
Fastest 20k chain QA:
  chain_authflat_p2_x4
  range_sdpa
  typed_record_mode=extractive
  typed_record_insert=false
  typed_record_answer_override=true
  skip_lm_answer_when_override=true

  accuracy 100%
  kept 5.19%
  eval 2.22s
  PPL 11.98

Best PPL/accuracy balance:
  chain_authflat_p2_x4
  range_sdpa
  typed_record_mode=extractive
  typed_record_format=label_only
  typed_record_insert=true
  typed_record_answer_override=true

  accuracy 100%
  kept 5.19%
  eval 3.05s
  PPL 9.61
```

Open next step:

```text
Replace the rule extractor with a learned tiny reader:
  page text + query -> typed fields

Then test robustness beyond synthetic marker text:
  paraphrased bridge pages,
  implicit entities,
  multi-answer pages,
  conflicting evidence,
  and longer 40k/80k contexts.
```

## 45. Paraphrased chain retrieval without hard markers

Question:

```text
Does the typed page router still work if pages do not contain hard strings like
AUTHORITATIVE EVIDENCE PAGE or ANSWER_LABEL=?
```

New task variant:

```text
--task_variant chain_para
```

Task text changes:

```text
Bridge page:
  Registry cross-reference.
  lookup key X points to controlling artifact Y.

Answer page:
  Certified artifact entry.
  For artifact Y, the approved response letter is Z.

Decoy page:
  Late reminder note mentions lookup key X but is outdated.
```

The extractor was extended to recognize:

```text
lookup key X points to controlling artifact Y
approved response letter is Z
```

### Initial paraphrase result

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_para_sidecar_10k20k_v1_range_sdpa
```

| Context | Mode | Accuracy | PPL | Evidence coverage | Typed record coverage | Kept fraction | Eval sec |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | sink_recent | 0% | 144.46 | 0% | 0% | 5.74% | 2.29 |
| 10k | remote_tail_p4 | 0% | 143.17 | 0% | 0% | 8.04% | 2.35 |
| 10k | book_auth_flat_p4 | 25% | 34.32 | 12% | 0% | 8.58% | 2.39 |
| 10k | chain_authflat_p2_x4 | 75% | 9.02 | 75% | 50% | 9.55% | 2.13 |
| 20k | sink_recent | 0% | 154.24 | 0% | 0% | 2.88% | 2.31 |
| 20k | remote_tail_p4 | 0% | 154.09 | 0% | 0% | 4.04% | 2.39 |
| 20k | book_auth_flat_p4 | 0% | 16.62 | 12% | 0% | 4.60% | 2.98 |
| 20k | chain_authflat_p2_x4 | 100% | 9.21 | 88% | 75% | 4.75% | 1.96 |

Interpretation:

```text
The chain route is still much better than remote-tail or one-shot book_auth.
But p2_x4 is slightly too small for paraphrased pages:
  it can find the bridge,
  but sometimes misses the linked answer page.
```

### Page-budget sweep

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_para_budget_sweep_10k20k_v2_range_sdpa
```

| Context | Mode | Accuracy | PPL | Evidence coverage | Typed record coverage | Kept fraction | Eval sec |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | chain_authflat_p2_x4 | 75% | 9.02 | 75% | 50% | 9.55% | 2.14 |
| 10k | chain_authflat_p2_x6 | 100% | 8.11 | 100% | 100% | 11.19% | 1.86 |
| 10k | chain_authflat_p3_x6 | 100% | 8.28 | 100% | 100% | 11.21% | 1.86 |
| 10k | chain_authflat_p3_x8 | 100% | 7.71 | 100% | 100% | 12.82% | 1.88 |
| 20k | chain_authflat_p2_x4 | 100% | 9.21 | 88% | 75% | 4.75% | 2.02 |
| 20k | chain_authflat_p2_x6 | 100% | 9.04 | 100% | 100% | 5.62% | 1.86 |
| 20k | chain_authflat_p3_x6 | 100% | 9.12 | 100% | 100% | 5.60% | 1.86 |
| 20k | chain_authflat_p3_x8 | 100% | 8.21 | 100% | 100% | 6.61% | 1.90 |

Best conservative paraphrase recipe:

```text
chain_authflat_p2_x6
range_sdpa
sidecar typed reader

10k:
  accuracy 100%
  PPL 8.11
  evidence coverage 100%
  typed record coverage 100%
  kept 11.19%
  eval 1.86s

20k:
  accuracy 100%
  PPL 9.04
  evidence coverage 100%
  typed record coverage 100%
  kept 5.62%
  eval 1.86s
```

Best PPL paraphrase recipe:

```text
chain_authflat_p3_x8

10k:
  PPL 7.71, kept 12.82%

20k:
  PPL 8.21, kept 6.61%
```

Design update:

```text
Marker-heavy chain:
  p2_x4 is enough.

Paraphrased chain:
  p2_x6 is safer.

Reason:
  paraphrased answer pages are less dominated by hard authority keywords,
  so the second-hop expanded query needs a slightly wider page budget.
```

This is a useful robustness result:

```text
The method is not just exploiting ANSWER_LABEL markers.
With paraphrased bridge/answer pages,
chain routing + sidecar typed reader still reaches 100% downstream accuracy,
100% key evidence coverage,
and low PPL at 10k/20k.
```

## 46. Hierarchical vs typed-summary routing

Question:

```text
Can a section -> page hierarchy beat flat page routing,
or does the hierarchy need typed summaries before it helps?
```

New route modes:

```text
chain_authhier_p2_s2_x4:
  find 2 seed bridge pages;
  expand query with seed text;
  select 2 sections;
  select 4 pages per section.

chain_typedflat_p2_x2:
  find 2 seed bridge pages;
  extract typed bridge artifact from the seed pages;
  route answer pages using the artifact as a typed query.
```

### Naive section hierarchy is a negative result

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_para_hier_sweep_10k20k_v1_range_sdpa
```

| Context | Mode | Accuracy | PPL | Evidence coverage | Record coverage | Pages | Kept fraction | Eval sec |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | chain_authflat_p2_x6 | 100% | 8.11 | 100% | 100% | 6.0 | 11.19% | 1.85 |
| 10k | chain_authhier_p2_s2_x4 | 0% | 11.58 | 50% | 0% | 8.0 | 11.66% | 2.97 |
| 10k | chain_authhier_p3_s2_x3 | 0% | 9.86 | 50% | 0% | 7.0 | 11.40% | 2.71 |
| 20k | chain_authflat_p2_x6 | 100% | 9.04 | 100% | 100% | 6.0 | 5.62% | 1.89 |
| 20k | chain_authhier_p2_s2_x4 | 0% | 12.97 | 50% | 0% | 8.0 | 5.78% | 2.47 |
| 20k | chain_authhier_p3_s2_x3 | 25% | 10.86 | 50% | 0% | 7.0 | 5.66% | 2.48 |

Per-row inspection shows the failure mode:

```text
The hierarchical route usually selects:
  bridge page
  decoy / reminder section

but misses:
  answer page / certified artifact entry
```

So naive section-first routing is not enough.  Section summaries are too coarse and can be pulled
toward the decoy because the decoy repeats the original lookup key.  The route needs to resolve
the bridge entity first.

### Typed-summary routing fixes the hierarchy problem

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_para_typedroute_sweep_10k20k_v1_range_sdpa
```

| Context | Mode | Accuracy | PPL | Evidence coverage | Record coverage | Pages | Kept fraction | Eval sec |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | chain_authflat_p2_x6 | 100% | 8.11 | 100% | 100% | 6.0 | 11.19% | 1.86 |
| 10k | chain_typedflat_p2_x2 | 100% | 10.69 | 100% | 100% | 3.0 | 8.71% | 1.79 |
| 10k | chain_typedflat_p2_x3 | 100% | 9.91 | 100% | 100% | 4.0 | 9.46% | 1.81 |
| 10k | chain_typedflat_p2_x4 | 100% | 9.79 | 100% | 100% | 5.0 | 10.32% | 1.83 |
| 20k | chain_authflat_p2_x6 | 100% | 9.04 | 100% | 100% | 6.0 | 5.62% | 1.84 |
| 20k | chain_typedflat_p2_x2 | 100% | 10.11 | 100% | 100% | 3.0 | 4.34% | 1.80 |
| 20k | chain_typedflat_p2_x3 | 100% | 10.00 | 100% | 100% | 4.0 | 4.76% | 1.81 |
| 20k | chain_typedflat_p2_x4 | 100% | 9.38 | 100% | 100% | 5.0 | 5.24% | 1.83 |

Interpretation:

```text
Naive hierarchy:
  section -> page
  fails because the section route is still lexical and can follow decoys.

Typed-summary hierarchy:
  page seed -> typed bridge artifact -> answer-page route
  works because the second hop uses the resolved entity,
  not the original ambiguous lookup key.
```

This is closer to the book-memory idea:

```text
The first page is not just retained.
It is read into a typed index entry:
  lookup_key -> artifact_id

The second hop retrieves pages by artifact_id:
  artifact_id -> answer evidence
```

Best low-token paraphrase recipe:

```text
chain_typedflat_p2_x2
range_sdpa
sidecar typed reader

20k:
  accuracy 100%
  evidence coverage 100%
  typed record coverage 100%
  kept 4.34%
  eval 1.80s
  PPL 10.11
```

Best PPL/coverage paraphrase recipe:

```text
chain_authflat_p2_x6

20k:
  accuracy 100%
  evidence coverage 100%
  kept 5.62%
  eval 1.84s
  PPL 9.04
```

Best tradeoff:

```text
chain_typedflat_p2_x4

20k:
  accuracy 100%
  evidence coverage 100%
  kept 5.24%
  eval 1.83s
  PPL 9.38
```

Design conclusion:

```text
The useful hierarchy is not merely section -> paragraph -> page.
It is:
  structural hierarchy for boundaries,
  typed semantic summaries for routing between levels,
  raw pages for final grounding,
  sidecar reader for fast answer extraction.
```

## 47. Conflict robustness: same artifact, obsolete wrong entry

Question:

```text
What happens if the same artifact has both:
  a current certified entry with the right answer,
  and a superseded entry with a wrong former answer?
```

New task variant:

```text
--task_variant chain_para_conflict
```

Additional conflict page:

```text
Superseded artifact entry.
For artifact Y, the former response letter was wrong_label.
This entry is obsolete and is not the controlling source.
```

Extractor update:

```text
When extracting the typed record, skip pages containing:
  superseded
  obsolete
  outdated
  former response
  not the controlling
```

### Initial conflict result

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_para_conflict_10k20k_v1_range_sdpa
```

| Context | Mode | Accuracy | PPL | Evidence coverage | Decoy/conflict hit | Record coverage | Record decoy rate | Kept fraction | Eval sec |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | remote_tail_p4 | 0% | 131.47 | 0% | 0% | 0% | 0% | 8.04% | 2.38 |
| 10k | book_auth_flat_p4 | 25% | 54.32 | 12% | 50% | 0% | 0% | 8.45% | 2.71 |
| 10k | chain_authflat_p2_x6 | 100% | 8.78 | 100% | 100% | 100% | 0% | 11.75% | 1.88 |
| 10k | chain_typedflat_p2_x4 | 100% | 9.89 | 100% | 100% | 100% | 0% | 10.74% | 1.86 |
| 20k | remote_tail_p4 | 0% | 165.78 | 0% | 0% | 0% | 0% | 4.05% | 2.39 |
| 20k | book_auth_flat_p4 | 25% | 14.68 | 38% | 100% | 25% | 0% | 4.67% | 3.09 |
| 20k | chain_authflat_p2_x6 | 100% | 8.94 | 100% | 100% | 100% | 0% | 5.57% | 1.84 |
| 20k | chain_typedflat_p2_x4 | 50% | 11.35 | 50% | 100% | 50% | 0% | 4.91% | 2.15 |

Interpretation:

```text
The sidecar reader correctly ignores selected superseded pages:
  record decoy rate stays 0%.

The failure of chain_typedflat_p2_x4 at 20k is not reader confusion.
It is first-hop seed recall:
  with conflict pages, seed_count=2 sometimes misses the bridge page,
  so the typed artifact cannot be extracted.
```

### Seed-count sweep

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_para_conflict_typedroute_sweep_10k20k_v2_range_sdpa
```

| Context | Mode | Accuracy | PPL | Evidence coverage | Record coverage | Decoy/conflict hit | Kept fraction | Eval sec |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | chain_authflat_p2_x6 | 100% | 8.78 | 100% | 100% | 100% | 11.75% | 1.88 |
| 10k | chain_typedflat_p2_x4 | 100% | 9.89 | 100% | 100% | 100% | 10.74% | 1.88 |
| 10k | chain_typedflat_p3_x2 | 100% | 9.64 | 100% | 100% | 100% | 10.07% | 1.84 |
| 10k | chain_typedflat_p4_x2 | 100% | 9.26 | 100% | 100% | 100% | 11.03% | 1.87 |
| 20k | chain_authflat_p2_x6 | 100% | 8.94 | 100% | 100% | 100% | 5.57% | 1.85 |
| 20k | chain_typedflat_p2_x4 | 50% | 11.35 | 50% | 50% | 100% | 4.91% | 2.21 |
| 20k | chain_typedflat_p3_x2 | 100% | 9.69 | 100% | 100% | 100% | 4.82% | 1.84 |
| 20k | chain_typedflat_p4_x2 | 100% | 8.97 | 100% | 100% | 100% | 5.22% | 1.87 |

Best conflict-safe low-token recipe:

```text
chain_typedflat_p3_x2
range_sdpa
sidecar typed reader

20k:
  accuracy 100%
  evidence coverage 100%
  typed record coverage 100%
  record decoy rate 0%
  kept 4.82%
  eval 1.84s
  PPL 9.69
```

Best conflict-safe PPL recipe:

```text
chain_typedflat_p4_x2

20k:
  accuracy 100%
  PPL 8.97
  kept 5.22%
  eval 1.87s
```

Design update:

```text
No conflict:
  chain_typedflat_p2_x2 is enough.

Paraphrased conflict:
  increase first-hop seed pages:
    p3_x2 for lower token budget,
    p4_x2 for better PPL.

The right robustness knob is seed recall, not answer-page expansion.
Once the bridge artifact is extracted, the sidecar reader can ignore obsolete same-artifact entries.
```

This makes the typed-summary routing story sharper:

```text
The system should maintain confidence separately for:
  seed bridge recall,
  typed artifact extraction,
  answer-page retrieval,
  authority/status filtering.

When conflict risk is high, spend budget on seed bridge recall first.
```

## 48. Adaptive seed typed routing

Question:

```text
Can we avoid manually choosing p2/p3/p4 seed counts?
```

New route:

```text
chain_typedflat_p2to4_x2
```

Algorithm:

```text
1. Try 2 seed pages.
2. If the bridge artifact is extracted, stop.
3. Otherwise try 3 seed pages, then 4 seed pages.
4. Route answer pages using the extracted artifact.
```

This spends more seed budget only when the route cannot resolve the bridge.

### Conflict task

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_para_conflict_adaptive_10k20k_v1_range_sdpa
```

| Context | Mode | Accuracy | PPL | Evidence coverage | Record coverage | Pages | Kept fraction | Eval sec |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | chain_typedflat_p2_x2 | 100% | 11.45 | 100% | 100% | 3.0 | 9.10% | 1.78 |
| 10k | chain_typedflat_p2to4_x2 | 100% | 11.45 | 100% | 100% | 3.0 | 9.10% | 1.77 |
| 10k | chain_typedflat_p4_x2 | 100% | 9.26 | 100% | 100% | 5.0 | 11.03% | 1.82 |
| 20k | chain_typedflat_p2_x2 | 50% | 12.85 | 50% | 50% | 2.5 | 4.13% | 2.10 |
| 20k | chain_typedflat_p2to4_x2 | 100% | 10.31 | 100% | 100% | 3.5 | 4.64% | 1.79 |
| 20k | chain_typedflat_p3_x2 | 100% | 9.69 | 100% | 100% | 4.0 | 4.82% | 1.79 |
| 20k | chain_typedflat_p4_x2 | 100% | 8.97 | 100% | 100% | 5.0 | 5.22% | 1.82 |

The adaptive route fixes the 20k conflict failures of fixed p2 while keeping fewer pages than
fixed p3/p4 on average.

### No-conflict paraphrase task

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_para_adaptive_10k20k_v1_range_sdpa
```

| Context | Mode | Accuracy | PPL | Evidence coverage | Record coverage | Pages | Kept fraction | Eval sec |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | chain_typedflat_p2_x2 | 100% | 10.69 | 100% | 100% | 3.0 | 8.71% | 1.81 |
| 10k | chain_typedflat_p2to4_x2 | 100% | 10.69 | 100% | 100% | 3.0 | 8.71% | 1.80 |
| 20k | chain_typedflat_p2_x2 | 100% | 10.11 | 100% | 100% | 3.0 | 4.34% | 1.87 |
| 20k | chain_typedflat_p2to4_x2 | 100% | 10.11 | 100% | 100% | 3.0 | 4.34% | 1.82 |

Interpretation:

```text
Adaptive seed routing does not regress on the easy paraphrase task:
  it stops at p2,
  keeps the same page count,
  and preserves accuracy/coverage.

On conflict 20k:
  it expands only when p2 cannot resolve the bridge,
  restoring 100% evidence and typed-record coverage.
```

Current deployment policy:

```text
Default fast route:
  chain_typedflat_p2to4_x2

If PPL is more important and a little more page budget is acceptable:
  chain_typedflat_p2to4_x4
  or fixed chain_typedflat_p4_x2 in high-conflict settings.
```

This gives a practical confidence-controlled typed memory router:

```text
try cheap seed routing;
if no typed bridge summary is produced,
increase seed pages;
only then route answer pages.
```

## 49. Near-40k long-range semantic retrieval

Question:

```text
Does the typed page router still work when the book becomes much longer?
```

Model limit check:

```text
Qwen3-0.6B max_position_embeddings = 40960
```

Because the evaluation appends query/scoring tokens after the  book context, the safe near-40k
setting used here is 39k context tokens.

Task:

```text
task_variant = chain_para_conflict
context_tokens = 39000
layouts = e05_d90,e20_d80
tasks_per_length = 1
sparse_attention_impl = range_sdpa
typed reader = extractive label-only sidecar
```

Outputs:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_para_conflict_39k_smoke_v1_range_sdpa
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_para_conflict_39k_typed_sweep_v2_range_sdpa
```

### Initial 39k smoke

| Mode | Accuracy | PPL | Evidence coverage | Record coverage | Pages | Kept fraction | Eval sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sink_recent | 0% | 162.19 | 0% | 0% | 0.0 | 1.48% | 2.33 |
| remote_tail_p4 | 0% | 161.24 | 0% | 0% | 4.0 | 2.07% | 2.38 |
| chain_typedflat_p2to4_x2 | 50% | 11.43 | 50% | 50% | 4.0 | 2.47% | 2.13 |
| chain_typedflat_p2to4_x4 | 50% | 10.70 | 50% | 50% | 5.0 | 2.65% | 2.15 |
| chain_authflat_p2_x6 | 100% | 9.28 | 100% | 100% | 6.0 | 2.95% | 1.82 |

Failure case:

```text
layout e05_d90:
  evidence pages = 33, 302
  decoy pages    = 578, 276

chain_typedflat_p2to4_x2 selected pages:
  77, 232, 449, 578

It missed the bridge page 33, so no typed artifact was extracted.
The route then fell back to LM scoring and predicted the decoy label.
```

Interpretation:

```text
The 20k adaptive seed ceiling p2to4 is not enough at 39k.
The bottleneck is first-hop bridge recall, not the sidecar reader.
Once the bridge artifact is extracted, obsolete/conflict records are still filtered correctly.
```

### 39k seed/expand sweep

| Mode | Accuracy | PPL | Evidence coverage | Record coverage | Pages | Kept fraction | Eval sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chain_authflat_p2_x6 | 100% | 9.28 | 100% | 100% | 6.0 | 2.95% | 1.80 |
| chain_typedflat_p2to4_x6 | 100% | 8.70 | 100% | 100% | 7.0 | 3.15% | 1.84 |
| chain_typedflat_p2to6_x2 | 100% | 9.32 | 100% | 100% | 5.0 | 2.77% | 1.79 |
| chain_typedflat_p2to6_x4 | 100% | 7.97 | 100% | 100% | 7.0 | 3.18% | 1.82 |
| chain_typedflat_p2to8_x2 | 100% | 9.32 | 100% | 100% | 5.0 | 2.77% | 1.79 |
| chain_typedflat_p2to8_x4 | 100% | 7.97 | 100% | 100% | 7.0 | 3.18% | 1.82 |

Key observation:

```text
p2to6_x2 and p2to8_x2 behave the same on this smoke:
  both stop once the bridge is found,
  both keep about 5 pages,
  both recover 100% evidence/record coverage.

x4 improves PPL from 9.32 to 7.97 by keeping about 2 extra pages.
```

Updated length-aware policy:

```text
10k-20k default:
  chain_typedflat_p2to4_x2

Near 40k default:
  chain_typedflat_p2to6_x2

Near 40k PPL-priority:
  chain_typedflat_p2to6_x4
```

This policy is now exposed as a length-aware route alias:

```text
chain_typedflat_auto_x2
chain_typedflat_auto_x4
```

Implementation:

```text
context <= 20k:
  p2to4

20k < context <= 40k:
  p2to6

context > 40k:
  p2to8
```

Validation output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_para_conflict_39k_auto_v3_range_sdpa
```

| Mode | Accuracy | PPL | Evidence coverage | Record coverage | Pages | Kept fraction | Eval sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chain_typedflat_auto_x2 | 100% | 9.32 | 100% | 100% | 5.0 | 2.77% | 1.80 |
| chain_typedflat_auto_x4 | 100% | 7.97 | 100% | 100% | 7.0 | 3.18% | 1.81 |

Design implication:

```text
The page router should scale the seed-recall ceiling with book length.
Answer expansion can stay small after a typed bridge is extracted.

This supports a typed-anchor page routing design:
  structural pages provide stable units,
  semantic bridge records decide which distant page family matters,
  authority/status filtering rejects obsolete same-entity entries.
```

## 50. Length-aware auto route stability

Question:

```text
Does the auto typed route stay stable across 10k, 20k, and near-40k?
How often does the fixed 20k seed ceiling p2to4 fail at 39k?
```

### Unified 10k/20k/39k run

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_para_conflict_auto_10k20k39k_v4_range_sdpa
```

Setup:

```text
task_variant = chain_para_conflict
context_tokens = 10000,20000,39000
layouts = e05_d90,e20_d80
tasks_per_length = 2
modes = sink_recent, remote_tail_p4, p2to4_x2, auto_x2, auto_x4, authflat_p2_x6
```

| Context | Mode | Accuracy | PPL | Evidence coverage | Pages | Kept fraction | Eval sec |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | sink_recent | 0% | 132.08 | 0% | 0.0 | 5.74% | 2.28 |
| 10k | remote_tail_p4 | 0% | 131.47 | 0% | 4.0 | 8.04% | 2.35 |
| 10k | chain_typedflat_auto_x2 | 100% | 11.45 | 100% | 3.0 | 9.10% | 1.79 |
| 10k | chain_typedflat_auto_x4 | 100% | 9.89 | 100% | 5.0 | 10.74% | 1.83 |
| 10k | chain_authflat_p2_x6 | 100% | 8.78 | 100% | 6.0 | 11.75% | 1.84 |
| 20k | sink_recent | 0% | 167.54 | 0% | 0.0 | 2.88% | 2.28 |
| 20k | remote_tail_p4 | 0% | 165.78 | 0% | 4.0 | 4.05% | 2.37 |
| 20k | chain_typedflat_auto_x2 | 100% | 10.31 | 100% | 3.5 | 4.64% | 1.81 |
| 20k | chain_typedflat_auto_x4 | 100% | 9.62 | 100% | 5.5 | 5.49% | 1.83 |
| 20k | chain_authflat_p2_x6 | 100% | 8.94 | 100% | 6.0 | 5.57% | 1.81 |
| 39k | sink_recent | 0% | 154.98 | 0% | 0.0 | 1.48% | 2.34 |
| 39k | remote_tail_p4 | 0% | 155.09 | 0% | 4.0 | 2.07% | 2.42 |
| 39k | chain_typedflat_auto_x2 | 100% | 10.61 | 100% | 3.5 | 2.41% | 1.85 |
| 39k | chain_typedflat_auto_x4 | 100% | 9.21 | 100% | 5.5 | 2.84% | 1.88 |
| 39k | chain_authflat_p2_x6 | 100% | 9.30 | 100% | 6.0 | 2.96% | 1.90 |

Interpretation:

```text
The absolute token budget stays almost flat while context length grows.
Therefore kept fraction improves with length:
  auto_x2: 9.10% at 10k, 4.64% at 20k, 2.41% at 39k.

For long semantic retrieval, remote_tail is not a useful substitute:
  it keeps remote tokens but never hits the bridge/answer pages in this setup.
```

### 39k tail-risk stress

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_para_conflict_39k_tailrisk_v5_range_sdpa
```

Setup:

```text
context_tokens = 39000
layouts = e05_d90,e20_d80
tasks_per_length = 4
modes = p2to4_x2, p2to6_x2, auto_x2, auto_x4, authflat_p2_x6
```

| Mode | Accuracy | PPL | Evidence coverage | Pages | Kept fraction | Eval sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chain_typedflat_p2to4_x2 | 87.5% | 10.85 | 87.5% | 3.125 | 2.30% | 1.96 |
| chain_typedflat_p2to6_x2 | 100% | 10.33 | 100% | 3.375 | 2.37% | 1.87 |
| chain_typedflat_auto_x2 | 100% | 10.33 | 100% | 3.375 | 2.37% | 1.87 |
| chain_typedflat_auto_x4 | 100% | 9.19 | 100% | 5.375 | 2.80% | 1.89 |
| chain_authflat_p2_x6 | 100% | 9.10 | 100% | 6.0 | 2.92% | 1.91 |

Failure sample for fixed p2to4:

```text
layout = e05_d90
target = A
decoy = B
evidence pages = 33, 302
decoy pages = 578, 276

p2to4_x2 selected:
  77, 232, 449, 578
  evidence_hit = 0
  typed_record_present = 0
  PPL = 13.49

p2to6_x2 / auto_x2 selected:
  33, 77, 232, 302, 449, 578
  evidence_hit = 1
  typed_record_present = 1
  PPL = 8.98

auto_x4 selected:
  33, 77, 155, 232, 276, 302, 449, 578
  PPL = 7.48
```

Conclusion:

```text
At 39k, p2to4 is usually enough but has a real tail failure mode.
The length-aware p2to6 ceiling removes that failure with tiny extra budget:
  +0.25 selected pages on average,
  +0.08 kept-fraction percentage points relative to p2to4,
  87.5% -> 100% accuracy on the 39k stress.

auto_x2 is the best current default:
  compute is close to p2to4,
  recall matches p2to6,
  and kept fraction continues to shrink as context length increases.

auto_x4 is the PPL-priority variant:
  it keeps about two more pages,
  usually lowers PPL by about 1 point,
  and remains much cheaper than dense/full context.
```

Next design direction:

```text
Replace the hard length thresholds with a confidence rule:
  keep increasing seed pages until a bridge record is found
  and the bridge page score clears a margin over decoy-like pages.

The current auto route is a length-aware approximation of that policy.
```

## 51. Confidence-style typed routing

Question:

```text
Can we remove the hard length thresholds in auto_x2?
```

New route aliases:

```text
chain_typedflat_conf_x2
chain_typedflat_conf_x4
chain_typedflat_conf_s10_x2  # optional max seed override
```

Policy:

```text
1. Start with 2 seed pages.
2. Try to extract the bridge artifact from those pages.
3. If no bridge artifact is found, increase seed pages one by one.
4. Stop as soon as the bridge artifact is found.
5. Default max seed pages = 8.
6. Route answer pages from the extracted artifact.
```

This removes the 20k/40k threshold from `auto_x2`. It is confidence-style because the route
spends more seed budget only when the typed bridge record is missing.

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_para_conflict_39k_conf_v6_range_sdpa
```

Setup:

```text
context_tokens = 39000
layouts = e05_d90,e20_d80
tasks_per_length = 4
task_variant = chain_para_conflict
sparse_attention_impl = range_sdpa
typed reader = extractive label-only sidecar
```

| Mode | Accuracy | PPL | Evidence coverage | Pages | Kept fraction | Eval sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chain_typedflat_p2to4_x2 | 87.5% | 10.85 | 87.5% | 3.125 | 2.30% | 1.95 |
| chain_typedflat_p2to6_x2 | 100% | 10.33 | 100% | 3.375 | 2.37% | 1.86 |
| chain_typedflat_auto_x2 | 100% | 10.33 | 100% | 3.375 | 2.37% | 1.86 |
| chain_typedflat_conf_x2 | 100% | 10.33 | 100% | 3.375 | 2.37% | 1.86 |
| chain_typedflat_conf_x4 | 100% | 9.19 | 100% | 5.375 | 2.80% | 1.88 |
| chain_authflat_p2_x6 | 100% | 9.10 | 100% | 6.0 | 2.92% | 1.90 |

Short-context no-regression output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_para_conflict_conf_10k20k_v7_range_sdpa
```

| Context | Mode | Accuracy | PPL | Evidence coverage | Pages | Kept fraction | Eval sec |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | chain_typedflat_auto_x2 | 100% | 11.45 | 100% | 3.0 | 9.10% | 1.82 |
| 10k | chain_typedflat_conf_x2 | 100% | 11.45 | 100% | 3.0 | 9.10% | 1.81 |
| 10k | chain_typedflat_conf_x4 | 100% | 9.89 | 100% | 5.0 | 10.74% | 1.86 |
| 20k | chain_typedflat_auto_x2 | 100% | 10.31 | 100% | 3.5 | 4.64% | 1.83 |
| 20k | chain_typedflat_conf_x2 | 100% | 10.31 | 100% | 3.5 | 4.64% | 1.83 |
| 20k | chain_typedflat_conf_x4 | 100% | 9.62 | 100% | 5.5 | 5.49% | 1.85 |

Interpretation:

```text
conf_x2 exactly matches p2to6/auto_x2 on the 39k stress set:
  same accuracy,
  same PPL,
  same selected pages,
  same kept fraction.

conf_x2 also matches auto_x2 on 10k and 20k:
  no short-context budget regression,
  no PPL regression,
  no coverage regression.

The difference is policy, not current metrics:
  auto_x2 uses context length to choose a seed ceiling;
  conf_x2 uses typed bridge presence to decide whether to keep searching.
```

Updated recommended routes:

```text
Default:
  chain_typedflat_conf_x2

PPL priority:
  chain_typedflat_conf_x4

Conservative fixed fallback:
  chain_typedflat_p2to6_x2 for near-40k
```

Design implication:

```text
The routing controller should not primarily ask:
  "How long is the context?"

It should ask:
  "Have I found the typed bridge record yet?"

Length still matters only as a soft prior for the maximum search budget.
The stop condition should be evidence of a usable semantic anchor.
```

## 52. Typed hierarchical confidence routing

Question:

```text
Can the book structure become genuinely hierarchical instead of only flat page routing?
```

New route aliases:

```text
chain_typedhier_conf_s1_p1
chain_typedhier_conf_s2_p1
chain_typedhier_conf_s2_p2
chain_typedhier_conf_s3_p1
```

Policy:

```text
1. Find the bridge artifact with the same confidence seed loop as conf_x2.
2. Score sections with an artifact query.
3. Inside selected sections, score pages with the same artifact query.
4. Keep top P pages from each of top S sections.
```

This is a conservative two-level design:

```text
bridge recall:
  still global page-level, because missing the bridge is catastrophic.

answer routing:
  hierarchical section -> page, because the artifact gives a strong semantic anchor.
```

### 39k stress

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_para_conflict_39k_typedhier_v8_range_sdpa
```

| Mode | Accuracy | PPL | Evidence coverage | Pages | Kept fraction | Eval sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chain_typedflat_conf_x2 | 100% | 10.33 | 100% | 3.375 | 2.37% | 1.87 |
| chain_typedflat_conf_x4 | 100% | 9.19 | 100% | 5.375 | 2.80% | 1.88 |
| chain_typedhier_conf_s1_p1 | 100% | 10.33 | 100% | 3.375 | 2.37% | 1.86 |
| chain_typedhier_conf_s2_p1 | 100% | 10.33 | 100% | 3.375 | 2.37% | 1.86 |
| chain_typedhier_conf_s2_p2 | 100% | 10.32 | 100% | 5.375 | 2.68% | 1.88 |
| chain_typedhier_conf_s3_p1 | 100% | 9.46 | 100% | 4.375 | 2.61% | 1.87 |
| chain_authflat_p2_x6 | 100% | 9.10 | 100% | 6.0 | 2.92% | 1.89 |

Example:

```text
task 3900000000

flat conf_x2:
  pages 33, 77, 232, 302, 449, 578
  PPL 8.98

flat conf_x4:
  pages 33, 77, 155, 232, 276, 302, 449, 578
  PPL 7.48

typedhier s3_p1:
  pages 33, 77, 232, 276, 302, 449, 578
  PPL 8.08
```

### 20k stress

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_para_conflict_20k_typedhier_v9_range_sdpa
```

| Mode | Accuracy | PPL | Evidence coverage | Pages | Kept fraction | Eval sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chain_typedflat_conf_x2 | 100% | 10.59 | 100% | 3.5 | 4.69% | 1.82 |
| chain_typedflat_conf_x4 | 100% | 9.68 | 100% | 5.5 | 5.55% | 1.85 |
| chain_typedhier_conf_s1_p1 | 100% | 10.59 | 100% | 3.5 | 4.69% | 1.82 |
| chain_typedhier_conf_s2_p2 | 100% | 10.58 | 100% | 5.5 | 5.29% | 1.85 |
| chain_typedhier_conf_s3_p1 | 100% | 9.91 | 100% | 4.5 | 5.21% | 1.84 |
| chain_authflat_p2_x6 | 100% | 9.21 | 87.5% hit / 93.75% coverage | 6.0 | 5.71% | 2.08 |

Interpretation:

```text
typedhier_s1_p1 is equivalent to flat_conf_x2 on these tasks.
The top artifact section contains the answer page, so one section and one page are enough.

typedhier_s3_p1 is the useful hierarchical PPL tradeoff:
  it keeps one page from each of three artifact-related sections,
  improves PPL versus conf_x2,
  and usually keeps fewer tokens than flat_conf_x4.

flat_conf_x4 still gives slightly better PPL in some cases,
but it spends budget as global extra pages.
typedhier_s3_p1 spends budget as section fanout:
  more breadth across related sections,
  only one page per section.
```

Updated route policy:

```text
Fast default:
  chain_typedflat_conf_x2
  or chain_typedhier_conf_s1_p1

PPL/quality tradeoff:
  chain_typedhier_conf_s3_p1

PPL-priority flat fallback:
  chain_typedflat_conf_x4
```

Design implication:

```text
The hierarchy should not replace semantic anchors.
It should organize how budget is spent after a semantic anchor exists.

For long-range semantic retrieval:
  page-level confidence seed finds the bridge;
  typed semantic anchor identifies the artifact;
  section fanout chooses a few related regions;
  page fanout keeps only the strongest page per region.

This is closer to a book workflow:
  find the index entry,
  jump to the relevant chapter/section,
  read one key page from several related sections.
```

## 53. Section granularity sweep

Question:

```text
How sensitive is typed hierarchical routing to section size?
```

Sweep:

```text
section_max_paragraphs = 4, 8, 16
context_tokens = 20000,39000
tasks_per_length = 2
task_variant = chain_para_conflict
modes = flat_conf_x2, typedhier_s1_p1, typedhier_s3_p1, typedhier_s5_p1
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_para_conflict_section_granularity_v10_range_sdpa
```

### section_max_paragraphs = 4

| Context | Mode | Accuracy | PPL | Evidence coverage | Pages | Kept fraction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 20k | flat_conf_x2 | 100% | 10.57 | 100% | 3.75 | 4.79% |
| 20k | typedhier_s1_p1 | 100% | 10.57 | 100% | 3.75 | 4.79% |
| 20k | typedhier_s3_p1 | 100% | 10.16 | 100% | 4.75 | 5.31% |
| 20k | typedhier_s5_p1 | 100% | 9.81 | 100% | 6.75 | 6.13% |
| 39k | flat_conf_x2 | 100% | 11.70 | 100% | 3.25 | 2.39% |
| 39k | typedhier_s1_p1 | 100% | 11.70 | 100% | 3.25 | 2.39% |
| 39k | typedhier_s3_p1 | 100% | 10.47 | 100% | 4.25 | 2.65% |
| 39k | typedhier_s5_p1 | 100% | 9.75 | 100% | 6.25 | 3.09% |

### section_max_paragraphs = 8

`section_max_paragraphs=8` produced the same selected-page pattern and metrics as section size 4
on this sweep. The artifact signal is strong enough that both granularities route to the same
answer sections.

### section_max_paragraphs = 16

| Context | Mode | Accuracy | PPL | Evidence coverage | Pages | Kept fraction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 20k | flat_conf_x2 | 100% | 10.57 | 100% | 3.75 | 4.79% |
| 20k | typedhier_s1_p1 | 75% | 11.07 | 87.5% | 3.5 | 4.69% |
| 20k | typedhier_s3_p1 | 100% | 10.16 | 100% | 4.75 | 5.31% |
| 20k | typedhier_s5_p1 | 100% | 9.58 | 100% | 6.75 | 6.16% |
| 39k | flat_conf_x2 | 100% | 11.70 | 100% | 3.25 | 2.39% |
| 39k | typedhier_s1_p1 | 100% | 11.70 | 100% | 3.25 | 2.39% |
| 39k | typedhier_s3_p1 | 100% | 10.47 | 100% | 4.25 | 2.65% |
| 39k | typedhier_s5_p1 | 100% | 9.94 | 100% | 6.25 | 3.10% |

Failure case for coarse sections:

```text
section_max_paragraphs = 16
context = 20k
layout = e20_d80
target = B
decoy = C
evidence pages = 67, 163
decoy pages = 262, 151

typedhier_s1_p1 selected:
  67, 262
  missing answer page 163
  predicted decoy C
  PPL 14.05

typedhier_s3_p1 selected:
  67, 151, 163, 262
  recovered answer page 163
  PPL 10.42

typedhier_s5_p1 selected:
  67, 118, 151, 163, 189, 262
  PPL 9.40
```

Interpretation:

```text
Fine/medium sections are stable:
  s1_p1 is enough because the top artifact section localizes the answer page.

Coarse sections make top-1 section/page brittle:
  the top section can contain both useful and misleading pages,
  and top-1 page selection can prefer a conflict/decoy page.

Section fanout fixes coarse-section brittleness:
  s3_p1 recovers the missed answer page with modest extra budget,
  s5_p1 buys more PPL at a larger token cost.
```

Updated hierarchy recommendation:

```text
Default section size:
  section_max_paragraphs = 8

Fast route:
  chain_typedhier_conf_s1_p1
  if section size is <= 8 and the task distribution is similar.

Robust/PPL route:
  chain_typedhier_conf_s3_p1
  especially when sections are coarse or documents are heterogeneous.

Avoid:
  coarse sections with s1_p1 only.
```

Design implication:

```text
The hierarchy has two independent knobs:
  section granularity controls localization noise;
  section fanout controls robustness and PPL.

A practical book router should make section fanout adaptive:
  if top section/page confidence is high, use s1_p1;
  if section is coarse or page margin is weak, expand to s3_p1;
  if PPL/quality is prioritized, expand further toward s5_p1.
```

## 54. Adaptive section fanout

Question:

```text
Can the hierarchical route avoid manually choosing s1 or s3?
```

New route:

```text
chain_typedhier_auto_p1
```

Policy:

```text
1. Use the same typed bridge confidence loop as chain_typedhier_conf.
2. Inspect the typical number of paragraph pages per section.
3. If typical section size <= 8 pages, use s1_p1.
4. If typical section size > 8 pages, use s3_p1.
```

This route targets a fast default, not PPL maximization. It keeps the cheap `s1_p1` behavior when
sections are fine/medium, and automatically increases section fanout when sections are coarse.

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_para_conflict_typedhier_auto_v11_range_sdpa
```

### section_max_paragraphs = 4

| Context | Mode | Accuracy | PPL | Evidence coverage | Pages | Kept fraction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 20k | typedhier_auto_p1 | 100% | 10.57 | 100% | 3.75 | 4.79% |
| 20k | typedhier_s1_p1 | 100% | 10.57 | 100% | 3.75 | 4.79% |
| 20k | typedhier_s3_p1 | 100% | 10.16 | 100% | 4.75 | 5.31% |
| 39k | typedhier_auto_p1 | 100% | 11.70 | 100% | 3.25 | 2.39% |
| 39k | typedhier_s1_p1 | 100% | 11.70 | 100% | 3.25 | 2.39% |
| 39k | typedhier_s3_p1 | 100% | 10.47 | 100% | 4.25 | 2.65% |

### section_max_paragraphs = 8

| Context | Mode | Accuracy | PPL | Evidence coverage | Pages | Kept fraction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 20k | typedhier_auto_p1 | 100% | 10.57 | 100% | 3.75 | 4.79% |
| 20k | typedhier_s1_p1 | 100% | 10.57 | 100% | 3.75 | 4.79% |
| 20k | typedhier_s3_p1 | 100% | 10.16 | 100% | 4.75 | 5.31% |
| 39k | typedhier_auto_p1 | 100% | 11.70 | 100% | 3.25 | 2.39% |
| 39k | typedhier_s1_p1 | 100% | 11.70 | 100% | 3.25 | 2.39% |
| 39k | typedhier_s3_p1 | 100% | 10.47 | 100% | 4.25 | 2.65% |

### section_max_paragraphs = 16

| Context | Mode | Accuracy | PPL | Evidence coverage | Pages | Kept fraction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 20k | typedhier_auto_p1 | 100% | 10.16 | 100% | 4.75 | 5.31% |
| 20k | typedhier_s1_p1 | 75% | 11.07 | 87.5% | 3.5 | 4.69% |
| 20k | typedhier_s3_p1 | 100% | 10.16 | 100% | 4.75 | 5.31% |
| 39k | typedhier_auto_p1 | 100% | 10.47 | 100% | 4.25 | 2.65% |
| 39k | typedhier_s1_p1 | 100% | 11.70 | 100% | 3.25 | 2.39% |
| 39k | typedhier_s3_p1 | 100% | 10.47 | 100% | 4.25 | 2.65% |

Interpretation:

```text
For section size 4 and 8:
  auto_p1 exactly matches s1_p1.
  It keeps the fast route and does not spend extra pages.

For section size 16:
  auto_p1 exactly matches s3_p1.
  It fixes the 20k s1_p1 failure and improves 39k PPL.
```

Updated default:

```text
Fast adaptive hierarchical route:
  chain_typedhier_auto_p1

Explicit PPL/quality route:
  chain_typedhier_conf_s3_p1
  or chain_typedhier_conf_s5_p1 when token budget allows.
```

Design implication:

```text
The router now has adaptive control at two levels:
  seed-page fanout adapts until a typed bridge exists;
  section fanout adapts to section granularity.

This is closer to the intended book index behavior:
  find the semantic index entry,
  choose chapter/section breadth based on section coarseness,
  then read only the best page per chosen section.
```

## 55. Margin-aware section fanout

Question:

```text
Can section fanout be controlled by score confidence instead of only section granularity?
```

New route family:

```text
chain_typedhier_margin_p1_m5
chain_typedhier_margin_p1_m10
chain_typedhier_margin_p1_m20
```

Policy:

```text
1. Find the typed bridge artifact.
2. Score sections with the artifact query.
3. Score pages in the top section with the artifact query.
4. Use s3_p1 if:
     typical section size > 8 pages, or
     top section score margin < threshold, or
     top page score margin < threshold.
5. Otherwise use s1_p1.
```

Threshold format:

```text
m5  = 0.05 score margin
m10 = 0.10 score margin
m20 = 0.20 score margin
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_para_conflict_typedhier_margin_v12_range_sdpa
```

### section_max_paragraphs = 8

| Context | Mode | Accuracy | PPL | Evidence coverage | Pages | Kept fraction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 20k | typedhier_auto_p1 | 100% | 10.57 | 100% | 3.75 | 4.79% |
| 20k | typedhier_s1_p1 | 100% | 10.57 | 100% | 3.75 | 4.79% |
| 20k | typedhier_s3_p1 | 100% | 10.16 | 100% | 4.75 | 5.31% |
| 20k | margin_m5/m10/m20 | 100% | 10.57 | 100% | 3.75 | 4.79% |
| 39k | typedhier_auto_p1 | 100% | 11.70 | 100% | 3.25 | 2.39% |
| 39k | typedhier_s1_p1 | 100% | 11.70 | 100% | 3.25 | 2.39% |
| 39k | typedhier_s3_p1 | 100% | 10.47 | 100% | 4.25 | 2.65% |
| 39k | margin_m5/m10/m20 | 100% | 11.70 | 100% | 3.25 | 2.39% |

### section_max_paragraphs = 16

| Context | Mode | Accuracy | PPL | Evidence coverage | Pages | Kept fraction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 20k | typedhier_auto_p1 | 100% | 10.16 | 100% | 4.75 | 5.31% |
| 20k | typedhier_s1_p1 | 75% | 11.07 | 87.5% | 3.5 | 4.69% |
| 20k | typedhier_s3_p1 | 100% | 10.16 | 100% | 4.75 | 5.31% |
| 20k | margin_m5/m10/m20 | 100% | 10.16 | 100% | 4.75 | 5.31% |
| 39k | typedhier_auto_p1 | 100% | 10.47 | 100% | 4.25 | 2.65% |
| 39k | typedhier_s1_p1 | 100% | 11.70 | 100% | 3.25 | 2.39% |
| 39k | typedhier_s3_p1 | 100% | 10.47 | 100% | 4.25 | 2.65% |
| 39k | margin_m5/m10/m20 | 100% | 10.47 | 100% | 4.25 | 2.65% |

Interpretation:

```text
On this synthetic task, the margin thresholds m5/m10/m20 did not add behavior beyond
the section-granularity rule:

section size 8:
  margins were confident enough, so margin routes stayed at s1_p1.

section size 16:
  coarse-section guard triggered, so margin routes matched s3_p1.
```

Conclusion:

```text
Margin-aware routing is implemented and safe on this suite,
but this run does not prove it is better than chain_typedhier_auto_p1.

Current default should remain:
  chain_typedhier_auto_p1

Keep margin routes for future harder data:
  chain_typedhier_margin_p1_m10
  chain_typedhier_margin_p1_m20
```

Design implication:

```text
The synthetic artifact signal is strong, so section/page score margins are not yet the limiting
factor. The observed failure was explained by section coarseness, and the granularity rule already
captures that.

Margin control should be revisited on less templated long-range semantic tasks where:
  entity names are ambiguous,
  sections contain multiple competing artifacts,
  and no hard artifact wording dominates the lexical score.
```

## 56. Less-template story chain task

Question:

```text
What happens when the long-range semantic task removes the strong artifact/certified wording?
```

New task variant:

```text
chain_story_conflict
```

Task structure:

```text
bridge page:
  badge KEY is logged under river-name ALIAS.

answer page:
  resolution memo says river-name ALIAS closes with option LABEL.

decoy page:
  old desk slip repeats badge KEY with a wrong option, but is withdrawn.

conflict page:
  earlier ruling note for ALIAS leaned toward the wrong option, but is superseded.
```

This keeps the same controllable bridge -> answer -> conflict structure, but avoids repeated
`artifact`, `certified`, and `approved response` keywords.

Implementation changes:

```text
Router:
  extracts badge -> river-name aliases;
  scores resolution memo / current ruling pages;
  penalizes old desk slip / withdrawn / earlier ruling pages.

Typed reader / verifier:
  extracts closes-with-option labels;
  filters withdrawn and earlier-ruling notes.
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_conflict_20k39k_v13_range_sdpa
```

Setup:

```text
context_tokens = 20000,39000
tasks_per_length = 2
layouts = e05_d90,e20_d80
section_max_paragraphs = 8
modes = sink_recent, remote_tail_p4, flat_conf_x2, typedhier_auto_p1, typedhier_s3_p1, margin_m10
```

### Results

| Context | Mode | Accuracy | PPL | Evidence coverage | Record coverage | Pages | Kept fraction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20k | sink_recent | 25% | 172.96 | 0% | 0% | 0.0 | 2.88% |
| 20k | remote_tail_p4 | 0% | 173.40 | 0% | 0% | 4.0 | 4.07% |
| 20k | flat_conf_x2 | 100% | 20.22 | 100% | 100% | 5.25 | 5.29% |
| 20k | typedhier_auto_p1 | 100% | 20.22 | 100% | 100% | 5.25 | 5.29% |
| 20k | typedhier_s3_p1 | 100% | 20.55 | 100% | 100% | 6.25 | 5.80% |
| 20k | margin_m10 | 100% | 20.22 | 100% | 100% | 5.25 | 5.29% |
| 39k | sink_recent | 0% | 176.55 | 0% | 0% | 0.0 | 1.48% |
| 39k | remote_tail_p4 | 0% | 174.25 | 0% | 0% | 4.0 | 2.07% |
| 39k | flat_conf_x2 | 100% | 16.99 | 100% | 100% | 6.5 | 2.81% |
| 39k | typedhier_auto_p1 | 100% | 17.03 | 100% | 100% | 6.0 | 2.69% |
| 39k | typedhier_s3_p1 | 100% | 16.97 | 100% | 100% | 7.0 | 2.92% |
| 39k | margin_m10 | 100% | 17.03 | 100% | 100% | 6.0 | 2.69% |

Example selected pages:

```text
20k task 2000000000

flat_conf_x2 / auto_p1:
  selected pages = 17, 40, 120, 156, 231, 297
  evidence pages = 17, 156
  decoy pages = 297, 143
  PPL = 20.62

typedhier_s3_p1:
  selected pages = 17, 40, 120, 143, 156, 231, 297
  added page 143, a decoy/old-slip page
  PPL = 21.28
```

Interpretation:

```text
The typed anchor mechanism still works after removing hard artifact/certified wording:
  typed routes recover 100% evidence coverage,
  typed record coverage stays 100%,
  remote_tail and sink/recent still fail.

But the task is harder:
  PPL rises to about 17-20 instead of the 9-11 range from the templated artifact task.

Blind section fanout is less reliable:
  s3_p1 can add conflict/old-slip pages,
  so PPL can worsen even when downstream accuracy remains 100%.
```

Design implication:

```text
For less-template semantic retrieval, hierarchy needs typed page roles, not only section fanout.

The next router should distinguish:
  bridge pages,
  current-ruling / answer pages,
  withdrawn / superseded / conflict pages,
  unrelated distractor pages.

Then section fanout should keep answer-like pages and optionally include conflict pages only as
negative evidence, not as ordinary context.
```

Updated recommendation:

```text
Templated artifact task:
  typedhier_auto_p1 is a good fast default;
  typedhier_s3_p1 is a useful PPL/quality tradeoff.

Less-template story task:
  typedhier_auto_p1 remains the best fast default;
  do not blindly expand to s3_p1 unless page-role filtering is added.
```

## 57. Role-filtered page routing

Question:

```text
Can typed page-role filtering fix the story-task problem where s3_p1 adds old-slip/conflict pages?
```

New route aliases:

```text
chain_typedhier_role_auto_p1
chain_typedhier_role_s3_p1
```

Policy:

```text
1. Find the typed bridge as before.
2. Keep bridge-like seed pages only if they contain the current alias/key and are not negative pages.
3. During section fanout, prefer answer-like pages:
     certified/current/resolution/approved pages.
4. Exclude negative pages:
     old desk slip, withdrawn, earlier ruling, superseded, obsolete, outdated.
5. Fall back to neutral pages only when no answer-like page exists in a selected section.
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_rolefilter_20k39k_v14_range_sdpa
```

Setup:

```text
task_variant = chain_story_conflict
context_tokens = 20000,39000
tasks_per_length = 2
section_max_paragraphs = 8
```

### Summary

| Context | Mode | Accuracy | PPL | Evidence coverage | Decoy hit | Pages | Kept fraction | Eval sec |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20k | typedhier_auto_p1 | 100% | 20.22 | 100% | 75% | 5.25 | 5.29% | 1.86 |
| 20k | typedhier_s3_p1 | 100% | 20.55 | 100% | 100% | 6.25 | 5.80% | 1.84 |
| 20k | typedhier_role_auto_p1 | 100% | 22.96 | 100% | 0% | 2.0 | 3.85% | 1.79 |
| 20k | typedhier_role_s3_p1 | 100% | 22.86 | 100% | 0% | 3.0 | 4.14% | 1.80 |
| 39k | typedhier_auto_p1 | 100% | 17.03 | 100% | 100% | 6.0 | 2.69% | 1.89 |
| 39k | typedhier_s3_p1 | 100% | 16.97 | 100% | 100% | 7.0 | 2.92% | 1.91 |
| 39k | typedhier_role_auto_p1 | 100% | 23.09 | 100% | 0% | 2.0 | 1.84% | 1.83 |
| 39k | typedhier_role_s3_p1 | 100% | 23.34 | 100% | 0% | 3.0 | 2.00% | 1.83 |

Example:

```text
20k task 2000000000

typedhier_auto_p1:
  selected = 17, 40, 120, 156, 231, 297
  decoy_hit = 1
  PPL = 20.62

typedhier_s3_p1:
  selected = 17, 40, 120, 143, 156, 231, 297
  decoy_hit = 1
  PPL = 21.28

typedhier_role_auto_p1:
  selected = 17, 156
  decoy_hit = 0
  PPL = 22.84

typedhier_role_s3_p1:
  selected = 17, 137, 156
  decoy_hit = 0
  PPL = 22.72
```

Interpretation:

```text
Role filtering works for information cleanliness:
  decoy hit falls from 75-100% to 0%;
  selected pages fall from 5-7 to 2-3;
  eval time improves slightly;
  downstream accuracy stays 100% because the typed reader still sees bridge + answer.

But role filtering hurts query PPL:
  PPL rises from 17-20 to about 23.
```

This exposes a real tradeoff:

```text
The cleanest evidence path is not necessarily the best LM context.

For downstream answer accuracy:
  bridge + current answer page is enough.

For PPL:
  the model benefits from broader topical/background pages,
  even when some of those pages are decoy or conflict pages that the typed reader must ignore.
```

Updated design:

```text
Separate the memory budget into two lanes:

1. Typed evidence lane:
   bridge pages + current answer pages;
   role-filtered;
   used for downstream answer routing and sidecar records.

2. Context/PPL lane:
   a small number of topical neighbor/section pages;
   allowed to include neutral background;
   conflict pages should be compressed or tagged, not blindly inserted as raw context.
```

Current recommendation:

```text
For answer accuracy and speed:
  chain_typedhier_role_auto_p1

For balanced PPL + accuracy:
  chain_typedhier_auto_p1

Avoid using role filtering alone as the PPL route.
```

## 58. Role + seed-context attempt

Question:

```text
Can we keep the clean role-filtered evidence path while recovering PPL by also keeping
non-negative pages from the bridge seed set?
```

New routes:

```text
chain_typedhier_rolectx_auto_p1
chain_typedhier_rolectx_s3_p1
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_rolectx_20k39k_v15_range_sdpa
```

### Summary

| Context | Mode | Accuracy | PPL | Evidence coverage | Decoy hit | Pages | Kept fraction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20k | typedhier_auto_p1 | 100% | 20.22 | 100% | 75% | 5.25 | 5.29% |
| 20k | typedhier_role_auto_p1 | 100% | 22.96 | 100% | 0% | 2.0 | 3.85% |
| 20k | typedhier_rolectx_auto_p1 | 100% | 22.96 | 100% | 0% | 2.0 | 3.85% |
| 20k | typedhier_role_s3_p1 | 100% | 22.86 | 100% | 0% | 3.0 | 4.14% |
| 20k | typedhier_rolectx_s3_p1 | 100% | 22.86 | 100% | 0% | 3.0 | 4.14% |
| 39k | typedhier_auto_p1 | 100% | 17.03 | 100% | 100% | 6.0 | 2.69% |
| 39k | typedhier_role_auto_p1 | 100% | 23.09 | 100% | 0% | 2.0 | 1.84% |
| 39k | typedhier_rolectx_auto_p1 | 100% | 23.09 | 100% | 0% | 2.0 | 1.84% |
| 39k | typedhier_role_s3_p1 | 100% | 23.34 | 100% | 0% | 3.0 | 2.00% |
| 39k | typedhier_rolectx_s3_p1 | 100% | 23.34 | 100% | 0% | 3.0 | 2.00% |

Interpretation:

```text
This route did not actually add a useful context lane.

Reason:
  the bridge seed set is optimized for finding the key/alias/artifact;
  once role filtering is applied, those seeds collapse back to bridge-like evidence pages.

So "context" must be selected independently from the bridge seed loop.
```

## 59. Independent global context lane

Question:

```text
Can a separate global semantic context lane recover PPL while preserving 0% decoy hit?
```

New routes:

```text
chain_typedhier_rolectxflat_auto_p1_c2
chain_typedhier_rolectxflat_auto_p1_c4
chain_typedhier_rolectxart_auto_p1_c2
chain_typedhier_rolectxart_auto_p1_c4
```

Definitions:

```text
rolectxflat:
  evidence lane = role-filtered bridge + current answer pages
  context lane = top-N non-negative global pages from the original query

rolectxart:
  evidence lane = role-filtered bridge + current answer pages
  context lane = top-N non-negative global pages from the discovered alias/artifact
```

Outputs:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_rolectxflat_20k39k_v16_range_sdpa
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_rolectxart_20k39k_v17_range_sdpa
```

### Summary

| Context | Mode | Accuracy | PPL | Evidence coverage | Decoy hit | Pages | Kept fraction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20k | typedhier_auto_p1 | 100% | 18.74 | 100% | 100% | 6.0 | 5.57% |
| 20k | typedhier_role_auto_p1 | 100% | 21.86 | 100% | 0% | 2.0 | 3.81% |
| 20k | rolectxflat_c2 | 100% | 21.72 | 100% | 0% | 4.0 | 4.63% |
| 20k | rolectxflat_c4 | 100% | 21.81 | 100% | 0% | 6.0 | 5.22% |
| 20k | rolectxart_c2 | 100% | 21.76 | 100% | 0% | 4.0 | 4.55% |
| 20k | rolectxart_c4 | 100% | 22.61 | 100% | 0% | 6.0 | 5.19% |
| 39k | typedhier_auto_p1 | 100% | 16.24 | 100% | 100% | 6.0 | 2.72% |
| 39k | typedhier_role_auto_p1 | 100% | 19.40 | 100% | 0% | 2.0 | 1.84% |
| 39k | rolectxflat_c2 | 100% | 19.36 | 100% | 0% | 4.0 | 2.22% |
| 39k | rolectxflat_c4 | 100% | 19.24 | 100% | 0% | 6.0 | 2.58% |
| 39k | rolectxart_c2 | 100% | 18.87 | 100% | 0% | 4.0 | 2.19% |
| 39k | rolectxart_c4 | 100% | 18.38 | 100% | 0% | 6.0 | 2.52% |

Interpretation:

```text
Independent context selection works mechanically:
  selected pages increase from 2 to 4/6;
  decoy hit remains 0%;
  evidence coverage and accuracy remain 100%.

But the PPL gain is limited:
  query-only global context barely helps;
  artifact-conditioned global context helps more at 39k,
  but is unstable at 20k and can worsen when c4 adds weakly related pages.
```

Example:

```text
20k e05 task 2000000000

auto_p1:
  pages = 17, 40, 120, 156, 231, 297
  decoy_hit = 1
  PPL = 20.46

role_auto_p1:
  pages = 17, 156
  decoy_hit = 0
  PPL = 23.35

rolectxart_c4:
  pages = 17, 46, 106, 156, 166, 270
  decoy_hit = 0
  PPL = 25.57

39k e05 task 3900000001

role_auto_p1:
  pages = 33, 303
  PPL = 16.40

rolectxart_c4:
  pages = 33, 59, 66, 155, 303, 529
  PPL = 15.65
```

Design implication:

```text
Global semantic context is too noisy without a learned or better structured page summary.

For long semantic retrieval, context lane should not be just "top lexical pages".
It needs either:
  learned summaries/embeddings,
  typed page summaries,
  or constraints from the hierarchical section structure.
```

## 60. Section-local context lane

Question:

```text
Can a hierarchical section-local context lane recover the useful context pages from auto_p1
without keeping conflict/decoy pages?
```

New routes:

```text
chain_typedhier_rolectxsec_auto_p1_c2
chain_typedhier_rolectxsec_auto_p1_c4
```

Policy:

```text
1. Evidence lane keeps role-filtered bridge + current answer pages.
2. Use the discovered alias/artifact to rank sections.
3. Add top-N non-negative, non-answer pages from the selected sections.
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_rolectxsec_20k39k_v18_range_sdpa
```

### Summary

| Context | Mode | Accuracy | PPL | Evidence coverage | Decoy hit | Pages | Kept fraction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20k | typedhier_auto_p1 | 100% | 18.74 | 100% | 100% | 6.0 | 5.57% |
| 20k | typedhier_role_auto_p1 | 100% | 21.86 | 100% | 0% | 2.0 | 3.81% |
| 20k | rolectxsec_c2 | 100% | 21.78 | 100% | 0% | 4.0 | 4.41% |
| 20k | rolectxsec_c4 | 100% | 21.67 | 100% | 0% | 6.0 | 5.01% |
| 39k | typedhier_auto_p1 | 100% | 16.24 | 100% | 100% | 6.0 | 2.72% |
| 39k | typedhier_role_auto_p1 | 100% | 19.40 | 100% | 0% | 2.0 | 1.84% |
| 39k | rolectxsec_c2 | 100% | 19.50 | 100% | 0% | 4.0 | 2.15% |
| 39k | rolectxsec_c4 | 100% | 19.64 | 100% | 0% | 6.0 | 2.45% |

Example:

```text
20k e05 task 2000000000

auto_p1:
  pages = 17, 40, 120, 156, 231, 297
  PPL = 20.46

rolectxsec_c4:
  pages = 17, 152, 154, 156, 157, 158
  PPL = 23.10

39k e05 task 3900000000

auto_p1:
  pages = 33, 78, 233, 303, 451, 580
  PPL = 18.82

rolectxsec_c4:
  pages = 33, 297, 300, 301, 302, 303
  PPL = 22.33
```

Interpretation:

```text
Section-local context is clean but too local:
  it mostly adds pages adjacent to the answer page,
  not the broader topical pages that help LM PPL.

So the useful PPL context in auto_p1 is not simply "near the answer page".
It is broader, cross-section topic context, even though the same route also pulls in conflict pages.
```

Updated design conclusion:

```text
Typed evidence lane is solved for this synthetic story task:
  bridge + current answer page gives 100% evidence coverage and 100% accuracy with 0% decoy hit.

The unsolved part is the PPL/context lane:
  auto_p1 gives best PPL but includes conflict pages;
  role-only is clean and fast but hurts PPL;
  global lexical context is noisy;
  section-local context is too narrow.

The next useful design is a typed context page summary:
  keep broad topic pages,
  but insert them as compressed/tagged summaries,
  and explicitly mark conflict/withdrawn pages as non-current instead of dropping them or inserting raw text.
```

## 61. Typed summary context lane

Question:

```text
Can we keep raw attention clean and still recover PPL by inserting a compressed,
typed summary of broader context pages?
```

Implementation:

```text
--typed_record_format summary
--typed_summary_source_mode <route>
```

This separates two page sets:

```text
raw sparse pages:
  pages the LM can attend to in the original long context.

typed summary source pages:
  pages read by the synthetic summarizer and compressed into a short record
  inserted before the query.
```

For the main clean route:

```text
raw pages = chain_typedhier_role_auto_p1
  bridge + current answer only
  decoy_hit = 0

summary pages = chain_typedhier_auto_p1 or chain_typedhier_conf_s3_p1
  broader topic pages
  conflict/withdrawn pages are tagged as status=non_current
```

Outputs:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_typedsummary_20k39k_v19_range_sdpa
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_typedsummary_compact_20k39k_v20_range_sdpa
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_typedsummary_s3source_20k39k_v21_range_sdpa
```

### Long summary attempt

The first summary format kept too many background bridge lines:

| Context | Mode | Summary source | Accuracy | PPL | Record tokens | Raw pages | Decoy hit | Eval sec |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20k | auto_p1 | auto_p1 | 100% | 17.04 | 319 | 6.0 | 100% | 14.75 |
| 20k | role_auto_p1 | auto_p1 | 100% | 21.10 | 319 | 2.0 | 0% | 14.28 |
| 39k | auto_p1 | auto_p1 | 100% | 13.81 | 320 | 6.0 | 100% | 14.87 |
| 39k | role_auto_p1 | auto_p1 | 100% | 18.19 | 320 | 2.0 | 0% | 14.33 |

Interpretation:

```text
PPL improves, but the record is too long.
Average eval time rises from about 4s to about 14s.
```

### Compressed summary

The summary was then compressed to keep only:

```text
target bridge,
current ruling,
withdrawn badge note,
superseded alias note if present.
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_typedsummary_compact_20k39k_v20_range_sdpa
```

Results:

| Context | Mode | Summary source | Accuracy | PPL | Record tokens | Raw pages | Decoy hit | Eval sec | Kept fraction |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20k | auto_p1 | auto_p1 | 100% | 16.97 | 177 | 6.0 | 100% | 9.32 | 5.55% |
| 20k | role_auto_p1 | auto_p1 | 100% | 19.63 | 177 | 2.0 | 0% | 8.74 | 3.80% |
| 39k | auto_p1 | auto_p1 | 100% | 14.13 | 178 | 6.0 | 100% | 9.38 | 2.72% |
| 39k | role_auto_p1 | auto_p1 | 100% | 17.11 | 178 | 2.0 | 0% | 8.81 | 1.84% |

Compared with compact label-only typed memory from Section 60:

```text
20k role_auto_p1:
  PPL 21.86 -> 19.63
  raw pages stay 2.0
  decoy_hit stays 0%

39k role_auto_p1:
  PPL 19.40 -> 17.11
  raw pages stay 2.0
  decoy_hit stays 0%
```

Example summary:

```text
Typed memory summary: lookup_key=LR2000000000-TJFGVPBA; BRIDGE_ALIAS=RIVER-Y45JVZ; ANSWER_LABEL=C.
- page=17; role=bridge; lookup_key=LR2000000000-TJFGVPBA; alias=RIVER-Y45JVZ; status=route_only
- page=156; role=current_ruling; alias=RIVER-Y45JVZ; ANSWER_LABEL=C; status=current
- page=297; role=withdrawn_badge_note; lookup_key=LR2000000000-TJFGVPBA; option=A; status=non_current
Rule: answer only from status=current; ignore status=non_current as answers.
```

### S3-source summary

Then the summary source was expanded to `chain_typedhier_conf_s3_p1`, which often includes the
superseded alias page as well as the withdrawn badge note.

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_typedsummary_s3source_20k39k_v21_range_sdpa
```

Results:

| Context | Mode | Summary source | Accuracy | PPL | Record tokens | Raw pages | Decoy hit | Eval sec |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20k | role_auto_p1 | s3_p1 | 100% | 19.22 | 209 | 2.0 | 0% | 10.02 |
| 39k | role_auto_p1 | s3_p1 | 100% | 17.12 | 211 | 2.0 | 0% | 10.30 |

Example:

```text
Typed memory summary: lookup_key=LR2000000000-TJFGVPBA; BRIDGE_ALIAS=RIVER-Y45JVZ; ANSWER_LABEL=C.
- page=17; role=bridge; lookup_key=LR2000000000-TJFGVPBA; alias=RIVER-Y45JVZ; status=route_only
- page=143; role=superseded_alias_note; alias=RIVER-Y45JVZ; option=A; status=non_current
- page=156; role=current_ruling; alias=RIVER-Y45JVZ; ANSWER_LABEL=C; status=current
- page=297; role=withdrawn_badge_note; lookup_key=LR2000000000-TJFGVPBA; option=A; status=non_current
Rule: answer only from status=current; ignore status=non_current as answers.
```

Interpretation:

```text
This is the first route that cleanly separates the roles:

1. Raw evidence lane:
   bridge + current answer page only.
   It keeps raw remote attention small and avoids raw decoy/conflict exposure.

2. Typed summary context lane:
   broader pages are allowed,
   but conflict/withdrawn pages are converted into status=non_current facts.

This recovers a large part of the PPL lost by role filtering while preserving:
  100% accuracy,
  100% evidence coverage,
  0% raw decoy hit,
  about 2 raw remote pages.
```

Current recommendation:

```text
Best PPL:
  auto_p1 + typed summary
  but raw decoy pages are still present.

Best clean tradeoff:
  raw route = chain_typedhier_role_auto_p1
  summary source = chain_typedhier_auto_p1
  typed_record_format = summary

More complete conflict explanation:
  raw route = chain_typedhier_role_auto_p1
  summary source = chain_typedhier_conf_s3_p1
  It improves 20k PPL slightly but costs more summary tokens.
```

Design update:

```text
The "book page" method should not only retrieve pages.
It should retrieve page roles and page summaries.

A practical architecture is:
  sink + recent raw tokens,
  clean typed evidence raw pages,
  compressed typed summaries for broader context/conflict pages.

This is closer to the target:
  compute is controlled by a small raw evidence lane,
  PPL is helped by summaries,
  downstream answer is protected by status=current / status=non_current typing.
```

## 62. Summary compression and answer stability

Question:

```text
Can the typed summary be shortened without losing PPL or downstream answer stability?
```

The previous `summary` format worked, but was still expensive:

```text
~177-211 inserted record tokens,
~8.7-10.3 sec eval time,
100% model-side answer accuracy.
```

This section tests shorter formats on the clean raw route:

```text
raw route = chain_typedhier_role_auto_p1
raw pages = bridge + current answer only
raw decoy hit = 0%
summary source = chain_typedhier_auto_p1 or chain_typedhier_conf_s3_p1
```

Outputs:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_minisummary_autosource_20k39k_v22_range_sdpa
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_minisummary_s3source_20k39k_v23_range_sdpa
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_shortsummary_autosource_20k39k_v24_range_sdpa
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_shortsummary_s3source_20k39k_v25_range_sdpa
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_litesummary_autosource_20k39k_v26_range_sdpa
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_litesummary_s3source_20k39k_v27_range_sdpa
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_naturalsummary_autosource_20k39k_v28_range_sdpa
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_naturalsummary_s3source_20k39k_v29_range_sdpa
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_answerlinesummary_autosource_20k39k_v30_range_sdpa
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_answerlinesummary_s3source_20k39k_v31_range_sdpa
```

### Formats

`mini_summary`:

```text
Typed memory mini: key=...; alias=...; current=C; withdrawn_noncurrent=A; rule=current_only.
```

`short_summary`:

```text
Typed memory summary: badge ... routes to river-name ...
The current ruling for ... is option C.
Old badge option A is withdrawn and non-current.
Answer from the current ruling only.
```

`lite_summary`:

```text
Typed memory lite: lookup_key=...; BRIDGE_ALIAS=...; ANSWER_LABEL=C; status=current;
withdrawn_badge_option=A; status=non_current; rule=use_current_status_only.
```

`natural_summary`:

```text
The current ruling for ... is option C (ANSWER_LABEL=C; status=current).
```

`answerline_summary`:

```text
Typed memory summary: ANSWER_LABEL=C; status=current.
Badge ... routes to river-name ...; current ruling for that river-name is option C.
Withdrawn badge option A has status=non_current.
Use the current status only.
```

### Results

| Context | Format | Source | PPL | Record tokens | Eval sec | Model acc | Cal model acc | Raw pages | Raw decoy hit |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20k | mini | auto | 21.58 | 46 | 4.84 | 25% | 50% | 2.0 | 0% |
| 20k | short | auto | 17.67 | 67 | 5.71 | 50% | 50% | 2.0 | 0% |
| 20k | lite | auto | 21.15 | 63 | 5.01 | 100% | 100% | 2.0 | 0% |
| 20k | natural | auto | 19.16 | 76 | 5.23 | 25% | 50% | 2.0 | 0% |
| 20k | answerline | auto | 17.16 | 70 | 5.52 | 100% | 100% | 2.0 | 0% |
| 20k | summary | auto | 19.63 | 177 | 8.74 | 100% | 100% | 2.0 | 0% |
| 20k | short | s3 | 17.65 | 84 | 6.29 | 50% | 50% | 2.0 | 0% |
| 20k | lite | s3 | 20.64 | 74 | 5.39 | 100% | 100% | 2.0 | 0% |
| 20k | natural | s3 | 19.20 | 95 | 6.18 | 50% | 75% | 2.0 | 0% |
| 20k | answerline | s3 | 17.54 | 88 | 6.24 | 100% | 100% | 2.0 | 0% |
| 20k | summary | s3 | 19.22 | 209 | 10.02 | 100% | 100% | 2.0 | 0% |
| 39k | mini | auto | 19.58 | 47 | 4.62 | 75% | 0% | 2.0 | 0% |
| 39k | short | auto | 15.24 | 68 | 6.05 | 50% | 50% | 2.0 | 0% |
| 39k | lite | auto | 19.13 | 64 | 4.96 | 100% | 100% | 2.0 | 0% |
| 39k | natural | auto | 15.10 | 77 | 5.81 | 100% | 100% | 2.0 | 0% |
| 39k | answerline | auto | 14.74 | 71 | 5.27 | 100% | 100% | 2.0 | 0% |
| 39k | summary | auto | 17.11 | 178 | 8.81 | 100% | 100% | 2.0 | 0% |
| 39k | short | s3 | 14.76 | 86 | 6.66 | 75% | 50% | 2.0 | 0% |
| 39k | lite | s3 | 17.84 | 75 | 5.44 | 100% | 100% | 2.0 | 0% |
| 39k | natural | s3 | 15.49 | 97 | 6.27 | 100% | 100% | 2.0 | 0% |
| 39k | answerline | s3 | 14.68 | 89 | 6.03 | 100% | 100% | 2.0 | 0% |
| 39k | summary | s3 | 17.12 | 211 | 10.30 | 100% | 100% | 2.0 | 0% |

Interpretation:

```text
The record has two different jobs:

1. Lower query PPL:
   natural language helps most.
   short_summary gives very low PPL, but model answer accuracy is unstable.

2. Preserve downstream answer stability:
   explicit ANSWER_LABEL=...; status=current is important.
   lite_summary is stable but too field-like, so PPL is worse.
```

The best balance is `answerline_summary`:

```text
It puts the answer anchor first:
  ANSWER_LABEL=C; status=current.

Then it gives a short natural sentence for the route:
  badge -> river-name -> current ruling.

Then it tags conflicts as non_current.
```

Why it works:

```text
Compared with short_summary:
  keeps almost the same PPL benefit,
  but fixes model-side answer accuracy.

Compared with lite_summary:
  keeps ANSWER_LABEL/status anchors,
  but uses natural language enough to reduce PPL.

Compared with full summary:
  much fewer tokens and faster,
  while preserving model-side answer accuracy in this small run.
```

Current best clean route:

```text
raw route:
  chain_typedhier_role_auto_p1

typed_record_format:
  answerline_summary

summary source:
  chain_typedhier_auto_p1 for the fastest clean default;
  chain_typedhier_conf_s3_p1 if explicit superseded-alias context is desired.
```

Current best clean metrics:

```text
answerline_summary + auto source:
  20k: PPL 17.16, record 70 tokens, eval 5.52s, model acc 100%, raw decoy 0%
  39k: PPL 14.74, record 71 tokens, eval 5.27s, model acc 100%, raw decoy 0%

answerline_summary + s3 source:
  20k: PPL 17.54, record 88 tokens, eval 6.24s, model acc 100%, raw decoy 0%
  39k: PPL 14.68, record 89 tokens, eval 6.03s, model acc 100%, raw decoy 0%
```

Updated architecture:

```text
1. Structural page routing:
   use structural anchors to cut pages and find bridge/current pages.

2. Clean raw evidence lane:
   keep only bridge + current answer pages in raw remote attention.

3. Typed answerline summary lane:
   read broader pages,
   summarize them as current/non_current facts,
   insert a short answerline summary before the query.

This is now closer to the desired target:
  raw KV compute is small,
  PPL is competitive with broader raw retrieval,
  downstream answer stability is preserved,
  conflict/decoy pages are not exposed as raw attention context.
```

## 63. Answerline stability and production speed

Question:

```text
Does the best clean route remain stable on more tasks and at 10k/20k/39k?
Can production mode skip LM option scoring once the typed summary has an answer?
```

Best clean route:

```text
raw route = chain_typedhier_role_auto_p1
typed_record_format = answerline_summary
typed_summary_source_mode = chain_typedhier_auto_p1
sparse_attention_impl = range_sdpa
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_answerline_stability_10k20k39k_v32_range_sdpa
```

Setup:

```text
context_tokens = 10000,20000,39000
tasks_per_length = 4
layouts = e05_d90,e20_d80
total tasks per length = 8
modes = auto_p1, role_auto_p1, s3_p1
skip_lm_answer_when_override = false
```

### Stability results

| Context | Mode | Accuracy | Model acc | Cal model acc | PPL | Record tokens | Raw pages | Evidence coverage | Raw decoy hit | Eval sec | Kept fraction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | auto_p1 | 100% | 100% | 100% | 18.52 | 70 | 5.75 | 100% | 100% | 5.31 | 10.92% |
| 10k | s3_p1 | 100% | 100% | 100% | 18.95 | 70 | 6.75 | 100% | 100% | 5.35 | 11.71% |
| 10k | role_auto_p1 | 100% | 100% | 100% | 18.44 | 70 | 2.0 | 100% | 0% | 5.27 | 7.54% |
| 20k | auto_p1 | 100% | 100% | 100% | 17.95 | 70 | 6.0 | 100% | 100% | 5.35 | 5.51% |
| 20k | s3_p1 | 100% | 100% | 100% | 19.32 | 70 | 7.0 | 100% | 100% | 5.37 | 6.02% |
| 20k | role_auto_p1 | 100% | 100% | 100% | 18.16 | 70 | 2.0 | 100% | 0% | 5.15 | 3.73% |
| 39k | auto_p1 | 100% | 100% | 100% | 12.85 | 72 | 5.75 | 100% | 100% | 5.93 | 2.69% |
| 39k | s3_p1 | 100% | 100% | 100% | 13.01 | 72 | 6.75 | 100% | 100% | 5.93 | 2.89% |
| 39k | role_auto_p1 | 100% | 100% | 100% | 13.09 | 72 | 2.0 | 100% | 0% | 5.66 | 1.89% |

Interpretation:

```text
The answerline route is stable in this larger small-scale run:
  final accuracy = 100%;
  model-side answer accuracy = 100%;
  calibrated model-side answer accuracy = 100%;
  evidence coverage = 100%;
  raw decoy hit = 0% for role_auto_p1;
  raw selected pages stay at 2.0.
```

Compared with raw broader routes:

```text
auto_p1 and s3_p1 still expose raw conflict/decoy pages:
  decoy hit = 100%.

role_auto_p1 avoids raw conflict/decoy pages:
  decoy hit = 0%.

With answerline_summary, the PPL gap is now small:
  10k role_auto_p1 PPL is slightly better than auto_p1;
  20k role_auto_p1 is only slightly worse than auto_p1;
  39k role_auto_p1 is close to auto_p1.
```

### Production speed test

Question:

```text
If the typed summary already contains ANSWER_LABEL, can production inference skip LM option scoring?
```

Output:

```text
/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_answerline_skipscore_10k20k39k_v33_range_sdpa
```

Setup:

```text
same tasks as v32
mode = chain_typedhier_role_auto_p1
skip_lm_answer_when_override = true
```

Results:

| Context | PPL | Accuracy | Raw pages | Raw decoy hit | Eval sec with scoring | Eval sec skip scoring | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | 18.44 | 100% | 2.0 | 0% | 5.27 | 4.32 | 1.22x |
| 20k | 18.16 | 100% | 2.0 | 0% | 5.15 | 4.35 | 1.18x |
| 39k | 13.09 | 100% | 2.0 | 0% | 5.66 | 4.40 | 1.28x |

Interpretation:

```text
Skipping option scoring does not change PPL or final accuracy because:
  query PPL is still scored;
  the final answer comes from the typed answerline record;
  LM option scoring is only used for diagnostic model-side accuracy.

Production mode can therefore use:
  skip_lm_answer_when_override = true

Research/evaluation mode should keep:
  skip_lm_answer_when_override = false

because it verifies whether the model itself would select the right answer from the typed summary.
```

Updated recommendation:

```text
Evaluation setting:
  raw route = chain_typedhier_role_auto_p1
  typed_record_format = answerline_summary
  typed_summary_source_mode = chain_typedhier_auto_p1
  skip_lm_answer_when_override = false

Production-like setting:
  same route and summary,
  skip_lm_answer_when_override = true
```

Current best architecture:

```text
1. Keep sink + recent.
2. Use structural/typed routing to keep only clean raw evidence pages.
3. Use broader page routing only to build a typed answerline summary.
4. Insert the answerline summary before the query.
5. In production, use the typed answerline as the answer and skip option scoring.
```

## 64. Overnight exploration summary

This section summarizes the overnight work from role-filtered page routing through typed
answerline summaries.

### Starting point

Before this set of experiments, the best story-conflict routes had a clear tradeoff:

```text
typedhier_auto_p1:
  good PPL,
  100% evidence coverage,
  but it often kept raw conflict/decoy pages.

typedhier_role_auto_p1:
  clean evidence path,
  0% raw decoy hit,
  only bridge + current answer pages,
  but PPL became much worse.
```

Representative earlier result:

| Context | Route | PPL | Raw pages | Raw decoy hit | Evidence coverage |
| ---: | --- | ---: | ---: | ---: | ---: |
| 20k | auto_p1 | 20.22 | 5.25 | 75% | 100% |
| 20k | role_auto_p1 | 22.96 | 2.0 | 0% | 100% |
| 39k | auto_p1 | 17.03 | 6.0 | 100% | 100% |
| 39k | role_auto_p1 | 23.09 | 2.0 | 0% | 100% |

The goal was to keep the clean raw evidence path while recovering PPL.

### Experiments run

The overnight run explored four designs:

```text
1. Seed-context lane
   Keep non-negative pages from the bridge seed set.

2. Independent raw context lane
   Add extra neutral context pages from query/artifact/section retrieval.

3. Typed summary context lane
   Do not expose broader pages as raw attention;
   summarize them into current/non_current typed facts.

4. Answerline summary
   Compress the typed summary into a short natural-language record with an explicit
   ANSWER_LABEL=...; status=current line.
```

Main output directories:

```text
v15 seed context:
  /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_rolectx_20k39k_v15_range_sdpa

v16-v18 raw context lane:
  /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_rolectxflat_20k39k_v16_range_sdpa
  /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_rolectxart_20k39k_v17_range_sdpa
  /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_rolectxsec_20k39k_v18_range_sdpa

v19-v21 typed summary:
  /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_typedsummary_20k39k_v19_range_sdpa
  /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_typedsummary_compact_20k39k_v20_range_sdpa
  /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_typedsummary_s3source_20k39k_v21_range_sdpa

v22-v31 summary compression sweep:
  mini_summary, short_summary, lite_summary, natural_summary, answerline_summary

v32-v33 stability and production-speed:
  /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_answerline_stability_10k20k39k_v32_range_sdpa
  /home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl/outputs/longrange_book_index_chain_story_answerline_skipscore_10k20k39k_v33_range_sdpa
```

### What failed

Seed-context did not help:

```text
rolectx_auto_p1 produced the same selected pages and PPL as role_auto_p1.

Reason:
  bridge seeds are optimized to discover the alias/artifact,
  not to provide broader topical context.
```

Independent raw context pages were also not enough:

```text
rolectxflat:
  added global query-neighbor pages,
  but PPL barely improved.

rolectxart:
  artifact-conditioned global pages helped 39k somewhat,
  but were unstable at 20k.

rolectxsec:
  section-local context was clean,
  but too local around the answer page and did not recover useful broad context.
```

Key negative result:

```text
Adding more raw pages is not the right abstraction.
The model benefits from broader context, but raw conflict pages can pollute answer selection.
```

### What worked

Typed summary context worked:

```text
raw evidence lane:
  keep bridge + current answer page as raw tokens.

typed summary lane:
  allow broader pages as summary sources,
  but compress conflict/withdrawn pages into status=non_current facts.
```

The best final format is `answerline_summary`:

```text
Typed memory summary: ANSWER_LABEL=C; status=current.
Badge LR... routes to river-name RIVER-...
current ruling for that river-name is option C.
Withdrawn badge option A has status=non_current.
Use the current status only.
```

Why this format works:

```text
The first line gives a strong answer anchor:
  ANSWER_LABEL=C; status=current.

The following sentence is natural enough to help PPL:
  badge -> alias -> current ruling.

Conflict facts remain visible but typed:
  status=non_current.
```

### Best current result

Best clean evaluation route:

```text
raw route = chain_typedhier_role_auto_p1
typed_record_format = answerline_summary
typed_summary_source_mode = chain_typedhier_auto_p1
sparse_attention_impl = range_sdpa
skip_lm_answer_when_override = false
```

Stability result on 10k/20k/39k, 8 tasks per length:

| Context | PPL | Final acc | Model acc | Raw pages | Raw decoy hit | Evidence coverage | Eval sec |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | 18.44 | 100% | 100% | 2.0 | 0% | 100% | 5.27 |
| 20k | 18.16 | 100% | 100% | 2.0 | 0% | 100% | 5.15 |
| 39k | 13.09 | 100% | 100% | 2.0 | 0% | 100% | 5.66 |

Compared with broader raw `auto_p1` under the same answerline-summary setting:

| Context | auto_p1 PPL | role_auto_p1 + answerline PPL | auto_p1 raw decoy hit | role_auto_p1 raw decoy hit |
| ---: | ---: | ---: | ---: | ---: |
| 10k | 18.52 | 18.44 | 100% | 0% |
| 20k | 17.95 | 18.16 | 100% | 0% |
| 39k | 12.85 | 13.09 | 100% | 0% |

Interpretation:

```text
PPL is close to the broader raw route,
but raw conflict/decoy pages are removed from attention.

This is the best tradeoff found so far:
  clean raw evidence,
  low raw page count,
  stable answer accuracy,
  PPL close to broader retrieval.
```

### Baseline clarification

The main PPL baseline in the overnight experiments is not full dense forward.

The comparisons mean:

```text
PPL baseline:
  chain_typedhier_auto_p1
  This is a broader sparse page retrieval route that keeps raw conflict/decoy pages.

Clean baseline:
  chain_typedhier_role_auto_p1 without typed summary
  This is clean but high-PPL.

Production speed baseline:
  same clean route with LM option scoring enabled.
```

The production speed comparison is:

```text
skip_lm_answer_when_override=false
vs
skip_lm_answer_when_override=true
```

It is not:

```text
sparse route vs full dense prefill/forward.
```

Reason:

```text
The evaluator still builds the full-context KV cache first.
The measured eval_seconds cover typed-record/query/answer scoring under the sparse query path,
not a production fused sparse-prefill system.
```

### Production-like speed

With the same answerline route, skipping LM option scoring gives:

| Context | With option scoring | Skip option scoring | Speedup | PPL | Accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10k | 5.27s | 4.32s | 1.22x | 18.44 | 100% |
| 20k | 5.15s | 4.35s | 1.18x | 18.16 | 100% |
| 39k | 5.66s | 4.40s | 1.28x | 13.09 | 100% |

This speedup is from avoiding the diagnostic option scoring step.

### Method summary

The current method is:

```text
1. Keep sink + recent.

2. Build book-like pages and sections.

3. Use structural/typed routing to find:
   bridge page,
   current answer page.

4. Keep only those clean evidence pages as raw remote attention.

5. Use broader routing only to read extra pages for summary construction.

6. Convert broader context into an answerline summary:
   current facts stay current,
   withdrawn/superseded/conflict facts become non_current.

7. Insert that summary before the query.
```

Conceptually:

```text
Raw KV memory:
  small and clean.

Typed page-summary memory:
  broader and safer.
```

### Current recommendation

For evaluation:

```text
--modes chain_typedhier_role_auto_p1
--typed_record_mode extractive
--typed_record_format answerline_summary
--typed_summary_source_mode chain_typedhier_auto_p1
--typed_record_answer_override true
--skip_lm_answer_when_override false
--sparse_attention_impl range_sdpa
```

For production-like speed:

```text
same settings,
but set:
--skip_lm_answer_when_override true
```

### Remaining gaps

This is still not the final answer to the broader goal:

```text
1. The current evidence is synthetic story-conflict retrieval.
   It should be tested on more realistic long-document QA / multi-hop semantic retrieval.

2. The summary builder is rule-based/extractive.
   A learned small summarizer or MLP/NLP router has not been tested yet.

3. The speed numbers are query-path eval seconds.
   They are not full dense-vs-sparse end-to-end serving numbers.

4. Full dense baseline was not rerun in the latest v32/v33 suite.
   It should be added if we need a direct full-forward comparison.
```

Next useful experiments:

```text
1. Add full and sink_recent to the answerline suite for explicit dense/sparse baseline comparison.
2. Run answerline_summary on a less synthetic long-document QA task.
3. Replace rule summaries with learned page summaries and compare PPL/accuracy/token cost.
4. Test larger task counts to reduce variance in 10k/20k/39k.
```
