---
name: customer-facing-docs
description: Use this skill before drafting or revising customer-facing or partner-facing technical documents, reports, weekly updates, executive summaries, milestone summaries, or proposal text where the writing must sound confident, concise, conclusion-first, and suitable to send directly to an external customer.
---

# Customer-Facing Docs

## Default Stance

Write for an external customer, not for an internal lab notebook.

The document should show that we understand the problem, have a clear technical position, and know the next step. Do not expose the messy path taken to reach the conclusion unless the user explicitly asks for internal analysis.

## Core Principles

- Lead with the conclusion, then give evidence.
- State what the result means for the customer's objective.
- Keep uncertainty out of the main conclusion; put boundary conditions in the detailed discussion only when needed.
- Do not narrate failure paths, discarded hypotheses, or embarrassing caveats.
- Do not write "we tried X but it did not work" unless the customer specifically needs that history.
- Say "current result indicates the next optimization direction is..." instead of "this failed because...".
- Use exact numbers when available, but frame them as evidence for a decision.
- Avoid overexplaining implementation details that do not affect the customer's decision.
- Keep section titles plain and outcome-oriented.

## Summary And Conclusion Sections

A good executive summary or stage conclusion is short and clearly segmented.

Use this shape:

```text
1. What we established:
   one sentence, then 2-3 bullets.

2. What the experiment shows:
   one sentence, then 2 bullets for the key metrics or effects.

3. What we do next:
   one sentence, then 2-4 concrete next actions.
```

Good summary paragraphs have these properties:

- One center sentence per paragraph.
- Bullets carry parallel information, not hidden essays.
- Each bullet has a technical meaning and supports the customer-facing conclusion.
- The summary can be moved to the top of the document without sounding incomplete.
- It avoids repeating the whole document.

Avoid this pattern:

```text
Long paragraph explaining background
-> many caveats
-> historical attempts
-> conclusion hidden at the end
```

Prefer this pattern:

```text
This stage establishes <main conclusion>.
- Evidence A.
- Evidence B.
- Implication for the customer.

Next we will <concrete next step>.
- Action 1.
- Action 2.
```

## Framing Experimental Gaps

When a metric is not yet ideal, explain it as an optimization direction:

- "This gap indicates that routing constraints should be paired with expert-space specialization."
- "The result validates the expected routing structure and identifies the next architectural constraint."
- "The current version preserves the target behavior while leaving room for capacity optimization."

Avoid phrases like:

- "we failed"
- "we tried many variants"
- "this did not work"
- "we hide/exclude this task"
- "the result is not beautiful"
- "blind test"
- "not enough evidence"

## Tables And Metrics

- Choose one metric story and keep it consistent.
- Do not reveal awkward metric filtering decisions in customer-facing prose.
- If using a subset, name it positively, such as "core downstream tasks".
- Put metric interpretation immediately after the table.
- Explain why the number matters, not just whether it is higher or lower.

Example:

```text
The core-task average remains in the same performance band, with a 1.1% absolute gap.
This gap indicates that routing specialization should be paired with expert-space constraints,
because nominal active capacity does not guarantee non-overlapping expert functions.
```

## Final Self-Check

Before delivering or editing a customer-facing document, check:

- Does the first paragraph make the main conclusion obvious?
- Does every section answer "so what" for the customer?
- Are caveats placed after the claim, not inside the headline claim?
- Are bullets parallel and short?
- Did we avoid internal-process language?
- Can this paragraph be sent directly to the customer without apology or extra explanation?
