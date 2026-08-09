# Iteration ledger

## 2026-08-08: replace synthetic-only objective

- Observation: the completed 2.5B-token runs reached low template loss but had
  zero retrieval accuracy and answer NLL near random chance.
- Diagnosis: sparse supervision on a low-entropy artificial stream allowed format
  learning to dominate and did not establish that the model learned retrieval.
- Decision: preserve those runs, introduce ordinary natural text, full-token LM
  loss, deterministic mixed examples and a short-retrieval gate.

