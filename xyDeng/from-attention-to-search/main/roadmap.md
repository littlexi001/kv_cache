# Research Roadmap - 0523

## Roadmap Conclusion

当前路线图应从 **router factor attribution** 开始，而不是继续堆 router 变体。

```text
router factor audit
  -> disentangled / frequency-balanced diagnostic
  -> induced specialization candidates
  -> specialization-to-Reverse-Index utility
  -> real representation transfer
```

## Gate 1: Existing-Run Router Factor Audit

Question:

```text
vanilla top-1 MoE expert assignment 主要由什么 factor 决定？
```

Candidate factors:

```text
token identity
position
left context
target label
frequency
common / high-frequency / residual-preserved component
rule / function class
```

Use existing H0521/H0521b runs first. Do not retrain before attribution.

Metrics:

```text
factor -> router_logit explained variance / R^2
factor -> expert_id MI / NMI
component removal -> route flip rate
```

Decision logic:

- If identity/frequency/common component dominates: vanilla router is not a clean feature-level learned index; move to frequency-balanced or disentangled setup.
- If rule/function component dominates route flips: partial rule specialization exists; test whether it helps Reverse Index.
- If all factors are weak or seed-dependent: routing is mixed/unstable; isolate mechanism before any retrieval claim.

## Gate 2: Disentangled / Frequency-Balanced Synthetic Setup

Question:

```text
如果 common/high-frequency component 干扰 routing，控制频率和可分性后，
expert 是否能形成更干净的 rule/function bucket？
```

Purpose:

```text
separate "MoE cannot specialize" from "current data/representation makes the
intended factor non-dominant".
```

## Gate 3: Induced Specialization Candidates

Compare only after Gate 1 clarifies the bottleneck:

```text
oracle MoE routing
attention-relation constrained routing
hierarchical / frequency-aware expert grouping
```

Claim boundary:

Bucket purity is not enough. Each candidate must later pass a retrieval utility test.

## Gate 4: Specialization To Reverse Index Utility

Question:

```text
specialized bucket 是否真的提高 high-mass KV candidate quality？
```

Behavior metric first:

```text
CE delta over random same-count candidate set
```

Supporting metrics:

```text
attention mass recall
candidate size
expert count / compute budget
```

Do not treat high feature-expert NMI as sufficient evidence.

## Gate 5: Real Representation Transfer

Only enter this gate after synthetic diagnostics show:

- candidate generation beats random by behavior metric;
- attention mass is not catastrophically lost;
- the mechanism is not only a synthetic label artifact;
- claim boundary is clear enough for advisor-facing reporting.

## Stop Rules

- Do not optimize router architecture before Gate 1 is decided.
- Do not call the current failure "token ID routing" unless factor audit supports it.
- Do not claim attention-relation routing solves Reverse Index before CE/candidate-quality tests.
- Do not claim speedup before exact sparse attention and candidate generation are validated together.
