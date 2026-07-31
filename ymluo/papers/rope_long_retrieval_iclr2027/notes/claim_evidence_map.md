# Claim–evidence map

| Paper claim | Current evidence | Strength | Required follow-up |
|---|---|---|---|
| Fixed semantic Q/K can receive different post-RoPE scores as distance changes | Identical layer-0 pre-RoPE Q/K at four distances; 64-pair reconstruction error ≤ 0.00198 | Exact for the audited Qwen3-8B case | Repeat across seeds/models and native windows |
| Phase-triggered attention changes enter and propagate through residual computation | 36-layer BF16 finite-difference replay; final readout reconstruction | Exact descriptive accounting for one transition | Repeat on registered transitions |
| Intermediate state divergence causally affects the first answer token | Residual-input patch at L16/L20 changes failed margin from −2.75 to positive | Strong causal case study | Population effect sizes and negative controls |
| Pre-RoPE remote proposal can improve matched-budget retrieval | 24 held-out seeds per length; significant paired NLL gains vs post-RoPE Top-2% at 16K/32K | Controlled held-out evidence | Public tasks and additional architectures |
| SAGE-Post is a practical accelerator | Not established; current proposal scans full pre-RoPE history | Unsupported | Approximate index and end-to-end profiling |
| The mechanism is universal across RoPE LLMs | Not established | Unsupported | Cross-model, cross-task replication |

