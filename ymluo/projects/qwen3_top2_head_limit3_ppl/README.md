# Qwen3 Top2 Historical Attention With Max 3 Heads Per Token

This project compares three PPL modes:

- `baseline`: original full attention.
- `top2`: each attention head can attend only to its top 2% historical tokens;
  the current token attends to itself unchanged.
- `top2limit3`: start from `top2`, then each historical token may be kept by at
  most 3 heads in the same layer and query. If more than 3 heads selected it,
  the script randomly keeps 3 of those heads.
- `top2limit3score`: same cap, but keep the 3 selected heads with the largest
  pre-softmax score for that token.
- `top2limit3scorefill`: keep the highest-score 3 heads, then fill each head
  back to its original top2 load using next-best historical tokens that do not
  violate the cap.
- `top2limit3gapM`: keep the highest-score 3 selected heads, then also keep
  any other selected head whose score is within margin `M` of the third-best
  selected head for that token. Example: `top2limit3gap8p0` means margin `8.0`.
- `top2limit3protectsSrR`: keep all top2-selected heads for protected tokens,
  then apply the score-based top3-head cap only to unprotected historical
  tokens. `S` is the number of protected sink positions at the start of the
  sequence; `R` is the protected recent-token percentage. Example:
  `top2limit3protects64r16p0` protects the first 64 historical positions and
  the most recent 16% historical positions.

The selection is applied before softmax by masking non-kept attention scores to
`-inf`.

## Run

```bash
bash ymluo/projects/qwen3_top2_head_limit3_ppl/scripts/run_analysis.sh
```

PowerShell:

```powershell
ymluo\.venv-qwen3\Scripts\python.exe `
  ymluo\projects\qwen3_top2_head_limit3_ppl\src\evaluate_qwen3_top2_head_limit3_ppl.py `
  --model_name_or_path ymluo\models\Qwen3-0.6B `
  --text_path external\needle-in-a-haystack\needlehaystack\PaulGrahamEssays\worked.txt `
  --output_dir ymluo\projects\qwen3_top2_head_limit3_ppl\outputs\top2_head_limit3_ppl `
  --prefill_tokens 1024 `
  --eval_tokens 512 `
  --chunk_size 128 `
  --top_fraction 0.02 `
  --max_heads_per_token 3 `
  --always_keep_self true
```

Score-based sweep:

```powershell
ymluo\.venv-qwen3\Scripts\python.exe `
  ymluo\projects\qwen3_top2_head_limit3_ppl\src\evaluate_qwen3_top2_head_limit3_ppl.py `
  --model_name_or_path ymluo\models\Qwen3-0.6B `
  --text_path external\needle-in-a-haystack\needlehaystack\PaulGrahamEssays\worked.txt `
  --output_dir ymluo\projects\qwen3_top2_head_limit3_ppl\outputs\score_sweep `
  --prefill_tokens 1024 `
  --eval_tokens 512 `
  --chunk_size 128 `
  --modes baseline,top2,top2limit3score,top2limit4score,top2limit6score,top2limit8score,top2limit12score,top2limit16score
```

Score-gap fine sweep:

```powershell
ymluo\.venv-qwen3\Scripts\python.exe `
  ymluo\projects\qwen3_top2_head_limit3_ppl\src\evaluate_qwen3_top2_head_limit3_ppl.py `
  --model_name_or_path ymluo\models\Qwen3-0.6B `
  --text_path external\needle-in-a-haystack\needlehaystack\PaulGrahamEssays\worked.txt `
  --output_dir ymluo\projects\qwen3_top2_head_limit3_ppl\outputs\gap_sweep_5_7 `
  --prefill_tokens 1024 `
  --eval_tokens 512 `
  --chunk_size 128 `
  --modes baseline,top2,top2limit3gap5p0,top2limit3gap6p0,top2limit3gap7p0,top2limit3gap8p0
```

Removal diagnostic:

```powershell
ymluo\.venv-qwen3\Scripts\python.exe `
  ymluo\projects\qwen3_top2_head_limit3_ppl\src\diagnose_top2_limit_removal.py `
  --model_name_or_path ymluo\models\Qwen3-0.6B `
  --text_path external\needle-in-a-haystack\needlehaystack\PaulGrahamEssays\worked.txt `
  --output_dir ymluo\projects\qwen3_top2_head_limit3_ppl\outputs\removal_diagnostics `
  --prefill_tokens 1024 `
  --eval_tokens 512 `
  --chunk_size 128 `
  --caps 3,8,12,15
```

Sink/recent protection threshold sweep:

```powershell
ymluo\.venv-qwen3\Scripts\python.exe `
  ymluo\projects\qwen3_top2_head_limit3_ppl\src\evaluate_qwen3_top2_head_limit3_ppl.py `
  --model_name_or_path ymluo\models\Qwen3-0.6B `
  --text_path external\needle-in-a-haystack\needlehaystack\PaulGrahamEssays\worked.txt `
  --output_dir ymluo\projects\qwen3_top2_head_limit3_ppl\outputs\protect_sink_recent_threshold_sweep `
  --prefill_tokens 1024 `
  --eval_tokens 512 `
  --chunk_size 128 `
  --modes top2,top2limit3protects64r10p0,top2limit3protects64r12p0,top2limit3protects64r14p0,top2limit3protects32r16p0,top2limit3protects128r12p0,top2limit3gap8p0
```

Head-count position distribution:

```powershell
ymluo\.venv-qwen3\Scripts\python.exe `
  ymluo\projects\qwen3_top2_head_limit3_ppl\src\analyze_top2_head_count_positions.py `
  --model_name_or_path ymluo\models\Qwen3-0.6B `
  --text_path external\needle-in-a-haystack\needlehaystack\PaulGrahamEssays\worked.txt `
  --output_dir ymluo\projects\qwen3_top2_head_limit3_ppl\outputs\head_count_position_distribution `
  --prefill_tokens 1024 `
  --eval_tokens 512 `
  --chunk_size 128 `
  --top_fraction 0.02
```

Smoke test:

```powershell
ymluo\.venv-qwen3\Scripts\python.exe `
  ymluo\projects\qwen3_top2_head_limit3_ppl\src\evaluate_qwen3_top2_head_limit3_ppl.py `
  --model_name_or_path ymluo\models\Qwen3-0.6B `
  --text_path external\needle-in-a-haystack\needlehaystack\PaulGrahamEssays\worked.txt `
  --output_dir ymluo\projects\qwen3_top2_head_limit3_ppl\outputs\smoke `
  --prefill_tokens 64 `
  --eval_tokens 32 `
  --chunk_size 16
```

## Outputs

- `ppl_by_mode.csv`: PPL for the three modes.
- `limit_load_by_head.csv`: per-layer and per-head load for every limit mode.
- `top2limit3score_load_by_head.csv`: focused load table when
  `top2limit3score` is present.
- `plots/ppl_by_mode.png`: PPL comparison.
- `plots/top2limit3_head_load_heatmap.png`: mean kept historical tokens per
  query for every layer/head.
- `plots/top2limit3_head_load_by_layer.png`: layer-level mean and min-max load.
- `plots/top2limit3_kept_fraction_heatmap.png`: fraction of original top2
  selections kept after the limit3 rule.
- `outputs/removal_diagnostics/removed_weight_summary_by_cap.csv`: offline
  estimate of how much original top2 attention weight each cap removes.
- `outputs/final_fine_rules/fine_rule_summary.csv`: compact comparison of the
  best tested random, score, score-fill, and score-gap rules.
- `outputs/protect_sink_recent_combined/combined_protect_sink_recent_summary.csv`:
  combined sink/recent protection results. Best tested mode in this run:
  `top2limit3protects64r16p0`, PPL `36.462`.
- `outputs/head_count_position_distribution/head_count_position_summary.csv`:
  where tokens selected by exactly 1..16 heads are located under the original
  top2 rule.
