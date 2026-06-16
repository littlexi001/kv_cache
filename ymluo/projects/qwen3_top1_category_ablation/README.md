# Qwen3 Top1 Category Ablation

This project runs a needle-in-haystack retrieval ablation on Qwen3-0.6B.

For each sample, it evaluates gold-answer PPL under:

- `full_attention`: no pruning.
- `top1_all`: keep per-layer/per-head attention top 1% tokens.
- `answer_only`: from the top 1%, keep only tokens overlapping the answer evidence span.
- `front_only`: from the top 1%, keep only the first 1% visible tokens.
- `end_only`: from the top 1%, keep only the last 1% visible tokens.
- `other_only`: from the top 1%, keep the remaining top tokens.

It also collects the final hidden states used to predict the answer tokens. These
states are the X matrix immediately before the final `lm_head` matrix
(`hidden_size x vocab_size`, about `1024 x 151k` for Qwen3-0.6B), and writes their
singular value spectra.

## Data

Default data is reused from Section 17-related local needle-in-haystack data:

```text
ymluo/projects/qwen3_kcache_l2_neighbor_analysis/data/needle_in_haystack/needle_in_haystack.jsonl
```

## Windows Setup

The default system Python in this workspace currently has an old `transformers`
version, so use a local venv:

```powershell
.\ymluo\projects\qwen3_top1_category_ablation\scripts\setup_windows_env.ps1
```

Download Qwen3-0.6B into the git-ignored model cache:

```powershell
.\ymluo\projects\qwen3_top1_category_ablation\scripts\download_qwen3_0p6b.ps1
```

The download script uses `https://hf-mirror.com` by default and writes to:

```text
ymluo/models/Qwen3-0.6B
```

Run a smoke test:

```powershell
$env:PYTHON_EXE="F:\desktop\学习\codex_workspace\kvcache\kv_cache\ymluo\.venv-qwen3\Scripts\python.exe"
$env:MAX_SAMPLES="1"
$env:MAX_CONTEXT_CHARS="4000"
$env:SVD_MAX_VECTORS="256"
.\ymluo\projects\qwen3_top1_category_ablation\scripts\run_ablation.ps1
```

Fuller run:

```powershell
$env:MAX_SAMPLES="8"
$env:MAX_CONTEXT_CHARS="24000"
.\ymluo\projects\qwen3_top1_category_ablation\scripts\run_ablation.ps1
```

With a 4GB RTX 3050 Laptop GPU, use small smoke tests first. If CUDA OOMs, set:

```powershell
$env:DEVICE="cpu"
$env:DEVICE_MAP="none"
$env:DTYPE="float32"
```

CPU will be much slower, but should avoid VRAM limits.

## Outputs

Default output directory:

```text
ymluo/projects/qwen3_top1_category_ablation/outputs/top1_category_ablation/
```

Main files:

- `ppl_summary.csv`: mean answer loss/PPL by ablation mode.
- `per_sample_results.csv` and `.jsonl`: sample-level answer PPL.
- `top1_category_counts_by_layer_head.csv`: top1% category distribution by sample/layer/head.
- `x_singular_values.csv`: singular values and cumulative energy of the final hidden-state X matrix.
- `summary.json`: run configuration and paths.
