# QABS8Cand3Reuse quality suite

This project evaluates `qabs8cand3reuse` quality against dense `baseline` and a SparQ attention baseline across:

1. topic-specific text PPL,
2. needle-in-a-haystack prompt PPL,
3. optional needle-in-a-haystack generation accuracy.

It intentionally reuses the implementation in:

```text
ymluo/projects/qwen3_top2_head_limit3_ppl/src/evaluate_qwen3_top2_head_limit3_ppl.py
```

so the test stays focused on quality measurement rather than duplicating attention code.

## Default modes

```text
baseline,qabs8cand3reuse,sparqfast8cand3
```

`sparqfast8cand3` uses the existing SparQ fast path with the same 8 query dimensions and 3% candidate fraction as `qabs8cand3reuse`.

## Server run

From repository root:

```bash
bash ymluo/projects/qabs8cand3reuse_quality_suite/scripts/run_quality_suite_server.sh
```

Useful overrides:

```bash
MODEL_PATH=/mnt/workspace/Qwen3-0.6B \
PREFILL_TOKENS=4096 \
EVAL_TOKENS=512 \
RUN_NEEDLE_GENERATION=true \
bash ymluo/projects/qabs8cand3reuse_quality_suite/scripts/run_quality_suite_server.sh
```

Outputs are written to:

```text
ymluo/projects/qabs8cand3reuse_quality_suite/outputs/
```

Main files:

```text
outputs/quality_suite_combined.csv
outputs/quality_suite_combined.md
outputs/needle_generation/needle_generation_results.csv
outputs/needle_generation/needle_generation_summary.csv
```

## 本地 7900XTX 结果

第一轮 Windows + Radeon RX 7900 XTX 本地 PPL 结果已整理到：

```text
docs/local_7900xtx_quality_results.md
```

简要结论：`qabs8cand3reuse` 明显比 `sparqfast8cand3` 更接近 dense `baseline`，但在这轮 `prefill=512, eval=32` 的短上下文测试里还没有超过 baseline PPL。

## Data

The suite creates a small controlled topic-text set under `data/topic_texts/` if it is missing. These files are for relative PPL deltas across modes, not for claiming absolute benchmark quality.

Needle prompts reuse the existing dataset if present:

```text
ymluo/projects/qwen3_kcache_l2_neighbor_analysis/data/needle_in_haystack/
```

## Windows local run on Radeon 7900XTX

On Windows, use the AMD ROCm PyTorch build if available. The suite still passes `--device cuda` because PyTorch exposes HIP/ROCm devices through the CUDA-compatible API. Keep `QabsCudaFinalKernel` disabled on AMD; the custom final-attention extension is CUDA/NVIDIA-specific.

Install into a venv:

```powershell
powershell -ExecutionPolicy Bypass -File ymluo\projects\qabs8cand3reuse_quality_suite\scripts\setup_rocm_windows.ps1 -CreateVenv
```

Run the suite:

```powershell
powershell -ExecutionPolicy Bypass -File ymluo\projects\qabs8cand3reuse_quality_suite\scripts\run_quality_suite_windows.ps1 `
  -Python .\.venv-qabs-rocm\Scripts\python.exe `
  -ModelPath C:\models\Qwen3-0.6B `
  -QabsCudaFinalKernel $false
```

CPU smoke test:

```powershell
powershell -ExecutionPolicy Bypass -File ymluo\projects\qabs8cand3reuse_quality_suite\scripts\run_quality_suite_windows_cpu_smoke.ps1 `
  -ModelPath C:\models\Qwen3-0.6B
```
