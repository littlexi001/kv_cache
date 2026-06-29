# Influence-Bounded Synthetic KV

This project is a local smoke-test implementation for the research idea in:

```text
ymluo/doc/section25_influence_bounded_synthetic_kv.md
```

The first script does not require a language model. It builds a controlled
attention problem, replaces a long remote KV cache with a small number of
synthetic K/V prototypes, and measures how well the compressed attention output
matches full remote attention.

## Run

From repository root on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File ymluo\projects\influence_bounded_synthetic_kv\scripts\run_smoke_local.ps1
```

Or run Python directly:

```bash
python ymluo/projects/influence_bounded_synthetic_kv/src/run_synthetic_kv_smoke.py
```

## Outputs

Default output directory:

```text
ymluo/projects/influence_bounded_synthetic_kv/outputs/smoke/
```

Main files:

- `metrics.csv`: train/test output reconstruction metrics for each method.
- `summary.json`: run config and best method summary.

## Methods in the smoke test

- `random_real_kv`: keep random real remote KV tokens.
- `top_mass_real_kv`: keep real tokens with highest calibration attention mass.
- `kmeans_k_ridge_v`: use k-means centers as synthetic keys and solve synthetic
  values by ridge regression.
- `joint_kv`: initialize from `kmeans_k_ridge_v`, then jointly optimize K/V with
  norm clipping.

The smoke test is intentionally offline and CPU friendly. Its purpose is to
validate the project mechanics before adding model-cache dumping and PPL eval.

## 本地 7900XTX calibrated PPL 初测

正式 calibration 版本和第一轮 PPL 结果已整理到：

```text
docs/calibrated_ppl_7900xtx_results.md
```

简要结论：`mass oracle` 说明少量原型有表达上界，但当前 per-layer independent calibration 在替换全部层后 PPL 仍然明显崩，需要继续做 layer ablation 和短 unroll logits-KL 目标。
