$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Resolve-Path (Join-Path $ScriptDir "..")
$RepoDir = Resolve-Path (Join-Path $ProjectDir "..\..\..")

$env:TOKENIZERS_PARALLELISM = "false"

if (-not $env:MODEL_PATH) {
  $env:MODEL_PATH = Join-Path $RepoDir "ymluo\models\Qwen3-0.6B"
}
if (-not $env:DATA_PATH) {
  $env:DATA_PATH = Join-Path $RepoDir "ymluo\projects\qwen3_kcache_l2_neighbor_analysis\data\needle_in_haystack\needle_in_haystack.jsonl"
}
if (-not $env:OUTPUT_DIR) {
  $env:OUTPUT_DIR = Join-Path $ProjectDir "outputs\top1_category_ablation"
}

$PythonExe = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }

& $PythonExe (Join-Path $ProjectDir "src\run_top1_category_ablation.py") `
  --model_name_or_path $env:MODEL_PATH `
  --data_path $env:DATA_PATH `
  --output_dir $env:OUTPUT_DIR `
  --max_samples $(if ($env:MAX_SAMPLES) { $env:MAX_SAMPLES } else { "8" }) `
  --max_context_chars $(if ($env:MAX_CONTEXT_CHARS) { $env:MAX_CONTEXT_CHARS } else { "24000" }) `
  --top_ratio $(if ($env:TOP_RATIO) { $env:TOP_RATIO } else { "0.01" }) `
  --modes $(if ($env:MODES) { $env:MODES } else { "full_attention,top1_all,answer_only,front_only,end_only,other_only" }) `
  --svd_max_vectors $(if ($env:SVD_MAX_VECTORS) { $env:SVD_MAX_VECTORS } else { "4096" }) `
  --svd_top_k $(if ($env:SVD_TOP_K) { $env:SVD_TOP_K } else { "128" }) `
  --dump_top_tokens $(if ($env:DUMP_TOP_TOKENS) { $env:DUMP_TOP_TOKENS } else { "true" }) `
  --dtype $(if ($env:DTYPE) { $env:DTYPE } else { "float16" }) `
  --device $(if ($env:DEVICE) { $env:DEVICE } else { "cuda" }) `
  --device_map $(if ($env:DEVICE_MAP) { $env:DEVICE_MAP } else { "auto" }) `
  --attn_implementation $(if ($env:ATTN_IMPLEMENTATION) { $env:ATTN_IMPLEMENTATION } else { "eager" }) `
  --trust_remote_code $(if ($env:TRUST_REMOTE_CODE) { $env:TRUST_REMOTE_CODE } else { "true" })
