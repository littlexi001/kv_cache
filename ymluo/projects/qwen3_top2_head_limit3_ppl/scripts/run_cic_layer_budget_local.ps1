param(
    [string]$Python = ".\.venv-qabs-rocm\Scripts\python.exe",
    [string]$ModelPath = "Qwen/Qwen3-0.6B",
    [string]$TextPath = "ymluo\projects\qabs8cand3reuse_quality_suite\data\topic_texts\literature.txt",
    [string]$OutputDir = "ymluo\projects\qwen3_top2_head_limit3_ppl\outputs\local_cic_literature_2048",
    [int]$PrefillTokens = 2048,
    [int]$EvalTokens = 64,
    [int]$ChunkSize = 8,
    [int]$LandmarkRecent = 4096,
    [int]$LandmarkStride = 64,
    [string]$Layers = "",
    [string]$CandidateCounts = "1,2,3,4"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")
Set-Location $RepoRoot

$env:TOKENIZERS_PARALLELISM = "false"

$ArgsList = @(
    "ymluo\projects\qwen3_top2_head_limit3_ppl\src\run_cic_layer_budget_local.py",
    "--model_name_or_path", $ModelPath,
    "--text_path", $TextPath,
    "--output_dir", $OutputDir,
    "--prefill_tokens", $PrefillTokens,
    "--eval_tokens", $EvalTokens,
    "--chunk_size", $ChunkSize,
    "--eval_chunk_size", 1,
    "--dtype", "bfloat16",
    "--device", "cuda",
    "--device_map", "auto",
    "--attn_implementation", "eager",
    "--landmark_recent", $LandmarkRecent,
    "--landmark_stride", $LandmarkStride,
    "--candidate_counts", $CandidateCounts,
    "--reuse_prefill_cache", "true",
    "--log_every", 1000
)
if ($Layers) {
    $ArgsList += @("--layers", $Layers)
}

& $Python @ArgsList
