param(
    [string]$Python = ".\.venv-qabs-rocm\Scripts\python.exe",
    [string]$ModelPath = "Qwen/Qwen3-0.6B",
    [string]$TextPath = "ymluo\projects\qwen3_top2_head_limit3_ppl\data\war_and_peace_pg2600.txt",
    [string]$OutputDir = "ymluo\projects\qwen3_top2_head_limit3_ppl\outputs\local_cic_combo_war",
    [int]$PrefillTokens = 4096,
    [int]$EvalTokens = 128,
    [int]$ChunkSize = 16,
    [int]$LandmarkRecent = 512,
    [int]$LandmarkStride = 64,
    [string]$PairwiseLayers = "",
    [string]$Combos = "",
    [string]$IncludeSingletons = "false",
    [string]$IncludePrefixes = "false"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..\..")
Set-Location $RepoRoot

$env:TOKENIZERS_PARALLELISM = "false"

$ArgsList = @(
    "ymluo\projects\qwen3_top2_head_limit3_ppl\src\run_cic_combo_local.py",
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
    "--include_singletons", $IncludeSingletons.ToString().ToLower(),
    "--include_prefixes", $IncludePrefixes.ToString().ToLower(),
    "--log_every", 1000
)
if ($PairwiseLayers) {
    $ArgsList += @("--pairwise_layers", $PairwiseLayers)
}
if ($Combos) {
    $ArgsList += @("--combos", $Combos)
}

& $Python @ArgsList
