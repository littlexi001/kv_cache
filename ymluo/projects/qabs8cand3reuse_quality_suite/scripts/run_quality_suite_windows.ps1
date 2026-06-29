param(
    [string]$Python = "python",
    [string]$ModelPath = "C:\models\Qwen3-0.6B",
    [string]$OutputRoot = "",
    [string]$Modes = "baseline,qabs8cand3reuse,sparqfast8cand3",
    [int]$PrefillTokens = 4096,
    [int]$EvalTokens = 512,
    [int]$MaxNeedleCases = 12,
    [int]$NeedlePrefillTokens = 0,
    [int]$NeedleEvalTokens = 0,
    [string]$Device = "cuda",
    [string]$DeviceMap = "auto",
    [string]$DType = "bfloat16",
    [bool]$QabsCudaFinalKernel = $false,
    [bool]$RunNeedleGeneration = $false
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$RepoRoot = Resolve-Path (Join-Path $ProjectDir "..\..\..")
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $ProjectDir "outputs"
}

Set-Location $RepoRoot
$env:TOKENIZERS_PARALLELISM = "false"

& $Python (Join-Path $ProjectDir "src\build_topic_texts.py")

& $Python (Join-Path $ProjectDir "src\run_quality_suite.py") `
    --model_name_or_path $ModelPath `
    --output_root $OutputRoot `
    --modes $Modes `
    --prefill_tokens $PrefillTokens `
    --eval_tokens $EvalTokens `
    --chunk_size 8 `
    --eval_chunk_size 1 `
    --max_needle_cases $MaxNeedleCases `
    --needle_prefill_tokens $NeedlePrefillTokens `
    --needle_eval_tokens $NeedleEvalTokens `
    --dtype $DType `
    --device $Device `
    --device_map $DeviceMap `
    --attn_implementation eager `
    --top_fraction 0.02 `
    --protect_sink_tokens 10 `
    --protect_recent_tokens 10 `
    --qabs_cuda_final_kernel ($QabsCudaFinalKernel.ToString().ToLower()) `
    --qabs_cuda_candidate_kernel false `
    --qabs_cuda_reuse_select_kernel false `
    --make_plots false

if ($RunNeedleGeneration) {
    & $Python (Join-Path $ProjectDir "src\evaluate_needle_generation.py") `
        --model_name_or_path $ModelPath `
        --output_dir (Join-Path $OutputRoot "needle_generation") `
        --modes $Modes `
        --max_cases $MaxNeedleCases `
        --dtype $DType `
        --device $Device `
        --device_map $DeviceMap `
        --attn_implementation eager `
        --top_fraction 0.02 `
        --protect_sink_tokens 10 `
        --protect_recent_tokens 10 `
        --qabs_cuda_final_kernel ($QabsCudaFinalKernel.ToString().ToLower()) `
        --qabs_cuda_candidate_kernel false `
        --qabs_cuda_reuse_select_kernel false
}

Write-Host "quality suite complete: $OutputRoot"
