param(
    [string]$Python = "python",
    [string]$ModelPath = "C:\models\Qwen3-0.6B"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

& (Join-Path $ScriptDir "run_quality_suite_windows.ps1") `
    -Python $Python `
    -ModelPath $ModelPath `
    -Modes "baseline,qabs8cand3reuse,sparqfast8cand3" `
    -PrefillTokens 128 `
    -EvalTokens 8 `
    -MaxNeedleCases 1 `
    -Device "cpu" `
    -DeviceMap "none" `
    -DType "float32" `
    -QabsCudaFinalKernel $false `
    -RunNeedleGeneration $false

