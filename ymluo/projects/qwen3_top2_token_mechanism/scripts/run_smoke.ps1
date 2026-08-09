param(
    [string]$Python = "python",
    [string]$Model = "ymluo/models/Qwen3-0.6B",
    [string]$Text = "external/needle-in-a-haystack/needlehaystack/PaulGrahamEssays/worked.txt",
    [string]$OutputDir = ""
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot
if (-not $OutputDir) {
    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $OutputDir = Join-Path $ProjectDir "outputs/smoke_$Stamp"
}

& $Python (Join-Path $ProjectDir "src/run_selector_ppl.py") `
    --model_name_or_path $Model `
    --text_path $Text `
    --output_dir $OutputDir `
    --prefill_tokens 64 `
    --eval_tokens 16 `
    --chunk_size 8 `
    --dtype float32 `
    --device cpu `
    --device_map none `
    --attn_implementation eager `
    --ratio_grid 0.02,1.0 `
    --target_ratio 0.02 `
    --control_selectors sink_recent_s0,sink_recent_s1,recent,random,bottom_attention `
    --diagnostic_sink_sweep 0,1,2 `
    --role_sink_tokens 2 `
    --role_recent_tokens 8 `
    --make_plots false

& $Python (Join-Path $ProjectDir "src/analyze_diagnostics.py") --run_dir $OutputDir
& $Python (Join-Path $ProjectDir "src/compare_selectors.py") `
    --run_dir $OutputDir `
    --bootstrap_repetitions 100

Write-Host "[top2-mechanism] smoke done: $OutputDir"
