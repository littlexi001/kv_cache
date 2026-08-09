param(
    [string]$Remote = "fdong@10.176.37.31",
    [ValidateSet("legacy", "single_token", "english_single_token", "full_pre_softmax_128k", "rope_pair_64k")]
    [string]$Mode = "legacy",
    [string]$RemoteOutput = ""
)

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $PSScriptRoot
$RemoteDefaults = @{
    legacy = "/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/attention_confidence_qwen3_8b_20260717"
    single_token = "/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/attention_confidence_qwen3_8b_single_token_20260717"
    english_single_token = "/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/attention_confidence_qwen3_8b_english_single_token_128k_20260718"
    full_pre_softmax_128k = "/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/full_pre_softmax_qwen3_8b_english_128k_20260718"
    rope_pair_64k = "/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/rope_pair_contributions_qwen3_8b_64k_20260720"
}
if ([string]::IsNullOrWhiteSpace($RemoteOutput)) {
    $RemoteOutput = $RemoteDefaults[$Mode]
}
$DataRoot = Join-Path $Project "attention_confidence_dashboard\public\data"
$Destination = switch ($Mode) {
    "legacy" { $DataRoot }
    "full_pre_softmax_128k" { Join-Path $DataRoot "english_single_token\full_pre_softmax_128k" }
    "rope_pair_64k" { Join-Path $DataRoot "english_single_token\rope_pair_64k" }
    default { Join-Path $DataRoot $Mode }
}
New-Item -ItemType Directory -Force $Destination | Out-Null

if ($Mode -eq "rope_pair_64k") {
    $RemoteData = "${RemoteOutput}/site_data"
    scp -r "${Remote}:${RemoteData}/." $Destination
} elseif ($Mode -eq "full_pre_softmax_128k") {
    $RemoteData = "${RemoteOutput}/full_pre_softmax/length_128000"
    scp -r "${Remote}:${RemoteData}/." $Destination
    $Python = (Get-Command python).Source
    & $Python (Join-Path $Project "src\build_full_pre_softmax_token_heatmap.py") `
        --root $Destination `
        --output (Join-Path $Destination "token_type_heatmap.json.gz")
} else {
    scp "${Remote}:${RemoteOutput}/site_data/manifest.json" $Destination
    scp "${Remote}:${RemoteOutput}/site_data/length_*.json.gz" $Destination
    if ($Mode -eq "english_single_token") {
        scp "${Remote}:${RemoteOutput}/site_data/pre_softmax_head_length_summary.json.gz" $Destination
    }
}

Write-Host "Dashboard mode '$Mode' synchronized to $Destination"
