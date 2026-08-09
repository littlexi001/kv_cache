param(
    [string]$Remote = "fdong@10.176.37.31",
    [int]$PollSeconds = 15
)

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $PSScriptRoot
$Output = "/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/full_pre_softmax_qwen3_8b_english_128k_20260718"
$LocalLog = Join-Path $Project "attention_confidence_dashboard\full_pre_softmax_sync.log"

"$(Get-Date -Format o) watcher started" | Out-File -FilePath $LocalLog -Encoding utf8 -Append
while ($true) {
    $State = ssh -o BatchMode=yes -o ConnectTimeout=8 $Remote "if test -f '$Output/launcher.done'; then echo done; elif test -f '$Output/launcher.failed'; then echo failed; else echo running; fi" 2>&1
    if ($LASTEXITCODE -ne 0) {
        "$(Get-Date -Format o) ssh unavailable: $State" | Out-File -FilePath $LocalLog -Encoding utf8 -Append
        Start-Sleep -Seconds $PollSeconds
        continue
    }
    if ($State -match "failed") {
        "$(Get-Date -Format o) remote experiment failed" | Out-File -FilePath $LocalLog -Encoding utf8 -Append
        exit 1
    }
    if ($State -match "done") {
        "$(Get-Date -Format o) remote experiment complete; synchronizing" | Out-File -FilePath $LocalLog -Encoding utf8 -Append
        & (Join-Path $PSScriptRoot "pull_attention_confidence_dashboard_data.ps1") -Remote $Remote -Mode full_pre_softmax_128k -RemoteOutput $Output *>> $LocalLog
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        "$(Get-Date -Format o) synchronization complete" | Out-File -FilePath $LocalLog -Encoding utf8 -Append
        exit 0
    }
    Start-Sleep -Seconds $PollSeconds
}
