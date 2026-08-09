param(
    [string]$Remote = "fdong@10.176.37.31",
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $PSScriptRoot
$RemoteRoot = "/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/attention_confidence_qwen3_8b_fixed_relative_328_128k_20260719"
$Destination = Join-Path $Project "artifacts\20260719_fixed_relative_328_128k"
$LocalLog = Join-Path $Project "artifacts\20260719_fixed_relative_328_128k_sync.log"

"$(Get-Date -Format o) watcher started" | Out-File -FilePath $LocalLog -Encoding utf8 -Append
while ($true) {
    $status = (& ssh -o BatchMode=yes -o ConnectTimeout=10 $Remote "if test -f '$RemoteRoot/launcher.done'; then echo done; elif test -f '$RemoteRoot/launcher.failed'; then echo failed; else echo running; fi" 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0) {
        "$(Get-Date -Format o) ssh unavailable: $status" | Out-File -FilePath $LocalLog -Encoding utf8 -Append
        Start-Sleep -Seconds $PollSeconds
        continue
    }
    if ($status -eq "failed") {
        New-Item -ItemType Directory -Force -Path $Destination | Out-Null
        & scp -r "${Remote}:$RemoteRoot/logs" $Destination *>> $LocalLog
        "$(Get-Date -Format o) remote experiment failed" | Out-File -FilePath $LocalLog -Encoding utf8 -Append
        exit 1
    }
    if ($status -eq "done") {
        New-Item -ItemType Directory -Force -Path $Destination | Out-Null
        & scp -r "${Remote}:$RemoteRoot/analysis" $Destination *>> $LocalLog
        & scp "${Remote}:$RemoteRoot/analysis_report.md" $Destination *>> $LocalLog
        & scp "${Remote}:$RemoteRoot/analysis_summary.csv" $Destination *>> $LocalLog
        & scp "${Remote}:$RemoteRoot/config_shard*.json" $Destination *>> $LocalLog
        & scp "${Remote}:$RemoteRoot/launcher.done" $Destination *>> $LocalLog
        if ($LASTEXITCODE -ne 0) {
            "$(Get-Date -Format o) synchronization failed" | Out-File -FilePath $LocalLog -Encoding utf8 -Append
            exit 1
        }
        "$(Get-Date -Format o) synchronization complete" | Out-File -FilePath $LocalLog -Encoding utf8 -Append
        exit 0
    }
    "$(Get-Date -Format o) remote experiment running" | Out-File -FilePath $LocalLog -Encoding utf8 -Append
    Start-Sleep -Seconds $PollSeconds
}
