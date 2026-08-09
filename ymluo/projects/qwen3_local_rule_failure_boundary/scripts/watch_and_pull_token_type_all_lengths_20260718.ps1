param(
    [string]$RemoteHost = "fdong@10.176.37.31",
    [string]$RemoteOutput = "/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary/outputs/token_type_pre_softmax_all_lengths_20260718",
    [int]$PollSeconds = 60
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$destination = Join-Path $projectRoot "attention_confidence_dashboard\public\data\english_single_token"
$marker = Join-Path $destination ".token_type_all_lengths_synced"

New-Item -ItemType Directory -Force -Path $destination | Out-Null

while ($true) {
    $status = (& ssh $RemoteHost "if test -f '$RemoteOutput/launcher.done'; then echo done; elif test -f '$RemoteOutput/launcher.failed'; then echo failed; else echo running; fi").Trim()
    if ($LASTEXITCODE -ne 0) {
        Write-Output "$(Get-Date -Format o) ssh status check failed; retrying"
        Start-Sleep -Seconds $PollSeconds
        continue
    }
    if ($status -eq "failed") {
        throw "remote token-type attention experiment failed; inspect $RemoteOutput/logs"
    }
    if ($status -eq "done") {
        & scp -r "${RemoteHost}:$RemoteOutput/site_data/token_type_all_lengths" $destination
        if ($LASTEXITCODE -ne 0) {
            throw "scp of token_type_all_lengths failed with exit code $LASTEXITCODE"
        }
        Set-Content -Path $marker -Value "$(Get-Date -Format o)`n$RemoteOutput" -Encoding UTF8
        Write-Output "$(Get-Date -Format o) synchronized token_type_all_lengths to $destination"
        break
    }
    Write-Output "$(Get-Date -Format o) remote experiment still running"
    Start-Sleep -Seconds $PollSeconds
}
