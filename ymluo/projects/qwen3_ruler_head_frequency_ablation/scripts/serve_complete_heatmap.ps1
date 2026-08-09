$ErrorActionPreference = "Stop"

$Port = 4196
$Root = "C:\Users\27814\.codex\visualizations\2026\07\18\019f7582-4120-7bc0-8df4-ec00ebf1ab44"
$PidFile = Join-Path $Root "viewer-server-4196.pid"

$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if (-not $listener) {
    $python = (Get-Command python).Source
    $process = Start-Process `
        -FilePath $python `
        -ArgumentList @("-m", "http.server", "$Port", "--bind", "127.0.0.1", "--directory", $Root) `
        -WindowStyle Hidden `
        -PassThru
    Set-Content -LiteralPath $PidFile -Value $process.Id
    Start-Sleep -Seconds 1
}

$response = Invoke-WebRequest `
    -UseBasicParsing `
    -Uri "http://127.0.0.1:$Port/rope-head-frequency-complete-preview.html" `
    -TimeoutSec 10
Write-Output "HTTP_STATUS=$($response.StatusCode)"
Write-Output "URL=http://127.0.0.1:$Port/rope-head-frequency-complete-preview.html"
