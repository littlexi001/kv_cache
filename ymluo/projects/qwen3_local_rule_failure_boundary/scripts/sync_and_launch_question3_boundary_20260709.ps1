param(
    [string]$RemoteHost = "10.176.37.31",
    [string]$RemoteUser = "fdong",
    [string]$RemoteRoot = "/home/fdong/ymluo/projects/qwen3_local_rule_failure_boundary",
    [string]$RemotePython = "/home/fdong/miniconda3/envs/moe/bin/python",
    [ValidateSet("smoke", "phase1", "qwen8b", "none")]
    [string]$Launch = "phase1",
    [string]$Gpu = "",
    [int]$MaxAttempts = 4,
    [int]$RetryDelaySeconds = 8
)

$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptRoot "..")
$RemoteTarget = "$RemoteUser@$RemoteHost"
$SshOptions = @(
    "-o", "ConnectTimeout=15",
    "-o", "ServerAliveInterval=10",
    "-o", "ServerAliveCountMax=3"
)

function Run-Checked {
    param([string[]]$Command)
    $lastExit = 1
    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        Write-Host ("+ " + ($Command -join " "))
        & $Command[0] @($Command | Select-Object -Skip 1)
        $lastExit = $LASTEXITCODE
        if ($lastExit -eq 0) {
            return
        }
        if ($attempt -lt $MaxAttempts) {
            Write-Warning "Command failed with exit code $lastExit; retrying in $RetryDelaySeconds seconds"
            Start-Sleep -Seconds $RetryDelaySeconds
        }
    }
    throw "Command failed with exit code $lastExit"
}

Write-Host "Checking SSH connectivity to $RemoteTarget..."
Run-Checked (@("ssh") + $SshOptions + @($RemoteTarget, "date; hostname"))

Run-Checked (@("ssh") + $SshOptions + @($RemoteTarget, "mkdir -p '$RemoteRoot/src' '$RemoteRoot/scripts' '$RemoteRoot/doc' '$RemoteRoot/outputs'"))

$files = @(
    @{ Local = "README.md"; Remote = "README.md" },
    @{ Local = "doc/experiment_design_20260709.md"; Remote = "doc/experiment_design_20260709.md" },
    @{ Local = "src/run_local_rule_failure_boundary.py"; Remote = "src/run_local_rule_failure_boundary.py" },
    @{ Local = "scripts/summarize_question3_boundary.py"; Remote = "scripts/summarize_question3_boundary.py" },
    @{ Local = "scripts/run_question3_boundary_smoke_server.sh"; Remote = "scripts/run_question3_boundary_smoke_server.sh" },
    @{ Local = "scripts/run_question3_boundary_phase1_qwen06_server.sh"; Remote = "scripts/run_question3_boundary_phase1_qwen06_server.sh" },
    @{ Local = "scripts/run_question3_boundary_qwen8b_compare_server.sh"; Remote = "scripts/run_question3_boundary_qwen8b_compare_server.sh" }
)

foreach ($item in $files) {
    $localPath = Join-Path $ProjectRoot $item.Local
    $remotePath = "${RemoteTarget}:$RemoteRoot/$($item.Remote)"
    Run-Checked (@("scp") + $SshOptions + @($localPath, $remotePath))
}

$remoteCheck = @"
cd $RemoteRoot &&
$RemotePython -m py_compile src/run_local_rule_failure_boundary.py scripts/summarize_question3_boundary.py &&
bash -n scripts/run_question3_boundary_smoke_server.sh &&
bash -n scripts/run_question3_boundary_phase1_qwen06_server.sh &&
bash -n scripts/run_question3_boundary_qwen8b_compare_server.sh &&
chmod +x scripts/run_question3_boundary_smoke_server.sh scripts/run_question3_boundary_phase1_qwen06_server.sh scripts/run_question3_boundary_qwen8b_compare_server.sh
"@ -replace "`r?`n", " "
Run-Checked (@("ssh") + $SshOptions + @($RemoteTarget, $remoteCheck))

if ($Launch -eq "none") {
    Write-Host "Synced and checked. Launch skipped."
    exit 0
}

$script = switch ($Launch) {
    "smoke" { "scripts/run_question3_boundary_smoke_server.sh" }
    "phase1" { "scripts/run_question3_boundary_phase1_qwen06_server.sh" }
    "qwen8b" { "scripts/run_question3_boundary_qwen8b_compare_server.sh" }
}

$gpuPrefix = ""
if ($Gpu -ne "") {
    $gpuPrefix = "CUDA_VISIBLE_DEVICES='$Gpu' "
}

$remoteLaunch = @"
cd $RemoteRoot &&
if ps -u $RemoteUser -o cmd | grep -F '$script' | grep -v grep >/dev/null; then
  echo '$script is already running';
  exit 0;
fi &&
${gpuPrefix}bash $script
"@ -replace "`r?`n", " "

Write-Host "Launching $Launch..."
Run-Checked (@("ssh") + $SshOptions + @($RemoteTarget, $remoteLaunch))
Write-Host "Monitor with:"
$monitorDir = switch ($Launch) {
    "smoke" { "question3_boundary_smoke_20260709" }
    "phase1" { "question3_boundary_qwen06_phase1_20260709" }
    "qwen8b" { "question3_boundary_qwen8b_compare_20260709" }
}
Write-Host "ssh $RemoteTarget `"tail -f $RemoteRoot/outputs/$monitorDir/run.log`""
