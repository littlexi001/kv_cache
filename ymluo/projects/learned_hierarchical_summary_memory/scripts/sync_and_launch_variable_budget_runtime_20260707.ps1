param(
    [string]$RemoteHost = "10.176.37.31",
    [string]$RemoteUser = "fdong",
    [string]$RemoteRoot = "/home/fdong/ymluo/projects/learned_hierarchical_summary_memory",
    [string]$RemotePython = "/home/fdong/miniconda3/envs/moe/bin/python",
    [int]$MaxAttempts = 6,
    [int]$RetryDelaySeconds = 10
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
        if ($MaxAttempts -gt 1) {
            Write-Host "Attempt $attempt/$MaxAttempts"
        }
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
$connectCommand = @("ssh") + $SshOptions + @($RemoteTarget, "date; hostname")
Run-Checked $connectCommand

$files = @(
    @{ Local = "src/run_rope_aware_kv_repack_benchmark.py"; Remote = "src/run_rope_aware_kv_repack_benchmark.py" },
    @{ Local = "scripts/summarize_runtime_scaling_20260707.py"; Remote = "scripts/summarize_runtime_scaling_20260707.py" },
    @{ Local = "scripts/run_variable_budget_runtime_sweep_20260707.sh"; Remote = "scripts/run_variable_budget_runtime_sweep_20260707.sh" },
    @{ Local = "scripts/run_variable_budget_longbench_mixed_expansion_20260707.sh"; Remote = "scripts/run_variable_budget_longbench_mixed_expansion_20260707.sh" },
    @{ Local = "scripts/run_variable_budget_ruler_scaling_20260707.sh"; Remote = "scripts/run_variable_budget_ruler_scaling_20260707.sh" },
    @{ Local = "scripts/run_variable_budget_ruler_expansion_m5_m3_20260707.sh"; Remote = "scripts/run_variable_budget_ruler_expansion_m5_m3_20260707.sh" },
    @{ Local = "scripts/run_variable_budget_ruler_m5_minsafe_20260707.sh"; Remote = "scripts/run_variable_budget_ruler_m5_minsafe_20260707.sh" },
    @{ Local = "scripts/run_variable_budget_ruler_m5_conformal_floor2_20260707.sh"; Remote = "scripts/run_variable_budget_ruler_m5_conformal_floor2_20260707.sh" },
    @{ Local = "scripts/run_variable_budget_16k_missing_recovery_20260707.sh"; Remote = "scripts/run_variable_budget_16k_missing_recovery_20260707.sh" },
    @{ Local = "scripts/run_output_verifier_floor_sweep_20260707.sh"; Remote = "scripts/run_output_verifier_floor_sweep_20260707.sh" },
    @{ Local = "scripts/run_ruler_scaling_expansion_20260707.sh"; Remote = "scripts/run_ruler_scaling_expansion_20260707.sh" },
    @{ Local = "scripts/run_ruler16k_case1_shards_20260707.sh"; Remote = "scripts/run_ruler16k_case1_shards_20260707.sh" },
    @{ Local = "scripts/run_ruler16k_floor2_recovery_20260707.sh"; Remote = "scripts/run_ruler16k_floor2_recovery_20260707.sh" }
)

foreach ($item in $files) {
    $localPath = Join-Path $ProjectRoot $item.Local
    $remotePath = "${RemoteTarget}:$RemoteRoot/$($item.Remote)"
    $scpCommand = @("scp") + $SshOptions + @($localPath, $remotePath)
    Run-Checked $scpCommand
}

$remoteCheck = @"
cd $RemoteRoot &&
$RemotePython -m py_compile src/run_rope_aware_kv_repack_benchmark.py scripts/summarize_runtime_scaling_20260707.py &&
bash -n scripts/run_variable_budget_runtime_sweep_20260707.sh &&
bash -n scripts/run_variable_budget_longbench_mixed_expansion_20260707.sh &&
bash -n scripts/run_variable_budget_ruler_scaling_20260707.sh &&
bash -n scripts/run_variable_budget_ruler_expansion_m5_m3_20260707.sh &&
bash -n scripts/run_variable_budget_ruler_m5_minsafe_20260707.sh &&
bash -n scripts/run_variable_budget_ruler_m5_conformal_floor2_20260707.sh &&
bash -n scripts/run_variable_budget_16k_missing_recovery_20260707.sh &&
bash -n scripts/run_output_verifier_floor_sweep_20260707.sh &&
bash -n scripts/run_ruler_scaling_expansion_20260707.sh &&
bash -n scripts/run_ruler16k_case1_shards_20260707.sh &&
bash -n scripts/run_ruler16k_floor2_recovery_20260707.sh &&
chmod +x scripts/run_variable_budget_runtime_sweep_20260707.sh scripts/run_variable_budget_longbench_mixed_expansion_20260707.sh scripts/run_variable_budget_ruler_scaling_20260707.sh scripts/run_variable_budget_ruler_expansion_m5_m3_20260707.sh scripts/run_variable_budget_ruler_m5_minsafe_20260707.sh scripts/run_variable_budget_ruler_m5_conformal_floor2_20260707.sh scripts/run_variable_budget_16k_missing_recovery_20260707.sh scripts/run_output_verifier_floor_sweep_20260707.sh scripts/run_ruler_scaling_expansion_20260707.sh scripts/run_ruler16k_case1_shards_20260707.sh scripts/run_ruler16k_floor2_recovery_20260707.sh
"@ -replace "`r?`n", " "
$remoteCheckCommand = @("ssh") + $SshOptions + @($RemoteTarget, $remoteCheck)
Run-Checked $remoteCheckCommand

$remoteLaunch = @"
cd $RemoteRoot &&
mkdir -p outputs &&
if ps -u $RemoteUser -o cmd | grep -F 'run_variable_budget_runtime_sweep_20260707.sh' | grep -v grep >/dev/null; then
  echo 'variable-budget runtime sweep is already running'
  exit 0
fi &&
setsid scripts/run_variable_budget_runtime_sweep_20260707.sh > outputs/variable_budget_runtime_sweep_20260707.nohup.log 2>&1 < /dev/null &
echo `$!
"@ -replace "`r?`n", " "

Write-Host "Launching variable-budget runtime sweep..."
$remoteLaunchCommand = @("ssh") + $SshOptions + @($RemoteTarget, $remoteLaunch)
Run-Checked $remoteLaunchCommand

Write-Host "Launch requested. Monitor with:"
Write-Host "ssh $RemoteTarget `"tail -f $RemoteRoot/outputs/variable_budget_runtime_sweep_20260707.nohup.log`""
