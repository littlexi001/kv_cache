param(
    [int]$MaxAttempts = 360,
    [int]$SleepSeconds = 120
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Resolve-Path (Join-Path $ScriptDir "..")
$Remote = "fdong@10.176.37.31"
$RemoteHost = "10.176.37.31"
$RemoteRoot = "/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl"
$LogDir = Join-Path $Root "outputs\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir "retry_launch_b16_windowvote_20260711.log"

function Write-Log {
    param([string]$Message)
    $Line = "$(Get-Date -Format o) $Message"
    Add-Content -Path $LogPath -Value $Line
}

function Invoke-Checked {
    param(
        [string]$Label,
        [string[]]$Command
    )
    Write-Log "RUN $Label :: $($Command -join ' ')"
    & $Command[0] @($Command[1..($Command.Length - 1)]) 2>&1 | ForEach-Object {
        Add-Content -Path $LogPath -Value $_
    }
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

function Invoke-RemoteScript {
    param(
        [string]$Label,
        [string]$Script
    )
    Write-Log "RUN_REMOTE $Label"
    $Script | ssh -o ConnectTimeout=30 $Remote "bash -s" 2>&1 | ForEach-Object {
        Add-Content -Path $LogPath -Value $_
    }
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

Write-Log "retry launcher started root=$Root"

for ($Attempt = 1; $Attempt -le $MaxAttempts; $Attempt++) {
    try {
        Write-Log "attempt=$Attempt checking ${RemoteHost}:22"
        $Reachable = Test-NetConnection $RemoteHost -Port 22 -InformationLevel Quiet
        if (-not $Reachable) {
            Write-Log "server not reachable"
            Start-Sleep -Seconds $SleepSeconds
            continue
        }

        Push-Location $Root
        try {
            Invoke-Checked "scp-src" @(
                "scp",
                "src\run_controlled_public_kv_benchmark_v1.py",
                "${Remote}:${RemoteRoot}/src/run_controlled_public_kv_benchmark_v1.py"
            )
            Invoke-Checked "scp-configs" @(
                "scp",
                "configs\riskkv_task_policy_v312_b16_windowvote_quality_20260711.json",
                "configs\riskkv_task_policy_v313_b16_windowvote_speed_20260711.json",
                "${Remote}:${RemoteRoot}/configs/"
            )
            Invoke-Checked "scp-scripts" @(
                "scp",
                "scripts\launch_b16_windowvote_sweep_20260711.sh",
                "scripts\watch_combine_b16_windowvote_sweep_20260711.sh",
                "scripts\update_riskkv_live_dashboard_20260711.py",
                "${Remote}:${RemoteRoot}/scripts/"
            )
            Invoke-Checked "scp-doc" @(
                "scp",
                "..\..\doc\section155_b16_window_vote_span_repack_20260711.md",
                "${Remote}:/home/fdong/ymluo/doc/section155_b16_window_vote_span_repack_20260711.md"
            )
        }
        finally {
            Pop-Location
        }

        Invoke-RemoteScript "validate" @"
cd "$RemoteRoot" || exit 1
/home/fdong/miniconda3/envs/moe/bin/python -m py_compile src/run_controlled_public_kv_benchmark_v1.py scripts/update_riskkv_live_dashboard_20260711.py
/home/fdong/miniconda3/envs/moe/bin/python -m json.tool configs/riskkv_task_policy_v312_b16_windowvote_quality_20260711.json >/dev/null
/home/fdong/miniconda3/envs/moe/bin/python -m json.tool configs/riskkv_task_policy_v313_b16_windowvote_speed_20260711.json >/dev/null
bash -n scripts/launch_b16_windowvote_sweep_20260711.sh
bash -n scripts/watch_combine_b16_windowvote_sweep_20260711.sh
chmod +x scripts/launch_b16_windowvote_sweep_20260711.sh scripts/watch_combine_b16_windowvote_sweep_20260711.sh
echo VALIDATION_OK
"@

        Invoke-RemoteScript "launch" @"
cd "$RemoteRoot" || exit 1
mkdir -p outputs/logs
nohup env SAMPLES=100 GPUS=0,1,2,3,4,5,6,7 bash scripts/launch_b16_windowvote_sweep_20260711.sh > outputs/logs/launch_b16_windowvote_sweep_20260711.log 2>&1 < /dev/null &
nohup bash scripts/watch_combine_b16_windowvote_sweep_20260711.sh > outputs/logs/watch_combine_b16_windowvote_sweep_20260711.log 2>&1 < /dev/null &
nohup bash scripts/watch_update_riskkv_live_dashboard_20260711.sh > outputs/logs/watch_update_riskkv_live_dashboard_20260711.log 2>&1 < /dev/null &
echo STARTED_B16_WINDOWVOTE
"@

        Write-Log "launch completed"
        exit 0
    }
    catch {
        Write-Log "attempt failed: $($_.Exception.Message)"
        Start-Sleep -Seconds $SleepSeconds
    }
}

Write-Log "giving up after $MaxAttempts attempts"
exit 1
