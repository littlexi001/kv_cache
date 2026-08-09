$ErrorActionPreference = "Stop"

$Remote = if ($env:RISKKV_REMOTE) { $env:RISKKV_REMOTE } else { "fdong@10.176.37.31" }
$RemoteRoot = if ($env:RISKKV_REMOTE_ROOT) { $env:RISKKV_REMOTE_ROOT } else { "/home/fdong/ymluo/projects/qwen3_top2_head_limit3_ppl" }
$LocalRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

function Retry-Command {
    param(
        [Parameter(Mandatory = $true)] [string] $Command,
        [int] $Attempts = 20,
        [int] $DelaySeconds = 30
    )
    for ($i = 1; $i -le $Attempts; $i++) {
        Write-Host "[$i/$Attempts] $Command"
        powershell -NoProfile -ExecutionPolicy Bypass -Command $Command
        if ($LASTEXITCODE -eq 0) {
            return
        }
        Start-Sleep -Seconds $DelaySeconds
    }
    throw "Command failed after $Attempts attempts: $Command"
}

Push-Location $LocalRoot
try {
    Retry-Command "ssh -o ConnectTimeout=15 $Remote 'mkdir -p $RemoteRoot/configs $RemoteRoot/scripts $RemoteRoot/src /home/fdong/ymluo/doc && echo remote_ready'"

    Retry-Command "scp src/run_controlled_public_kv_benchmark_v1.py ${Remote}:$RemoteRoot/src/run_controlled_public_kv_benchmark_v1.py"
    Retry-Command "scp configs/riskkv_task_policy_v306_repobench_bounded_retry_no_full_20260711.json configs/riskkv_task_policy_v307_b16_purefine_highrecall_20260711.json configs/riskkv_task_policy_v308_b16_purefine_window_highrecall_20260711.json configs/riskkv_task_policy_v309_b16_microspan_repack_quality_20260711.json configs/riskkv_task_policy_v310_b16_microspan_repack_speed_20260711.json ${Remote}:$RemoteRoot/configs/"
    Retry-Command "scp scripts/launch_b16_purefine_sweep_20260711.sh scripts/watch_combine_b16_purefine_sweep_20260711.sh scripts/watch_combine_v306_repobench_retry_20260711.sh scripts/launch_b16_microspan_sweep_20260711.sh scripts/watch_combine_b16_microspan_sweep_20260711.sh ${Remote}:$RemoteRoot/scripts/"
    Retry-Command "scp ..\..\doc\section150_b16_purefine_moreblocks_20260711.md ..\..\doc\section151_microblock_span_repack_20260711.md ${Remote}:/home/fdong/ymluo/doc/"

    $remoteValidate = "cd $RemoteRoot && /home/fdong/miniconda3/envs/moe/bin/python -m py_compile src/run_controlled_public_kv_benchmark_v1.py && /home/fdong/miniconda3/envs/moe/bin/python -m json.tool configs/riskkv_task_policy_v306_repobench_bounded_retry_no_full_20260711.json >/dev/null && /home/fdong/miniconda3/envs/moe/bin/python -m json.tool configs/riskkv_task_policy_v307_b16_purefine_highrecall_20260711.json >/dev/null && /home/fdong/miniconda3/envs/moe/bin/python -m json.tool configs/riskkv_task_policy_v308_b16_purefine_window_highrecall_20260711.json >/dev/null && /home/fdong/miniconda3/envs/moe/bin/python -m json.tool configs/riskkv_task_policy_v309_b16_microspan_repack_quality_20260711.json >/dev/null && /home/fdong/miniconda3/envs/moe/bin/python -m json.tool configs/riskkv_task_policy_v310_b16_microspan_repack_speed_20260711.json >/dev/null && bash -n scripts/launch_b16_purefine_sweep_20260711.sh && bash -n scripts/watch_combine_b16_purefine_sweep_20260711.sh && bash -n scripts/watch_combine_v306_repobench_retry_20260711.sh && bash -n scripts/launch_b16_microspan_sweep_20260711.sh && bash -n scripts/watch_combine_b16_microspan_sweep_20260711.sh && echo remote_validation_ok"
    Retry-Command "ssh -o ConnectTimeout=15 $Remote '$remoteValidate'"

    $remoteLaunch = "cd $RemoteRoot && mkdir -p outputs/logs && chmod +x scripts/launch_b16_purefine_sweep_20260711.sh scripts/watch_combine_b16_purefine_sweep_20260711.sh scripts/watch_combine_v306_repobench_retry_20260711.sh scripts/launch_b16_microspan_sweep_20260711.sh scripts/watch_combine_b16_microspan_sweep_20260711.sh && { nohup bash scripts/launch_b16_purefine_sweep_20260711.sh > outputs/logs/launch_b16_purefine_sweep_20260711.log 2>&1 < /dev/null & nohup bash scripts/watch_combine_b16_purefine_sweep_20260711.sh > outputs/logs/watch_combine_b16_purefine_sweep_20260711.log 2>&1 < /dev/null & nohup env GPUS=0,1,2,3,4,5,6,7 SAMPLES=100 LABEL=v306_repobench_bounded_retry_no_full STAMP=20260711_repobench_retry POLICY=configs/riskkv_task_policy_v306_repobench_bounded_retry_no_full_20260711.json TASKS=repobench-p bash scripts/run_riskkv_task_policy_v19_one_20260709.sh > outputs/logs/launch_v306_repobench_bounded_retry_no_full_20260711_repobench_retry_m100.log 2>&1 < /dev/null & nohup bash scripts/watch_combine_v306_repobench_retry_20260711.sh > outputs/logs/watch_combine_v306_repobench_retry_20260711.log 2>&1 < /dev/null & nohup bash scripts/launch_b16_microspan_sweep_20260711.sh > outputs/logs/launch_b16_microspan_sweep_20260711.log 2>&1 < /dev/null & nohup bash scripts/watch_combine_b16_microspan_sweep_20260711.sh > outputs/logs/watch_combine_b16_microspan_sweep_20260711.log 2>&1 < /dev/null & } && echo submitted_pending_riskkv"
    Retry-Command "ssh -o ConnectTimeout=15 $Remote '$remoteLaunch'"
}
finally {
    Pop-Location
}
