param(
    [string]$RemoteHost = "10.176.37.31",
    [string]$RemoteUser = "fdong",
    [string]$RemoteRoot = "/home/fdong/ymluo/projects/learned_hierarchical_summary_memory",
    [string]$RemotePython = "/home/fdong/miniconda3/envs/moe/bin/python",
    [string]$LocalOutputDir = "ymluo/projects/learned_hierarchical_summary_memory/outputs/remote_runtime_status_20260707",
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
$LocalOutputPath = Resolve-Path -Path "." | ForEach-Object { Join-Path $_ $LocalOutputDir }
New-Item -ItemType Directory -Force -Path $LocalOutputPath | Out-Null

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

$remoteStatus = @"
cd $RemoteRoot &&
echo '=== date ===' &&
date &&
echo '=== gpu ===' &&
(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits || true) &&
echo '=== active processes ===' &&
 (ps -u $RemoteUser -o pid,ppid,stat,etime,cmd | grep -E 'run_rope_aware|variable_budget_runtime|variable_budget_ruler|ruler16k|ruler_scaling|output_verifier_floor' | grep -v grep || true) &&
echo '=== expected summaries ===' &&
for d in
outputs/variable_budget_runtime_qwen8b_longbench_m4_bestcal_tail035_20260707
outputs/variable_budget_runtime_qwen8b_longbench_m4_minsafe_tail035_20260707
outputs/variable_budget_runtime_qwen8b_longbench_m4_conformal_auto_20260707
outputs/variable_budget_runtime_qwen8b_longbench_m8_bestcal_tail035_20260707
outputs/variable_budget_runtime_qwen8b_longbench_m8_minsafe_tail035_20260707
outputs/variable_budget_runtime_qwen8b_longbench_m8_conformal_auto_20260707
outputs/variable_budget_runtime_qwen8b_mixed13_m1_bestcal_tail035_20260707
outputs/variable_budget_runtime_qwen8b_mixed13_m1_minsafe_tail035_20260707
outputs/variable_budget_runtime_qwen8b_mixed13_m1_conformal_auto_20260707
outputs/variable_budget_runtime_qwen8b_mixed13_m2_minsafe_tail035_20260707
outputs/variable_budget_runtime_qwen8b_mixed13_m2_conformal_auto_20260707
outputs/variable_budget_runtime_qwen8b_ruler8k_m1_bestcal_tail035_20260707
outputs/variable_budget_runtime_qwen8b_ruler8k_m1_conformal_auto_20260707
outputs/variable_budget_runtime_qwen8b_ruler4096_m5_bestcal_tail035_20260707
outputs/variable_budget_runtime_qwen8b_ruler8192_m5_bestcal_tail035_20260707
outputs/variable_budget_runtime_qwen8b_ruler4096_m5_minsafe_tail035_20260707
outputs/variable_budget_runtime_qwen8b_ruler8192_m5_minsafe_tail035_20260707
outputs/variable_budget_runtime_qwen8b_ruler4096_m5_conformal_auto_20260707
outputs/variable_budget_runtime_qwen8b_ruler8192_m5_conformal_auto_20260707
outputs/variable_budget_runtime_qwen8b_ruler4096_m5_conformal_floor2_20260707
outputs/variable_budget_runtime_qwen8b_ruler8192_m5_conformal_floor2_20260707
outputs/output_verifier_runtime_qwen8b_ruler_4096_m3_tau07_prefix_floor2_20260707
outputs/output_verifier_runtime_qwen8b_ruler_8192_m3_tau07_prefix_floor2_20260707; do
  if [ -f "`$d/summary.csv" ]; then
    echo "DONE `$d";
  else
    echo "MISSING `$d";
  fi;
done &&
echo '=== run aggregate ===' &&
$RemotePython scripts/summarize_runtime_scaling_20260707.py --root .
"@ -replace "`r?`n", " "

$remoteStatusCommand = @("ssh") + $SshOptions + @($RemoteTarget, $remoteStatus)
Run-Checked $remoteStatusCommand

$remoteSummaryDir = "$RemoteRoot/outputs/runtime_scaling_summary_20260707"
$scpCommand = @("scp") + $SshOptions + @("-r", "${RemoteTarget}:$remoteSummaryDir", $LocalOutputPath)
Run-Checked $scpCommand

$SummaryRoot = Join-Path $LocalOutputPath "runtime_scaling_summary_20260707"
$SummaryCsv = Join-Path $SummaryRoot "runtime_scaling_summary.csv"
$TableOutputDir = Join-Path $SummaryRoot "icml_tables"
$FigureOutputDir = Join-Path $SummaryRoot "icml_figures"
$ReadinessOutputDir = Join-Path $SummaryRoot "icml_readiness"
$OverheadOutputDir = Join-Path $SummaryRoot "icml_overhead"
$PaperTableOutputDir = Join-Path $SummaryRoot "icml_paper_tables"

Run-Checked @(
    "python",
    (Join-Path $ProjectRoot "scripts/make_icml_runtime_tables_20260707.py"),
    "--summary_csv",
    $SummaryCsv,
    "--output_dir",
    $TableOutputDir
)
Run-Checked @(
    "python",
    (Join-Path $ProjectRoot "scripts/plot_icml_runtime_figures_20260707.py"),
    "--summary_csv",
    $SummaryCsv,
    "--output_dir",
    $FigureOutputDir
)
Run-Checked @(
    "python",
    (Join-Path $ProjectRoot "scripts/make_icml_readiness_report_20260707.py"),
    "--summary_csv",
    $SummaryCsv,
    "--output_dir",
    $ReadinessOutputDir
)
Run-Checked @(
    "python",
    (Join-Path $ProjectRoot "scripts/make_icml_overhead_report_20260707.py"),
    "--summary_csv",
    $SummaryCsv,
    "--output_dir",
    $OverheadOutputDir
)
Run-Checked @(
    "python",
    (Join-Path $ProjectRoot "scripts/make_icml_paper_tables_20260707.py"),
    "--summary_csv",
    $SummaryCsv,
    "--output_dir",
    $PaperTableOutputDir
)

Write-Host "Collected summary under $LocalOutputPath"
Write-Host "Generated ICML tables under $TableOutputDir"
Write-Host "Generated ICML figures under $FigureOutputDir"
Write-Host "Generated ICML readiness report under $ReadinessOutputDir"
Write-Host "Generated ICML overhead report under $OverheadOutputDir"
Write-Host "Generated ICML paper tables under $PaperTableOutputDir"
