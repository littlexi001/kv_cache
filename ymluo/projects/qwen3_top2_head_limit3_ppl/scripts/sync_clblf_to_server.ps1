param(
    [string]$HostName = "10.176.34.117",
    [string]$UserName = "u21307130306",
    [string]$RemoteProject = "/home/u21307130306/kvcache/kv_cache/ymluo/projects/qwen3_top2_head_limit3_ppl",
    [string]$RemoteLog = "/home/u21307130306/kvcache/kv_cache/remote-change-log.md"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$RepoRoot = Resolve-Path (Join-Path $ProjectRoot "..\..\..")
$Target = "${UserName}@${HostName}"

$files = @(
    @{
        Local = Join-Path $ProjectRoot "src\evaluate_qwen3_top2_head_limit3_ppl.py"
        Remote = "${RemoteProject}/src/evaluate_qwen3_top2_head_limit3_ppl.py"
    },
    @{
        Local = Join-Path $ProjectRoot "src\select_safe_layer_budget.py"
        Remote = "${RemoteProject}/src/select_safe_layer_budget.py"
    },
    @{
        Local = Join-Path $ProjectRoot "scripts\run_clblf_safe_layer_search_server.sh"
        Remote = "${RemoteProject}/scripts/run_clblf_safe_layer_search_server.sh"
    },
    @{
        Local = Join-Path $ProjectRoot "docs\icml_clblf_candidate.md"
        Remote = "${RemoteProject}/docs/icml_clblf_candidate.md"
    },
    @{
        Local = Join-Path $RepoRoot "remote-change-log.md"
        Remote = $RemoteLog
    }
)

ssh -o BatchMode=yes -o ConnectTimeout=10 $Target "mkdir -p '${RemoteProject}/src' '${RemoteProject}/scripts' '${RemoteProject}/docs'"

foreach ($file in $files) {
    if (-not (Test-Path $file.Local)) {
        throw "Missing local file: $($file.Local)"
    }
    scp $file.Local "${Target}:$($file.Remote)"
}

ssh $Target "cd '${RemoteProject}' && python -m py_compile src/evaluate_qwen3_top2_head_limit3_ppl.py src/select_safe_layer_budget.py && chmod +x scripts/run_clblf_safe_layer_search_server.sh"

Write-Host "Synced CLB-LF files to ${Target}:${RemoteProject}"
