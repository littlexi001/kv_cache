param(
    [int]$Port = 3000
)

$ErrorActionPreference = "Stop"
$project = (Resolve-Path (Join-Path $PSScriptRoot "..\attention_confidence_dashboard")).Path
$runtime = Join-Path $HOME ".cache\codex-runtimes\codex-primary-runtime\dependencies"
$nodeBin = Join-Path $runtime "node\bin"
$overrideBin = Join-Path $runtime "bin\override"
$fallbackBin = Join-Path $runtime "bin\fallback"
$env:PATH = "$nodeBin;$overrideBin;$fallbackBin;$env:PATH"
$env:WRANGLER_LOG_PATH = ".wrangler/wrangler.log"

Set-Location $project
pnpm exec vinext dev --host 127.0.0.1 --port $Port
