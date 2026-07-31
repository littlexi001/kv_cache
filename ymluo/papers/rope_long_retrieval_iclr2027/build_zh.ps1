$ErrorActionPreference = "Stop"

$PaperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $PaperDir "..\..\..")
$Tectonic = "C:\Users\27814\.codex\.tmp\bundled-marketplaces\openai-bundled\plugins\latex\bin\tectonic.exe"
$BuildDir = Join-Path $PaperDir "build"
$OutputDir = Join-Path $RepoRoot "output\pdf"

if (-not (Test-Path $Tectonic)) {
    throw "Bundled Tectonic was not found at $Tectonic"
}

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Push-Location $PaperDir
try {
    & python scripts\check_bilingual_sync.py
    if ($LASTEXITCODE -ne 0) { throw "Bilingual synchronization check failed." }

    & python scripts\make_figures.py
    if ($LASTEXITCODE -ne 0) { throw "Figure generation failed." }

    & python scripts\make_figures_zh.py
    if ($LASTEXITCODE -ne 0) { throw "Chinese figure generation failed." }

    & $Tectonic -X compile main_zh.tex --outdir $BuildDir --keep-logs --keep-intermediates
    if ($LASTEXITCODE -ne 0) { throw "Chinese companion PDF compilation failed." }
}
finally {
    Pop-Location
}

$ChinesePdf = Join-Path $OutputDir "SAGE_RoPE_ICLR2027_draft_zh.pdf"
Copy-Item -Force (Join-Path $BuildDir "main_zh.pdf") $ChinesePdf

Write-Host "Built:"
Write-Host $ChinesePdf
