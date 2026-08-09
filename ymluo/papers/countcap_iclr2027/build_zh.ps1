$ErrorActionPreference = "Stop"

$PaperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $PaperDir "..\..\..")
$Tectonic = "C:\Users\27814\.codex\.tmp\bundled-marketplaces\openai-bundled\plugins\latex\bin\tectonic.exe"
$BuildDir = Join-Path $PaperDir "build_zh"
$OutputDir = Join-Path $RepoRoot "output\pdf"

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Push-Location $PaperDir
try {
    & python scripts\make_qksieve_rtx3090_system_rows.py
    if ($LASTEXITCODE -ne 0) { throw "RTX 3090 system table generation failed." }

    & $Tectonic -X compile main_zh.tex --outdir $BuildDir --keep-logs --keep-intermediates
    if ($LASTEXITCODE -ne 0) { throw "Chinese PDF compilation failed." }
}
finally {
    Pop-Location
}

$SourcePdf = Join-Path $BuildDir "main_zh.pdf"
$TargetPdf = Join-Path $OutputDir "QKSieve_ICLR2027_Chinese_Reading_Version.pdf"
Copy-Item -Force $SourcePdf $TargetPdf

Write-Host "Built:"
Write-Host $TargetPdf
