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
    & python scripts\make_figures.py
    if ($LASTEXITCODE -ne 0) { throw "Figure generation failed." }

    & $Tectonic -X compile main.tex --outdir $BuildDir --keep-logs --keep-intermediates
    if ($LASTEXITCODE -ne 0) { throw "Anonymous PDF compilation failed." }

    & $Tectonic -X compile main_author.tex --outdir $BuildDir --keep-logs --keep-intermediates
    if ($LASTEXITCODE -ne 0) { throw "Author PDF compilation failed." }
}
finally {
    Pop-Location
}

$AnonymousPdf = Join-Path $OutputDir "SAGE_RoPE_ICLR2027_draft_anonymous.pdf"
$AuthorPdf = Join-Path $OutputDir "SAGE_RoPE_ICLR2027_draft_author.pdf"
Copy-Item -Force (Join-Path $BuildDir "main.pdf") $AnonymousPdf
Copy-Item -Force (Join-Path $BuildDir "main_author.pdf") $AuthorPdf

Write-Host "Built:"
Write-Host $AnonymousPdf
Write-Host $AuthorPdf

