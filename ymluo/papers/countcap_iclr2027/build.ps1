$ErrorActionPreference = "Stop"

$PaperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $PaperDir "..\..\..")
$Tectonic = "C:\Users\27814\.codex\.tmp\bundled-marketplaces\openai-bundled\plugins\latex\bin\tectonic.exe"
$BuildDir = Join-Path $PaperDir "build"
$OutputDir = Join-Path $RepoRoot "output\pdf"

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Push-Location $PaperDir
try {
    & python scripts\make_method_figure.py
    if ($LASTEXITCODE -ne 0) { throw "Method figure generation failed." }

    & python scripts\make_qksieve_speed_figure.py
    if ($LASTEXITCODE -ne 0) { throw "Speed figure generation failed." }

    & python scripts\make_qksieve_system_figure.py
    if ($LASTEXITCODE -ne 0) { throw "System figure generation failed." }

    & python scripts\make_qksieve_rtx3090_system_rows.py
    if ($LASTEXITCODE -ne 0) { throw "RTX 3090 system table generation failed." }

    & python scripts\make_qksieve_256k_oracle_gap_figure.py `
        --summary data\qksieve_256k_oracle_gap.json `
        --output_pdf figures\qksieve_256k_oracle_gap.pdf `
        --output_png figures\qksieve_256k_oracle_gap.png
    if ($LASTEXITCODE -ne 0) { throw "256K oracle-gap figure generation failed." }

    & $Tectonic -X compile main.tex --outdir $BuildDir --keep-logs --keep-intermediates
    if ($LASTEXITCODE -ne 0) { throw "Anonymous PDF compilation failed." }

    & $Tectonic -X compile main_author.tex --outdir $BuildDir --keep-logs --keep-intermediates
    if ($LASTEXITCODE -ne 0) { throw "Author PDF compilation failed." }
}
finally {
    Pop-Location
}

Copy-Item -Force (Join-Path $BuildDir "main.pdf") `
    (Join-Path $OutputDir "QKSieve_ICLR2027_draft_anonymous.pdf")
Copy-Item -Force (Join-Path $BuildDir "main_author.pdf") `
    (Join-Path $OutputDir "QKSieve_ICLR2027_draft_author.pdf")

Write-Host "Built:"
Write-Host (Join-Path $OutputDir "QKSieve_ICLR2027_draft_anonymous.pdf")
Write-Host (Join-Path $OutputDir "QKSieve_ICLR2027_draft_author.pdf")
