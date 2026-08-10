$ErrorActionPreference = "Stop"

$PaperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $PaperDir "..\..\..")
$ProjectDir = Join-Path $RepoRoot "ymluo\projects\qwen3_top2_head_limit3_ppl"
$DataDir = Join-Path $PaperDir "data"
$GeneratedDir = Join-Path $DataDir "generated"
$PdfDir = Join-Path $RepoRoot "output\pdf"

$PersistentSummary = Join-Path $ProjectDir `
    "docs\qksieve_persistent_kv_20260810\raw_results\20260810_qksieve_persistent_kv_v2\independent_summary.json"
$LongBenchSummary = Join-Path $DataDir "qksieve_robust_longbench_summary.json"
$RulerSummary = Join-Path $DataDir "qksieve_robust_ruler_summary.json"
$MultimodelSummary = Join-Path $DataDir "qksieve_robust_multimodel_summary.json"
$ShrinkageSummary = Join-Path $DataDir "qksieve_shrinkage_sensitivity_summary.json"
$H100Summary = Join-Path $DataDir "qksieve_h100_matched_summary.json"
$EvidenceReport = Join-Path $GeneratedDir "qksieve_frozen_evidence_report.json"
$PdfAudit = Join-Path $GeneratedDir "qksieve_final_pdf_audit.json"

$RequiredEvidence = @(
    $PersistentSummary,
    $LongBenchSummary,
    $RulerSummary,
    $MultimodelSummary,
    $ShrinkageSummary,
    $H100Summary
)
foreach ($Path in $RequiredEvidence) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required frozen evidence is missing: $Path"
    }
}

$TexContracts = @(
    @{
        Path = Join-Path $PaperDir "sections\05_experiments.tex"
        Snippets = @(
            "\input{data/generated/qksieve_quality_tables.tex}",
            "\input{data/generated/qksieve_h100_tables.tex}"
        )
    },
    @{
        Path = Join-Path $PaperDir "sections\appendix.tex"
        Snippets = @("\input{data/generated/qksieve_quality_appendix.tex}")
    },
    @{
        Path = Join-Path $PaperDir "sections_zh\05_experiments.tex"
        Snippets = @(
            "\input{data/generated/qksieve_quality_tables_zh.tex}",
            "\input{data/generated/qksieve_h100_tables_zh.tex}"
        )
    },
    @{
        Path = Join-Path $PaperDir "sections_zh\appendix.tex"
        Snippets = @("\input{data/generated/qksieve_quality_appendix_zh.tex}")
    }
)
foreach ($Contract in $TexContracts) {
    $Text = Get-Content -Raw -Encoding UTF8 -LiteralPath $Contract.Path
    foreach ($Snippet in $Contract.Snippets) {
        if (-not $Text.Contains($Snippet)) {
            throw "Final paper does not consume generated evidence: $($Contract.Path): $Snippet"
        }
    }
}

New-Item -ItemType Directory -Force -Path $GeneratedDir | Out-Null

Push-Location $PaperDir
try {
    & python "$ProjectDir\src\verify_qksieve_robust_paper_evidence_20260810.py" `
        --project_root $ProjectDir `
        --persistent_summary $PersistentSummary `
        --longbench_summary $LongBenchSummary `
        --ruler_summary $RulerSummary `
        --multimodel_summary $MultimodelSummary `
        --shrinkage_summary $ShrinkageSummary `
        --h100_summary $H100Summary `
        --output $EvidenceReport
    if ($LASTEXITCODE -ne 0) { throw "Frozen evidence verification failed." }

    & python scripts\make_qksieve_quality_tables.py
    if ($LASTEXITCODE -ne 0) { throw "Quality table generation failed." }

    & python scripts\make_qksieve_quality_generalization_figure.py
    if ($LASTEXITCODE -ne 0) { throw "Quality figure generation failed." }

    & python scripts\make_qksieve_h100_tables.py
    if ($LASTEXITCODE -ne 0) { throw "H100 table generation failed." }

    & python scripts\make_qksieve_rtx3090_system_rows.py
    if ($LASTEXITCODE -ne 0) { throw "RTX 3090 table generation failed." }
}
finally {
    Pop-Location
}

& (Join-Path $PaperDir "build.ps1")
if ($LASTEXITCODE -ne 0) { throw "English paper build failed." }

& (Join-Path $PaperDir "build_zh.ps1")
if ($LASTEXITCODE -ne 0) { throw "Chinese paper build failed." }

& python (Join-Path $PaperDir "scripts\audit_qksieve_final_pdf.py") `
    --anonymous (Join-Path $PdfDir "QKSieve_ICLR2027_draft_anonymous.pdf") `
    --author (Join-Path $PdfDir "QKSieve_ICLR2027_draft_author.pdf") `
    --chinese (Join-Path $PdfDir "QKSieve_ICLR2027_Chinese_Reading_Version.pdf") `
    --output $PdfAudit
if ($LASTEXITCODE -ne 0) { throw "Final PDF audit failed." }

Write-Host "Frozen evidence, generated tables, figures, and PDFs passed."
Write-Host $EvidenceReport
Write-Host $PdfAudit
