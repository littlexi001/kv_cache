$ErrorActionPreference = "Stop"

$RepoDir = Resolve-Path (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "..\..\..")
$VenvDir = Join-Path $RepoDir ".venv-qwen3"

if (-not (Test-Path $VenvDir)) {
  python -m venv --system-site-packages $VenvDir
}

$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install "numpy<2,>=1.26.0" "urllib3<1.27,>=1.25.4" "transformers==4.51.3" "huggingface_hub<1.0,>=0.30.0" "tokenizers<0.22,>=0.21" "accelerate>=0.26.0" safetensors sentencepiece protobuf

Write-Host "Python ready: $PythonExe"
Write-Host "Use: `$env:PYTHON_EXE='$PythonExe'; .\ymluo\projects\qwen3_top1_category_ablation\scripts\run_ablation.ps1"
