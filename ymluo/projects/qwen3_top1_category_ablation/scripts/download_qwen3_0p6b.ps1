$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoDir = Resolve-Path (Join-Path $ScriptDir "..\..\..\..")
$ModelDir = if ($env:MODEL_PATH) { $env:MODEL_PATH } else { Join-Path $RepoDir "ymluo\models\Qwen3-0.6B" }
$Endpoint = if ($env:HF_ENDPOINT) { $env:HF_ENDPOINT.TrimEnd("/") } else { "https://hf-mirror.com" }
$BaseUrl = "$Endpoint/Qwen/Qwen3-0.6B/resolve/main"

New-Item -ItemType Directory -Force $ModelDir | Out-Null

$Files = @(
  ".gitattributes",
  "LICENSE",
  "README.md",
  "config.json",
  "generation_config.json",
  "merges.txt",
  "model.safetensors",
  "tokenizer.json",
  "tokenizer_config.json",
  "vocab.json"
)

foreach ($File in $Files) {
  $OutPath = Join-Path $ModelDir $File
  Write-Host "Downloading $File"
  curl.exe -L --ssl-no-revoke --retry 5 --retry-delay 2 --continue-at - "$BaseUrl/$File" -o $OutPath
  if ($LASTEXITCODE -ne 0) {
    throw "curl failed for $File with exit code $LASTEXITCODE"
  }
}

Write-Host "Downloaded Qwen/Qwen3-0.6B to $ModelDir"
