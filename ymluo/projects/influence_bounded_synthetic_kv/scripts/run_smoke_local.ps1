param(
  [string]$Python = "python",
  [string]$OutputDir = "ymluo/projects/influence_bounded_synthetic_kv/outputs/smoke",
  [int]$RemoteTokens = 2048,
  [int]$TrainQueries = 128,
  [int]$TestQueries = 128,
  [int]$Dim = 64,
  [int]$ValueDim = 64,
  [int]$Prototypes = 16,
  [int]$JointSteps = 200
)

$ErrorActionPreference = "Stop"

& $Python ymluo/projects/influence_bounded_synthetic_kv/src/run_synthetic_kv_smoke.py `
  --output-dir $OutputDir `
  --remote-tokens $RemoteTokens `
  --train-queries $TrainQueries `
  --test-queries $TestQueries `
  --dim $Dim `
  --value-dim $ValueDim `
  --prototypes $Prototypes `
  --joint-steps $JointSteps
