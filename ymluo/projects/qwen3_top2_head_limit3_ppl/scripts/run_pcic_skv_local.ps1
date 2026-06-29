param(
  [string]$Python = ".\.venv-qabs-rocm\Scripts\python.exe",
  [string]$Model = "Qwen/Qwen3-0.6B",
  [Parameter(Mandatory=$true)][string]$TextPath,
  [Parameter(Mandatory=$true)][string]$OutputDir,
  [int]$PrefillTokens = 4096,
  [int]$EvalTokens = 128,
  [int]$ChunkSize = 16,
  [int]$EvalChunkSize = 1,
  [int]$RecentTokens = 512,
  [int]$LandmarkStride = 64,
  [int]$SyntheticPrototypes = 16,
  [string]$SyntheticMethods = "mass",
  [string]$Fallbacks = "landmark,synthetic",
  [string]$Combos = "",
  [string]$PairwiseLayers = "",
  [string]$IncludeSingletons = "false",
  [string]$IncludePrefixes = "false",
  [string]$DType = "bfloat16",
  [string]$Device = "cuda",
  [string]$DeviceMap = "auto",
  [string]$AttnImplementation = "eager",
  [int]$LogEvery = 1000
)

$ArgsList = @(
  "ymluo\projects\qwen3_top2_head_limit3_ppl\src\run_pcic_skv_local.py",
  "--model_name_or_path", $Model,
  "--text_path", $TextPath,
  "--output_dir", $OutputDir,
  "--prefill_tokens", $PrefillTokens,
  "--eval_tokens", $EvalTokens,
  "--chunk_size", $ChunkSize,
  "--eval_chunk_size", $EvalChunkSize,
  "--recent_tokens", $RecentTokens,
  "--landmark_stride", $LandmarkStride,
  "--synthetic_prototypes", $SyntheticPrototypes,
  "--synthetic_methods", $SyntheticMethods,
  "--fallbacks", $Fallbacks,
  "--dtype", $DType,
  "--device", $Device,
  "--device_map", $DeviceMap,
  "--attn_implementation", $AttnImplementation,
  "--log_every", $LogEvery
)

if ($Combos -ne "") {
  $ArgsList += @("--combos", $Combos)
}
if ($PairwiseLayers -ne "") {
  $ArgsList += @("--pairwise_layers", $PairwiseLayers)
}
$ArgsList += @("--include_singletons", $IncludeSingletons)
$ArgsList += @("--include_prefixes", $IncludePrefixes)

& $Python @ArgsList
