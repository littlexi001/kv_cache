param(
  [string]$Python = ".\.venv-qabs-rocm\Scripts\python.exe",
  [string]$ModelPath = "Qwen/Qwen3-0.6B",
  [string]$TextPath = "ymluo\projects\qabs8cand3reuse_quality_suite\data\topic_texts\science.txt",
  [string]$OutputDir = "ymluo\projects\influence_bounded_synthetic_kv\outputs\calibrated_ppl_science",
  [int]$PrefillTokens = 512,
  [int]$CalibTokens = 16,
  [int]$EvalTokens = 32,
  [int]$Prototypes = 16,
  [int]$JointSteps = 0,
  [double]$JointLr = 0.03,
  [string]$LayerSets = "all"
)

$ErrorActionPreference = "Stop"

& $Python ymluo/projects/influence_bounded_synthetic_kv/src/run_calibrated_synthetic_kv_ppl.py `
  --model_name_or_path $ModelPath `
  --text_path $TextPath `
  --output_dir $OutputDir `
  --prefill_tokens $PrefillTokens `
  --calib_tokens $CalibTokens `
  --eval_tokens $EvalTokens `
  --prototypes $Prototypes `
  --protect_sink_tokens 10 `
  --protect_recent_tokens 10 `
  --ridge 0.001 `
  --joint_steps $JointSteps `
  --joint_lr $JointLr `
  --layer_sets $LayerSets `
  --chunk_size 8 `
  --dtype bfloat16 `
  --device cuda `
  --device_map auto `
  --attn_implementation eager `
  --log_every 8
