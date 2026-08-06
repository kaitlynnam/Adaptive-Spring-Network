param(
  [int[]]$Seeds = @(202, 303)
)

$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = "$env:USERPROFILE\anaconda3\envs\adaptive-spring-network\python.exe"
$trainer = Join-Path $repo "spring-network\04_adaptive_learning\train_profile_conditioned_passive_3d.py"
$topology = Join-Path $repo "spring-network\topologies\spatial\internal_fan_3d_60_spring.json"

foreach ($seed in $Seeds) {
  & $python -u $trainer `
    --topology $topology `
    --profiles-per-family 2000 `
    --test-profiles-per-family 400 `
    --samples 160 `
    --motion-mode triangular `
    --fixed-frequency-hz 0.2 `
    --iterations 10000 `
    --hidden-dim 256 `
    --device cuda `
    --relaxation-steps 300 `
    --mechanics-batch-size 512 `
    --mechanics-progress-batches 10 `
    --mechanics-correction-phases 2 `
    --mechanics-correction-profiles 600 `
    --mechanics-correction-samples 0 `
    --mechanics-correction-iterations 1500 `
    --mechanics-correction-learning-rate 0.001 `
    --nonlinear-power 1 `
    --nonlinear-ratio 0 `
    --progress-interval 500 `
    --seed $seed `
    --output-name "passive_60spring_seed$seed"
  if ($LASTEXITCODE -ne 0) {
    throw "Passive seed $seed failed with native exit code $LASTEXITCODE."
  }
}
