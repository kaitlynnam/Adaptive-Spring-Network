$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$trainer = Join-Path $repo "spring-network\04_adaptive_learning\train_profile_conditioned_passive_3d.py"
$topology = Join-Path $repo "spring-network\topologies\spatial\internal_fan_3d_60_spring.json"

python $trainer `
  --topology $topology `
  --profiles-per-family 2000 `
  --test-profiles-per-family 400 `
  --samples 160 `
  --iterations 10000 `
  --hidden-dim 256 `
  --device cuda `
  --relaxation-steps 300 `
  --mechanics-batch-size 512 `
  --mechanics-correction-phases 2 `
  --mechanics-correction-profiles 0 `
  --mechanics-correction-samples 0 `
  --mechanics-correction-iterations 1500 `
  --mechanics-correction-learning-rate 0.001 `
  --nonlinear-power 1 `
  --nonlinear-ratio 0 `
  --progress-interval 500 `
  --seed 101 `
  --output-name profile_passive_3d_60spring_full_refresh_seed101
