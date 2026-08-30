$ErrorActionPreference = 'Stop'

$python = 'C:/Users/kn109/anaconda3/envs/adaptive-spring-passive/python.exe'
$train = 'spring-network/04_adaptive_learning/train_period_adaptive_3d.py'
$benchmark = 'spring-network/04_adaptive_learning/benchmark_period_adaptive_deployment.py'
$modelRoot = 'spring-network/models/period_adaptive_3d'
$tableRoot = 'spring-network/tables/period_adaptive_3d'
$stiffnessValues = @(70, 80, 85, 90, 100)

foreach ($stiffness in $stiffnessValues) {
    $name = "uniform_k${stiffness}_compact4_preload065_seed101"
    & $python $train `
        --training-profiles 1200 `
        --test-profiles 200 `
        --training-periods 6 `
        --iterations 1334 `
        --mechanics-refreshes 0 `
        --rest-length-scale 0.65 `
        --uniform-initial-stiffness $stiffness `
        --observation-channels theta target_torque spring_torque motor_torque `
        --compact-observation-channels `
        --device cuda `
        --seed 101 `
        --output-name $name
    if ($LASTEXITCODE -ne 0) { throw "Training failed for $name" }

    & $python $benchmark `
        --checkpoint "$modelRoot/$name.npz" `
        --profiles 200 `
        --periods 6 `
        --relaxation-steps 300 `
        --mechanics-batch-size 1024 `
        --device cuda `
        --seed 901 `
        --output-name "${name}_benchmark"
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed for $name" }
}

$ranked = foreach ($stiffness in $stiffnessValues) {
    $name = "uniform_k${stiffness}_compact4_preload065_seed101"
    $summary = Import-Csv -LiteralPath "$tableRoot/${name}_benchmark_summary.csv" |
        Select-Object -First 1
    [pscustomobject]@{
        Stiffness = $stiffness
        MeanOffload = [double]$summary.settled_mean_profile_offload_pct
    }
}
$best = $ranked | Sort-Object MeanOffload -Descending | Select-Object -First 1
$bestStiffness = [int]$best.Stiffness
Write-Output "Selected uniform stiffness $bestStiffness with mean offload $($best.MeanOffload)%"

$fullName = "input_full_compact4_uniform_k${bestStiffness}_preload065_6x1000_seed101"
& $python $train `
    --training-profiles 6000 `
    --test-profiles 1200 `
    --training-periods 6 `
    --iterations 6000 `
    --mechanics-refreshes 5 `
    --rest-length-scale 0.65 `
    --uniform-initial-stiffness $bestStiffness `
    --observation-channels theta target_torque spring_torque motor_torque `
    --compact-observation-channels `
    --device cuda `
    --seed 101 `
    --output-name $fullName
if ($LASTEXITCODE -ne 0) { throw "Full training failed for $fullName" }

& $python $benchmark `
    --checkpoint "$modelRoot/$fullName.npz" `
    --profiles 200 `
    --periods 6 `
    --relaxation-steps 300 `
    --mechanics-batch-size 1024 `
    --device cuda `
    --seed 901 `
    --output-name "${fullName}_benchmark"
if ($LASTEXITCODE -ne 0) { throw "Full benchmark failed for $fullName" }

Write-Output 'Uniform-stiffness selection, full training, and final benchmark completed.'
