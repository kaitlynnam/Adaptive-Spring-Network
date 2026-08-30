$ErrorActionPreference = 'Stop'

$python = 'C:/Users/kn109/anaconda3/envs/adaptive-spring-passive/python.exe'
$train = 'spring-network/04_adaptive_learning/train_period_adaptive_3d.py'
$benchmark = 'spring-network/04_adaptive_learning/benchmark_period_adaptive_deployment.py'
$modelRoot = 'spring-network/models/period_adaptive_3d'
$tableRoot = 'spring-network/tables/period_adaptive_3d'

function Invoke-Candidate([int]$stiffness) {
    $name = "uniform_k${stiffness}_compact4_preload065_seed101"
    $summaryPath = "$tableRoot/${name}_benchmark_summary.csv"
    if (-not (Test-Path -LiteralPath $summaryPath)) {
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
            --output-name $name | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Training failed for $name" }

        & $python $benchmark `
            --checkpoint "$modelRoot/$name.npz" `
            --profiles 200 `
            --periods 6 `
            --relaxation-steps 300 `
            --mechanics-batch-size 1024 `
            --device cuda `
            --seed 901 `
            --output-name "${name}_benchmark" | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "Benchmark failed for $name" }
    }
    $summary = Import-Csv -LiteralPath $summaryPath | Select-Object -First 1
    return [pscustomobject]@{
        Stiffness = $stiffness
        MeanOffload = [double]$summary.settled_mean_profile_offload_pct
    }
}

$results = [System.Collections.Generic.List[object]]::new()
$best = Invoke-Candidate 180
$results.Add($best)
$candidate = 210
while ($true) {
    $result = Invoke-Candidate $candidate
    $results.Add($result)
    Write-Output "Uniform stiffness ${candidate}: mean offload $($result.MeanOffload)%"
    if ($result.MeanOffload -le $best.MeanOffload) {
        break
    }
    $best = $result
    $candidate += 30
    if ($candidate -gt 600) {
        throw 'Uniform stiffness search exceeded the 600 N/m safety ceiling without turning down.'
    }
}

foreach ($refinement in @(($best.Stiffness - 10), ($best.Stiffness + 10))) {
    if ($refinement -gt 0 -and -not ($results.Stiffness -contains $refinement)) {
        $result = Invoke-Candidate $refinement
        $results.Add($result)
        Write-Output "Uniform stiffness ${refinement}: mean offload $($result.MeanOffload)%"
    }
}

$winner = $results | Sort-Object MeanOffload -Descending | Select-Object -First 1
$winningStiffness = [int]$winner.Stiffness
Write-Output "Selected uniform stiffness $winningStiffness with mean offload $($winner.MeanOffload)%"

$fullName = "input_full_compact4_uniform_k${winningStiffness}_preload065_6x1000_seed101"
& $python $train `
    --training-profiles 6000 `
    --test-profiles 1200 `
    --training-periods 6 `
    --iterations 6000 `
    --mechanics-refreshes 5 `
    --rest-length-scale 0.65 `
    --uniform-initial-stiffness $winningStiffness `
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

Write-Output 'Adaptive uniform-stiffness search, full training, and benchmark completed.'
