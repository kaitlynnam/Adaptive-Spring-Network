$ErrorActionPreference = 'Stop'

$python = 'C:/Users/kn109/anaconda3/envs/adaptive-spring-passive/python.exe'
$trainScript = 'spring-network/04_adaptive_learning/train_period_adaptive_3d.py'
$benchmarkScript = 'spring-network/04_adaptive_learning/benchmark_period_adaptive_deployment.py'
$modelRoot = 'spring-network/models/period_adaptive_3d'
$tableRoot = 'spring-network/tables/period_adaptive_3d'
$allChannels = @('theta', 'theta_dot', 'theta_ddot', 'target_torque', 'spring_torque', 'motor_torque')

$preloads = @(
    @{ Scale = '0.500'; Tag = '050' },
    @{ Scale = '0.575'; Tag = '0575' },
    @{ Scale = '0.600'; Tag = '060' },
    @{ Scale = '0.625'; Tag = '0625' },
    @{ Scale = '0.650'; Tag = '065' },
    @{ Scale = '0.675'; Tag = '0675' },
    @{ Scale = '0.700'; Tag = '070' },
    @{ Scale = '0.725'; Tag = '0725' },
    @{ Scale = '0.750'; Tag = '075' }
)

foreach ($candidate in $preloads) {
    $name = "preload_gpu_confirm_$($candidate.Tag)_seed101"
    & $python $trainScript `
        --training-profiles 1200 `
        --test-profiles 400 `
        --training-periods 6 `
        --iterations 1334 `
        --mechanics-refreshes 0 `
        --rest-length-scale $candidate.Scale `
        --observation-channels $allChannels `
        --device cuda `
        --seed 101 `
        --output-name $name
    if ($LASTEXITCODE -ne 0) { throw "Training failed for $name" }

    & $python $benchmarkScript `
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

$allPreloadTags = @('050', '0575', '060', '0625', '065', '0675', '070', '0725', '075')
$ranked = foreach ($tag in $allPreloadTags) {
    $summaryPath = "$tableRoot/preload_gpu_confirm_${tag}_seed101_benchmark_summary.csv"
    if (Test-Path -LiteralPath $summaryPath) {
        $row = Import-Csv -LiteralPath $summaryPath | Select-Object -First 1
        $checkpoint = "$modelRoot/preload_gpu_confirm_${tag}_seed101.npz"
        $scale = & $python -c "import numpy as np; print(float(np.load(r'$checkpoint', allow_pickle=True)['rest_length_scale']))"
        if ($LASTEXITCODE -ne 0) { throw "Could not read preload metadata from $checkpoint" }
        [pscustomobject]@{
            Scale = [double]$scale
            MeanOffload = [double]$row.settled_mean_profile_offload_pct
        }
    }
}
$best = $ranked | Sort-Object MeanOffload -Descending | Select-Object -First 1
$bestScale = $best.Scale.ToString('0.###', [Globalization.CultureInfo]::InvariantCulture)
$bestTag = $bestScale.Replace('.', '')
Write-Output "Selected preload $bestScale with mean offload $($best.MeanOffload)%"

$inputConfigurations = @(
    @{ Name = "input_gpu_${bestTag}_target_spring_seed101"; Channels = @('target_torque', 'spring_torque') },
    @{ Name = "input_gpu_${bestTag}_target_spring_motor_seed101"; Channels = @('target_torque', 'spring_torque', 'motor_torque') },
    @{ Name = "input_gpu_${bestTag}_theta_target_spring_motor_seed101"; Channels = @('theta', 'target_torque', 'spring_torque', 'motor_torque') },
    @{ Name = "input_gpu_${bestTag}_theta_target_spring_seed101"; Channels = @('theta', 'target_torque', 'spring_torque') }
)

foreach ($configuration in $inputConfigurations) {
    & $python $trainScript `
        --training-profiles 1200 `
        --test-profiles 400 `
        --training-periods 6 `
        --iterations 1334 `
        --mechanics-refreshes 0 `
        --rest-length-scale $bestScale `
        --observation-channels $configuration.Channels `
        --device cuda `
        --seed 101 `
        --output-name $configuration.Name
    if ($LASTEXITCODE -ne 0) { throw "Training failed for $($configuration.Name)" }

    & $python $benchmarkScript `
        --checkpoint "$modelRoot/$($configuration.Name).npz" `
        --profiles 200 `
        --periods 6 `
        --relaxation-steps 300 `
        --mechanics-batch-size 1024 `
        --device cuda `
        --seed 901 `
        --output-name "$($configuration.Name)_benchmark"
    if ($LASTEXITCODE -ne 0) { throw "Benchmark failed for $($configuration.Name)" }
}

Write-Output 'Expanded preload and requested-input queue completed.'
