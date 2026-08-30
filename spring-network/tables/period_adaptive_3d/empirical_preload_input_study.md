# Empirical preload and controller-input study

## Matched protocol

- Linear-spring 60-spring spatial topology
- Training: 1,200 profiles, 400 held-out profiles, 1,334 iterations, 6 training periods, 0 mechanics refreshes, seed 101
- Deployment benchmark: 200 profiles, 6 periods, 300 relaxation steps, seed 901
- Primary metric: settled mean profile motor-work offload

## Preload comparison

| Rest-length scale | Mean profile offload | Aggregate offload | Settled RMSE (N m) | Negative profiles |
|---:|---:|---:|---:|---:|
| 0.500 | 30.917% | 32.713% | 36.766 | 4.0% |
| 0.575 | 32.361% | 33.894% | 36.118 | 2.5% |
| 0.600 | 32.865% | 34.408% | 35.856 | 2.5% |
| 0.625 | 33.181% | 34.662% | 35.699 | 2.0% |
| **0.650** | **33.552%** | **34.953%** | **35.563** | **1.5%** |

The best tested preload is 0.650.

## One-input removals at rest-length scale 0.650

| Removed input | Mean profile offload | Change from six-input reference |
|---|---:|---:|
| None (six-input reference) | 33.552% | - |
| `theta` | 33.821% | +0.269 pp |
| `theta_dot` | 34.090% | +0.538 pp |
| **`theta_ddot`** | **35.199%** | **+1.647 pp** |
| `target_torque` | 32.896% | -0.656 pp |
| `spring_torque` | 33.356% | -0.196 pp |
| `motor_torque` | 32.763% | -0.789 pp |

## Two-input confirmation around the best removal

| Removed inputs | Mean profile offload | Change versus removing only `theta_ddot` |
|---|---:|---:|
| `theta_ddot` only | 35.199% | - |
| `theta` + `theta_ddot` | 35.122% | -0.077 pp |
| `theta_dot` + `theta_ddot` | 35.025% | -0.174 pp |
| `theta_ddot` + `target_torque` | 35.156% | -0.043 pp |
| `theta_ddot` + `spring_torque` | 35.042% | -0.157 pp |
| `theta_ddot` + `motor_torque` | 35.119% | -0.081 pp |

Within this matched single-seed experiment, the selected short-study controller uses rest-length scale 0.650 and retains `theta`, `theta_dot`, `target_torque`, `spring_torque`, and `motor_torque`. It removes only `theta_ddot`.

## Fully trained checkpoint (different training budget)

`input_full_target_spring_preload060_6x1000_seed101.npz` uses rest-length scale 0.600 and only `target_torque` and `spring_torque`. It was trained with 6,000 profiles, 1,200 held-out profiles, 6,000 iterations, and 5 mechanics refreshes. Under the same deployment benchmark it achieved:

- Mean profile offload: 45.661%
- Aggregate offload: 49.615%
- Settled RMSE: 30.022 N m
- Negative-offload profiles: 1.5%

This is the best currently trained checkpoint, but it is not an architecture comparison against the short-study candidates because its training budget is much larger.

## Reproduction commands

Train the selected short-study configuration:

```powershell
python spring-network/04_adaptive_learning/train_period_adaptive_3d.py --training-profiles 1200 --test-profiles 400 --training-periods 6 --iterations 1334 --mechanics-refreshes 0 --rest-length-scale 0.65 --observation-channels theta theta_dot target_torque spring_torque motor_torque --device cpu --seed 101 --output-name input065_confirm_drop_theta_ddot_seed101
```

Benchmark it:

```powershell
python spring-network/04_adaptive_learning/benchmark_period_adaptive_deployment.py --checkpoint spring-network/models/period_adaptive_3d/input065_confirm_drop_theta_ddot_seed101.npz --profiles 200 --periods 6 --relaxation-steps 300 --mechanics-batch-size 1024 --device cpu --seed 901 --output-name input065_confirm_drop_theta_ddot_seed101_benchmark
```

The per-profile, per-period, summary, and complete compressed arrays are saved alongside this report.

## Expanded all-GPU confirmation

Because CPU and CUDA optimization are not numerically identical, the complete preload range was rerun using the same CUDA environment before comparing the new values. Every candidate used 1,200 training profiles, 400 held-out profiles, 1,334 iterations, 6 training periods, 0 mechanics refreshes, seed 101, and all six inputs. Deployment used the same 200 profiles, 6 periods, and seed 901.

| Rest-length scale | Mean profile offload | Aggregate offload | Settled RMSE (N m) | Negative profiles |
|---:|---:|---:|---:|---:|
| 0.500 | 30.819% | 32.615% | 36.821 | 3.5% |
| 0.575 | 32.336% | 33.971% | 36.079 | 2.5% |
| 0.600 | 32.981% | 34.545% | 35.775 | 2.5% |
| 0.625 | 33.372% | 34.852% | 35.613 | 2.0% |
| **0.650** | **33.681%** | **35.097%** | **35.500** | **1.0%** |
| 0.675 | 33.473% | 34.679% | 35.677 | 1.0% |
| 0.700 | 33.589% | 34.657% | 35.666 | 1.0% |
| 0.725 | 33.237% | 34.175% | 35.885 | 0.5% |
| 0.750 | 32.388% | 33.192% | 36.367 | 1.0% |

The expanded sweep confirms 0.650 as the best tested preload; performance falls on both sides over the tested grid.

### Requested input configurations at preload 0.650

| Active inputs | Mean profile offload | Aggregate offload | Settled RMSE (N m) | Negative profiles |
|---|---:|---:|---:|---:|
| `target_torque`, `spring_torque` | 34.742% | 36.295% | 34.959 | 1.0% |
| `target_torque`, `spring_torque`, `motor_torque` | 34.669% | 36.189% | 35.009 | 1.0% |
| **`theta`, `target_torque`, `spring_torque`, `motor_torque`** | **35.107%** | **36.580%** | **34.810** | **0.5%** |
| `theta`, `target_torque`, `spring_torque` | 34.979% | 36.513% | 34.835 | 0.5% |

The best requested configuration retains angle and all three torque channels. Its checkpoint is `input_gpu_065_theta_target_spring_motor_seed101.npz`.

Reproduce its training and benchmark with the CUDA-enabled environment:

```powershell
C:/Users/kn109/anaconda3/envs/adaptive-spring-passive/python.exe spring-network/04_adaptive_learning/train_period_adaptive_3d.py --training-profiles 1200 --test-profiles 400 --training-periods 6 --iterations 1334 --mechanics-refreshes 0 --rest-length-scale 0.65 --observation-channels theta target_torque spring_torque motor_torque --device cuda --seed 101 --output-name input_gpu_065_theta_target_spring_motor_seed101

C:/Users/kn109/anaconda3/envs/adaptive-spring-passive/python.exe spring-network/04_adaptive_learning/benchmark_period_adaptive_deployment.py --checkpoint spring-network/models/period_adaptive_3d/input_gpu_065_theta_target_spring_motor_seed101.npz --profiles 200 --periods 6 --relaxation-steps 300 --mechanics-batch-size 1024 --device cuda --seed 901 --output-name input_gpu_065_theta_target_spring_motor_seed101_benchmark
```
