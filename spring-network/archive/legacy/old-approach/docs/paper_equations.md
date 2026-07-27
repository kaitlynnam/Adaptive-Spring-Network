# Paper Equation Reproduction Notes

Source PDF: `/Users/kaitlyn/Desktop/OG Paper.pdf`

Paper: Huyue Ma, Yurui Jin, Helmut Hauser, Rui Wu, "Physical Imitation Learning: Distilling Control Policies into Passive Elasticity", arXiv:2604.00611v2, 16 Jun 2026.

## Symbols

- `theta` or `q`: joint angle, rad.
- `theta_dot` or `q_dot`: joint angular velocity, rad/s.
- `tau_total`: total joint torque demanded by the learned policy, N m.
- `tau_PEJ`: passive Parallel Elastic Joint torque, N m.
- `tau_residual`: active residual motor torque, N m.
- `P`: positive-only motor mechanical power consumption, W.

## Equations Implemented

1. Torque decomposition:

   `tau_total = tau_PEJ + tau_residual`

2. Residual motor mechanical power before applying zero-regeneration:

   `P_mechanical = (tau_total - tau_PEJ) * theta_dot`

3. PEJ distillation objective, per angle bin `b`:

   `tau_PEJ*(theta_b) = arg min_tau_PEJ sum_{t in T_b} P(t)`

   where `P(t)` is positive motor power after subtracting the PEJ torque.

4. Base reward:

   `r_base = sum_k w_k r_k`

5. Cost of Transport:

   `CoT = sum_{i in thigh,calf} P_i(t) / (m g v_scalar)`

   The paper uses `m = 15 kg`, `g = 9.81 m/s^2`, and clamps low speed to `0.1 m/s`.

6. Projected scalar speed:

   `v_scalar = max(0, v_actual dot vhat_cmd)`

   The paper averages this over a sliding window of 10 simulation steps for training.

7. Stage-2 total reward:

   `r_total = r_base - alpha * CoT`

8. PEJ offload percentage:

   `R_offload = ((P_without_PEJ - P_with_PEJ) / P_without_PEJ) * 100%`

9. Positive-only motor power:

   `P(t) = max(0, tau(t) * q_dot(t))`

10. Mean absolute velocity tracking error:

   `e_v = (1/T) integral_0^T |v_cmd(t) - v_actual(t)| dt`

11. Cam manufacturability mapping:

   `tau(theta) = -dU/dtheta`

   `U(theta) = U0 - integral tau_t(theta) dtheta`

   `r(theta) = R_b + sqrt(2 U(theta) / k) - delta_pre`

   Use SI units consistently: N, m, rad, J.

## Paper Constants And Reported Values

- Control frequency: `50 Hz`, `dt = 0.02 s`.
- Parallel simulated robots: `4096`.
- Experience buffer: `200000` tuples per joint group.
- Distillation minibatch: `20%` of buffer, up to `40000` samples.
- PEJ profile knots: `20`.
- Observed angle distribution trimming: lowest and highest `5%`.
- Outside PEJ active range: linearly ramp torque to zero over `5 deg` on each side.
- Heavy-ball momentum for online distillation: learning rate `0.15`, momentum `0.8`.
- Unitree Go2 mass used for CoT: `15 kg`.
- Maximum motor torque: `23.5 N m` per joint.
- Target PEJ joint groups: front thigh, rear thigh, front calf, rear calf.

### Table 2: CoT Weights

| Policy | Flat | L1 | L2 | L3 | L4 | L5 | L6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Co-design | 2.0 | 1.0 | 0.9 | 0.5 | 0.4 | 0.2 | 0.1 |
| Reference | 0.6 | 0.6 | 0.5 | 0.4 | 0.3 | 0.15 | 0.1 |

### Table 4: Average Motor Power, W

| Policy | Metric | Flat | L1 | L2 | L3 | L4 | L5 | L6 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Co-design | Before | 21.7 | 19.6 | 24.3 | 26.8 | 31.0 | 39.4 | 49.5 |
| Co-design | After | 1.12 | 3.70 | 8.65 | 16.6 | 22.3 | 32.3 | 43.0 |
| Co-design | Offload | 94.8 | 81.2 | 64.4 | 38.1 | 28.0 | 18.2 | 13.1 |
| Reference | Before | 18.5 | 21.3 | 22.6 | 25.4 | 29.8 | 38.2 | 48.6 |
| Reference | After | 15.4 | 17.8 | 18.5 | 22.4 | 25.2 | 33.0 | 43.0 |
| Reference | Offload | 16.6 | 16.3 | 18.1 | 11.7 | 15.4 | 13.6 | 11.6 |

