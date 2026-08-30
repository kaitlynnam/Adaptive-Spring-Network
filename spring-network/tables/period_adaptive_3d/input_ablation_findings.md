# Preload and observation-input study

All promoted comparisons used matched seeds, generated profiles, training budgets,
stiffness constraints, and exact relaxed-mechanics evaluation. Every controller was
retrained for its active preload and observation mask.

The selected rest-length scale is 0.60 (40% nominal preload). In the 200-profile,
six-period benchmark it achieved 35.856 N m settled RMSE, 32.865% mean profile
offload, and 34.408% aggregate motor-work offload, outperforming confirmed scales
0.55 and 0.575. Mean profile offload is the primary offload metric for this study.

At scale 0.60, spring torque plus motor torque had the highest mean profile offload
at 34.760%. Target torque plus measured spring torque was effectively tied at
34.750% and had the slightly lower settled RMSE (34.926 versus 34.935 N m). Target
torque alone remained viable and outperformed the original six-channel input, with
the lowest negative-offload rate.

All mechanics in this study used linear springs. No cubic force term was enabled.

For this fixed repeated triangular-motion experiment, angle, velocity, and
acceleration were unnecessary. This does not establish that they are unnecessary
for variable, unknown, or nonrepeating motion; that requires a separate randomized-
motion ablation with matched training and deployment trajectories.
