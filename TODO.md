# Future Work

## Determine what is actually required to build an effective network

Deferred at mentor suggestion. Do not begin this study until the main
one-period-buffer paper results and methods are stable.

Run controlled ablations to identify which mechanical and controller features
are necessary, rather than assuming the complete 60-spring design is required.

Candidate comparisons:

- Spring count and connection density.
- Skin anchors versus direct limb attachments.
- Free internal nodes versus fully prescribed nodes.
- Number and placement of internal nodes.
- Direct cross-joint torque lanes versus internal-node paths.
- Split-skin enclosure versus simpler attachment geometry.
- Required stiffness range and whether every spring must be adjustable.
- Uniform, grouped, and independently controlled stiffnesses.
- Which observation channels are necessary.
- Required samples per period and buffer duration.
- Once-per-period updates versus less frequent updates.
- Performance versus mechanical complexity, packaging volume, and collision clearance.
- Performance versus computation, sensing, and likely hardware cost.
- Rest-length/preload ratio, using separately retrained controllers for every
  candidate value. Compare no preload against a coarse sweep and then refine
  around the best region; do not reuse a controller trained at one preload to
  make the final comparison.

Use matched datasets, seeds, training budgets, stiffness constraints, and exact
mechanics evaluation. Compare torque RMSE, motor-work offload, equilibrium
residual, robustness, parameter count, spring count, sensor requirements, and
physical feasibility. The intended outcome is a minimal-design recommendation,
not merely the highest-performing topology.

### Preliminary collision-free spring-count screen (2026-08-11)

Matched 1,334-iteration, pre-refresh screens compared optimized 60-, 70-, and
80-spring routed topologies. All passed the 300-step, 19-angle collision audit.
Exact adapted held-out RMSE was 46.52, 46.60, and 47.66 N m, respectively.
The 70-spring design improved the linear bounded-fit estimate but not the exact
relaxed result, so increasing spring count alone did not improve performance and
did not justify another full-refresh run.

### Preliminary connection-pattern search (2026-08-11)

A collision-constrained search rewired the 60-spring design with rotationally
staggered outer and internal routes. Four of 49 patterns passed the full-angle
screen; the others were rejected for physical-clearance violations. The best
candidate shifted each outer cross-joint spring by one circumferential lane and
kept the internal distal routes local. It retained 34.9 mm minimum spring
clearance against the configured 12 mm requirement.

The candidate improved shortened exact held-out RMSE from 46.52 to 45.07 N m,
so it was promoted to a matched 4,000-iteration run with two exact mechanics
refreshes. Refreshed held-out RMSE improved from 41.99 to 41.30 N m. However,
the authoritative 200-profile, 12-period benchmark did not improve: settled
mean RMSE was 39.95 rather than 39.12 N m and aggregate motor-work offload was
27.10% rather than 28.91%. It also reduced the negative-offload profile rate
from 9.0% to 6.5%. Retain this as a robustness tradeoff, not the main topology.

Future topology ranking should include multi-period motor work in the promotion
criterion; one-period RMSE and bounded torque fitting were insufficient proxies
for the paper's primary metric.

### Preliminary motor-work-loss experiment (2026-08-11)

The controller trainer now optionally replaces torque MSE with mean absolute
residual motor power, `abs((target torque - spring torque) * angular velocity)`.
With uniform time samples this minimizes the assisted-work numerator used by the
aggregate offload benchmark. The stiffness-change penalty and hard stiffness
bounds remain active.

On the best collision-free topology, matched 4,000-iteration training with two
exact mechanics refreshes produced 41.69 N m exact held-out RMSE. However, the
200-profile, 12-period benchmark gave 40.85 N m settled mean RMSE and 27.27%
aggregate motor-work offload, compared with 39.12 N m and 28.91% for MSE
training. Direct motor-work loss therefore did not improve final deployment and
should not replace the current objective. A future test could align training and
evaluation horizons at 12 periods or use a mixed MSE/work objective.

A shortened 1,334-iteration follow-up restored MSE and increased the closed-loop
training horizon from 6 to 12 periods. Exact held-out RMSE improved from 46.52
to 45.60 N m relative to the matched six-period pre-refresh screen. Its
200-profile deployment benchmark reached 17.18% aggregate offload without any
mechanics refreshes. This is promising relative to early pre-refresh models but
is not comparable to the fully refreshed 28.91% result; a full 12-period run
would require substantially more refresh computation.

### Stiffness turn-down screen (2026-08-11)

The original collision-free controller placed 21.7% of settled spring commands
at its 0.5x lower stiffness bound and none at the upper bound. Widening only the
lower range was the strongest tested intervention. With matched full training,
a 0.1x minimum increased aggregate offload from 28.91% to 34.86%, reduced
settled mean RMSE from 39.12 to 36.06 N m, and eliminated negative-offload
profiles. A 0.01x minimum produced 34.93% offload and 36.02 N m RMSE, an
insignificant improvement for a much more demanding 100:1 turn-down ratio.

Use the 10:1 turn-down model as the practical performance candidate. Additional
springs are not currently the priority: earlier 70- and 80-spring designs added
anchor, limb, and internal lanes but failed to improve exact performance. A
future internal-node ablation should hold spring count and attachment points
fixed so the effect of internal routing can be isolated.

### Preliminary cubic-spring screen (2026-08-11)

A small paired mechanics test used 20 held-out profiles, their existing settled
stiffness schedules, and 300 relaxation steps. Relative to linear springs, a
cubic contribution equal to 10% of the linear force at 0.6 m extension reduced
mean RMSE by 0.96%; a 30% contribution reduced it by 1.86%. Mean motor-work
offload changed from 49.05% (linear) to 49.73% and 50.57%, respectively.

This small improvement is not evidence that cubic springs are required. A full
claim would require separately retraining linear and cubic controllers across
multiple matched seeds and comparing the benefit against added physical and
identification complexity.

On 2026-08-11, a second paired screen used 24 held-out profiles from the fully
refreshed collision-free topology. Keeping its settled linear-controller
stiffness schedules fixed, cubic ratios from 0.05 to 0.30 slightly worsened both
RMSE and motor-work offload. At ratio 0.30, mean RMSE increased by 0.58% and
subset aggregate offload decreased from 20.36% to 19.91%. This is not a matched
retraining result, but it makes cubic springs a lower-priority follow-up than
topology, stiffness-family, and loss-function studies.
