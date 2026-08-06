# Deferred paper-quality 3D runs

These commands freeze the current best-found linear topology after the
spring-count/position search. They use independent seeds that were not used to
select candidate 131, 300-step surrogate mechanics, and the convergence-tested
800-step final evaluation.

Current selected topology:

`topologies/spatial/global_search/candidate_0131_56s.json`

Dense feasibility result:

- 56 linear springs
- 181 joint angles from -45 to +45 degrees
- 50 mm required spring centerline spacing
- 53.63 mm measured minimum spacing
- no detected spring, limb, or bearing intersections

Run each command from the repository root.

```powershell
python spring-network\04_adaptive_learning\train_adaptive_3d.py `
  --topology spring-network\topologies\spatial\global_search\candidate_0131_56s.json `
  --profiles-per-family 2000 --test-profiles-per-family 400 `
  --iterations 5000 --surrogate-refreshes 2 `
  --relaxation-steps 300 --evaluation-relaxation-steps 800 `
  --mechanics-batch-size 4096 --device cuda --seed 401 `
  --output-name paper_global56_linear_seed401

python spring-network\04_adaptive_learning\train_adaptive_3d.py `
  --topology spring-network\topologies\spatial\global_search\candidate_0131_56s.json `
  --profiles-per-family 2000 --test-profiles-per-family 400 `
  --iterations 5000 --surrogate-refreshes 2 `
  --relaxation-steps 300 --evaluation-relaxation-steps 800 `
  --mechanics-batch-size 4096 --device cuda --seed 503 `
  --output-name paper_global56_linear_seed503

python spring-network\04_adaptive_learning\train_adaptive_3d.py `
  --topology spring-network\topologies\spatial\global_search\candidate_0131_56s.json `
  --profiles-per-family 2000 --test-profiles-per-family 400 `
  --iterations 5000 --surrogate-refreshes 2 `
  --relaxation-steps 300 --evaluation-relaxation-steps 800 `
  --mechanics-batch-size 4096 --device cuda --seed 607 `
  --output-name paper_global56_linear_seed607
```

The cubic model is implemented consistently in force and energy, but has only
passed a static-authority sanity check. Do not combine it with the three final
linear seeds. Run this matched pilot first:

```powershell
python spring-network\04_adaptive_learning\train_adaptive_3d.py `
  --topology spring-network\topologies\spatial\global_search\candidate_0131_56s.json `
  --profiles-per-family 250 --test-profiles-per-family 50 `
  --iterations 1000 --surrogate-refreshes 1 `
  --relaxation-steps 300 --evaluation-relaxation-steps 800 `
  --cubic-ratio 0.5 --cubic-reference-extension 0.6 `
  --mechanics-batch-size 4096 --device cuda --seed 701 `
  --output-name global56_cubic05_pilot
```

The final paper should describe this as the best topology found under the
documented randomized search budget, not a proof of a global mathematical
optimum.
