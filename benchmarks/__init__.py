"""Benchmark harness, task families and measurement scripts.

Run everything with ``python -m benchmarks.harness --all``. Individual pieces:

* ``benchmarks.profile_models`` -- Phase 2, measure one model's card
* ``benchmarks.harness`` -- Phase 12, the full suite and ``docs/RESULTS.md``
* ``benchmarks.regret`` -- C1, scheduler regret and the swap-awareness ablation
* ``benchmarks.train_router`` -- Phase 7, train and evaluate the learned router
* ``benchmarks.calibrate`` -- Phase 10 / C3, risk-coverage and the precision contract
"""
