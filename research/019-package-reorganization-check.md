# Regression check after the package reorganization

| | |
|---|---|
| **Date** | 2026-10-24 |
| **Path** | ZeroMQ, in-memory, 2 workers (lifecycle harness) plus the unit-test suite |
| **Result** | Suite: 77 tests, 51 passed, 26 skipped (database-only). Lifecycle harness: 100/100 updates matched PyTorch; max weight error 1.55e-14. |
| **Evidence** | `runs/layout-regression.log`, `runs/ayuma-layout-lifecycle/verification.json` (local, not in git) |

After separating the compiler and webapp into packages, the full regression suite ran **77 tests: 51 passed, 26 skipped**. The skips require explicitly configured disposable databases; neither Redis nor PostgreSQL integration was rerun for this change. A focused production-policy rerun also passed its four available tests, including the guard against importing the local oracle in production SQL execution.

The [tiny lifecycle harness](018-tiny-lstm-100-updates-recovery.md) completed **100 distributed updates** in about **90.5 seconds**, comparing every update against PyTorch at `rtol=2e-7, atol=2e-9`. Maximum absolute weight error was about 1.55e-14. Worker interruption, rejection of a stale attempt, checkpoint resume after the partial update 51, and the data cursor checks passed. Artifacts occupied **928,027 bytes**, with two retained checkpoints.

This confirms the reorganized code preserves the tested tiny Python workflow. It is not a new large-model, real-database or browser WebGPU benchmark. (The project was then called Ayuma, hence the run directory name.)

## Reproduce

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python -m unittest discover -s tests
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  ygaraty experiment tiny_lstm_training --output runs/tiny-lstm-repeat
```
