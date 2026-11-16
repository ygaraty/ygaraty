# ygaraty.analysis

Pure report calculations shared by experiment drivers and offline tools.

## Responsibilities and entry points

Use metrics, task_families and convnet_scaling to summarize existing records. Keep service startup and artifact collection in callers.

## Modules

- [convnet_scaling](convnet_scaling.py)
- [metrics](metrics.py)
- [task_families](task_families.py)

Use canonical `ygaraty.analysis` imports in new code. Keep compatibility facades thin; avoid duplicating implementations across packages.

## Validation

From the repository root, with development dependencies already installed:

```bash
ruff check ygaraty/analysis
ruff format --check ygaraty/analysis
python -m unittest discover -s tests -p "test_component_boundaries.py" -v
```

The selected suite checks offline structure or mocked contracts, not a complete training run. Numerical accuracy, live transport behavior and convergence require separate authorized integration checks. Do not run unrestricted test discovery on an offline-only machine.

See the [package map](../../../README.md), [maintainer tools](../../../../../tools/README.md) and [verification backlog](../../../../webgpu/tests/README.md).
