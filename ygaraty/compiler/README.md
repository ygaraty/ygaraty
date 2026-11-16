# Ygaraty compiler

The compiler turns tensor graphs into transport-independent execution plans.
It is the contract between model builders and Python/browser workers: shapes,
gradients, task dependencies, tensor regions, packed outputs and optimizer state
must agree across those components.

It records computation; it does not run training, initialize remote stores,
schedule workers or start services. `Tape` stores explicit NumPy input arrays;
`ShapeTape` stores shape/initializer descriptions for large symbolic graphs.
Neither lowering nor differentiation evaluates workload kernels.

## Start here

Import the supported entry points from `ygaraty.compiler`. This complete example
constructs and checks a plan entirely in memory:

```python
from ygaraty.compiler import (
    Tape, admit_plan, bounded_reductions, split, with_optimizer,
)

# Build the forward graph; these are small explicit input values.
tape = Tape()
x = tape.input("x", [[1., 2.]])
w = tape.parameter("w", [[0.1], [0.2]])
loss = tape.sum(tape.matmul(x, w))

# Record derivative operations, then lower to worker tasks.
tape.backward(loss)
tasks = split(tape, tile_size=2)
tasks, updated = with_optimizer(
    tasks, tape.shapes, tape.parameters, tape.gradients,
    tile_size=2, learning_rate=0.01, optimizer="sgd",
)
tasks = bounded_reductions(tasks, fan_in=8)
peak_estimate = admit_plan(tasks, tape.shapes, budget=1 << 20)
assert updated["w"] == "updated/w"
assert peak_estimate > 0
```

`updated` maps state names to task output IDs, not trained arrays. Executing the
plan and storing those outputs are separate runtime responsibilities. Adam also
requires external `adam_m_<parameter>` and `adam_v_<parameter>` leaves; the caller
provides their values and shapes. Use a fresh tape per graph and a fresh optimizer
builder per append operation.

For the supported server workflow, see [model bundles](../../../../../docs/model-bundles.md).
The compiler CLI is `ygaraty-compile`. To inspect an already saved task list:

```bash
ygaraty-compile saved-plan.json --text plan.txt --jsonl plan.jsonl
```

The exporter exclusively creates both outputs. They must not already exist;
creation is sequential, so failure on the second can leave the first behind.
Task-plan export is distinct from trained `.ygmodel` export.

## Architecture and ownership

The basic path is forward graph → derivative graph → tiled/fused task plan →
optimizer tasks → bounded reductions → admission → runtime handoff. Grouping,
wire narrowing and ownership passes are options selected by higher-level callers;
there is no universal pass sequence for every engine.

| Modules | Responsibility |
|---|---|
| `tape.py` | Rank-two graph operations, shape rules and explicit reverse-mode graph construction |
| `shapes.py`, `parameters.py` | Shape-only declarations, regions, model-name binding and tape-local transpose reuse |
| `tasks.py` | `Input` and `Task` records, dependencies and serialized task descriptions |
| `plan.py` | Main lowering orchestration, fusion, grouping, reduction trees, aliases and `AdmissionPolicy` |
| `sequence.py` | Sequence lowering context, dispatch, task emission and assembly/placement |
| `sequence_attention.py` | Attention products, softmax and row losses |
| `sequence_recurrent.py` | LSTM recurrent products and forward/backward cell or chain tasks |
| `sequence_transformer.py` | Fused layers, vocabulary shards and owner load/update/publish tasks |
| `sequence_views.py`, `sequence_common.py` | Alias/concatenation resolution, pruning, iteration and checkpoint wire dtype |
| `optimizer.py` | `OptimizerPlanBuilder`: norm/clipping, SGD/Adam updates and state assembly |
| `layouts.py`, `convolution.py` | Shared packed tensor layouts and convolution shape contracts |
| `spec.py` | Restricted data-only forward model specification, reconstruction and validation |
| `cost.py`, `export.py` | Work estimates and readable/JSONL task-plan export |

Operation-family sequence mixins share `Lowering` state; they do not own separate
lifecycle state. Model recipes belong in `ygaraty.models`. Numerical kernels
belong in the runtime and browser SDK. Storage, transport and coordinator code
must not become compiler dependencies, including through lazy imports.

The package facade is the preferred import surface. Existing imports of
`Input`/`Task` from `plan`, and view/iteration helpers from `sequence`, remain
compatible. Internal builder classes are implementation seams, not a promise of
thread-safe shared instances or a stable third-party plugin API.

## Contracts to preserve

- Tensor shapes are two-dimensional. Regions are half-open
  `[row_start, row_stop, column_start, column_stop]`; `None` means the whole value.
- IDs, dependency order, region offsets and attributes form the worker protocol.
  Packed LSTM/Transformer layouts and Adam's moment-m/moment-v/parameter bands
  must change together with their consumers.
- `backward()` records derivatives explicitly. A forward model specification
  must be captured before differentiation; it supports a restricted operation
  allowlist, not every internal Tape operation.
- Specifications allow 1–128 nodes and inputs, input dimensions of 1–128,
  at most 65,536 input elements total and 65,536 elements per operation output.
  These bounds do not make arbitrary raw task lists safe or valid.
- View resolution mutates task input references; whole-value assemble/transpose
  consumers are preserved. Some planner passes mutate tasks. Do not assume
  immutable plans or concurrent reuse.
- Admission estimates four times float64 input-plus-output bytes per task.
  It is not measured RSS, total job memory, numerical validation or full graph
  validation. Region estimates can be computed without a known leaf shape.
  Special chain/output budgets can supersede ordinary output limits.
- Export checks unique IDs and internal dependency order. Unknown references
  are external inputs; the caller must establish that those leaves exist.
- Reduction trees require `fan_in >= 2`; smaller values are rejected before
  construction because they cannot shrink a join.
- Work estimates describe units for timing models, not measured performance.
  Owner checkpoints retain float64 to preserve reload state.

## Extending the compiler

1. Define the operation's shape, input, attribute and output-layout contract.
   Add a Tape method and derivative rule if differentiation is supported.
2. Add lowering to the responsible operation family. Preserve external leaves,
   dependency order, tail tiles and assembly regions. Keep task emission separate
   from numerical execution.
3. Update admission/work estimates and runtime/browser capabilities as needed.
   Extend `spec.OPERATIONS` only when the operation is intended for public model
   specifications, with attribute and size rejection tests.
4. Add explicit structural tests and representative graph fixtures. Test empty
   or invalid inputs, uneven boundaries, fan-out and budget edges where relevant.
   Explain any changed fingerprint rather than merely replacing its hash.
5. Verify arithmetic, gradients and cross-backend parity on an authorized host
   before claiming numerical support or production readiness.

## Style and build checks

The compiler shares the project's package-wide Ruff configuration.
Install the `dev` extra in the active development environment to get the pinned
formatter/linter and Python build frontend:

```bash
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
```

The root commands select Python files throughout the repository. Rules cover
PEP 8 errors (including 79-character lines), Python correctness, import sorting
and Python 3.12 idioms. Use `ruff check --fix ygaraty/compiler` and
`ruff format ygaraty/compiler` for safe automatic cleanup, then review the diff.
Tests and external Python scripts use the same repository-wide checks.

The project already uses `pyproject.toml` metadata, setuptools package discovery,
optional dependency extras and a console-script entry point. Build a wheel with
`python -m build --wheel`; see the [maintainer tools](../../../../../tools/README.md)
for an offline build command and prerequisites. Building and importing a wheel
checks packaging; it does not run or validate training.

## Verification and maturity

The graph/task regression suite covers primitive and fused Transformer, banded
and owned vocabulary plans, LSTM, synthetic and factorization graphs, SGD/Adam,
public specifications, exports and package boundaries. New boundary tests cover
specification caps, admission precedence, view concatenation edges, fused Adam
output caps, reduction leaf preservation and graph-to-optimizer pass composition.

Run the reviewed offline compiler set from the repository root with an existing
Python environment containing the project dependencies:

```bash
PYTHONPATH=tests python -m unittest \
  test_compiler_boundaries_offline \
  test_optimizer_sequence_offline \
  test_plan_spec_offline \
  test_compiler_cost_export_offline \
  test_convolution_contract_offline \
  test_model_graphs_offline \
  test_translation_components_offline \
  test_synthetic_factorization_offline \
  test_package_layout \
  test_factorization_simulation
```

These are metadata, mocked-dispatch and simulation checks. They do not establish
numerical gradient correctness, convergence, measured memory or a complete
training run. Full repository discovery also includes workload and service tests;
it is not an offline-safe substitute for this list.

Further tests **are necessary**: per-operation derivative/numerical parity,
randomized transformation equivalence, fused/unfused execution agreement,
Python/browser/WebGPU compatibility, ownership reloads and complete training.
Some numerical tests already exist (for example `tests/test_core.py`), but were
not run under this machine's offline policy. `plan.split` and parts of Tape still
need decomposition; raw pass arguments and all malformed task schemas are not
uniformly validated. This is a tested compiler core with remaining verification
and design work, not a fully verified production compiler.

See the [verification backlog](../../../../webgpu/tests/README.md)
for completed evidence and remaining gates, and the
[model module notes](../models/README.md) for model details.
Internal design notes live in source-adjacent README files; user guides live in
`docs/`. Experiment reports, JSON fixtures and generated plans remain outside
the source package.

For inference without a training loss, use `tape.forward(outputs={"name": tensor})`.
It validates and snapshots a bounded forward graph without calling `backward()`.
See [spec.py](spec.py) for serialization and input-binding validation.
