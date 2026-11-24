# Python API and local training

Install with the [quickstart](quickstart.md). From a source checkout:

```sh
python -m pip install -e .
```

Import the compiler, analyzer, runtime and training library through `ygaraty`:

```python
from ygaraty import Tape, ShapeTape, Parameters
from ygaraty.compiler import split, group, with_optimizer
from ygaraty.analysis import summarize, useful_flops
from ygaraty.runtime import evaluate, execute_plan
from ygaraty.training import Coordinator, DiLoCoConfig, Trainer, TransformerRecipe
from ygaraty.backends.webgpu import WebGPUFactory
```

All implementations live directly in `ygaraty/` in the repository. The root
exports are conveniences for the same classes in its submodules.

## Differentiate a graph

```python
from ygaraty import Tape
from ygaraty.runtime import evaluate

tape = Tape()
x = tape.input("x", [[2.0, -1.0]])
w = tape.parameter("w", [[0.3], [0.7]])
y = tape.matmul(x, w)
loss = tape.sum(tape.mul(y, y))
tape.backward(loss)
values = evaluate(tape)
print("loss:", values[loss])
print("gradient:", values[tape.gradients[w]])
```

## Run training locally

This small Transformer example uses synthetic data, NumPy and the public Tape
runtime. It needs no model download, private service, GPU or PyTorch:

```sh
ygaraty train --rounds 2 --workers 2 --local-steps 2 --portable
```

The workers run sequentially in this local demonstration. It validates DiLoCo
rounds; it does not launch remote contributors. Use `ygaraty launch` for the
browser inference service described in the quickstart.

For networked training, use [`ygaraty train serve`](distributed-training.md).
It launches a residual ConvNet session with browser contributors, deterministic
dataset shards, held-out evaluation and complete round-boundary recovery.

`ygaraty train --help` lists model config, batch and weight inputs, learning
rates and portable round export. `Trainer` performs local Adam updates;
`Coordinator` collects worker deltas and commits outer SGD or Nesterov updates.
MLP, Transformer and ConvNet recipes, recovery and growth helpers are included.
PyTorch recipes and independent reference oracles are optional; install
`python -m pip install '.[training-torch]'` from a checkout.

## Compile and use browser kernels

`ygaraty compile --help` describes exporting saved Tape plans. The WebGPU
package includes the browser SDK, forward/backward kernels, DiLoCo client and
training workers. Locate the installed assets with:

```python
from ygaraty.backends.webgpu.browser_assets import BROWSER_ROOT, SDK_ASSETS
```

Serve ES modules with their sibling assets intact. The optional TensorFlow.js
comparison worker expects a separately served `/tf.min.js`; the native research
runner requires Node's `webgpu` package or Deno WebGPU. Neither is needed by the
local training command. Real GPU execution depends on WebGPU hardware support.
