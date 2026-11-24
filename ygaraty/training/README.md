# Training implementation

`Trainer` runs local updates using a recipe. `Coordinator` assigns rounds,
collects deltas and commits outer SGD or Nesterov updates. Recipes include
Transformer, MLP and ConvNet; optional Torch variants import Torch explicitly.
Recovery, convolution growth and batch validation are included here.

Run `ygaraty train --rounds 1 --workers 2 --local-steps 1 --portable` for a tiny
local Transformer round. This runs workers sequentially and uses no network.
The browser DiLoCo client and kernels live in the WebGPU package's `browser/`.

`ygaraty train serve` launches the separate browser training service. See the
[user guide](../../docs/distributed-training.md) for commands.

`TrainingSession` owns fixed logical worker slots and a `Coordinator`.
`server.py` supplies authenticated HTTP transport; `launch.py` owns the server
and optional public tunnel. `datasets.py` validates IDX archives and derives
each shuffled shard from seed, logical worker and epoch. The committed Adam
step determines the next sample cursor. `convnet_webgpu.py` builds its graph
using `ShapeTape`, reverse-mode differentiation and the shared compiler.

Only a completed round is published durably. All replica Adam states, global
weights, outer momentum, configuration/dataset/code identity, metrics and retry
receipts are in one atomic checkpoint. An aborted or interrupted round discards
every tentative update. Membership replacement preserves the logical slot's
committed state. Browser credentials rotate on server restart and are not
written into checkpoints.

Tests cover finite-difference gradients, JavaScript/Python HTTP parity, retry
conflicts, expired leases, replacement, disk failure and recovery. Run
`python -m unittest discover -s tests -p test_training_session.py`.
`tools/training_e2e.py` exercises actual Chromium contributors with bounded
runtime and compares the resulting model against the Python reference.

`gradient_reduction.py` isolates synchronous gradient reduction from DiLoCo.
`gradient_plan` prunes optimizer tasks from the Tape backward graph;
`reduce_gradients` performs deterministic adjacent-pair, sample-weighted means,
carrying an odd leaf unchanged. `SharedAdam` applies one compiled update after
reduction. Tests cover unequal partitions, repeated updates, optimizer moments,
and parity with the existing compiled trainer. The hardware-browser experiment
is `tools/tree_gradient_experiment.py`; its worker source is
`browser/gradient_client.mts`.

`lan.py` adds a separate synchronous, fixed-membership experiment with native
Deno/wgpu workers. It compiles gradient-only graphs and the root optimizer graph,
and sends inputs with `tensor_transport.py`. `browser/peer_training.mts` owns
worker state and the direct binary-heap collective: child gradients reduce at
parents, root Adam executes once, and weights broadcast down the tree. This is
host-memory HTTP transport, not GPU-direct or WebRTC. A failed collective requires
a new session; partial broadcasts are not recoverable checkpoints.

`tensor_transport.py` and `browser/tensor_transport.mts` implement YGT1:
four magic bytes, a uint32 little-endian header length, a UTF-8 JSON header, then
contiguous little-endian f64 tensors. The header lists names, shapes and dtype in
payload order, metadata and the payload SHA-256. Bounds are 4 MiB for the header
and 64 MiB for the whole frame. Decoders reject duplicate tensor names, unsupported
dtypes, malformed shapes, nonfinite data, corruption and trailing bytes. The
checksum is corruption detection; session authentication and a trusted private
network provide the transport boundary.

Run the binary interoperability and topology/failure tests through the normal
Python and Node suites. `tools/native_tree_pilot.py` starts real native processes
and sockets, verifies CPU parity, and checks process/port cleanup. See the
[native training guide](../../docs/native-training.md) for LAN commands.

Muon is available for the public ConvNet recipe through `optimizer_kind="muon"`
and the training CLI's `--optimizer muon`. See the
[Muon user guide](../../docs/distributed-training.md#experimental-muon-local-optimizer)
for parameter groups, optimizer state and the bounded comparison command.
