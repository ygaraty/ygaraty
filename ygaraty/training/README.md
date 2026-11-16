# Training implementation

`Trainer` runs local updates using a recipe. `Coordinator` assigns rounds,
collects deltas and commits outer SGD or Nesterov updates. Recipes include
Transformer, MLP and ConvNet; optional Torch variants import Torch explicitly.
Recovery, convolution growth and batch validation are included here.

Run `ygaraty train --rounds 1 --workers 2 --local-steps 1 --portable` for a tiny
local Transformer round. This runs workers sequentially and uses no network.
The browser DiLoCo client and kernels live in the WebGPU package's `browser/`.
The inference launch server does not expose a distributed training service.
