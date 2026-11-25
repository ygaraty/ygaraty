# Use Ygaraty

Ygaraty runs a model across contributors' browser GPUs. You host the coordinator,
share a contributor link, and use the model through chat or an API.

Choose [S5-Nano or Qwen3.5-9B from public Spaces](public-bundles.md), then:

1. [Install and start your server](quickstart.md).
2. [Invite browser contributors](contributing-gpu.md) or start
   [contributors from command line](contributing-gpu.md#contributors-from-command-line).
3. [Connect an app](api.md), or open your private chat link.

For more control, see [advanced launch settings](advanced.md),
[model bundle preparation](model-bundles.md), and [troubleshooting](troubleshooting.md).

For WebRTC contributor links, see [TURN setup and command-line routes](turn.md).

To use a compressed English Qwen3.5-0.8B checkpoint, see [Kyse models](kyse.md).

Use the [Python API and local training](python-api.md) to build, analyze,
compile and execute Tape graphs from your own code.

For open training research, see [DiLoCo: implementation, commands and measured results](distributed-training.md).
Train a residual ConvNet on MNIST or Fashion-MNIST with browser contributors,
Adam or Muon, GPU-resident local steps and round-boundary recovery.

For native wgpu contributors and 1/2/4/10-worker LAN experiments, see
[native peer-tree training](native-training.md).
