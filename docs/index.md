# Use Ygaraty

**Ygaraty is an open-source framework for distributed computation.** We aim to
provide the building blocks for any kind of mathematical computation to run
across heterogeneous devices and networks, with compute contributed by one
person or many collaborators.

The compiler, analyzer, Tape and execution engine are reusable pieces for
describing, planning and running computations. The [Python API](python-api.md)
lets you work with these pieces directly.

Today, Ygaraty supports **inference with or without distribution** through
browser or native WebGPU workers. **Distributed training is ongoing research**,
including [DiLoCo and other training experiments](distributed-training.md).
These are current applications of the framework; support for arbitrary
mathematical workloads remains a goal.

To start with inference, host a coordinator and use the model through chat or
an API. Run a contributor yourself, or share the contributor link to invite
others to provide compute.

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
