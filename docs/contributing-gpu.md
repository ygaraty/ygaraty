# Contribute your GPU

1. Open the contributor link shared by the server owner in a WebGPU-capable
   browser. For another machine's server, the link must use HTTPS.
2. Click **Accept and contribute**. The link supplies your contributor key;
   no account or manual key entry is needed.
3. Wait while your assigned model stage downloads and passes device checks.
4. Keep the tab open while people use the model. Closing it stops your
   contribution and may make the model unavailable until capacity returns.

Contribution uses GPU memory, network bandwidth and power. The page shows
loading and serving status. The server owner chooses layer splits and weight
budgets; if your device cannot fit its assigned stage, ask the owner to adjust
those settings and prepare a new workspace.

A contributor link grants contributor access only. It cannot be used as a chat
API key. Ask the owner for authorized chat access separately.

## Contributors from command line

For experimental **native training** workers with direct LAN peer reduction,
see [Native training contributors](native-training.md). The commands below are
for inference contributors.

After [installing Ygaraty](quickstart.md), choose a runtime and pass the exact
contributor link printed by `ygaraty launch`. Run this on each machine that will
contribute GPU capacity:

```sh
ygaraty contribute 'CONTRIBUTOR_LINK' --runtime headless --num-workers 2
ygaraty contribute 'CONTRIBUTOR_LINK' --runtime deno --num-workers 1
ygaraty contribute 'CONTRIBUTOR_LINK' --runtime node --num-workers 1
ygaraty contribute 'CONTRIBUTOR_LINK' --runtime bun --num-workers 1
```

These commands accept participation automatically. Ctrl-C stops the processes
they started. Multiple workers on one GPU share that GPU's memory; more processes
do not create more capacity. A worker exiting stops the rest of its group.

| Runtime | GPU implementation | Setup | Transport |
| --- | --- | --- | --- |
| `headless` | Chromium WebGPU | Chromium; `CHROMIUM` can select its executable | WebRTC, TURN and relay fallback |
| `deno` or `wgpu` | Deno's built-in wgpu | Deno 2 with WebGPU | Coordinator relay |
| `node` | Dawn | Node and `webgpu@0.6.2` | Coordinator relay |
| `bun` | Dawn, experimental | Bun and `webgpu@0.6.2` | Coordinator relay |

`wgpu` is an alias for Deno, not a separate Rust executable. Native contributors
reuse the browser contributor's admission, numerical self-test, verified weight
downloads and inference kernels. They currently download weights on each run;
Chromium retains its browser cache. Native workers do not provide WebRTC or TURN.

For Node or Bun, install the public Dawn binding in your working directory:

```sh
npm install webgpu@0.6.2
# Or: bun add webgpu@0.6.2
```

Run the contributor from that directory, or pass
`--webgpu-module /absolute/path/to/node_modules/webgpu/index.js`. Dawn options
can be passed repeatedly, for example `--dawn-option backend=vulkan`.
Deno needs no npm dependency. Its launcher permits network access for the
coordinator and weight downloads; it loads only local Ygaraty modules.

Check native GPU access before joining or downloading a model:

```sh
ygaraty contribute --runtime deno --check
ygaraty contribute --runtime node --check
ygaraty contribute --runtime bun --check
```

The probe runs a tiny compute shader and verifies its output. It does not prove
that a particular model fits or that its inference kernels pass admission.
Native contributor support is experimental: local compute probes and float16
admission against a localhost fixture passed on Deno, Node and Bun. Full
live-session model inference still needs runtime-specific validation. The
archived Deno/Dawn benchmark results are not live contributor acceptance tests.

From a checkout, separate scripts are also available:

```sh
./tools/headless-contributor.sh 'CONTRIBUTOR_LINK' 2
./tools/deno-contributor.sh 'CONTRIBUTOR_LINK' --num-workers 2
./tools/wgpu-contributor.sh 'CONTRIBUTOR_LINK' --num-workers 2
./tools/node-contributor.sh 'CONTRIBUTOR_LINK' --num-workers 2
./tools/bun-contributor.sh 'CONTRIBUTOR_LINK' --num-workers 2
```

Native scripts use `python3`; set `PYTHON` to your Ygaraty Python interpreter if
needed. Each supports `--help` and `--check`.
