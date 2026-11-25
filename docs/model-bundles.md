# Prepare and serve model bundles

Most users should start with [`ygaraty launch --model NAME --num-workers 2`](advanced.md); see
[launch or serve](commands.md#launch-or-serve) for when `serve` fits better. These lower-level
commands let you inspect, prepare and validate bundles separately.

## Start one session

Use existing local GGUF, config and tokenizer files. No command below downloads
weights, contacts a provider or starts a contributor implicitly.

```bash
ygaraty model inspect --gguf ./checkpoint/model.gguf --config ./checkpoint/config.json
ygaraty model prepare --gguf ./checkpoint/model.gguf \
  --config ./checkpoint/config.json --tokenizer ./checkpoint/tokenizer.json \
  --splits 0,16,32 --context 4096 --generate 256 --output ./model
ygaraty model validate ./model/bundle.json
```

Choose split boundaries matching the inspected layer count; omit `--splits` for
one stage. Preparation derives supported ChatML pieces from the GGUF's template
or adjacent `tokenizer_config.json`. If unavailable or unsupported, pass
`--chat-format ./chat.json` containing explicit `first`, `generation` and optional
`sep`/`end` pieces. It never silently substitutes another model's chat format.

`inspect` reads local headers/configuration. `validate` checks paths, range hashes,
tokenizer/self-test digests and freshly compiled plans; neither executes model
inference. **`prepare` computes numerical device self-test references**, so it
must run on a host authorized for that computation. Offline tests replace that
step with synthetic references; they are not an actual checkpoint preparation run.

The output directory must be new. Preparation copies content-addressed weight
pieces with bounded buffers, verifies them, pins tokenizer/self-test contents and
publishes `bundle.json` atomically after validation. It cleans up a failed new
export and never overwrites an existing directory. The bundle is independent of the original checkpoint paths.
Legacy local bundles remain readable; validation reports `artifacts_pinned:false`
when their tokenizer/self-test digest inventory is absent. Direct/WebRTC and
remote-storage bundles are outside this host's supported scope.

The same API is available without CLI helpers:

```python
from pathlib import Path
from ygaraty import BundleBuilder, Checkpoint

checkpoint = Checkpoint(
    Path("checkpoint/model.gguf"), Path("checkpoint/config.json"),
    Path("checkpoint/tokenizer.json"),
)
report = BundleBuilder(checkpoint, splits=(0, 16, 32)).prepare("model")
print(report.path, report.sha256)
```

Once prepared, start the host:

```bash
ygaraty serve --bundle ./model/bundle.json --model my-model
```

`serve` can also download a supported model, like `launch`:

```bash
ygaraty serve --model s5-nano --num-workers 2
ygaraty serve --bundle-url https://example.com/my-model/bundle.json --model my-model --num-workers 2
```

It prints the contributor link to share, the session page, the web chat and the
API address. Options mean the same as in `launch`:

- `--workspace DIR`: keys and state, default `runs/MODEL/N-workers`. Keys are
  created on first start and reused.
- `--slots N`: chats at once, default 1, at most 16.
- `--context N`: default the bundle's context.
- `--host`, `--port`: default `127.0.0.1:8765`.
- `--public-url URL`: the address in the printed links, when a proxy serves
  this host.
- `--keys DIR`, `--state-dir DIR`: your own directories instead
  (`ygaraty init --directory DIR` creates keys).

The memory flag bounds declared stage weights, not total browser/GPU memory.
Each start creates a new session ID and epoch. Logs/artifact state go in that
epoch's state directory; chat history and the request ledger do not survive a
restart.

Open the contributor link in a WebGPU-capable browser and accept
contribution. The same page shows the model, the workers and live statistics.
Open `/chat`, enter the **owner** key and send a message. The host
becomes ready only after the required contributors finish preparation. The keys
are separate scopes: a contributor cannot call owner chat endpoints, and the
owner key is not a contributor credential. Use HTTPS when exposing the host
beyond loopback so browsers have a secure WebGPU context and credentials are
protected. TLS termination and account management are operator responsibilities.
