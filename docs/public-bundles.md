# Public Spaces models

These presets download prepared bundles from our public DigitalOcean Spaces
storage. They need no local checkpoint, cloud credentials or conversion step.

Choose **1, 2, 3 or 4 workers** for any model. `--num-workers` is mandatory
and selects the matching `MODEL/prepared/v1/N-workers/bundle.json` in Spaces.
The launcher reads that manifest and prints each worker's calculated GPU RAM
requirement **before downloading weights**.

```sh
ygaraty launch --model qwen3.5-0.8b --num-workers 2 --open-chat
ygaraty launch --model s5-nano --num-workers 2 --open-chat
ygaraty launch --model qwen3.5-9b --num-workers 2 --open-chat
ygaraty launch --model qwen3.8-27b --num-workers 4 --open-chat
# More contributors, smaller per-worker allocations:
ygaraty launch --model qwen3.5-9b --num-workers 4 --open-chat
```

## Calculated GPU RAM

At the default 4,096-token context and one chat slot, recommended memory per
worker is shown below in decimal GB. Values include planning headroom and are
rounded for display; startup recalculates from the selected manifest.

| Model | Workers | Recommended GPU RAM for workers 1…N |
| --- | --- | --- |
| Qwen3.5-0.8B | 1 | 1.02 GB |
| Qwen3.5-0.8B | 2 | 0.78 / 0.78 GB |
| Qwen3.5-0.8B | 3 | 0.69 / 0.44 / 0.70 GB |
| Qwen3.5-0.8B | 4 | 0.65 / 0.40 / 0.40 / 0.66 GB |
| S5-Nano | 1 | 3.82 GB |
| S5-Nano | 2 | 2.37 / 2.37 GB |
| S5-Nano | 3 | 1.74 / 1.54 / 1.76 GB |
| S5-Nano | 4 | 1.47 / 1.20 / 1.18 / 1.49 GB |
| Qwen3.5-9B | 1 | 7.39 GB |
| Qwen3.5-9B | 2 | 3.85 / 3.84 GB |
| Qwen3.5-9B | 3 | 2.69 / 2.62 / 2.68 GB |
| Qwen3.5-9B | 4 | 2.01 / 2.13 / 2.13 / 2.01 GB |
| Qwen3.8-27B | 1 | 20.86 GB |
| Qwen3.8-27B | 2 | 10.66 / 10.52 GB |
| Qwen3.8-27B | 3 | 7.28 / 7.06 / 7.14 GB |
| Qwen3.8-27B | 4 | 5.33 / 5.64 / 5.38 / 5.44 GB |

Qwen3.8-27B's bundles come from lmstudio-community's `Qwen3.8-27B-Q4_K_M.gguf`
(revision `5a7da68`, SHA-256 `e00082f7…e520`, Q4_K and Q6_K tensors only) with
Qwen's `config.json` and tokenizer (revision `3756181`). Unsloth's "UD" Q4_K_M
mixes in IQ4/IQ3 tensors the WebGPU kernels do not run. Splits: `0,64`,
`0,33,64`, `0,21,45,64`, `0,14,33,51,64`; the MTP and vision tensors are not used.

The calculation uses actual tensor bytes and layer assignments, tied embedding
reuse, context-sized KV caches, per-slot recurrent state, scratch buffers and
resident decoding snapshots. Recommendations add **20% + 256 MiB per worker**
for transient allocations and browser/driver overhead. The runtime budget is
rounded up to whole GiB from the largest recommendation. `--memory-gb` is an
optional override of that budget; values below the recommendation are rejected.
It does not change the estimate.

These are GPU-memory planning estimates, not measured peaks, guarantees for a
particular card, or server system-RAM requirements. Other applications and
browser/device limits can reduce available memory. Every contributor must pass
its device self-tests. More slots increase state allocations; `--context` can
reduce capacity within the prepared bundle's limits. Repacked/speculative
bundles are not supported by automatic estimation.

All presets have a 256-token generation cap; requests can lower it with
`max_tokens`. Workspaces are separated by model and worker count, so switching
configurations does not mix artifacts or keys. Run one server at a time unless
you assign different ports.

## Other prepared bundles

Use `--bundle-url https://.../bundle.json` with the required `--num-workers` for
another Qwen3.5-family prepared bundle. The stage count must match. `--splits`
and `--generate` are for [checkpoint preparation](model-bundles.md); published
plans are not rebuilt. Other model families remain available through the
lower-level preparation and `serve` workflow.

## Storage and verification

Artifacts are downloaded to `workspace/model/` from paths relative to the
manifest's URL. The manifest must use `ygaraty-webgpu-v1` and complete weight
pieces with offset zero. Anonymous direct HTTPS URLs are used; credentials,
presigned queries and redirects are not accepted. No private cloud package or
Spaces SDK is required.

Weights are streamed to disk with declared length and SHA-256 checks. Publisher
metadata digests are checked when supplied. Legacy metadata without published
digests is trusted over HTTPS on first download and pinned locally thereafter;
this does not prove it matches an independently known publisher hash. The cache
records both source and localized manifest hashes. Subsequent starts verify the
cache without network requests, and reject modified artifacts or a different
source URL. A cache is a snapshot: use a new workspace to pick up a new release.

The manifest is limited to 16 MiB, each metadata file to 64 MiB, and the download
inventory to 64 GiB. Artifact paths cannot escape the cache. A failed download
removes only the newly created cache; pre-existing data is never overwritten.
An interrupted process may leave an incomplete cache; inspect it and choose a
new workspace. Bundle validation recompiles metadata plans but executes no model.

The coordinator serves cached pieces through its existing HTTP routes. Browser
kernels and download behavior are unchanged. Direct browser-to-Spaces streaming
is not added by this feature.

## Public availability

On 2026-11-15, all eight 1–4-worker manifests were publicly readable.
The persistent allocation calculations matched the browser loader for all eight.
For the two-worker bundles, all 461 referenced
S5-Nano objects and all 469 Qwen objects passed anonymous HEAD checks without
declared size mismatches. Both manifests passed public metadata/fresh-plan
admission at the settings above. If an object later returns 403, that is a
publisher access issue; users should not provide Spaces credentials.

On 2026-11-16 the four Qwen3.8-27B manifests were published; all 3,452
objects they reference passed anonymous HEAD checks with matching sizes, and
the 1-, 2- and 4-worker bundles ran on an RTX 3090 (see [TESTED.md](https://github.com/ygaraty/ygaraty/blob/main/TESTED.md)).

Manifest and object-access checks do not establish weight-hash integrity or
GPU correctness. The installer checks hashes while downloading; each contributor
runs device self-tests before the session becomes ready.
