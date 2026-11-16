# Offline tensor serialization benchmark

Follow-up: [implemented binary tensor uploads and measured loopback evaluation](LOOPBACK.md). The findings below describe the original isolated serialization run.

Measured on 2026-10-30. This benchmark generates deterministic local arrays and only serializes them. It does not train models, run services, open datasets, or access the network.

## Run

With an installed `ygaraty` CLI and a locally available MessagePack dependency:

```bash
ygaraty benchmark-serialization --output /tmp/serialization.json
```

The optional `serialization` package extra declares MessagePack. The command never installs or downloads it. If only pip's bundled copy is available, explicitly select it:

```bash
ygaraty benchmark-serialization --pip-vendored-msgpack \
  --elements 1024 65536 1048576 --repeats 7 --seed 20261030 \
  --output /tmp/serialization.json
```

The recorded run used that local bundled copy: MessagePack 1.1.2, **pure-Python fallback**, Python 3.12.2, NumPy 2.1.1, macOS ARM64. The checkout's existing console-script shebang pointed to a missing former environment; execution used the existing interpreter to invoke that same CLI script with the checkout on `PYTHONPATH`. No dependency was fetched and no installed environment was changed.

## Method

Three formats carry the same envelope (`version`, `dtype`, `shape`, `data`): compact JSON numeric arrays; compact JSON containing base64 tensor bytes; MessagePack containing a native binary field. Every tensor retains its dtype and byte order. This isolates representation changes from float64-to-float32 quantization.

Fixtures contain seeded normal values plus positive/negative zero, a subnormal, and finite extrema, at 1,024, 65,536 and 1,048,576 elements for both float32 and float64. Fixture generation and correctness checks are outside timing. Each encode/decode has one warm-up followed by seven samples; the report retains all samples, minimum and median. Encoding includes list/bytes conversion and UTF-8 encoding; decoding returns a writable independent NumPy array for every format. There is no compression, hashing, HTTP, browser, disk I/O or training inside measured operations. Formats run in fixed order in one process; these are exploratory measurements, not controlled cross-machine performance claims.

Memory is measured in a **separate tracemalloc pass**, avoiding tracing overhead in speed measurements. It reports peak traced temporary plus output allocations per operation, excluding the pre-existing tensor or input payload. This is not total RSS; native allocator activity may be missed. Timing and memory results depend on the implementation and allocator.

## Results

All **18 cases passed bitwise round-trip checks**, including dtype and shape. Results for 1,048,576 elements follow; MiB means 1,048,576 bytes. Full samples, input hashes, environment and smaller cases are in [the JSON report](results-2026-10-30.json).

| Dtype | Format | Wire MiB | Encode median ms | Decode median ms | Encode peak traced MiB | Decode peak traced MiB |
|---|---|---:|---:|---:|---:|---:|
| <f4 | json | 19.63 | 459.37 | 287.19 | 71.26 | 51.69 |
| <f4 | base64 | 5.33 | 21.04 | 18.98 | 17.33 | 14.67 |
| <f4 | msgpack | 4.00 | 0.25 | 0.54 | 8.00 | 12.00 |
| <f8 | json | 19.63 | 550.37 | 255.01 | 71.26 | 51.69 |
| <f8 | base64 | 10.67 | 41.60 | 37.33 | 34.67 | 29.33 |
| <f8 | msgpack | 8.00 | 0.42 | 1.15 | 16.00 | 24.00 |

## What is worth changing

1. **Prioritize tensor bytes in uploads and weight/update transfers.** `submission/client.py` and `submission/tensors.py` currently encode/decode base64 chunks; `coordination/convnet_scaling.py` also supports base64 tensors. In this benchmark, MessagePack reduces the base64 envelope by approximately 25% and substantially reduces local encode/decode work without losing precision. This supports testing binary upload bodies or binary fields next. It does not prove MessagePack is superior to raw HTTP binary bodies with separate metadata; that alternative is not measured here.
2. **Keep metadata and human-facing output in JSON.** CLI output, manifests, configuration and reports benefit from readability. These tensor results do not justify replacing their format, or imply small control messages achieve the same gains.
3. **Centralize the tensor contract before migration.** Specify version, dtype, shape, byte order, contiguous layout, size limits and checksum rules. Keep current finite-value checks, quotas and idempotency behavior. The codec here is a benchmark utility, not an untrusted-input production decoder. Add Python/browser compatibility fixtures and negotiated content types before changing existing endpoints.
4. **Do not change model exports or hashes on this evidence alone.** File loading, compression, streaming and artifact compatibility are not measured. Existing canonical JSON hashing must retain its semantics or receive an explicit versioned replacement.

MessagePack's result is driven largely by preserving contiguous bytes instead of creating/parsing numeric text or base64. Even its local pure-Python implementation can copy a large binary field efficiently; this does not establish how its nested-object encoding or a JavaScript implementation performs. Browser measurements, transport integration, whole-process memory, representative real payloads and end-to-end throughput remain unmeasured. No production wire format has changed.

## Offline validation

`tests/test_serialization_benchmark.py` checks endian preservation, finite extrema, signed zero, non-contiguous arrays, empty arrays, writable independent output, metrics, relative payload size and CLI report generation. JSON explicitly rejects nonfinite values. Missing local MessagePack produces an actionable CLI error; tests skip when neither regular nor pip-bundled MessagePack is available. No workload kernels or services are used.
