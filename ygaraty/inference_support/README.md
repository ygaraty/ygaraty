# Backend-neutral inference support

Backend-neutral host utilities used by public WebGPU serving.
This package uses only the Python standard library. It does not import a backend,
load models, start servers or depend on NumPy, Django or WebGPU.

| Module | Purpose |
| --- | --- |
| `chat` | ChatML formatting, conversation identifiers and tokenizer adapter |
| `throughput` | Bounded aggregate token-rate accounting |
| `artifacts` | Local artifact lookup with directory confinement |
| `http` | HTTP replies and NDJSON streaming |
| `roster` | Contributor liveness and optional SQLite history |

Tokenization accepts a caller-provided object implementing `encode_batch`.
Static asset paths, authentication, model execution and backend-specific bundle
validation remain with each consumer. Interfaces live in the separate
`ygaraty.backend_contracts` package.

This module ships in the single `ygaraty` distribution. From the repository
root, run `python -m pytest tests/inference_support`.

Import helpers from `ygaraty.inference_support`, for example:

```python
from ygaraty.inference_support.roster import ContributorRoster
```
