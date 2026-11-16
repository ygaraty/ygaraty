# First HTTP and real-browser contributor runs

| | |
|---|---|
| **Date** | 2026-10-22 |
| **Path** | HTTP + browser (CPU and WebGPU) |
| **Result** | HTTP integration tests passed on memory and PostgreSQL; two real-browser LSTM runs with WebGPU matmuls and zero rejected results. |
| **Evidence** | `runs/http-webgpu-smoke.json`, `runs/http-two-browser-webgpu.json` (local, not in git) |

*Moved verbatim from the former project README. Commands assume Ygaraty is installed (`pip install -e .`) and use the `ygaraty` command; `runs/` paths refer to local generated output.*

`tests/test_http.py` starts real HTTP coordinators and two Node contributors,
checks MLP training against a local reference, and verifies full LSTM weights,
Adam moments, and recurrent state with the independent block-streaming reference.
It also tests invalid recipes, numerical rejection, expired/duplicate results,
authorization, and cancellation. Set `HTTP_POSTGRES_TEST_URL` to repeat those
checks against a disposable PostgreSQL database:

```sh
HTTP_POSTGRES_TEST_URL=postgresql://... python -m unittest discover -s tests -p test_http.py -v
```

On 2026-10-22 all four HTTP integration tests passed (memory and PostgreSQL).
An actual in-app browser also completed an LSTM job with 1,470 numerical tasks,
including 243 WebGPU matrix operations, three updates and zero rejected results.
That report is `runs/http-webgpu-smoke.json`; it records a small browser smoke
test, not a throughput or large-model claim.

A second actual-browser run used two contributors, hidden width 32, and three
LSTM/Adam updates: all 3,852 operations completed (3,249 browser computations,
including 702 WebGPU matrix operations), with zero retries or rejected results.
Both browsers contributed. See `runs/http-two-browser-webgpu.json` and
`runs/http-browser-dashboard.png`. GPU operations were verified at the float32
tolerance above; this is separate from the strict full-checkpoint CPU tests.

Dashboard access: open the administrator URL once per browser tab. Validated
access is retained in sessionStorage for that tab, so navigating to the plain
root URL keeps creation available. Contributor pages never use this cached
administrator access. Missing/expired access hides the creation form and shows
a link-entry recovery screen. After a server restart, use the newly printed
administrator link; the previous token is intentionally invalid.
