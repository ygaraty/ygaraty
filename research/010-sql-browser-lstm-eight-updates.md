# SQL-fed attention LSTM in two browser tabs, eight updates

| | |
|---|---|
| **Date** | 2026-10-22 |
| **Path** | HTTP + browser (JavaScript float64), SQLite dataset |
| **Result** | All 1,344 PyTorch checks passed over 8 updates and 2 epochs; max error 6.03e-15; one expired-task retry, zero rejections. |
| **Evidence** | `runs/http-lstm-780aaaacb3f04cca974b1ddcf14cb3d0/verification.json` (local, not in git) |

*Moved verbatim from the former project README. Commands assume Ygaraty is installed (`pip install -e .`) and use the `ygaraty` command; `runs/` paths refer to local generated output.*

**Local browser validation, 2026-10-22:** created the job through the dashboard
and ran two real browser tabs for eight updates (two ordered epochs of the eight
SQL pairs). All 1,344 PyTorch checks passed: 70,716 total operations, 42,818 numerical
operations on browser workers, maximum error `6.03e-15`. Reloading both contributor
tabs created four worker sessions and exercised one expired-task retry, with zero
rejected numerical results. The downloaded checkpoint was byte-identical to the
saved file; an independent eight-update PyTorch replay from SQL also matched its
weights and Adam moments. Training files occupied 808,930 bytes before the small
audit report, plus a 24,576-byte SQLite database. This was a correctness/UI run,
including time waiting for contributors and manual UI checks, not a throughput
benchmark. Evidence: `runs/http-lstm-780aaaacb3f04cca974b1ddcf14cb3d0/verification.json`
and `checkpoint-audit.json`; HTTP regressions: `runs/dashboard-http-tests.log`.
