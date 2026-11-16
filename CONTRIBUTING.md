# Contributing to Ygaraty

Help us make heterogeneous compute clusters seamless to use. Code, research,
hardware measurements, documentation and reproducible bug reports all count.
You do not need a large GPU or a finished research paper to contribute.

Open an issue or a pull request with a clear purpose. Small, focused changes and
early experiments are welcome. For a larger direction, an issue is a useful
place to compare ideas before investing in an implementation.

## Keep it easy to verify

Include these basics in your contribution:

- **Purpose:** the question, problem or behavior you want to improve.
- **Reproduction:** commands, dependencies and relevant model, runtime and
  hardware details. Include configuration and seeds when they affect results.
- **Evidence:** the checks you ran and their results. For numerical or performance
  claims, include a reference or baseline and explain how you compared them.
- **Limits:** what failed, what remains untested and what the evidence supports.

For a documentation fix, a checked example or corrected link can be enough.
For code, run the relevant tests and add a regression check when it helps catch
the bug. Report checks you could not run; do not present untested behavior as
verified. No particular machine or paid service is required to propose a change.

## Share research openly

Put experiments and reports in [research/](research/README.md), with a short
README beside any new reusable code explaining how to run it. A report can use
four headings: Question, Method, Results and Limitations. Link the code and
configuration needed to reproduce it, and separate simulations from measurements
on real hardware. Negative results are useful too.

Keep reusable implementations in the relevant `ygaraty/` module once they are
ready for the public API. User-facing instructions belong in `docs/`; developer
notes belong in the README of the relevant folder. Include only code and data
that can be shared publicly, and keep credentials out of reports and logs.

See [the package overview](ygaraty/README.md),
[WebGPU tests](tests/webgpu/README.md) and [tools](tools/README.md) for starting
points. Contributions are shared under the project's [MIT license](LICENSE).
