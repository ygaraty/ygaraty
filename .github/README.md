# GitHub automation

[CI](workflows/ci.yml) runs on pull requests, pushes to `main`, and manual runs
from the repository's Actions tab. It needs no repository secrets.

The Python/browser job checks generated files, installer behavior, package
boundaries, isolated runtime/research wheels, and the Python and Node test suites.
It uses Python 3.12 and Node 22. The Go job uses the version in `relay/go.mod`,
runs race-enabled tests, and builds both relay commands.

Successful package jobs attach both wheels as `ygaraty-wheels-<commit>` artifacts,
retained for 14 days. Download them from the workflow run. These are development
builds; this workflow does not publish to PyPI or deploy a server.

GPU and optional Torch tests retain their existing opt-in/skip behavior. These
checks do not certify real GPU performance or cross-machine networking.

For branch protection, require both **Python, browser and wheels** and **Go relay**.
Also require **Clean install (ubuntu-latest)** and **Clean install (macos-latest)**.
The installer jobs use fresh home directories, install real dependencies, check
CLI entry points and reuse, and verify that shell profiles are not modified.
Push runs fetch the exact tested revision from the public Git repository.

Preview releases are published only after all four jobs pass. Version `0.1.0a1`
is the first preview; install instructions pin `v0.1.0a1`. Release artifacts are
the verified wheels, with checksums. GPU readiness still needs real hardware.
