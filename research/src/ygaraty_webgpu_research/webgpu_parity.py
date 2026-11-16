"""Direct Transformer programs: browser CPU/WebGPU versus PyTorch float64.

No App, Job, scheduler, workers, task lowering or DiLoCo. The tiny local HTTP
server serves fixed JavaScript and a fixture, then collects one JSON result.
All training computation runs in one browser program and one PyTorch program.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from ygaraty_webgpu_research.host import component as _host_component

SDK_ASSETS = _host_component("browser_assets").SDK_ASSETS
from ygaraty_webgpu_research.host import component as _host_component

CASES = _host_component("parity.cases").CASES
stable_hash = _host_component("parity.cases").stable_hash
from ygaraty_webgpu_research.host import component as _host_component

check = _host_component("parity.compare").check
summarize = _host_component("parity.compare").summarize
from ygaraty_webgpu_research.host import component as _host_component

profile = _host_component("parity.tolerances").profile
from ygaraty_webgpu_research.host import component as _host_component

PROJECT_ROOT = _host_component("paths").PROJECT_ROOT

ASSETS = _host_component("browser_assets").BROWSER_ROOT
MODES = ("float64", "float32", "compensated")


def pack(array):
    array = np.asarray(array, dtype=np.float64)
    return dict(shape=list(array.shape), data=array.ravel().tolist())


def unpack(value):
    return np.asarray(value["data"], dtype=np.float64).reshape(value["shape"])


def make_fixture(case, steps, modes=MODES):
    """Use a registered case and its independent PyTorch reference
    unchanged."""
    if case.architecture != "transformer":
        raise ValueError("This standalone program expects a Transformer case")
    model, optimizer = case.reference()
    weights = {
        name: p.detach().numpy().copy() for name, p in model.named_parameters()
    }
    graphs, reference = [], []
    for cursor in range(steps):
        batch = case.batches(cursor)
        tape, logits, parameters, _ = case.build_tape(weights, batch)
        graphs.append(
            dict(
                nodes=[asdict(node) for node in tape.nodes],
                parameters=parameters,
                gradients=tape.gradients,
                loss=tape.loss,
                logits=logits,
                constants={
                    name: pack(array)
                    for name, array in tape.values.items()
                    if name not in tape.parameters
                },
            )
        )
        reference.append(case.reference_step(model, optimizer, batch))
    return dict(
        format="ygaraty-standalone-parity-v1",
        case=case.describe(),
        modes=list(modes),
        optimizer=case.optimizer,
        weights={
            name: pack(array.reshape(1, -1) if array.ndim == 1 else array)
            for name, array in weights.items()
        },
        graphs=graphs,
    ), reference


PARITY_ASSETS = (
    "standalone_parity.mjs",
    "standalone_gpu.mjs",
    "sdk.mjs",
    "budget.mjs",
    "convolution.mjs",
) + SDK_ASSETS


def run_browser(
    fixture,
    output,
    chrome=None,
    timeout=60.0,
    chrome_args=(),
    page="standalone_parity.mjs",
    assets=PARITY_ASSETS,
):
    """Launch one headless browser, exchange one fixture and one final result.
    `page` is the module
    the browser runs; `assets` are the static files it may load."""
    chrome = (
        chrome
        or os.environ.get("YGARATY_CHROME")
        or shutil.which("chromium")
        or shutil.which("google-chrome")
    )
    if not chrome:
        raise RuntimeError(
            "Chrome/Chromium required; set YGARATY_CHROME or pass --chrome"
        )
    body = json.dumps(fixture, allow_nan=False).encode()
    received, ready = [], threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path == "/":
                content, mime = (
                    f'<script type="module" src="/{page}"></script>'.encode(),
                    "text/html",
                )
            elif self.path == "/fixture.json":
                content, mime = body, "application/json"
            elif self.path[1:] in assets:
                content, mime = (
                    (ASSETS / self.path[1:]).read_bytes(),
                    "text/javascript",
                )
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self):
            if self.path != "/result":
                self.send_error(404)
                return
            received.append(
                json.loads(
                    self.rfile.read(int(self.headers["Content-Length"]))
                )
            )
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            ready.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with (output / "chromium.log").open("w") as log:
            command = [
                str(chrome),
                "--headless=new",
                "--enable-unsafe-webgpu",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-background-timer-throttling",
                "--disable-renderer-backgrounding",
                *chrome_args,
                f"--user-data-dir={output / 'browser-profile'}",
                f"http://127.0.0.1:{server.server_port}/",
            ]
            process = subprocess.Popen(command, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + timeout
                while not ready.wait(0.2):
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"Chromium exited; see {output / 'chromium.log'}"
                        )
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"Browser program timed out; see "
                            f"{output / 'chromium.log'}"
                        )
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    result = received[0]
    (output / "browser-results.json").write_text(
        json.dumps(result, allow_nan=False) + "\n"
    )
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def require_nvidia_adapter(adapter):
    """Independently reject missing/ambiguous evidence in a required-GPU
    run."""
    if not isinstance(adapter, dict):
        raise ValueError("Missing NVIDIA WebGPU adapter evidence")
    vendor = str(adapter.get("vendor", "")).strip().lower()
    text = " ".join(
        str(adapter.get(k, ""))
        for k in ("vendor", "architecture", "device", "description")
    ).lower()
    if (
        not re.fullmatch(r"nvidia(?:[- ]corporation)?|(?:0x)?10de", vendor)
        or adapter.get("isFallbackAdapter") is not False
        or re.search(
            r"swiftshader|llvmpipe|lavapipe|software|microsoft "
            r"basic",
            text,
        )
    ):
        raise ValueError(
            f"NVIDIA hardware WebGPU adapter required; got {adapter}"
        )
    return adapter


def compare_run(run, reference, config):
    """Per-element acceptance and float64 diagnostics, with existing
    budgets."""
    tolerance_name = "float64" if run["mode"] == "float64" else "webgpu"
    tolerances, strict = profile(tolerance_name), profile("float64")
    if len(run["records"]) != len(reference):
        raise ValueError("Browser returned an incomplete training program")
    records = []
    for actual, expected in zip(run["records"], reference):
        checks, strict_checks, key_biases = [], [], []

        def add(name, kind, actual_value, expected_value, before=None):
            checks.append(
                check(
                    name,
                    kind,
                    actual_value,
                    expected_value,
                    tolerances[kind],
                    before,
                )
            )
            strict_checks.append(
                check(
                    name,
                    kind,
                    actual_value,
                    expected_value,
                    strict[kind],
                    before,
                )
            )

        add(
            "loss",
            "loss",
            unpack(actual["loss"]),
            np.array([[expected["loss"]]]),
        )
        logits = np.stack([unpack(value) for value in actual["logits"]])
        add("logits", "logits", logits, expected["logits"])
        for name, after in expected["after"].items():
            shape = after.shape
            before_actual = unpack(actual["before"][name]).reshape(shape)
            after_actual = unpack(actual["after"][name]).reshape(shape)
            gradient = unpack(actual["gradients"][name]).reshape(shape)
            add(name, "gradient", gradient, expected["gradients"][name])
            add(name, "weight", after_actual, after)
            add(
                name,
                "update_delta",
                after_actual,
                after,
                before=(before_actual, expected["before"][name]),
            )
            for kind in ("m", "v"):
                add(
                    name,
                    kind,
                    unpack(actual["moments"][kind][name]).reshape(shape),
                    expected["moments"][kind][name],
                )
            if name.endswith(".in_proj_bias"):
                d = config["embedding"]
                section = slice(d, 2 * d)
                delta_error = (after_actual - before_actual) - (
                    after - expected["before"][name]
                )
                key_biases.append(
                    dict(
                        name=name,
                        reference_gradient_max_abs=float(
                            np.max(
                                np.abs(expected["gradients"][name][section])
                            )
                        ),
                        actual_gradient_max_abs=float(
                            np.max(np.abs(gradient[section]))
                        ),
                        update_error_max_abs=float(
                            np.max(np.abs(delta_error[section]))
                        ),
                    )
                )
        execution = actual["execution"]
        if run["mode"] != "float64" and (
            not execution["matmuls"]
            or execution["gpu_matmuls"] != execution["matmuls"]
        ):
            raise ValueError("Not every matrix multiplication ran on WebGPU")
        if run["mode"] == "float64" and execution["gpu_matmuls"]:
            raise ValueError("Float64 control executed a GPU product")
        records.append(
            dict(
                step=actual["step"],
                loss=float(unpack(actual["loss"])[0, 0]),
                reference_loss=expected["loss"],
                passed=all(c["passed"] for c in checks),
                float64_parity=all(c["passed"] for c in strict_checks),
                kinds=summarize(checks),
                checks=checks,
                key_biases=key_biases,
                execution=execution,
            )
        )
    return dict(
        mode=run["mode"],
        tolerance=tolerance_name,
        passed=all(r["passed"] for r in records),
        float64_parity=all(r["float64_parity"] for r in records),
        records=records,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=[
            n for n, c in CASES.items() if c.architecture == "transformer"
        ],
        default="transformer-tiny",
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--modes", nargs="+", choices=MODES, default=list(MODES)
    )
    parser.add_argument(
        "--chrome",
        help="Chrome/Chromium executable; defaults to YGARATY_CHROME or PATH",
    )
    parser.add_argument(
        "--chrome-arg",
        action="append",
        default=[],
        help="Extra browser argument; repeat as --chrome-arg=--flag",
    )
    parser.add_argument(
        "--require-nvidia",
        action="store_true",
        help="Stop unless the actual WebGPU device is verified NVIDIA "
        "hardware",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 1 or not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("steps and timeout must be positive")
    if args.require_nvidia and all(mode == "float64" for mode in args.modes):
        parser.error(
            "--require-nvidia needs at least one WebGPU arithmetic mode"
        )
    output = args.output or PROJECT_ROOT / "runs" / (
        "webgpu-direct-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%SZ")
    )
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    case = CASES[args.case]
    fixture, reference = make_fixture(case, args.steps, args.modes)
    fixture["require_nvidia"] = args.require_nvidia
    (output / "fixture.json").write_text(
        json.dumps(fixture, allow_nan=False) + "\n"
    )
    print(
        f"Direct programs: {args.case}, {args.steps} updates, "
        f"{', '.join(args.modes)}\nOutput: {output}",
        flush=True,
    )
    browser = run_browser(
        fixture, output, args.chrome, args.timeout, args.chrome_arg
    )
    if args.require_nvidia:
        require_nvidia_adapter(browser.get("adapter"))
    if [run["mode"] for run in browser["runs"]] != args.modes:
        raise ValueError("Browser returned different arithmetic modes")
    runs = [compare_run(run, reference, case.model) for run in browser["runs"]]
    report = dict(
        format="ygaraty-standalone-parity-report-v1",
        case=case.describe(),
        fixture_sha256=stable_hash(fixture),
        steps=args.steps,
        browser=browser["browser"],
        device_status=browser["device_status"],
        adapter=browser.get("adapter"),
        require_nvidia=args.require_nvidia,
        passed=all(run["passed"] for run in runs),
        runs=runs,
        tolerances={name: profile(name) for name in ("float64", "webgpu")},
    )
    (output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    for run in runs:
        print(
            f"{run['mode']}: {'PASS' if run['passed'] else 'FAIL'} ({run['tolerance']} budgets); "  # noqa: E501 -- preserve literal/report layout
            f"float64 parity: {run['float64_parity']}"
        )
        for record in run["records"]:
            kinds, stats = record["kinds"], record["execution"]
            print(
                f"  step {record['step']}: loss error {kinds['loss']['max_abs']:.3e}, "  # noqa: E501 -- preserve literal/report layout
                f"gradient error {kinds['gradient']['max_abs']:.3e}, "
                f"update error {kinds['update_delta']['max_abs']:.3e}, GPU {stats['gpu_matmuls']}/{stats['matmuls']}"  # noqa: E501 -- preserve literal/report layout
            )
            for c in sorted(
                (c for c in record["checks"] if not c["passed"]),
                key=lambda c: c["ratio"],
                reverse=True,
            )[:3]:
                print(
                    f"    FAIL {c['kind']} {c['name']}: max error "
                    f"{c['max_abs']:.3e}, budget ratio {c['ratio']:.3e}"
                )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
