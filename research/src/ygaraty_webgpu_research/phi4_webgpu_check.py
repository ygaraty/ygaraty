"""External-host Phi-4 WebGPU conformance runner (DO-23).

Consumes exported stage weights and a JSON sequence of steps. Executes actual
browser inference: do NOT run on the offline simulation workstation. Reference
calculations use NumPy on the identical exported F16/F32 weights, not Torch.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ygaraty_webgpu_research.browser_binary import run_page
from ygaraty.backends.webgpu.checkpoints import phi4_kernels as kernels


def compare(actual, expected, *, atol=1e-3, rtol=1e-3):
    if actual.shape != expected.shape or not np.isfinite(actual).all():
        raise ValueError("Nonfinite output or wrong output shape")
    error = float(np.max(np.abs(actual - expected), initial=0))
    scale = float(np.max(np.abs(expected), initial=0))
    return dict(
        max_absolute_error=error,
        relative_to_peak=error / max(scale, 1e-30),
        passes=bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
    )


def run(
    stage_dir,
    input_path,
    output,
    *,
    chrome=None,
    timeout=600,
    atol=1e-3,
    rtol=1e-3,
):
    if not all(np.isfinite(v) and v >= 0 for v in (atol, rtol)):
        raise ValueError("Tolerances must be finite and nonnegative")
    stage_dir, input_path, output = map(Path, (stage_dir, input_path, output))
    manifest = json.loads((stage_dir / "manifest.json").read_text())
    spec = json.loads(input_path.read_text())
    config = manifest["config"]
    files = {
        "manifest.json": stage_dir / "manifest.json",
        "input.json": input_path,
    }
    weights = {}
    for name, chunks in manifest["tensors"].items():
        parts = []
        for chunk in chunks:
            path = stage_dir / chunk["file"]
            if path.resolve().parent != stage_dir.resolve():
                raise ValueError("Tensor path escapes stage directory")
            files[chunk["file"]] = path
            dtype = "<f2" if chunk["kind"] == "F16" else "<f4"
            parts.append(
                np.fromfile(path, dtype=dtype)
                .astype(np.float32)
                .reshape(chunk["rows"], chunk["cols"])
            )
        value = np.concatenate(parts)
        weights[name] = value.reshape(-1) if "norm.weight" in name else value
    caches = {
        i: kernels.LayerCache()
        for i in range(manifest["first"], manifest["last"])
    }
    expected = []
    position = 0
    last_step = None
    for index, step in enumerate(spec["steps"]):
        at = step.get("position", position)
        if at != position:
            if last_step != (at, step["tokens"]):
                raise ValueError("Out-of-order fixture step")
            for cache in caches.values():
                cache.k = cache.k[:, :at]
                cache.v = cache.v[:, :at]
                cache.length, cache.last = at, None
        if manifest["first"] == 0:
            hidden = kernels.embed(
                step["ids"], weights["model.embed_tokens.weight"]
            )
        else:
            name = step["hidden"]
            path = input_path.parent / name
            if path.resolve().parent != input_path.parent.resolve():
                raise ValueError("Input path escapes fixture directory")
            files[name] = path
            hidden = np.fromfile(path, dtype="<f4").reshape(
                step["tokens"], config["hidden"]
            )
        # Match the browser's single-token submission policy.
        rows = []
        for t in range(step["tokens"]):
            x = hidden[t : t + 1]
            for layer, cache in caches.items():
                prefix = f"model.layers.{layer}."
                w = {
                    key: weights[prefix + key] for key in kernels.LAYER_WEIGHTS
                }
                x = kernels.block(x, w, config, at + t, cache)
            rows.append(x)
        result = np.concatenate(rows)
        logits = None
        if manifest["last"] == config["layers"]:
            logits = kernels.head(
                result,
                weights["model.norm.weight"],
                weights["lm_head.weight"],
                config["rms_epsilon"],
                rows=-1,
            )
        expected.append((result, logits))
        position = at + step["tokens"]
        last_step = (at, step["tokens"])
    report = run_page(
        "phi4_webgpu_check.mjs",
        files,
        output,
        modules=("phi4_gpu.mjs", "webgpu_kernels.mjs"),
        chrome=chrome,
        timeout=timeout,
    )
    if not report.get("ok"):
        raise RuntimeError(report)
    for index, (hidden, logits) in enumerate(expected):
        result = report["results"][index]
        actual = np.fromfile(
            output / f"hidden-{index}.bin", dtype="<f4"
        ).reshape(hidden.shape)
        result["hidden_comparison"] = compare(
            actual, hidden, atol=atol, rtol=rtol
        )
        if logits is not None:
            actual = np.fromfile(output / f"logits-{index}.bin", dtype="<f4")
            result["logits_comparison"] = compare(
                actual, logits, atol=atol, rtol=rtol
            )
            result["reference_token"] = kernels.greedy(logits)
            result["token_matches"] = (
                result["token"] == result["reference_token"]
            )
    report["tolerances"] = dict(atol=atol, rtol=rtol)
    report["passes"] = all(
        result["hidden_comparison"]["passes"]
        and result.get("logits_comparison", {"passes": True})["passes"]
        and result.get("token_matches", True)
        for result in report["results"]
    )
    (output / "comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if not report["passes"]:
        raise RuntimeError(f"Parity failed; see {output / 'comparison.json'}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chrome")
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.stage,
                args.inputs,
                args.output,
                chrome=args.chrome,
                timeout=args.timeout,
                atol=args.atol,
                rtol=args.rtol,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
