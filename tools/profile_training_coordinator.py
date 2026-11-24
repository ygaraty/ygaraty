"""Measure coordinator phases without changing the training protocol.

Pass training_e2e arguments after --profile-output. Inclusive scopes overlap;
use unprofiled trials for speedups. --preparation-only separately CPU-profiles
four assignment responses, avoiding concurrent Python profiler conflicts.
"""

import argparse
import cProfile
import functools
import json
import pstats
import runpy
import sys
import tempfile
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch


def preparation():
    from starlette.responses import JSONResponse
    from ygaraty.training.datasets import load_dataset
    from ygaraty.training.session import TrainingConfig, TrainingSession

    data = load_dataset(
        "fashion-mnist", Path.home() / ".cache/ygaraty/datasets"
    )
    with tempfile.TemporaryDirectory() as directory:
        session = TrainingSession(
            TrainingConfig(
                4,
                optimizer="muon",
                batch_size=16,
                local_steps=8,
                rounds=3,
                eval_examples=16,
            ),
            data,
            directory,
        )
        clients = [
            session.join(session.join_token, f"profile-client-{i:016}")
            for i in range(4)
        ]
        profiler = cProfile.Profile()
        profiler.enable()
        for client in clients:
            args = (client["worker"], client["token"])
            if hasattr(session, "assignment_bytes"):
                session.assignment_bytes(*args)
            else:
                JSONResponse(session.assignment(*args))
        profiler.disable()
    stats = pstats.Stats(profiler)
    rows = []
    for (file, line, name), (_, calls, own, cumulative, _) in sorted(
        stats.stats.items(), key=lambda item: item[1][3], reverse=True
    )[:40]:
        rows.append(
            dict(
                file=Path(file).name,
                line=line,
                function=name,
                calls=calls,
                self_seconds=own,
                cumulative_seconds=cumulative,
            )
        )
    return dict(
        kind="sequential-assignment-cpu-profile",
        workers=4,
        batch_size=16,
        local_steps=8,
        channels=[8, 16, 16],
        cpu_hotspots=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-output", type=Path, required=True)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--preparation-only", action="store_true")
    args, rest = parser.parse_known_args()
    from ygaraty.training.session import TrainingSession, implementation_hash
    from ygaraty.training.convnet_webgpu import ConvNetRecipe
    from ygaraty.training.datasets import ImageDataset
    from ygaraty.training import recovery
    import fastapi.routing
    from starlette.responses import JSONResponse

    loaded_root = (
        Path(sys.modules[TrainingSession.__module__].__file__)
        .resolve()
        .parents[2]
    )
    if loaded_root != args.source.resolve():
        parser.error(
            "--source must match imported Ygaraty; set PYTHONPATH to that root"
        )
    source_hash = implementation_hash()
    if args.preparation_only:
        if rest:
            parser.error(
                "--preparation-only uses its recorded fixed configuration"
            )
        report = preparation()
        report["source_sha256"] = source_hash
        args.profile_output.parent.mkdir(parents=True, exist_ok=True)
        args.profile_output.write_text(json.dumps(report, indent=2) + "\n")
        return
    totals, lock = {}, threading.Lock()

    def record(name, elapsed):
        with lock:
            row = totals.setdefault(
                name, dict(calls=0, seconds=0, max_seconds=0)
            )
            row["calls"] += 1
            row["seconds"] += elapsed
            row["max_seconds"] = max(row["max_seconds"], elapsed)

    with ExitStack() as stack:

        def measure(owner, name, label):
            original = getattr(owner, name)

            @functools.wraps(original)
            def wrapped(*a, **kw):
                start = time.perf_counter()
                try:
                    return original(*a, **kw)
                finally:
                    record(label, time.perf_counter() - start)

            stack.enter_context(patch.object(owner, name, wrapped))

        for name in (
            "assignment",
            "assignment_bytes",
            "_assignment",
            "submit",
            "_commit",
            "_save",
            "status",
        ):
            if hasattr(TrainingSession, name):
                measure(TrainingSession, name, "session." + name)
        for name in ("compile", "round_bundle", "predict"):
            measure(ConvNetRecipe, name, "recipe." + name)
        measure(ImageDataset, "evaluate", "evaluation")
        for name in ("save", "load"):
            measure(recovery, name, "checkpoint." + name)
        measure(JSONResponse, "render", "http.json_render")
        measure(json.JSONEncoder, "encode", "json.encode")
        serialize = fastapi.routing.serialize_response

        async def encoded(*a, **kw):
            start = time.perf_counter()
            try:
                return await serialize(*a, **kw)
            finally:
                record("http.fastapi_encode", time.perf_counter() - start)

        stack.enter_context(
            patch.object(fastapi.routing, "serialize_response", encoded)
        )
        script = args.source / "tools/training_e2e.py"
        stack.enter_context(patch.object(sys, "argv", [str(script), *rest]))
        start = time.perf_counter()
        succeeded = False
        try:
            try:
                runpy.run_path(str(script), run_name="__main__")
                succeeded = True
            except SystemExit as error:
                succeeded = error.code in (None, 0)
                if not succeeded:
                    raise
        finally:
            report = dict(
                kind="coordinator-phase-timers",
                succeeded=succeeded,
                source_sha256=source_hash,
                total_seconds=time.perf_counter() - start,
                scopes=totals,
                note="Inclusive scopes overlap across threads. assignment_bytes includes waiting; _assignment measures preparation. Initial evaluation is included here. Profiled wall time is diagnostic; use unprofiled paired trials for speedups.",
            )
            args.profile_output.parent.mkdir(parents=True, exist_ok=True)
            args.profile_output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
