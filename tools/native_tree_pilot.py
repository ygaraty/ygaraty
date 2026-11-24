"""Same-host native pilot; not evidence of physical GPU scaling."""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ygaraty.training.lan import PeerClient, initialize
from ygaraty.training.lan import main as lan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        nargs="+",
        type=int,
        choices=(1, 2, 4, 10),
        default=[1, 2, 4, 10],
    )
    parser.add_argument("--cpu-test", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--global-batch", type=int, default=20)
    parser.add_argument("--channels", nargs=3, type=int, default=[2, 4, 4])
    parser.add_argument(
        "--dataset",
        default="fixture",
        choices=("fixture", "mnist", "fashion-mnist"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path.home() / ".cache/ygaraty/datasets",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for workers in args.workers:
        processes, logs = [], []
        with tempfile.TemporaryDirectory(
            prefix="ygaraty-native-tree-"
        ) as temp:
            root = Path(temp)
            sockets = [socket.socket() for _ in range(workers)]
            try:
                for sock in sockets:
                    sock.bind(("127.0.0.1", 0))
                ports = [sock.getsockname()[1] for sock in sockets]
            finally:
                for sock in sockets:
                    sock.close()
            config = initialize(
                root / "session",
                [f"http://127.0.0.1:{p}" for p in ports],
                workers,
                "float64" if args.cpu_test else "webgpu",
            )
            client = PeerClient(config)
            try:
                for rank, port in enumerate(ports):
                    log = (root / f"worker-{rank}.log").open("w")
                    logs.append(log)
                    processes.append(
                        subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "ygaraty",
                                "train",
                                "lan",
                                "worker",
                                "--config",
                                str(root / "session" / f"worker-{rank}.json"),
                                "--port",
                                str(port),
                            ],
                            stdout=log,
                            stderr=log,
                            start_new_session=True,
                        )
                    )
                deadline = time.monotonic() + 60
                for rank, process in enumerate(processes):
                    while True:
                        if (
                            process.poll() is not None
                            or time.monotonic() > deadline
                        ):
                            raise RuntimeError(
                                f"Worker {rank} startup failed: "
                                + (root / f"worker-{rank}.log").read_text()
                            )
                        try:
                            client.request(rank, "/status", {})
                            break
                        except OSError:
                            time.sleep(0.1)
                lan(
                    [
                        "run",
                        "--config",
                        str(root / "session/coordinator.json"),
                        "--num-workers",
                        str(workers),
                        "--dataset",
                        args.dataset,
                        "--data-dir",
                        str(args.data_dir),
                        "--output",
                        str(args.output_dir / f"workers-{workers}.json"),
                        "--steps",
                        str(args.steps),
                        "--global-batch",
                        str(args.global_batch),
                        "--channels",
                        *(str(n) for n in args.channels),
                        "--eval-examples",
                        "16",
                        "--eval-interval",
                        "1",
                        "--verify-steps",
                        str(args.steps),
                        "--timeout",
                        "300",
                    ]
                )
            finally:
                for process in processes:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                for process in processes:
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=5)
                for log in logs:
                    log.close()
                ports_closed = True
                for port in ports:
                    with socket.socket() as check:
                        check.settimeout(1)
                        ports_closed &= (
                            check.connect_ex(("127.0.0.1", port)) != 0
                        )
                (args.output_dir / f"cleanup-{workers}.json").write_text(
                    json.dumps(
                        {
                            "all_supervisors_stopped": all(
                                p.poll() is not None for p in processes
                            ),
                            "all_worker_ports_closed": ports_closed,
                            "worker_processes": workers,
                            "physical_gpus": 0 if args.cpu_test else 1,
                            "machines": 1,
                            "external_spend_usd": 0,
                        },
                        indent=2,
                    )
                    + "\n"
                )
                if not ports_closed:
                    raise RuntimeError(
                        "A pilot worker port is still listening"
                    )


if __name__ == "__main__":
    main()
