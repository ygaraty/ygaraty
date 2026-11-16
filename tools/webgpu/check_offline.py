"""Run public metadata and fake-device checks using existing tools only."""

import os
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]


def main() -> None:
    env = os.environ | {"PYTHONPATH": str(PACKAGE)}
    commands = (
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests/webgpu",
            "-v",
        ],
        [sys.executable, "tools/webgpu/inference_schema.py", "--check"],
        ["npm", "test"],
    )
    for command in commands:
        subprocess.run(command, cwd=PACKAGE, env=env, check=True)


if __name__ == "__main__":
    main()
