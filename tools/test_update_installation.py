"""Exercise a real managed update against the checked-out Git revision in CI."""
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ygaraty import update as updater  # noqa: E402


def main():
    installation = Path(sys.argv[1]).resolve()
    source = ROOT
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    workspace = installation / "workspaces" / "update-check"
    workspace.mkdir(parents=True)
    expected = {"model.bin": b"model fixture", "owner.key": b"test-only-key", "checkpoint.json": b'{"step": 7}'}
    for name, data in expected.items():
        (workspace / name).write_bytes(data)
    updater.REPOSITORY = str(source)
    updater.update(installation, revision)
    python = installation / "venv/bin/python"
    result = subprocess.check_output(
        [str(python), "-I", "-c", "import json; from ygaraty.update import installation_root; print(json.dumps(str(installation_root())))"],
        text=True,
    )
    assert Path(json.loads(result)) == installation
    subprocess.run([str(installation / "venv/bin/ygaraty"), "update", "--help"], check=True)
    assert (installation / ".installed").read_text().strip() == revision
    for name, data in expected.items():
        assert (workspace / name).read_bytes() == data
    assert installation.joinpath("venv").is_symlink()
    print("Real environment update preserved models, keys and checkpoint fixtures.")


if __name__ == "__main__":
    main()
