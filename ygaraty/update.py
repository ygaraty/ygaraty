"""Update installer-managed environments without modifying user workspaces."""

import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid

REPOSITORY = "https://github.com/ygaraty/ygaraty.git"


def installation_root():
    prefix = Path(sys.prefix).resolve()
    root = prefix.parent
    if root.name == ".versions":
        root = root.parent
    if not (root / ".installed").is_file() or (root / "venv").resolve() != prefix:
        raise ValueError(
            "This is not an installer-managed environment. For a pip install, "
            "use that environment's python -m pip install --upgrade with your "
            "chosen Ygaraty source; source checkouts should use git pull."
        )
    return root


def update(root, ref="main", run=subprocess.run):
    root = Path(root).expanduser().resolve()
    active = root / "venv"
    marker = root / ".installed"
    if not marker.is_file() or not (active / "bin/python").is_file():
        raise ValueError("No complete Ygaraty installation at " + str(root))
    if not ref or ref.startswith("-") or any(c.isspace() for c in ref):
        raise ValueError("--ref must be a Git branch, tag or commit")
    with (root / ".update.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another Ygaraty update is running") from error
        versions = root / ".versions"
        versions.mkdir(exist_ok=True)
        candidate = versions / uuid.uuid4().hex
        link = root / (".venv-next-" + uuid.uuid4().hex)
        marker_next = root / (".installed-next-" + uuid.uuid4().hex)
        activated = False
        try:
            with tempfile.TemporaryDirectory(prefix="ygaraty-update-") as directory:
                source = Path(directory) / "source"
                run(["git", "init", "-q", str(source)], check=True)
                run(["git", "-C", str(source), "remote", "add", "origin", REPOSITORY], check=True)
                run(["git", "-C", str(source), "fetch", "--depth", "1", "origin", ref], check=True)
                run(["git", "-C", str(source), "checkout", "-q", "--detach", "FETCH_HEAD"], check=True)
                run([str(active / "bin/python"), "-m", "venv", str(candidate)], check=True)
                python = str(candidate / "bin/python")
                # Never relocate this environment: console-script shebangs are absolute.
                run([python, "-m", "pip", "install", str(source) + "[server,prepare]"], check=True)
                run([python, "-m", "pip", "check"], check=True)
                for args in (["--help"], ["launch", "--help"], ["update", "--help"]):
                    run([str(candidate / "bin/ygaraty"), *args], check=True, stdout=subprocess.DEVNULL)
            link.symlink_to(candidate, target_is_directory=True)
            marker_next.write_text(ref + "\n")
            # Existing installer versions used a directory; retain it for recovery.
            backup = None
            previous_link = active.readlink() if active.is_symlink() else None
            if not active.is_symlink():
                backup = versions / ("previous-" + uuid.uuid4().hex)
                active.rename(backup)
            try:
                os.replace(link, active)
            except OSError:
                if backup is not None:
                    backup.rename(active)
                raise
            try:
                os.replace(marker_next, marker)
            except OSError:
                if backup is not None:
                    active.unlink()
                    backup.rename(active)
                else:
                    link.symlink_to(previous_link, target_is_directory=True)
                    os.replace(link, active)
                raise
            activated = True
        finally:
            link.unlink(missing_ok=True)
            marker_next.unlink(missing_ok=True)
            if not activated and candidate.exists():
                shutil.rmtree(candidate)
    print(f"Ygaraty updated to {ref}: {active / 'bin/ygaraty'}")
    print("Workspaces, model caches, keys and checkpoints were not modified.")
    print("Restart your server with the same workspace, then reconnect contributors.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Update Ygaraty; preserve model and session files. Stop running servers first.")
    parser.add_argument("--ref", default="main", help="Public Git branch, tag or commit (default: main)")
    parser.add_argument("--install-dir", type=Path, help="Installer directory (normally detected automatically)")
    args = parser.parse_args(argv)
    try:
        update(args.install_dir or installation_root(), args.ref)
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"ygaraty update: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
