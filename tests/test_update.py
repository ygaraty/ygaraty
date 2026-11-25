"""Managed updates preserve user data and do not activate failed environments."""
import fcntl
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from ygaraty.update import installation_root, update


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "install with spaces"
        (self.root / "venv/bin").mkdir(parents=True)
        (self.root / "venv/bin/python").write_text("original interpreter")
        (self.root / ".installed").write_text("old\n")
        self.data = self.root / "workspaces/session"
        self.data.mkdir(parents=True)
        for name in ("model.bin", "owner.key", "checkpoint.json"):
            (self.data / name).write_bytes(b"preserve exactly")
        self.calls = []
        self.failure = None

    def run_command(self, command, **kwargs):
        self.calls.append(command)
        if self.failure and self.failure(command):
            raise subprocess.CalledProcessError(1, command)
        if command[1:3] == ["-m", "venv"]:
            directory = Path(command[-1]) / "bin"
            directory.mkdir(parents=True)
            (directory / "python").write_text("new interpreter")
            (directory / "ygaraty").write_text("new command")

    def check_data(self):
        for p in self.data.iterdir():
            self.assertEqual(p.read_bytes(), b"preserve exactly")

    def test_success_and_repeated_update(self):
        update(self.root, run=self.run_command)
        self.assertTrue((self.root / "venv").is_symlink())
        first = (self.root / "venv").resolve()
        self.assertEqual((self.root / ".installed").read_text(), "main\n")
        update(self.root, "feature/test", run=self.run_command)
        self.assertNotEqual(first, (self.root / "venv").resolve())
        self.assertTrue(first.exists())
        self.assertEqual((self.root / ".installed").read_text(), "feature/test\n")
        self.check_data()

    def test_failures_leave_original_active(self):
        for predicate in (
            lambda c: "fetch" in c,
            lambda c: "install" in c,
            lambda c: "check" in c,
            lambda c: c[-2:] == ["update", "--help"],
        ):
            self.failure = predicate
            with self.assertRaises(subprocess.CalledProcessError):
                update(self.root, run=self.run_command)
            self.assertFalse((self.root / "venv").is_symlink())
            self.assertEqual((self.root / "venv/bin/python").read_text(), "original interpreter")
            self.assertEqual((self.root / ".installed").read_text(), "old\n")
            self.assertEqual(list((self.root / ".versions").iterdir()), [])
            self.check_data()

    def test_failed_activation_restores_directory(self):
        with patch("ygaraty.update.os.replace", side_effect=OSError("disk error")):
            with self.assertRaises(OSError):
                update(self.root, run=self.run_command)
        self.assertEqual((self.root / "venv/bin/python").read_text(), "original interpreter")
        self.check_data()

    def test_marker_failure_restores_previous_install(self):
        replace = os.replace
        def fail_marker(source, target):
            if Path(target).name == ".installed":
                raise OSError("marker write failed")
            return replace(source, target)
        for already_updated in (False, True):
            if already_updated:
                update(self.root, run=self.run_command)
            old = (self.root / "venv").resolve()
            with patch("ygaraty.update.os.replace", side_effect=fail_marker):
                with self.assertRaises(OSError):
                    update(self.root, run=self.run_command)
            self.assertEqual((self.root / "venv").resolve(), old)
            self.check_data()

    def test_concurrent_update_rejected(self):
        with (self.root / ".update.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(ValueError, "Another"):
                update(self.root, run=self.run_command)
        self.assertEqual(self.calls, [])

    def test_root_detection_before_and_after_update(self):
        with patch("sys.prefix", str(self.root / "venv")):
            self.assertEqual(installation_root(), self.root.resolve())
        update(self.root, run=self.run_command)
        with patch("sys.prefix", str((self.root / "venv").resolve())):
            self.assertEqual(installation_root(), self.root.resolve())
        with patch("sys.prefix", str(self.root / "unrelated")):
            with self.assertRaisesRegex(ValueError, "not an installer-managed"):
                installation_root()

    def test_workspace_symlink_untouched(self):
        target = Path(self.tmp.name) / "external"
        self.data.rename(target)
        self.data.symlink_to(target, target_is_directory=True)
        update(self.root, run=self.run_command)
        self.assertTrue(self.data.is_symlink())
        self.check_data()


if __name__ == "__main__":
    unittest.main()
