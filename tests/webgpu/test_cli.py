"""CLI/help/composition tests; uvicorn.run is always replaced with a mock."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fixtures import bundle

from ygaraty.backends.webgpu.cli import initialize, main


class CliTests(unittest.TestCase):
    def test_help_does_not_construct_or_launch_a_host(self):
        with patch("ygaraty.backends.webgpu.cli.serve") as run:
            for args in (["--help"], ["serve", "--help"]):
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(SystemExit) as exit:
                        main(args)
                self.assertEqual(exit.exception.code, 0)
            run.assert_not_called()

    def test_credentials_are_separate_private_files_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            with contextlib.redirect_stdout(io.StringIO()):
                initialize(path)
            owner, contributor = (
                (path / "owner.key"),
                (path / "contributor.key"),
            )
            self.assertNotEqual(owner.read_text(), contributor.read_text())
            self.assertEqual(owner.stat().st_mode & 0o777, 0o600)
            original = owner.read_bytes()
            with self.assertRaises(ValueError):
                initialize(path)
            self.assertEqual(owner.read_bytes(), original)

    def test_serve_composes_public_factory_without_running_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            data = bundle()
            data["tokenizer"] = "tokenizer.json"
            data["files"]["synthetic-weight"]["sha256"] = "a" * 64
            (path / "tokenizer.json").write_text("{}")
            (path / "bundle.json").write_text(json.dumps(data))
            with contextlib.redirect_stdout(io.StringIO()):
                initialize(path)
                with patch("uvicorn.run") as run:
                    main(
                        [
                            "serve",
                            "--bundle",
                            str(path / "bundle.json"),
                            "--model-id",
                            "fixture",
                            "--keys",
                            str(path),
                            "--state-dir",
                            str(path / "state"),
                        ]
                    )
                    self.assertEqual(run.call_count, 1)
                    self.assertEqual(run.call_args.kwargs["workers"], 1)
                    self.assertEqual(run.call_args.kwargs["host"], "127.0.0.1")
