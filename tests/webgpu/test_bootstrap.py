"""Bootstrap orchestration with fake preparation and serving, no workloads."""

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from test_memory import bundle

from ygaraty.backends.webgpu.bootstrap import PROFILES, Bootstrap
from ygaraty.backends.webgpu.cli import main


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.setup = Bootstrap(
            PROFILES["qwen3.5-9b"], self.root / "input", self.root / "run"
        )
        self.output = io.StringIO()
        self.enterContext(contextlib.redirect_stdout(self.output))
        self.inspect = self.enterContext(
            patch(
                "ygaraty.backends.webgpu.bundles.inspect_checkpoint",
                return_value={
                    "compatible": True,
                    "family": "qwen35",
                    "geometry": {"layers": 32},
                },
            )
        )
        self.enterContext(patch("ygaraty.backends.webgpu.bundles.Checkpoint.validate"))
        self.builder = self.enterContext(
            patch("ygaraty.backends.webgpu.bundles.BundleBuilder")
        )

        def write_bundle(path):
            path.mkdir(parents=True, exist_ok=True)
            (path / "bundle.json").write_text(json.dumps(bundle(4)))

        self.builder.return_value.prepare.side_effect = write_bundle
        self.validate = self.enterContext(
            patch(
                "ygaraty.backends.webgpu.bundles.validate_bundle",
                return_value=SimpleNamespace(sha256="a" * 64),
            )
        )
        self.serve = self.enterContext(patch("ygaraty.backends.webgpu.cli.serve"))

    def arguments(self):
        return [
            "bootstrap",
            "--num-workers",
            "4",
            "--checkpoint",
            str(self.setup.checkpoint),
            "--workspace",
            str(self.setup.workspace),
        ]

    def test_dry_run_has_no_preparation_or_files(self):
        self.assertEqual(main([*self.arguments(), "--dry-run"]), 0)
        plan = json.loads(self.output.getvalue())
        self.assertEqual(plan["contributors"], 4)
        self.assertIn("ygaraty serve", plan["resume_command"])
        self.inspect.assert_not_called()
        self.builder.assert_not_called()
        self.serve.assert_not_called()
        self.assertFalse(self.setup.workspace.exists())

    def test_prepare_and_serve_reuses_existing_public_components(self):
        self.assertEqual(main(self.arguments()), 0)
        self.assertEqual(
            self.builder.call_args.kwargs["splits"], (0, 8, 16, 24, 32)
        )
        self.assertEqual(self.builder.call_args.kwargs["context_tokens"], 4096)
        self.builder.return_value.prepare.assert_called_once_with(
            self.setup.workspace / "model"
        )
        self.validate.assert_called_once_with(
            self.setup.workspace / "model/bundle.json"
        )
        self.serve.assert_called_once()
        keys = self.setup.workspace / "keys"
        self.assertNotEqual(
            (keys / "owner.key").read_bytes(),
            (keys / "contributor.key").read_bytes(),
        )
        self.assertEqual((keys / "owner.key").stat().st_mode & 0o777, 0o600)
        self.assertNotIn(
            (keys / "owner.key").read_text().strip(), self.output.getvalue()
        )

    def test_repeat_revalidates_without_repreparing_or_rotating_keys(self):
        self.setup.prepare()
        owner = (self.setup.workspace / "keys/owner.key").read_bytes()
        self.setup.prepare()
        self.assertEqual(self.builder.call_count, 1)
        self.assertEqual(self.validate.call_count, 2)
        self.assertEqual(
            (self.setup.workspace / "keys/owner.key").read_bytes(), owner
        )
        with self.assertRaisesRegex(ValueError, "different bootstrap"):
            replace(self.setup, context=8192).prepare()
        (self.setup.workspace / "keys/owner.key").unlink()
        with self.assertRaisesRegex(ValueError, "credentials"):
            self.setup.prepare()

    def test_wrong_family_rejected_before_creating_workspace(self):
        self.inspect.return_value["family"] = "dense"
        with self.assertRaisesRegex(ValueError, "profile"):
            self.setup.prepare()
        self.assertFalse(self.setup.workspace.exists())
        self.builder.assert_not_called()

    def test_unrelated_workspace_never_overwritten(self):
        self.setup.workspace.mkdir()
        marker = self.setup.workspace / "keep"
        marker.write_text("unchanged")
        with self.assertRaisesRegex(ValueError, "incomplete or unrelated"):
            self.setup.prepare()
        self.assertEqual(marker.read_text(), "unchanged")
        self.inspect.assert_not_called()

    def test_failed_preparation_never_serves_or_marks_complete(self):
        self.builder.return_value.prepare.side_effect = ValueError(
            "bad weights"
        )
        with self.assertRaises(SystemExit):
            main(self.arguments())
        self.assertFalse((self.setup.workspace / "bootstrap.json").exists())
        self.serve.assert_not_called()

    def test_invalid_reused_bundle_never_serves(self):
        self.setup.prepare()
        self.validate.side_effect = ValueError("changed bundle")
        with self.assertRaises(SystemExit):
            main(self.arguments())
        self.serve.assert_not_called()

    def test_prepare_only_and_argument_validation(self):
        self.assertEqual(main([*self.arguments(), "--prepare-only"]), 0)
        self.serve.assert_not_called()
        for changes in (
            {"generate": 4096},
            {"port": 0},
            {"gguf_name": "../model.gguf"},
        ):
            with self.assertRaises(ValueError):
                replace(self.setup, **changes)

    def test_replaced_valid_bundle_is_rejected(self):
        self.setup.prepare()
        self.validate.return_value = SimpleNamespace(sha256="b" * 64)
        with self.assertRaisesRegex(ValueError, "bootstrap receipt"):
            self.setup.prepare()
        self.serve.assert_not_called()

    def test_default_public_url_dry_run_never_downloads(self):
        with patch("ygaraty.backends.webgpu.downloads.fetch_workspace_bundle") as download:
            self.assertEqual(
                main(["bootstrap", "--num-workers", "2", "--dry-run"]), 0
            )
            download.assert_not_called()
        self.assertIn("2-workers/bundle.json", self.output.getvalue())

    def test_default_public_bundle_reaches_existing_local_host(self):
        bundle = self.root / "downloaded/bundle.json"
        with patch(
            "ygaraty.backends.webgpu.downloads.fetch_workspace_bundle", return_value=bundle
        ) as download:
            self.assertEqual(
                main(
                    [
                        "bootstrap",
                        "--num-workers",
                        "2",
                        "--workspace",
                        str(self.setup.workspace),
                    ]
                ),
                0,
            )
            self.assertIn("2-workers/bundle.json", download.call_args.args[0])
        self.assertEqual(self.serve.call_args.args[0].bundle, bundle.resolve())
        self.assertFalse(self.serve.call_args.args[0].share)
        self.builder.assert_not_called()

    def test_public_access_failure_does_not_create_keys_or_serve(self):
        with patch(
            "ygaraty.backends.webgpu.downloads.fetch_workspace_bundle",
            side_effect=ValueError("HTTP 403"),
        ):
            with self.assertRaises(SystemExit):
                main(
                    [
                        "bootstrap",
                        "--num-workers",
                        "2",
                        "--workspace",
                        str(self.setup.workspace),
                    ]
                )
        self.assertFalse(self.setup.workspace.exists())
        self.serve.assert_not_called()
