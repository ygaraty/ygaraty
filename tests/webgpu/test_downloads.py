"""Public bundle fetch/cache tests with in-memory responses, never sockets."""

import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

from ygaraty.backends.webgpu.downloads import fetch_bundle, open_public, public_url

URL = "https://models.example/model/bundle.json"


def digest(value):
    return hashlib.sha256(value).hexdigest()


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "model"
        self.artifacts = {
            "pieces/weight.bin": b"weights",
            "tokenizer.json": b"{}",
            "selftest/x.bin": b"x",
            "selftest/y.npy": b"reference",
        }
        self.data = {
            "format": "ygaraty-webgpu-v1",
            "storage": {"backend": "spaces"},
            "files": {
                "weight": {
                    "path": "pieces/weight.bin",
                    "offset": 0,
                    "length": 7,
                    "sha256": digest(b"weights"),
                    "object_key": "model/pieces/weight.bin",
                }
            },
            "tokenizer": "tokenizer.json",
            "test_files": {"x": "selftest/x.bin"},
            "expected": ["selftest/y.npy"],
            "artifact_digests": {
                p: digest(v)
                for p, v in self.artifacts.items()
                if not p.startswith("pieces/")
            },
            "distribution_plan": {"unchanged": True},
        }
        self.calls = []

        def validate(path):
            return SimpleNamespace(path=path, sha256=digest(path.read_bytes()))

        self.validate = self.enterContext(
            patch(
                "ygaraty.backends.webgpu.downloads.validate_bundle",
                side_effect=validate,
            )
        )

    def opener(self, url):
        self.calls.append(url)
        if url == URL:
            return io.BytesIO(json.dumps(self.data).encode())
        return io.BytesIO(
            self.artifacts[url.removeprefix("https://models.example/model/")]
        )

    def test_fetch_converts_only_storage_and_reuses_cache_without_network(
        self,
    ):
        result = fetch_bundle(URL, self.output, opener=self.opener)
        local = json.loads(result.read_text())
        self.assertNotIn("storage", local)
        self.assertNotIn("object_key", local["files"]["weight"])
        self.assertEqual(
            local["distribution_plan"], self.data["distribution_plan"]
        )
        self.assertEqual(len(self.calls), 5)
        with patch(
            "ygaraty.backends.webgpu.downloads.open_public",
            side_effect=AssertionError("network"),
        ):
            self.assertEqual(fetch_bundle(URL, self.output), result)
        with self.assertRaisesRegex(ValueError, "different public URL"):
            fetch_bundle(URL.replace("model/", "another/"), self.output)

    def test_remote_weights_fetch_only_metadata_and_keep_public_base(self):
        result = fetch_bundle(
            URL, self.output, opener=self.opener, weights=False
        )
        local = json.loads(result.read_text())
        self.assertEqual(
            local["storage"],
            {"backend": "url", "base_url": "https://models.example/model/"},
        )
        self.assertNotIn("object_key", local["files"]["weight"])
        self.assertNotIn(
            "https://models.example/model/pieces/weight.bin", self.calls
        )
        self.assertEqual(len(self.calls), 4)
        self.assertFalse((self.output / "pieces").exists())
        self.assertTrue((self.output / "tokenizer.json").is_file())
        with patch(
            "ygaraty.backends.webgpu.downloads.open_public",
            side_effect=AssertionError("network"),
        ):
            self.assertEqual(
                fetch_bundle(URL, self.output, weights=False), result
            )
        with self.assertRaisesRegex(ValueError, "without local weights"):
            fetch_bundle(URL, self.output)

    def test_local_weight_cache_is_not_reused_for_remote_delivery(self):
        fetch_bundle(URL, self.output, opener=self.opener)
        with self.assertRaisesRegex(ValueError, "with local weights"):
            fetch_bundle(URL, self.output, weights=False)

    def test_memory_admission_runs_before_downloads_and_on_cache_reuse(self):
        def reject(data):
            raise ValueError("Insufficient configured memory")

        with self.assertRaisesRegex(ValueError, "Insufficient"):
            fetch_bundle(
                URL, self.output, opener=self.opener, on_manifest=reject
            )
        self.assertEqual(self.calls, [URL])
        self.assertFalse(self.output.exists())
        fetch_bundle(URL, self.output, opener=self.opener)
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            fetch_bundle(
                URL, self.output, opener=self.opener, on_manifest=reject
            )
        self.assertEqual(self.calls, [])

    def test_wrong_stage_count_is_rejected_before_artifact_downloads(self):
        self.data["stages"] = [{"names": ["weight"]}]
        with self.assertRaisesRegex(ValueError, "Expected 2 contributors"):
            fetch_bundle(
                URL,
                self.output,
                opener=self.opener,
                contributors=2,
                memory_gb=3,
            )
        self.assertEqual(self.calls, [URL])
        self.assertFalse(self.output.exists())

    def test_over_budget_stage_is_rejected_before_artifact_downloads(self):
        self.data["stages"] = [{"names": ["weight"]}, {"names": ["weight"]}]
        self.data["files"]["weight"]["length"] = 3 * 1024**3 + 1
        with self.assertRaisesRegex(ValueError, "exceed the 3 GiB"):
            fetch_bundle(
                URL,
                self.output,
                opener=self.opener,
                contributors=2,
                memory_gb=3,
            )
        self.assertEqual(self.calls, [URL])
        self.assertFalse(self.output.exists())

    def test_cached_bundle_rechecks_contributor_count(self):
        self.data["stages"] = [{"names": ["weight"]}, {"names": ["weight"]}]
        path = fetch_bundle(
            URL, self.output, opener=self.opener, contributors=2, memory_gb=3
        )
        with self.assertRaisesRegex(ValueError, "Expected 1 contributors"):
            fetch_bundle(URL, self.output, contributors=1, memory_gb=3)
        self.assertTrue(path.exists())

    def test_legacy_metadata_is_pinned_on_first_download(self):
        del self.data["artifact_digests"]
        local = json.loads(
            fetch_bundle(URL, self.output, opener=self.opener).read_text()
        )
        self.assertEqual(
            local["artifact_digests"]["tokenizer.json"], digest(b"{}")
        )

    def test_corrupt_truncated_or_oversize_download_never_publishes(self):
        for content in (b"changed", b"short", b"too long"):
            with self.subTest(content=content):
                self.artifacts["pieces/weight.bin"] = content
                with self.assertRaises(ValueError):
                    fetch_bundle(URL, self.output, opener=self.opener)
                self.assertFalse(self.output.exists())
        self.validate.assert_not_called()

    def test_declared_metadata_digest_is_checked(self):
        self.artifacts["tokenizer.json"] = b'{"different":true}'
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            fetch_bundle(URL, self.output, opener=self.opener)
        self.assertFalse(self.output.exists())

    def test_untrusted_paths_fail_before_any_artifact_download(self):
        for path in (
            "../escape",
            "/absolute",
            "https://elsewhere/file",
            "x/%2e%2e/a",
            "bundle.json",
        ):
            with self.subTest(path=path):
                self.calls.clear()
                self.data["files"]["weight"]["path"] = path
                with self.assertRaises(ValueError):
                    fetch_bundle(URL, self.output, opener=self.opener)
                self.assertEqual(self.calls, [URL])
                self.assertFalse(self.output.exists())

    def test_http_403_explains_anonymous_access(self):
        with patch("ygaraty.backends.webgpu.downloads.build_opener") as opener:
            opener.return_value.open.side_effect = HTTPError(
                URL, 403, "Forbidden", {}, None
            )
            with self.assertRaisesRegex(
                ValueError, "HTTP 403.*anonymous reads"
            ):
                open_public(URL)

    def test_url_credentials_and_insecure_transport_rejected(self):
        for url in (
            "http://models.example/a",
            "https://user:pass@models.example/a",
            "https://models.example/a?token=secret",
        ):
            with self.assertRaises(ValueError):
                public_url(url)

    def test_failed_validation_and_existing_files_preserved(self):
        self.validate.side_effect = ValueError("bad plan")
        with self.assertRaisesRegex(ValueError, "bad plan"):
            fetch_bundle(URL, self.output, opener=self.opener)
        self.assertFalse(self.output.exists())
        self.output.mkdir()
        (self.output / "keep").write_text("existing")
        with self.assertRaisesRegex(ValueError, "Incomplete or unrelated"):
            fetch_bundle(URL, self.output, opener=self.opener)
        self.assertEqual((self.output / "keep").read_text(), "existing")

    def test_incomplete_pins_and_budget_fail_before_weights(self):
        del self.data["artifact_digests"]["tokenizer.json"]
        with self.assertRaisesRegex(ValueError, "Incomplete artifact"):
            fetch_bundle(URL, self.output, opener=self.opener)
        self.assertEqual(self.calls, [URL])
        del self.data["artifact_digests"]
        self.data["files"]["weight"]["length"] = 65 << 30
        self.calls.clear()
        with self.assertRaisesRegex(ValueError, "download budget"):
            fetch_bundle(URL, self.output, opener=self.opener)
        self.assertEqual(self.calls, [URL])
