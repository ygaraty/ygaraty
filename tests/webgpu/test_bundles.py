"""Local synthetic artifact preparation; numerical self-tests are mocked."""

import contextlib
import hashlib
import io
import json
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fixtures import cases, stages

from ygaraty.backends.webgpu.bundles import (
    BundleBuilder,
    Checkpoint,
    inspect_checkpoint,
    validate_bundle,
)
from ygaraty.backends.webgpu.checkpoints import gguf_remote
from ygaraty.backends.webgpu.cli import main
from ygaraty.backends.webgpu.distribution import compile_distribution


def gguf_bytes():
    def text(value):
        raw = value.encode()
        return struct.pack("<Q", len(raw)) + raw

    header = struct.pack("<4sIQQ", b"GGUF", 3, 1, 1)
    header += (
        text("general.architecture") + struct.pack("<I", 8) + text("fixture")
    )
    header += text("weight") + struct.pack("<IQQIQ", 2, 4, 2, 0, 0)
    return header + bytes((-len(header)) % 32) + bytes(32)


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.gguf = self.root / "model.gguf"
        self.gguf.write_bytes(gguf_bytes())
        self.config = self.root / "config.json"
        self.config.write_text('{"model_type":"fixture"}')
        self.tokenizer = self.root / "tokenizer.json"
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel

        Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")).save(
            str(self.tokenizer)
        )
        self.chat = self.root / "chat.json"
        self.chat.write_text(
            json.dumps(dict(first="", generation="<|im_start|>assistant\n"))
        )
        self.checkpoint = Checkpoint(
            self.gguf, self.config, self.tokenizer, self.chat
        )
        self.output = self.root / "bundle"
        self.session = self.synthetic_session(cases()[0])
        self.report = dict(
            compatible=True, geometry={"layers": 8}, family="qwen35"
        )

    def synthetic_session(self, case):
        source = stages(case)
        distribution = compile_distribution(source)
        weight = self.root / "weight.bin"
        weight.write_bytes(b"0123456789abcdef")
        return SimpleNamespace(
            files={"synthetic-weight": (weight, 0, 16)},
            hashes={
                "synthetic-weight": hashlib.sha256(
                    weight.read_bytes()
                ).hexdigest()
            },
            stages=source,
            distribution_plan=distribution,
            test_files={"w": b"abcd", "x": bytes(16), "f16": bytes(16)},
            expected=[np.array([[1.0]])],
            selftest={
                "cases": [dict(tokens=1, shape=[1, 4], file="w", x="x")],
                "f16": "f16",
            },
            eos=[2],
            chat_style=None,
            args=SimpleNamespace(
                capacity=1024, generate=32, chunk=128, kv="f16"
            ),
        )

    @contextlib.contextmanager
    def preparation(self):
        with (
            patch(
                "ygaraty.backends.webgpu.bundles.inspect_checkpoint",
                return_value=self.report,
            ),
            patch(
                "ygaraty.backends.webgpu.checkpoints.preparation.prepare_session",
                return_value=self.session,
            ) as compute,
        ):
            yield compute

    def build(self):
        with self.preparation():
            return BundleBuilder(self.checkpoint, context_tokens=1024).prepare(
                self.output
            )

    def test_prepare_reload_and_verify_detached_copies(self):
        report = self.build()
        self.assertTrue(report.artifacts_pinned)
        self.assertEqual(report, validate_bundle(report.path))
        data = json.loads(report.path.read_text())
        self.assertEqual(
            data["distribution_plan"], self.session.distribution_plan
        )
        source = self.session.files["synthetic-weight"][0]
        source.write_bytes(b"changed source")
        self.tokenizer.write_text('{"changed":true}')
        self.assertEqual(report, validate_bundle(report.path))
        piece = self.output / data["files"]["synthetic-weight"]["path"]
        self.assertEqual(piece.read_bytes(), b"0123456789abcdef")
        self.assertEqual(
            data["options"]["chat"]["generation"], "<|im_start|>assistant\n"
        )

    def test_mutated_weights_or_tokenizer_are_rejected(self):
        report = self.build()
        data = json.loads(report.path.read_text())
        piece = self.output / data["files"]["synthetic-weight"]["path"]
        piece.write_bytes(b"X" * 16)
        with self.assertRaisesRegex(ValueError, "Weight hash mismatch"):
            validate_bundle(report.path)
        piece.write_bytes(b"0123456789abcdef")
        (self.output / "tokenizer.json").write_text('{"other":true}')
        with self.assertRaisesRegex(ValueError, "Artifact hash mismatch"):
            validate_bundle(report.path)

    def test_invalid_tokenizer_fails_before_reference_computation(self):
        self.tokenizer.write_text("{}")
        with (
            self.preparation() as compute,
            self.assertRaisesRegex(ValueError, "tokenizer"),
        ):
            BundleBuilder(self.checkpoint).prepare(self.output)
        compute.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_manifest_is_published_only_after_validation(self):
        def validate_pending(path):
            self.assertFalse((self.output / "bundle.json").exists())
            self.assertEqual(path.name, ".bundle.pending.json")
            return validate_bundle(path)

        with (
            self.preparation(),
            patch(
                "ygaraty.backends.webgpu.bundles.validate_bundle",
                side_effect=validate_pending,
            ),
        ):
            report = BundleBuilder(self.checkpoint).prepare(self.output)
        self.assertEqual(report.path.name, "bundle.json")
        self.assertFalse((self.output / ".bundle.pending.json").exists())

    def test_existing_directory_is_preserved_before_computation(self):
        self.output.mkdir()
        marker = self.output / "mine"
        marker.write_text("keep")
        with self.preparation() as compute, self.assertRaises(FileExistsError):
            BundleBuilder(self.checkpoint).prepare(self.output)
        compute.assert_not_called()
        self.assertEqual(marker.read_text(), "keep")

    def test_source_hash_failure_removes_only_new_output(self):
        self.session.hashes["synthetic-weight"] = "0" * 64
        with (
            self.preparation(),
            self.assertRaisesRegex(ValueError, "source changed"),
        ):
            BundleBuilder(self.checkpoint).prepare(self.output)
        self.assertFalse(self.output.exists())
        self.assertTrue(self.gguf.exists())

    def test_invalid_splits_never_compute(self):
        for bounds in ((1, 8), (0, 9), (0, 4, 4, 8), (False, 8), [0, 8]):
            with self.subTest(bounds=bounds), self.preparation() as compute:
                with self.assertRaisesRegex(ValueError, "Splits"):
                    BundleBuilder(self.checkpoint, splits=bounds).prepare(
                        self.output
                    )
                compute.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_local_only_missing_files_and_context_limits(self):
        for source in (
            "https://example.invalid/model.gguf",
            self.root / "absent",
        ):
            with (
                self.preparation() as compute,
                self.assertRaises((ValueError, OSError)),
            ):
                BundleBuilder(replace(self.checkpoint, gguf=source)).prepare(
                    self.output
                )
            compute.assert_not_called()
        for context, generate in ((True, 1), (0, 1), (16, 32)):
            with self.assertRaises(ValueError):
                BundleBuilder(
                    self.checkpoint,
                    context_tokens=context,
                    max_output_tokens=generate,
                )
        with self.assertRaisesRegex(ValueError, "local"):
            gguf_remote.read_header("https://example.invalid/checkpoint")

    def test_unsupported_family_never_computes(self):
        self.report.update(
            compatible=False,
            findings=[{"level": "error", "message": "No stage"}],
        )
        with (
            self.preparation() as compute,
            self.assertRaisesRegex(ValueError, "Unsupported"),
        ):
            BundleBuilder(self.checkpoint).prepare(self.output)
        compute.assert_not_called()

    def test_missing_chat_template_does_not_guess(self):
        with (
            self.preparation() as compute,
            self.assertRaisesRegex(ValueError, "chat-format"),
        ):
            BundleBuilder(replace(self.checkpoint, chat_format=None)).prepare(
                self.output
            )
        compute.assert_not_called()

    def test_chat_template_comes_from_local_checkpoint(self):
        self.report["chat_template"] = (
            "{% for m in messages %}<|im_start|>{{m.role}}\n"
            "{{m.content}}<|im_end|>\n{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )
        with self.preparation():
            report = BundleBuilder(
                replace(self.checkpoint, chat_format=None)
            ).prepare(self.output)
        self.assertTrue(report.path.is_file())

    def test_validation_rejects_escape_and_bad_selftest_shapes(self):
        report = self.build()
        data = json.loads(report.path.read_text())
        data["tokenizer"] = "../tokenizer.json"
        report.path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "relative"):
            validate_bundle(report.path)
        data["tokenizer"] = "tokenizer.json"
        data["selftest"]["cases"][0]["tokens"] = 2
        report.path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "shape"):
            validate_bundle(report.path)
        data["selftest"]["cases"][0]["tokens"] = 1
        report.path.write_text(json.dumps(data))
        with (self.output / data["expected"][0]).open("wb") as archive:
            np.savez(archive, invalid=np.array([[1.0]]))
        with self.assertRaisesRegex(ValueError, "not an archive"):
            validate_bundle(report.path)

    def test_cli_prepare_and_validate_without_engine_or_network(self):
        with self.preparation(), patch("ygaraty.backends.webgpu.cli.serve") as serve:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "model",
                        "prepare",
                        "--gguf",
                        str(self.gguf),
                        "--config",
                        str(self.config),
                        "--tokenizer",
                        str(self.tokenizer),
                        "--chat-format",
                        str(self.chat),
                        "--output",
                        str(self.output),
                    ]
                )
            self.assertEqual(status, 0)
            self.assertTrue(json.loads(stdout.getvalue())["artifacts_pinned"])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        ["model", "validate", str(self.output / "bundle.json")]
                    ),
                    0,
                )
            serve.assert_not_called()

    def test_real_preparation_compiles_fresh_plans_with_synthetic_tensors(
        self,
    ):
        case = cases()[0]
        config = dict(case["config"], vocab=4)
        hidden = config["hidden"]
        tensors, entries, raw = {}, [], bytearray()

        def tensor(name, shape):
            count = 1
            for dimension in shape:
                count *= dimension
            entries.append(SimpleNamespace(name=name, data_offset=len(raw)))
            raw.extend(bytes(count * 4))
            tensors[name] = SimpleNamespace(
                name=name,
                base=None,
                first_row=0,
                shape=shape,
                kind="F32",
                row_bytes=shape[-1] * 4,
                data=SimpleNamespace(nbytes=count * 4),
            )

        tensor("token_embd.weight", (4, hidden))
        tensor("output_norm.weight", (hidden,))
        for index in range(8):
            tensor(f"blk.{index}.weight", (1, hidden))
        weight = self.root / "synthetic-tensors.bin"
        weight.write_bytes(raw)

        class Model:
            family = "qwen35"
            path = weight
            reader = SimpleNamespace(tensors=entries)

            def __getitem__(self, name):
                return tensors[
                    "token_embd.weight" if name == "output.weight" else name
                ]

            def validate_webgpu(self):
                pass

            def layer_type(self, index):
                return "full" if index % 4 == 3 else "linear"

            def layer_tensors(self, index):
                return {"weight": tensors[f"blk.{index}.weight"]}

        model = Model()
        model.config = SimpleNamespace(layers=8, manifest=lambda: config)

        def synthetic_selftest(session, _model):
            session.test_files = self.session.test_files
            session.selftest = self.session.selftest
            session.expected = self.session.expected

        with (
            patch(
                "ygaraty.backends.webgpu.bundles.inspect_checkpoint",
                return_value=self.report,
            ),
            patch(
                "ygaraty.backends.webgpu.checkpoints.families.load_model",
                return_value=model,
            ) as loader,
            patch(
                "ygaraty.backends.webgpu.checkpoints.preparation.selftest_fixture",
                side_effect=synthetic_selftest,
            ) as selftest,
        ):
            report = BundleBuilder(self.checkpoint, splits=(0, 4, 8)).prepare(
                self.output
            )
        loader.assert_called_once()
        self.assertTrue(loader.call_args.kwargs["tensor_only"])
        selftest.assert_called_once()
        data = json.loads(report.path.read_text())
        self.assertEqual(
            data["distribution_plan"], compile_distribution(data["stages"])
        )
        self.assertEqual(report.stages, 2)
        self.assertTrue(report.artifacts_pinned)

    def test_header_inspection_reads_bytes_without_loading_model(self):
        family = SimpleNamespace(
            name="fixture",
            check=lambda *_: [],
            geometry=lambda *_: {"layers": 1},
        )
        with (
            patch(
                "ygaraty.backends.webgpu.checkpoints.families.detect",
                return_value=family,
            ),
            patch("ygaraty.backends.webgpu.checkpoints.families.load_model") as loader,
        ):
            report = inspect_checkpoint(self.gguf, self.config)
            self.assertTrue(report["compatible"])
            self.assertEqual(report["architecture"], "fixture")
            loader.assert_not_called()
            self.gguf.write_bytes(self.gguf.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "range"):
                inspect_checkpoint(self.gguf, self.config)

    def test_remote_weight_bundle_validates_without_local_pieces(self):
        from ygaraty.backends.webgpu._legacy.prepared import PreparedSession
        from ygaraty.backends.webgpu.downloads import fetch_bundle

        report = self.build()
        original = json.loads(report.path.read_text())
        published = dict(
            original, storage={"backend": "spaces", "prefix": "model/v1"}
        )
        url = "https://models.example/model/v1/bundle.json"
        fetched = []

        def opener(address):
            fetched.append(address)
            if address == url:
                return io.BytesIO(json.dumps(published).encode())
            relative = address.removeprefix("https://models.example/model/v1/")
            return io.BytesIO((self.output / relative).read_bytes())

        downloaded = fetch_bundle(
            url, self.root / "remote", opener=opener, weights=False
        )
        piece = original["files"]["synthetic-weight"]
        self.assertNotIn(
            "https://models.example/model/v1/" + piece["path"], fetched
        )
        result = validate_bundle(downloaded)
        self.assertEqual(result.weight_bytes, piece["length"])
        # The session hands contributors the public piece URL.
        session = PreparedSession.__new__(PreparedSession)
        session.storage = "https://models.example/model/v1/"
        session.files = {"synthetic-weight": (None, 0, piece["length"])}
        session.hashes = {"synthetic-weight": piece["sha256"]}
        session.remote_paths = {"synthetic-weight": piece["path"]}
        self.assertEqual(
            session.download("synthetic-weight"),
            {
                "url": "https://models.example/model/v1/" + piece["path"],
                "bytes": piece["length"],
                "sha256": piece["sha256"],
            },
        )
        with self.assertRaisesRegex(ValueError, "No Spaces piece"):
            session.download("missing")
        data = json.loads(downloaded.read_text())
        for storage, error in (
            ({"backend": "url", "base_url": "http://x/"}, "public HTTPS"),
            ({"backend": "url", "base_url": "https://x/a"}, "public HTTPS"),
            ({"backend": "spaces"}, "Unsupported"),
        ):
            downloaded.write_text(json.dumps(dict(data, storage=storage)))
            with self.assertRaisesRegex(ValueError, error):
                validate_bundle(downloaded)
        data["files"]["synthetic-weight"]["path"] = "../escape.bin"
        downloaded.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "Unsafe remote weight path"):
            validate_bundle(downloaded)

    def test_public_download_validates_real_synthetic_bundle_and_fresh_plan(
        self,
    ):
        from ygaraty.backends.webgpu.downloads import fetch_bundle

        report = self.build()
        original = json.loads(report.path.read_text())
        published = dict(
            original, storage={"backend": "spaces", "prefix": "model/v1"}
        )
        url = "https://models.example/model/v1/bundle.json"

        def opener(address):
            if address == url:
                return io.BytesIO(json.dumps(published).encode())
            relative = address.removeprefix("https://models.example/model/v1/")
            return io.BytesIO((self.output / relative).read_bytes())

        downloaded = fetch_bundle(url, self.root / "download", opener=opener)
        self.assertTrue(validate_bundle(downloaded).artifacts_pinned)
        self.assertEqual(
            json.loads(downloaded.read_text())["distribution_plan"],
            original["distribution_plan"],
        )
