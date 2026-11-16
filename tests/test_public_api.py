"""Stable public imports and numerical behavior, without private services."""

import importlib
import importlib.abc
import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np


class PublicApiTests(unittest.TestCase):
    def test_public_exports_are_implemented_in_the_package(self):
        from pathlib import Path

        import ygaraty
        from ygaraty import Tape, Trainer
        from ygaraty.compiler.tape import Tape as compiler_tape
        from ygaraty.tape import Tape as root_tape
        from ygaraty.training.trainer import Trainer as trainer

        self.assertIs(Tape, root_tape)
        self.assertIs(Tape, compiler_tape)
        self.assertIs(Trainer, trainer)
        package_root = Path(ygaraty.__file__).parent
        for name in (
            "compiler.optimizer",
            "analysis.metrics",
            "runtime.local",
            "training.coordinator",
            "backends.webgpu.factory",
        ):
            module = importlib.import_module("ygaraty." + name)
            self.assertEqual(
                Path(module.__file__).resolve(),
                (package_root / (name.replace(".", "/") + ".py")).resolve(),
            )

    def test_cli_dispatch_and_help(self):
        from ygaraty.cli import main

        for command, target in (
            ("train", "ygaraty.training.__main__.main"),
            ("compile", "ygaraty.compiler.export.main"),
            ("launch", "ygaraty.backends.webgpu.cli.main"),
        ):
            with patch(target, return_value=0) as run:
                self.assertEqual(main([command, "--help"]), 0)
                run.assert_called_once_with(
                    [command, "--help"] if command == "launch" else ["--help"]
                )
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(["--help"]), 0)
        for command in ("launch", "train", "compile"):
            self.assertIn(command, output.getvalue())

    def test_reference_fingerprint_is_package_relative(self):
        from ygaraty.parity.cases import reference_fingerprint

        fingerprints = reference_fingerprint("transformer")
        self.assertEqual(set(fingerprints), {"reference/transformer.py"})
        self.assertTrue(
            all(len(value) == 64 for value in fingerprints.values())
        )

    def test_tape_backward_matches_finite_difference(self):
        from ygaraty import Tape
        from ygaraty.runtime import evaluate

        tape = Tape()
        x = tape.input("x", [[2.0, -1.0]])
        w = tape.parameter("w", [[0.3], [0.7]])
        y = tape.matmul(x, w)
        loss = tape.sum(tape.mul(y, y))
        tape.backward(loss)
        values = evaluate(tape)
        analytic = values[tape.gradients[w]]
        numeric = np.zeros((2, 1))
        for i in range(2):
            a, b = tape.values[w].copy(), tape.values[w].copy()
            a[i, 0] += 1e-6
            b[i, 0] -= 1e-6
            numeric[i, 0] = (
                ((tape.values[x] @ a) ** 2).sum()
                - ((tape.values[x] @ b) ** 2).sum()
            ) / 2e-6
        np.testing.assert_allclose(analytic, numeric, rtol=1e-7, atol=1e-9)

    def test_public_training_imports_without_cloud(self):
        class BlockPrivate(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] in {
                    "ygaraty_cloud",
                    "django",
                    "vllm",
                }:
                    raise AssertionError("Private import: " + fullname)

        blocker = BlockPrivate()
        sys.meta_path.insert(0, blocker)
        try:
            from ygaraty.training import Coordinator, DiLoCoConfig, Trainer
            from ygaraty.training.mlp import ImageBatch, MLPRecipe

            recipe = MLPRecipe(
                dict(inputs=3, hidden=[4], classes=2), learning_rate=0.01
            )
            c = Coordinator(
                recipe.initial_weights(),
                DiLoCoConfig(
                    local_steps=2,
                    outer_learning_rate=1,
                    momentum=0,
                    nesterov=False,
                ),
            )
            c.begin_round(["a", "b"])
            batch = ImageBatch(
                np.array([[255, 0, 0], [0, 255, 0]]), np.array([0, 1])
            )
            for worker in ("a", "b"):
                result = Trainer(worker, recipe).run_round(
                    c.assignment(worker), [batch, batch]
                )
                self.assertLess(result.losses[-1], result.losses[0])
                c.submit(result)
            self.assertEqual(c.commit_round()["workers"], ["a", "b"])
            self.assertEqual(c.version, 1)
        finally:
            sys.meta_path.remove(blocker)
