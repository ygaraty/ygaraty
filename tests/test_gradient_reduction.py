import unittest

import numpy as np

from ygaraty.training.convnet_webgpu import ConvBatch, ConvNetRecipe
from ygaraty.training.coordinator import Coordinator, DiLoCoConfig
from ygaraty.training.gradient_reduction import (
    GradientPart,
    SharedAdam,
    compute_gradients,
    reduce_gradients,
)
from ygaraty.training.trainer import Trainer


class GradientReductionTests(unittest.TestCase):
    def setUp(self):
        self.recipe = ConvNetRecipe(
            dict(
                in_channels=1,
                height=4,
                width=4,
                channels=[2, 2, 2],
                classes=2,
                residual=True,
            ),
            max_batch=8,
        )
        self.weights = self.recipe.initial_weights(3)
        self.batch = ConvBatch(
            np.random.default_rng(8).integers(
                0, 256, (7, 1, 4, 4), dtype=np.uint8
            ),
            np.arange(7) % 2,
        )

    def test_unequal_partitions_match_full_gradient_and_shared_adam(self):
        a, b = (
            SharedAdam(self.recipe, self.weights),
            SharedAdam(self.recipe, self.weights),
        )
        for _ in range(4):
            whole = compute_gradients(self.recipe, a.weights, self.batch)
            parts = [
                compute_gradients(
                    self.recipe,
                    b.weights,
                    ConvBatch(self.batch.images[ids], self.batch.labels[ids]),
                )
                for ids in np.array_split(np.arange(7), 3)
            ]
            reduced = reduce_gradients(parts)
            self.assertEqual(reduced.samples, 7)
            self.assertAlmostEqual(whole.loss, reduced.loss, places=13)
            for name in whole.gradients:
                np.testing.assert_allclose(
                    whole.gradients[name],
                    reduced.gradients[name],
                    atol=1e-12,
                    rtol=1e-11,
                )
            a.apply(whole.gradients)
            b.apply(reduced.gradients)
            for name in a.weights:
                np.testing.assert_allclose(
                    a.weights[name], b.weights[name], atol=1e-12, rtol=1e-11
                )
                for kind in ("m", "v"):
                    np.testing.assert_allclose(
                        a.moments[kind][name],
                        b.moments[kind][name],
                        atol=1e-12,
                        rtol=1e-11,
                    )
        self.assertEqual(a.step, 4)
        self.assertEqual(b.step, 4)

    def test_shared_adam_matches_existing_compiled_trainer(self):
        coordinator = Coordinator(self.weights, DiLoCoConfig(local_steps=3))
        coordinator.begin_round(["a"])
        trainer = Trainer("a", self.recipe)
        trainer.run_round(coordinator.assignment("a"), [self.batch] * 3)
        optimizer = SharedAdam(self.recipe, self.weights)
        for _ in range(3):
            optimizer.apply(
                compute_gradients(
                    self.recipe, optimizer.weights, self.batch
                ).gradients
            )
        for name in self.weights:
            np.testing.assert_array_equal(
                optimizer.weights[name], trainer.weights[name]
            )

    def test_tree_is_sample_weighted_not_an_unweighted_mean(self):
        result = reduce_gradients(
            [
                GradientPart(1, 1, {"x": np.array([[1.0]])}),
                GradientPart(2, 4, {"x": np.array([[4.0]])}),
                GradientPart(4, 8, {"x": np.array([[8.0]])}),
            ]
        )
        self.assertAlmostEqual(result.loss, 41 / 7)
        self.assertAlmostEqual(result.gradients["x"][0, 0], 41 / 7)
        for parts in (
            [],
            [GradientPart(0, 1, self.weights)],
            [GradientPart(1, float("nan"), self.weights)],
            [GradientPart(1, 1, {"x": np.array([[float("inf")]])})],
        ):
            with self.assertRaises(ValueError):
                reduce_gradients(parts)

    def test_invalid_gradient_does_not_advance_optimizer(self):
        optimizer = SharedAdam(self.recipe, self.weights)
        with self.assertRaises(ValueError):
            optimizer.apply({"bad": np.zeros((1, 1))})
        self.assertEqual(optimizer.step, 0)


if __name__ == "__main__":
    unittest.main()
