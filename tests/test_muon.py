"""Independent optimizer reference, mixed state, and durable resume tests."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ygaraty.compiler.muon import append_muon
from ygaraty.compiler.plan import split
from ygaraty.compiler.shapes import ShapeTape
from ygaraty.runtime.local import execute_plan
from ygaraty.training.convnet_webgpu import ConvBatch
from ygaraty.training.coordinator import Assignment
from ygaraty.training.session import TrainingConfig
from ygaraty.training.trainer import Trainer, optimizer_state


def reference(w, g, m, lr=0.02, beta=0.95):
    m = beta * m + (1 - beta) * g
    x = beta * m + (1 - beta) * g
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    x = x / (np.linalg.norm(x) + 1e-7)
    for _ in range(5):
        gram = x @ x.T
        x = 3.4445 * x + (-4.775 * gram + 2.0315 * (gram @ gram)) @ x
    if transpose:
        x = x.T
    return w - lr * max(1, w.shape[0] / w.shape[1]) ** 0.5 * x, m


class MuonTests(unittest.TestCase):
    def test_compiled_updates_match_independent_reference(self):
        rng = np.random.default_rng(42)
        for shape in [(3, 8), (8, 3), (1, 5), (4, 4)]:
            for zero in [False, True]:
                tape = ShapeTape()
                p = tape.declare("weight", shape, dict(kind="zeros"), True)
                g = tape.declare("gradient", shape, dict(kind="zeros"))
                tape.gradients[p] = g
                updates = append_muon(tape, [p], 0.02)
                # Small tiles verify normalization is global, not per tile.
                tasks = split(tape, 2)
                w = rng.normal(size=shape)
                grad = np.zeros(shape) if zero else rng.normal(size=shape)
                m = np.zeros(shape) if zero else rng.normal(size=shape)
                expected, expected_m = reference(w, grad, m)
                result = execute_plan(
                    tasks,
                    {p: w, g: grad, "muon_m_weight": m},
                    list(updates.values()),
                )
                np.testing.assert_allclose(
                    result[updates[p]], expected, atol=1e-12
                )
                np.testing.assert_allclose(
                    result[updates["muon_m_weight"]], expected_m, atol=1e-12
                )

    def test_browser_cpu_matches_reference(self):
        tape = ShapeTape()
        p = tape.declare("weight", (2, 3), dict(kind="zeros"), True)
        g = tape.declare("gradient", (2, 3), dict(kind="zeros"))
        tape.gradients[p] = g
        updates = append_muon(tape, [p], 0.02)
        arrays = {
            p: np.arange(6).reshape(2, 3) / 10,
            g: np.array([[1.0, 2.0, 3.0], [-2.0, 0.0, 1.0]]),
            "muon_m_weight": np.ones((2, 3)) / 10,
        }
        payload = dict(
            plan=dict(
                tasks=[t.describe() for t in split(tape, 2)],
                loss=updates[p],
                updates=updates,
            ),
            inputs={
                k: dict(shape=list(v.shape), data=v.ravel().tolist())
                for k, v in arrays.items()
            },
        )
        root = Path(__file__).resolve().parents[1]
        script = """import {executeLocalPlan} from './ygaraty/backends/webgpu/browser/diloco.mjs';
import {ComputeSDK} from './ygaraty/backends/webgpu/browser/sdk_compute.mjs';
let s='';
for await(const c of process.stdin)s+=c;
const d=JSON.parse(s);
const inputs=Object.fromEntries(Object.entries(d.inputs).map(([k,v])=>
[k,{shape:v.shape,data:Float64Array.from(v.data)}]));
const r=await executeLocalPlan(d.plan,inputs,new ComputeSDK());
console.log(JSON.stringify(Array.from(r.values.get(d.plan.loss).data)));"""
        result = subprocess.run(
            ["node", "--input-type=module", "-e", script],
            cwd=root,
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            check=True,
        )
        expected, _ = reference(arrays[p], arrays[g], arrays["muon_m_weight"])
        np.testing.assert_allclose(
            json.loads(result.stdout), expected.ravel(), atol=1e-12
        )

    def test_mixed_state_resume_and_layout_validation(self):
        recipe = TrainingConfig(
            1, optimizer="muon", channels=(2, 4, 4), batch_size=2
        ).recipe()
        trainer = Trainer("worker", recipe)
        rng = np.random.default_rng(9)
        batch = ConvBatch(
            rng.integers(0, 256, (2, 1, 28, 28), dtype=np.uint8),
            np.array([1, 2]),
        )
        weights = recipe.initial_weights()
        # Assignment argument order is explicit to keep the test readable.
        a = Assignment(
            round_id="r0", version=0, worker="worker", steps=1, weights=weights
        )
        trainer.run_round(a, [batch])
        saved = trainer.checkpoint()
        self.assertEqual(
            set(saved["moments"]["v"]), set(recipe.moment_names()["v"])
        )
        self.assertNotIn("conv0.weight", saved["moments"]["v"])
        restored = Trainer("worker", recipe)
        restored.restore(json.loads(json.dumps(saved)), trainer.weights)
        a = Assignment(
            round_id="r1",
            version=1,
            worker="worker",
            steps=1,
            weights=trainer.weights,
        )
        trainer.run_round(a, [batch])
        restored.run_round(a, [batch])
        for n in trainer.weights:
            np.testing.assert_array_equal(
                trainer.weights[n], restored.weights[n]
            )
        del saved["moments"]["m"]["conv0.weight"]
        with self.assertRaises(ValueError):
            optimizer_state(saved, weights, 1, recipe.moment_names())

    def test_session_checkpoint_resume_and_optimizer_mismatch(self):
        from dataclasses import replace

        from test_training_session import config, fixture, join_all, result_for

        from ygaraty.training.session import TrainingSession

        with tempfile.TemporaryDirectory() as directory:
            cfg = config(2, optimizer="muon")
            session = TrainingSession(cfg, fixture(), Path(directory))
            for credentials in join_all(session):
                payload = result_for(session, credentials)
                session.submit(
                    credentials["worker"], credentials["token"], payload
                )
            resumed = TrainingSession(
                cfg, fixture(), Path(directory), resume=True
            )
            self.assertEqual(resumed.states, session.states)
            self.assertEqual(resumed.coordinator.version, 1)
            for credentials in join_all(resumed):
                resumed.submit(
                    credentials["worker"],
                    credentials["token"],
                    result_for(resumed, credentials),
                )
            self.assertEqual(resumed.coordinator.version, 2)
            with self.assertRaises(ValueError):
                TrainingSession(
                    replace(cfg, optimizer="adam"),
                    fixture(),
                    Path(directory),
                    resume=True,
                )

    def test_options_reject_invalid_values(self):
        for args in [
            dict(optimizer="x"),
            dict(muon_steps=0),
            dict(muon_momentum=1),
            dict(muon_learning_rate=float("nan")),
        ]:
            with self.assertRaises(ValueError):
                TrainingConfig(1, **args)
