"""Public network protocol, recovery and Tape ConvNet correctness."""

import copy
import json
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ygaraty.runtime.local import execute_plan
from ygaraty.training import recovery
from ygaraty.training.convnet_webgpu import ConvBatch, ConvNetRecipe
from ygaraty.training.datasets import ImageDataset, read_idx
from ygaraty.training.server import create_app
from ygaraty.training.session import TrainingConfig, TrainingSession
from ygaraty.training.trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    rng = np.random.default_rng(44)
    return ImageDataset(
        "synthetic-protocol-fixture",
        rng.integers(0, 256, (8, 1, 28, 28), dtype=np.uint8),
        np.arange(8, dtype=np.uint8),
        rng.integers(0, 256, (4, 1, 28, 28), dtype=np.uint8),
        np.arange(4, dtype=np.uint8),
    )


def config(workers=2, **kwargs):
    return replace(
        TrainingConfig(
            workers,
            rounds=2,
            channels=(2, 2, 2),
            batch_size=1,
            local_steps=1,
            eval_examples=2,
            precision="float64",
        ),
        **kwargs,
    )


def join_all(session):
    return [
        session.join(session.join_token, f"client-{i:020}")
        for i in range(session.config.num_workers)
    ]


def result_for(session, credentials):
    w, token = credentials["worker"], credentials["token"]
    bundle = session.assignment(w, token)
    trainer = Trainer(w, session.recipe)
    trainer.restore(bundle["optimizer"], session.coordinator.weights)
    result = trainer.run_round(
        session.coordinator.assignment(w),
        session.dataset.batches(
            session.members.index(w),
            len(session.members),
            trainer.step,
            session.config.local_steps,
            session.config.batch_size,
            session.config.seed,
        ),
    )
    return dict(result=result.describe(), optimizer=trainer.checkpoint())


def run_round(session, credentials):
    for c in credentials:
        session.submit(c["worker"], c["token"], result_for(session, c))


class TrainingSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = fixture()
        self.now = 0.0

    def session(self, **kwargs):
        return TrainingSession(
            config(**kwargs), self.data, self.temp.name, clock=lambda: self.now
        )

    def test_retry_conflict_and_durable_restart(self):
        session = self.session()
        creds = join_all(session)
        first = result_for(session, creds[0])
        send = lambda payload: session.submit(
            creds[0]["worker"], creds[0]["token"], payload
        )
        send(first)
        send(first)
        bad = copy.deepcopy(first)
        bad["result"]["losses"][0] += 1
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            send(bad)
        second = result_for(session, creds[1])
        session.submit(creds[1]["worker"], creds[1]["token"], second)
        self.assertEqual(send(first)["version"], 1)
        resumed = TrainingSession(
            config(), self.data, self.temp.name, resume=True
        )
        resumed_creds = join_all(resumed)
        self.assertEqual(
            resumed.submit(
                resumed_creds[0]["worker"], resumed_creds[0]["token"], first
            )["version"],
            1,
        )
        run_round(resumed, resumed_creds)
        # Same training with no restart must preserve weights AND Adam/outer momentum.
        run_round(session, creds)
        a, b = session._snapshot(), resumed._snapshot()
        for name in session.coordinator.weights:
            np.testing.assert_array_equal(
                session.coordinator.weights[name],
                resumed.coordinator.weights[name],
            )
        self.assertEqual(a["states"], b["states"])
        self.assertEqual(
            a["coordinator"]["items"][1], b["coordinator"]["items"][1]
        )
        self.assertEqual(resumed.status()["state"], "complete")

    def test_deadline_rolls_back_moments_and_replaces_pending_worker(self):
        session = self.session(round_timeout=10, lease_seconds=100)
        creds = join_all(session)
        first, late = (
            result_for(session, creds[0]),
            result_for(session, creds[1]),
        )
        session.submit(creds[0]["worker"], creds[0]["token"], first)
        self.now = 11
        session.tick()
        self.assertEqual(session.states[creds[0]["worker"]]["step"], 0)
        with self.assertRaises(PermissionError):
            session.submit(creds[1]["worker"], creds[1]["token"], late)
        replacement = session.join(session.join_token, "replacement-000000000")
        self.assertEqual(replacement["worker"], creds[1]["worker"])
        new = session.assignment(creds[0]["worker"], creds[0]["token"])
        self.assertNotEqual(
            new["assignment"]["round_id"], first["result"]["round_id"]
        )
        with self.assertRaisesRegex(ValueError, "Stale"):
            session.submit(creds[0]["worker"], creds[0]["token"], first)
        run_round(session, [creds[0], replacement])
        self.assertEqual(session.coordinator.version, 1)

    def test_lease_expiry_and_retry_limit(self):
        session = self.session(max_retries=0, lease_seconds=5)
        creds = join_all(session)
        session.assignment(creds[0]["worker"], creds[0]["token"])
        self.now = 6
        session.tick()
        self.assertEqual(session.status()["state"], "failed")
        self.assertEqual(session.coordinator.version, 0)

    def test_crash_mid_round_discards_uncommitted_results(self):
        session = self.session()
        creds = join_all(session)
        session.submit(
            creds[0]["worker"],
            creds[0]["token"],
            result_for(session, creds[0]),
        )
        resumed = TrainingSession(
            config(), self.data, self.temp.name, resume=True
        )
        self.assertEqual(resumed.coordinator.version, 0)
        self.assertTrue(all(s["step"] == 0 for s in resumed.states.values()))
        run_round(resumed, join_all(resumed))
        run_round(session, creds[1:])
        for n in session.coordinator.weights:
            np.testing.assert_array_equal(
                session.coordinator.weights[n], resumed.coordinator.weights[n]
            )

    def test_failed_publication_restores_complete_committed_generation(self):
        session = self.session(workers=1)
        c = join_all(session)[0]
        payload = result_for(session, c)
        original = session.coordinator.weights
        with patch.object(recovery, "save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                session.submit(c["worker"], c["token"], payload)
        self.assertEqual(session.coordinator.version, 0)
        self.assertEqual(session.status()["state"], "failed")
        for n in original:
            np.testing.assert_array_equal(
                original[n], session.coordinator.weights[n]
            )

    def test_invalid_optimizer_and_foreign_result_leave_round_usable(self):
        session = self.session(workers=1)
        c = join_all(session)[0]
        payload = result_for(session, c)
        bad = copy.deepcopy(payload)
        bad["optimizer"]["step"] = 100
        with self.assertRaises(ValueError):
            session.submit(c["worker"], c["token"], bad)
        bad = copy.deepcopy(payload)
        next(iter(bad["optimizer"]["moments"]["v"].values()))["data"][0] = -1
        with self.assertRaises(ValueError):
            session.submit(c["worker"], c["token"], bad)
        session.submit(c["worker"], c["token"], payload)
        self.assertEqual(session.coordinator.version, 1)

    def test_dataset_stream_and_identity(self):
        a = self.data.indices(0, 2, 0, 4, 1)
        b = self.data.indices(1, 2, 0, 4, 1)
        self.assertFalse(set(a) & set(b))
        whole = self.data.indices(0, 2, 0, 12, 1)
        parts = np.concatenate(
            [self.data.indices(0, 2, i, 3, 1) for i in (0, 3, 6, 9)]
        )
        np.testing.assert_array_equal(whole, parts)
        self.session()
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            TrainingSession(
                config(seed=9), self.data, self.temp.name, resume=True
            )
        with self.assertRaises(ValueError):
            read_idx(b"invalid", images=True)

    def test_periodic_evaluation_always_includes_final_round(self):
        session = self.session(workers=1, eval_interval=3)
        credentials = join_all(session)
        run_round(session, credentials)
        self.assertNotIn("evaluation", session.metrics[-1])
        run_round(session, credentials)
        self.assertEqual(session.metrics[-1]["evaluation"]["examples"], 2)
        self.assertEqual(session.metrics[-1]["version"], 2)

    def test_http_auth_assets_and_no_private_files(self):
        from fastapi.testclient import TestClient

        session = self.session()
        with TestClient(create_app(session)) as client:
            self.assertEqual(client.get("/").status_code, 200)
            self.assertEqual(
                client.get("/assets/training_client.mjs").status_code, 200
            )
            self.assertEqual(
                client.get("/assets/checkpoint.json").status_code, 404
            )
            self.assertEqual(
                client.post(
                    "/api/join", json={"client_id": "x" * 16}
                ).status_code,
                401,
            )
            headers = {"Authorization": "Bearer " + session.join_token}
            response = client.post(
                "/api/join", headers=headers, json={"client_id": "x" * 16}
            )
            self.assertEqual(response.status_code, 200)
            c = response.json()
            self.assertEqual(
                client.post(
                    f"/api/workers/{c['worker']}/assignment"
                ).status_code,
                401,
            )
            self.assertIn("no-store", response.headers["cache-control"])

    def wait_for_assignment(self, session, credentials):
        """Start a waiter and prove it released the session lock."""
        from concurrent.futures import ThreadPoolExecutor

        pool = ThreadPoolExecutor(max_workers=1)
        self.addCleanup(pool.shutdown)
        entered = threading.Event()
        original = session.changed.wait

        def wait(timeout):
            entered.set()
            return original(timeout)

        patched = patch.object(session.changed, "wait", side_effect=wait)
        patched.start()
        self.addCleanup(patched.stop)
        future = pool.submit(
            session.assignment_bytes,
            credentials["worker"],
            credentials["token"],
            wait_seconds=3,
        )
        self.assertTrue(entered.wait(2), "Assignment did not wait")
        return future

    def test_assignment_bytes_cache_is_isolated_and_authenticated(self):
        session = self.session(workers=1)
        c = join_all(session)[0]
        args = (c["worker"], c["token"])
        encoded = session.assignment_bytes(*args)
        self.assertIs(encoded, session.assignment_bytes(*args))
        decoded = json.loads(encoded)
        value = session.assignment(*args)
        self.assertEqual(decoded, json.loads(json.dumps(value)))
        value["plans"][0]["tasks"].clear()
        value["assignment"]["weights"].clear()
        self.assertEqual(json.loads(session.assignment_bytes(*args)), decoded)
        with self.assertRaises(PermissionError):
            session.assignment_bytes(c["worker"], "wrong-token")
        session.submit(*args, result_for(session, c))
        self.assertFalse(session.encoded_bundles)
        following = json.loads(session.assignment_bytes(*args))
        self.assertEqual(following["assignment"]["version"], 1)
        self.assertNotEqual(
            following["assignment"]["round_id"],
            decoded["assignment"]["round_id"],
        )

    def test_large_assignment_is_serialized_without_retained_byte_cache(self):
        session = self.session(workers=1)
        c = join_all(session)[0]
        with patch.object(recovery, "LIMIT", 1):
            encoded = session.assignment_bytes(c["worker"], c["token"])
        self.assertFalse(session.encoded_bundles)
        self.assertEqual(json.loads(encoded)["state"], "assignment")

    def test_waiting_assignment_wakes_on_join_and_has_bounded_timeout(self):
        session = self.session()
        c = session.join(session.join_token, "first-client-000000")
        status = json.loads(
            session.assignment_bytes(c["worker"], c["token"], wait_seconds=0)
        )
        self.assertEqual(status["state"], "waiting")
        self.assertNotIn("metrics", status)
        self.assertIn("metrics", session.assignment(c["worker"], c["token"]))
        for invalid in (-1, 21, float("nan"), True):
            with self.assertRaises(ValueError):
                session.assignment_bytes(
                    c["worker"], c["token"], wait_seconds=invalid
                )
        future = self.wait_for_assignment(session, c)
        session.join(session.join_token, "second-client-00000")
        self.assertEqual(json.loads(future.result(2))["state"], "assignment")

    def test_waiting_assignment_wakes_only_after_durable_commit(self):
        session = self.session()
        a, b = join_all(session)
        session.submit(a["worker"], a["token"], result_for(session, a))
        future = self.wait_for_assignment(session, a)
        original = recovery.save

        def save(*args, **kwargs):
            self.assertFalse(future.done())
            return original(*args, **kwargs)

        with patch.object(recovery, "save", side_effect=save):
            session.submit(b["worker"], b["token"], result_for(session, b))
        bundle = json.loads(future.result(2))
        self.assertEqual(bundle["assignment"]["version"], 1)
        self.assertEqual(
            bundle["optimizer"]["step"], session.config.local_steps
        )
        self.assertEqual(
            recovery.load(session.path)["metrics"][-1]["version"], 1
        )

    def test_waiter_rechecks_lease_and_discards_aborted_response_cache(self):
        session = self.session(lease_seconds=5)
        a, b = join_all(session)
        session.assignment_bytes(a["worker"], a["token"])
        session.submit(a["worker"], a["token"], result_for(session, a))
        future = self.wait_for_assignment(session, a)
        self.now = 6
        session.tick()
        with self.assertRaises(PermissionError):
            future.result(2)
        self.assertFalse(session.encoded_bundles)
        self.assertEqual(session.coordinator.version, 0)
        replacement = join_all(session)
        new = json.loads(
            session.assignment_bytes(
                replacement[0]["worker"], replacement[0]["token"]
            )
        )
        self.assertEqual(new["optimizer"]["step"], 0)

    def test_commit_failure_wakes_waiter_without_publishing_new_state(self):
        session = self.session()
        a, b = join_all(session)
        session.submit(a["worker"], a["token"], result_for(session, a))
        future = self.wait_for_assignment(session, a)
        payload = result_for(session, b)
        with patch.object(recovery, "save", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                session.submit(b["worker"], b["token"], payload)
        self.assertEqual(json.loads(future.result(2))["state"], "failed")
        self.assertEqual(session.coordinator.version, 0)
        self.assertFalse(session.encoded_bundles)

    def test_http_assignment_is_protocol_json_without_recursive_conversion(
        self,
    ):
        from fastapi.testclient import TestClient

        session = self.session(workers=1)
        c = join_all(session)[0]
        with TestClient(create_app(session)) as client:
            with patch(
                "fastapi.routing.serialize_response",
                side_effect=AssertionError(
                    "Protocol response must bypass recursive conversion"
                ),
            ):
                response = client.post(
                    f"/api/workers/{c['worker']}/assignment",
                    headers={"Authorization": "Bearer " + c["token"]},
                )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.headers["content-type"], "application/json"
            )
            self.assertEqual(
                response.json(),
                json.loads(
                    json.dumps(session.assignment(c["worker"], c["token"]))
                ),
            )


class ConvNetTapeTests(unittest.TestCase):
    def test_gradients_match_finite_differences(self):
        recipe = ConvNetRecipe(
            dict(
                in_channels=1,
                height=4,
                width=4,
                channels=[2, 2, 2],
                classes=2,
                residual=True,
            )
        )
        batch = ConvBatch(
            np.random.default_rng(7).integers(
                0, 256, (1, 1, 4, 4), dtype=np.uint8
            ),
            np.array([1]),
        )
        plan = recipe.compile(batch, 1)
        weights = recipe.initial_weights(4)
        inputs = dict(plan.constants)
        for name, p in plan.parameters.items():
            inputs[p] = weights[name]
            inputs["adam_m_" + p] = np.zeros_like(weights[name])
            inputs["adam_v_" + p] = np.zeros_like(weights[name])
        values = execute_plan(
            plan.tasks, inputs, list(plan.gradients.values())
        )
        for name, p in plan.parameters.items():
            index = (0, 0)
            original = weights[name][index]
            losses = []
            for sign in (1, -1):
                weights[name][index] = original + sign * 1e-5
                logits = recipe.predict(weights, batch.images)
                losses.append(
                    float(
                        np.log(np.exp(logits - logits.max()).sum())
                        - (logits[0, 1] - logits.max())
                    )
                )
            weights[name][index] = original
            self.assertAlmostEqual(
                values[plan.gradients[p]][index],
                (losses[0] - losses[1]) / 2e-5,
                places=6,
                msg=name,
            )


@unittest.skipUnless(
    shutil.which("node"), "Node required for browser transport parity"
)
class BrowserTransportTests(unittest.TestCase):
    def test_two_javascript_contributors_match_python_over_http(self):
        import uvicorn

        with (
            tempfile.TemporaryDirectory() as directory,
            tempfile.TemporaryDirectory() as reference,
        ):
            data, cfg = fixture(), config(rounds=2)
            session = TrainingSession(cfg, data, directory)
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            server = uvicorn.Server(
                uvicorn.Config(create_app(session), log_level="error")
            )
            thread = threading.Thread(
                target=server.run, kwargs={"sockets": [sock]}, daemon=True
            )
            thread.start()
            try:
                deadline = time.monotonic() + 10
                while not server.started and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(server.started)
                module = (
                    ROOT
                    / "ygaraty/backends/webgpu/browser/training_client.mjs"
                )
                script = """
import {contribute} from MODULE;
const nativeFetch = globalThis.fetch;
let dropped = false;
globalThis.fetch = async (...args) => {
  const response = await nativeFetch(...args);
  if (!dropped && String(args[0]).endsWith('/result') && response.ok) {
    dropped = true;
    throw new TypeError('Simulated lost result acknowledgement');
  }
  return response;
};
const results = await Promise.all([0,1].map(i => contribute({baseURL: process.argv[1],
 joinToken: process.argv[2], clientId: 'node-worker-0000000'+i})));
if (results.some(r => r.state !== 'complete')) throw Error(JSON.stringify(results));
if (!dropped) throw Error('Result retry was not exercised');
""".replace("MODULE", json.dumps(module.as_uri()))
                result = subprocess.run(
                    [
                        "node",
                        "--input-type=module",
                        "-e",
                        script,
                        f"http://127.0.0.1:{port}",
                        session.join_token,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(
                    result.returncode, 0, result.stdout + result.stderr
                )
            finally:
                server.should_exit = True
                thread.join(10)
                sock.close()
            baseline = TrainingSession(cfg, data, reference)
            credentials = join_all(baseline)
            for _ in range(2):
                run_round(baseline, credentials)
            for name, expected in baseline.coordinator.weights.items():
                np.testing.assert_allclose(
                    session.coordinator.weights[name],
                    expected,
                    rtol=1e-9,
                    atol=1e-10,
                )
            self.assertEqual(session.states["worker-0"]["step"], 2)
            self.assertGreater(
                session.metrics[-1]["worker_stats"]["worker-0"]["cpu_tasks"], 0
            )
            self.assertEqual(
                session.metrics[-1]["worker_stats"]["worker-0"]["gpu_tasks"], 0
            )


if __name__ == "__main__":
    unittest.main()
