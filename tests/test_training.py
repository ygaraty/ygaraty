"""Round semantics, real Transformer training, and independent optimizer
checks."""

import json
import os
import shutil
import subprocess
import tempfile
import threading
import unittest
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from ygaraty.compiler import Tape, split
from ygaraty.runtime.kernels import evaluate
from ygaraty.runtime.local import execute_plan
from ygaraty.training import (
    Batch,
    Coordinator,
    DiLoCoConfig,
    RoundResult,
    Trainer,
    TransformerRecipe,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = dict(
    vocab=9,
    embedding=4,
    hidden=6,
    layers=1,
    heads=2,
    max_length=5,
    dropout=0.0,
)
BATCHES = [
    Batch(
        np.array([[4, 3, 0], [5, 6, 3]]),
        np.array([[2, 4], [2, 5]]),
        np.array([[4, 3], [5, 3]]),
    ),
    Batch(np.array([[6, 5, 3]]), np.array([[2, 6, 5]]), np.array([[6, 5, 3]])),
]


class CoordinatorTests(unittest.TestCase):
    def result(self, assignment, value):
        return RoundResult(
            assignment.round_id,
            assignment.version,
            assignment.worker,
            assignment.steps,
            {"w": np.array([[value]])},
            (1.0,) * assignment.steps,
        )

    def test_sgd_outer_is_fedavg_and_result_ownership_is_isolated(self):
        cfg = DiLoCoConfig(
            local_steps=2,
            outer_learning_rate=1.0,
            momentum=0.0,
            nesterov=False,
        )
        original = {"w": np.array([[10.0]])}
        c = Coordinator(original, cfg)
        original["w"][:] = 100
        c.begin_round(["b", "a"])
        a, b = c.assignment("a"), c.assignment("b")
        a.weights["w"][:] = 999
        self.assertEqual(b.weights["w"][0, 0], 10.0)
        result = self.result(b, 4.0)
        c.submit(result)
        result.pseudo_gradient["w"][:] = 100
        with self.assertRaises(ValueError):
            c.commit_round()
        self.assertEqual(c.version, 0)
        c.submit(self.result(a, 2.0))
        record = c.commit_round()
        np.testing.assert_array_equal(c.weights["w"], [[7.0]])
        self.assertEqual(record["workers"], ["a", "b"])
        returned = c.weights
        returned["w"][:] = 0
        np.testing.assert_array_equal(c.weights["w"], [[7.0]])

    def test_invalid_duplicate_stale_and_aborted_results_do_not_advance(self):
        c = Coordinator({"w": np.array([[10.0]])}, DiLoCoConfig(local_steps=2))
        c.begin_round(["a"])
        a = c.assignment("a")
        good = self.result(a, 2.0)
        for invalid in (
            replace(good, worker="intruder"),
            replace(good, round_id="foreign"),
            replace(good, version=-1),
            replace(good, steps=1),
            replace(good, losses=(float("nan"), 1.0)),
            replace(good, pseudo_gradient={"wrong": np.ones((1, 1))}),
            replace(good, pseudo_gradient={"w": np.ones((1, 2))}),
            replace(good, pseudo_gradient={"w": np.array([[np.inf]])}),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    c.submit(invalid)
                self.assertEqual(c.pending, ("a",))
                np.testing.assert_array_equal(c.weights["w"], [[10.0]])
        c.submit(good)
        with self.assertRaises(ValueError):
            c.submit(good)
        c.abort_round()
        c.begin_round(["a"])
        with self.assertRaises(ValueError):
            c.submit(good)
        c.submit(self.result(c.assignment("a"), 2.0))
        c.commit_round()
        # Fresh velocity: v=2, direction=2+.9*2, theta=10-.7*3.8.
        np.testing.assert_allclose(c.weights["w"], [[7.34]], atol=1e-15)

    def test_averaging_order_does_not_depend_on_completion_order(self):
        outputs = []
        for order in (["a", "b", "c"], ["c", "a", "b"]):
            c = Coordinator(
                {"w": np.array([[1.0]])}, DiLoCoConfig(local_steps=1)
            )
            c.begin_round(order)
            for worker in order:
                c.submit(
                    self.result(
                        c.assignment(worker),
                        {"a": 1e15, "b": -1e15, "c": 1.0}[worker],
                    )
                )
            c.commit_round()
            outputs.append(c.weights["w"])
        np.testing.assert_array_equal(*outputs)

    def test_overflow_is_atomic(self):
        c = Coordinator(
            {"w": np.array([[1.0]])},
            DiLoCoConfig(local_steps=1, outer_learning_rate=1e308),
        )
        c.begin_round(["a"])
        c.submit(self.result(c.assignment("a"), 1e308))
        with self.assertRaises(FloatingPointError):
            c.commit_round()
        self.assertEqual(c.version, 0)
        self.assertEqual(c.history, [])
        np.testing.assert_array_equal(c.weights["w"], [[1.0]])


class LocalPlanTests(unittest.TestCase):
    def test_repeated_dependencies_views_and_released_outputs(self):
        t = Tape()
        x = t.parameter("x", [[1.0, 2.0], [3.0, 4.0]])
        t.backward(t.sum(t.mul(x, x)))
        leaves = {k: v.copy() for k, v in t.values.items()}
        keep = [t.loss, t.gradients[x]]
        values = execute_plan(split(t, 1), leaves, keep)
        expected = evaluate(t)
        self.assertEqual(set(values), set(keep))
        for key in keep:
            np.testing.assert_array_equal(values[key], expected[key])
        for key in leaves:
            np.testing.assert_array_equal(leaves[key], t.values[key])


class TransformerRoundTests(unittest.TestCase):
    def test_fused_and_portable_replicas_match_over_multiple_rounds(self):
        outputs = []
        for portable in (False, True):
            recipe = TransformerRecipe(
                CONFIG, portable=portable, tile_size=4, learning_rate=0.002
            )
            c = Coordinator(
                recipe.initial_weights(), DiLoCoConfig(local_steps=2)
            )
            trainers = [Trainer("a", recipe), Trainer("b", recipe)]
            losses = []
            for _ in range(2):
                c.begin_round(t.worker for t in trainers)
                for trainer, batch in zip(trainers, BATCHES):
                    result = trainer.run_round(
                        c.assignment(trainer.worker), [batch, batch]
                    )
                    losses.extend(result.losses)
                    c.submit(result)
                c.commit_round()
            self.assertEqual([t.step for t in trainers], [4, 4])
            outputs.append((c.weights, trainers, losses))
        for name in outputs[0][0]:
            # Attention key biases have theoretically zero gradients. Tiny
            # summation differences are amplified by Adam's 1e-8 epsilon.
            np.testing.assert_allclose(
                outputs[0][0][name],
                outputs[1][0][name],
                rtol=1e-10,
                atol=1e-10,
                err_msg=name,
            )
            for a, b in zip(outputs[0][1], outputs[1][1]):
                for kind in ("m", "v"):
                    np.testing.assert_allclose(
                        a.moments[kind][name],
                        b.moments[kind][name],
                        atol=1e-12,
                        err_msg=name,
                    )
        np.testing.assert_allclose(outputs[0][2], outputs[1][2], atol=1e-12)

    def test_failed_round_does_not_change_local_optimizer_and_rejects_replay(
        self,
    ):
        recipe = TransformerRecipe(CONFIG, tile_size=4)
        trainer = Trainer("a", recipe)
        c = Coordinator(recipe.initial_weights(), DiLoCoConfig(local_steps=2))
        c.begin_round(["a"])
        assignment = c.assignment("a")
        with self.assertRaises(ValueError):
            trainer.run_round(assignment, [BATCHES[0]])
        self.assertEqual(trainer.step, 0)
        self.assertIsNone(trainer.weights)
        result = trainer.run_round(assignment, [BATCHES[0]] * 2)
        with self.assertRaises(ValueError):
            trainer.run_round(assignment, [BATCHES[0]] * 2)
        c.submit(result)
        c.commit_round()
        c.begin_round(["a"])
        before = {
            kind: {n: a.copy() for n, a in state.items()}
            for kind, state in trainer.moments.items()
        }
        with self.assertRaises(ValueError):
            trainer.run_round(c.assignment("a"), [BATCHES[1]])
        self.assertEqual(trainer.step, 2)
        for kind in before:
            for name in before[kind]:
                np.testing.assert_array_equal(
                    trainer.moments[kind][name], before[kind][name]
                )

    def test_real_updates_match_independent_pytorch_diloco_reference(self):
        try:
            import torch
            from torch import (
                nn,  # noqa: F401 — require a complete installation
            )
        except ImportError:
            self.skipTest(
                "Install translation requirements for independent parity"
            )
        from ygaraty.reference.transformer import TransformerTranslator

        torch.set_num_threads(1)
        recipe = TransformerRecipe(CONFIG, tile_size=4, learning_rate=0.002)
        initial = recipe.initial_weights()
        c = Coordinator(initial, DiLoCoConfig(local_steps=2))
        global_model = TransformerTranslator(**CONFIG).double().eval()
        global_model.load_state_dict(
            {
                n: torch.tensor(a.reshape(global_model.state_dict()[n].shape))
                for n, a in initial.items()
            }
        )
        outer = torch.optim.SGD(
            global_model.parameters(),
            lr=0.7,
            momentum=0.9,
            nesterov=True,
            foreach=False,
        )
        models = [
            TransformerTranslator(**CONFIG).double().eval() for _ in range(2)
        ]
        optimizers = [
            torch.optim.Adam(
                m.parameters(),
                lr=0.002,
                betas=(0.9, 0.98),
                eps=1e-8,
                foreach=False,
            )
            for m in models
        ]
        trainers = [Trainer("a", recipe), Trainer("b", recipe)]
        for _ in range(2):
            c.begin_round(t.worker for t in trainers)
            for model, optimizer, trainer, batch in zip(
                models, optimizers, trainers, BATCHES
            ):
                model.load_state_dict(
                    global_model.state_dict()
                )  # Preserve optimizer state.
                reference_losses = []
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(
                        torch.tensor(batch.source),
                        torch.tensor(batch.target_input),
                    )
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, CONFIG["vocab"]),
                        torch.tensor(batch.targets).reshape(-1),
                        ignore_index=0,
                        label_smoothing=0.1,
                    )
                    loss.backward()
                    optimizer.step()
                    reference_losses.append(loss.item())
                result = trainer.run_round(
                    c.assignment(trainer.worker), [batch] * 2
                )
                np.testing.assert_allclose(
                    result.losses, reference_losses, rtol=1e-10, atol=1e-12
                )
                for name, param in model.named_parameters():
                    np.testing.assert_allclose(
                        trainer.weights[name],
                        param.detach()
                        .numpy()
                        .reshape(trainer.weights[name].shape),
                        rtol=1e-9,
                        atol=1e-10,
                        err_msg=name,
                    )
                    for kind, key in [("m", "exp_avg"), ("v", "exp_avg_sq")]:
                        np.testing.assert_allclose(
                            trainer.moments[kind][name],
                            optimizer.state[param][key]
                            .numpy()
                            .reshape(trainer.weights[name].shape),
                            rtol=1e-9,
                            atol=1e-12,
                            err_msg=f"{kind} {name}",
                        )
                c.submit(result)
            outer.zero_grad(set_to_none=True)
            for name, parameter in global_model.named_parameters():
                parameter.grad = sum(
                    parameter.detach()
                    - dict(m.named_parameters())[name].detach()
                    for m in models
                ) / len(models)
            outer.step()
            c.commit_round()
            for name, parameter in global_model.named_parameters():
                np.testing.assert_allclose(
                    c.weights[name],
                    parameter.detach().numpy().reshape(initial[name].shape),
                    rtol=1e-9,
                    atol=1e-10,
                    err_msg=name,
                )


@unittest.skipUnless(
    shutil.which("node"), "Node required for browser SDK checks"
)
class BrowserRoundTests(unittest.TestCase):
    def test_javascript_trains_same_transformer_and_preserves_adam_between_rounds(  # noqa: E501 -- preserve test name
        self,
    ):
        recipe = TransformerRecipe(
            CONFIG, portable=True, tile_size=4, learning_rate=0.002
        )
        c = Coordinator(recipe.initial_weights(), DiLoCoConfig(local_steps=2))
        trainer = Trainer("a", recipe)
        bundles, expected = [], []
        for batch in BATCHES:
            c.begin_round(["a"])
            assignment = c.assignment("a")
            bundles.append(
                recipe.round_bundle(
                    assignment, [batch] * 2, start_step=trainer.step
                )
            )
            result = trainer.run_round(assignment, [batch] * 2)
            expected.append(result)
            c.submit(result)
            c.commit_round()
        code = """
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {DiLoCoTrainer} from './ygaraty/backends/webgpu/browser/diloco.mjs';
const bundles=JSON.parse(fs.readFileSync(0,'utf8')), trainer=new DiLoCoTrainer('a');
const results=[];
for (const bundle of bundles) results.push(await trainer.runRound(bundle.assignment,bundle.plans));
await assert.rejects(()=>trainer.runRound(bundles[1].assignment,bundles[1].plans),/Invalid round/);
const moments=Object.fromEntries(['m','v'].map(k=>[k,Object.fromEntries(Object.entries(trainer.moments[k]).map(([n,t])=>[n,Array.from(t.data)]))]));
console.log(JSON.stringify({results,step:trainer.step,moments}));
"""  # noqa: E501 -- preserve fixture or report layout
        process = subprocess.run(
            ["node", "--input-type=module", "-e", code],
            input=json.dumps(bundles),
            text=True,
            capture_output=True,
            cwd=ROOT,
            timeout=60,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        actual = json.loads(process.stdout)
        self.assertEqual(actual["step"], 4)
        for result, wanted in zip(actual["results"], expected):
            decoded = RoundResult.from_description(result)
            self.assertEqual(decoded.round_id, wanted.round_id)
            np.testing.assert_allclose(
                result["losses"], wanted.losses, rtol=1e-10, atol=1e-12
            )
            self.assertGreater(result["stats"]["matmul_tasks"], 0)
            self.assertEqual(result["stats"]["gpu_tasks"], 0)
            for name, value in wanted.pseudo_gradient.items():
                np.testing.assert_allclose(
                    decoded.pseudo_gradient[name],
                    value,
                    rtol=1e-8,
                    atol=1e-10,
                    err_msg=name,
                )
        for kind in ("m", "v"):
            for name, value in trainer.moments[kind].items():
                np.testing.assert_allclose(
                    np.array(actual["moments"][kind][name]).reshape(
                        value.shape
                    ),
                    value,
                    rtol=1e-8,
                    atol=1e-12,
                    err_msg=f"{kind} {name}",
                )

    def test_required_gpu_round_uses_sdk_and_fails_without_device(self):
        recipe = TransformerRecipe(CONFIG, portable=True, tile_size=4)
        c = Coordinator(recipe.initial_weights(), DiLoCoConfig(local_steps=1))
        c.begin_round(["a"])
        bundle = recipe.round_bundle(c.assignment("a"), [BATCHES[0]])
        code = """
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {DiLoCoTrainer} from './ygaraty/backends/webgpu/browser/diloco.mjs';
import {ComputeSDK,executeCPU} from './ygaraty/backends/webgpu/browser/sdk.mjs';
const bundle=JSON.parse(fs.readFileSync(0,'utf8'));
const sdk=new ComputeSDK(), trainer=new DiLoCoTrainer('a',{sdk,precision:'webgpu',arithmetic:'compensated'});
await assert.rejects(()=>trainer.runRound(bundle.assignment,bundle.plans),/requires WebGPU/);
assert.equal(trainer.step,0);assert.equal(trainer.weights,null);
let calls=0;sdk.device={};sdk.matmul=async(a,b,shape,arithmetic)=>{
  assert.equal(arithmetic,'compensated');calls++;return executeCPU('matmul',[a,b],{},shape);
};
const result=await trainer.runRound(bundle.assignment,bundle.plans);
assert.ok(calls>0);assert.equal(result.stats.gpu_tasks,calls);assert.equal(result.stats.matmul_tasks,calls);
console.log('ok');
"""  # noqa: E501 -- preserve fixture or report layout
        process = subprocess.run(
            ["node", "--input-type=module", "-e", code],
            input=json.dumps(bundle),
            text=True,
            capture_output=True,
            cwd=ROOT,
            timeout=60,
        )
        self.assertEqual(process.returncode, 0, process.stderr)

    @unittest.skipUnless(
        os.environ.get("YGARATY_TEST_WEBGPU"),
        "Set YGARATY_TEST_WEBGPU=1 for real WebGPU rounds",
    )
    def test_complete_round_on_chromium_webgpu(self):
        chrome = (
            os.environ.get("YGARATY_CHROME")
            or shutil.which("chromium")
            or shutil.which("google-chrome")
        )
        if not chrome:
            self.skipTest("Chrome/Chromium required")
        recipe = TransformerRecipe(CONFIG, portable=True, tile_size=4)
        c = Coordinator(recipe.initial_weights(), DiLoCoConfig(local_steps=2))
        c.begin_round(["gpu"])
        assignment = c.assignment("gpu")
        bundle = recipe.round_bundle(assignment, [BATCHES[0]] * 2)
        expected = Trainer("gpu", recipe).run_round(
            assignment, [BATCHES[0]] * 2
        )
        received, ready = [], threading.Event()
        script = b"""
import {DiLoCoTrainer} from '/diloco.mjs';
import {ComputeSDK} from '/sdk.mjs';
let result;
try {
  const bundle=await (await fetch('/bundle')).json(), sdk=new ComputeSDK();
  await sdk.initialize(true);
  result=await new DiLoCoTrainer('gpu',{sdk,precision:'webgpu',arithmetic:'compensated'})
    .runRound(bundle.assignment,bundle.plans);
} catch(e) {result={error:String(e.stack || e)};}
await fetch('/result',{method:'POST',body:JSON.stringify(result)});
"""  # noqa: E501 -- preserve fixture or report layout

        BROWSER = ROOT / "ygaraty/backends/webgpu/browser"

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                if self.path == "/":
                    content, mime = (
                        b'<script type="module" src="/smoke.mjs"></script>',
                        "text/html",
                    )
                elif self.path == "/smoke.mjs":
                    content, mime = script, "text/javascript"
                elif self.path == "/bundle":
                    content, mime = (
                        json.dumps(bundle).encode(),
                        "application/json",
                    )
                elif (
                    # The SDK is split into modules (sdk_*.mjs,
                    # webgpu_kernels.mjs, ...): serve any browser module
                    # by bare name, never a path.
                    self.path.endswith(".mjs")
                    and "/" not in self.path[1:]
                    and (BROWSER / self.path[1:]).is_file()
                ):
                    content, mime = (
                        (
                            ROOT
                            / "ygaraty/backends/webgpu/browser"
                            / self.path[1:]
                        ).read_bytes(),
                        "text/javascript",
                    )
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            def do_POST(self):
                received.append(
                    json.loads(
                        self.rfile.read(int(self.headers["Content-Length"]))
                    )
                )
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
                ready.set()

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with (
                tempfile.TemporaryDirectory() as directory,
                open(Path(directory) / "chrome.log", "w+") as log,
            ):
                process = subprocess.Popen(
                    [
                        chrome,
                        "--headless=new",
                        "--enable-unsafe-webgpu",
                        # Headless Chromium on Linux needs Vulkan for the
                        # real GPU; otherwise it has no adapter or falls
                        # back to SwiftShader on the CPU.
                        "--enable-features=Vulkan",
                        "--use-angle=vulkan",
                        "--no-first-run",
                        "--no-default-browser-check",
                        "--disable-background-timer-throttling",
                        "--disable-renderer-backgrounding",
                        f"--user-data-dir={directory}/profile",
                        f"http://127.0.0.1:{server.server_port}/",
                    ],
                    stdout=log,
                    stderr=log,
                )
                try:
                    done = ready.wait(55)
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                log.seek(0)
                self.assertTrue(done, log.read()[-3000:])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        result = received[0]
        self.assertNotIn("error", result, result.get("error"))
        self.assertGreater(result["stats"]["gpu_tasks"], 0)
        self.assertEqual(
            result["stats"]["gpu_tasks"], result["stats"]["matmul_tasks"]
        )
        np.testing.assert_allclose(
            result["losses"], expected.losses, rtol=2e-5, atol=2e-5
        )
        decoded = RoundResult.from_description(result)
        max_error = max_relative_l2 = 0.0
        strict_update_parity = True
        for name, value in expected.pseudo_gradient.items():
            difference = decoded.pseudo_gradient[name] - value
            max_error = max(max_error, float(np.max(np.abs(difference))))
            relative_l2 = float(
                np.linalg.norm(difference) / max(np.linalg.norm(value), 1e-12)
            )
            max_relative_l2 = max(max_relative_l2, relative_l2)
            # This is a functional training budget, not the existing
            # per-element
            # parity guarantee. Compensated GPU products can perturb
            # theoretically
            # zero key-bias gradients, which Adam amplifies. Report that
            # separately.
            self.assertLess(relative_l2, 0.01, name)
            strict_update_parity &= bool(
                np.allclose(
                    decoded.pseudo_gradient[name], value, rtol=1e-2, atol=1e-6
                )
            )
        print(
            json.dumps(
                dict(
                    diloco_webgpu=True,
                    gpu_tasks=result["stats"]["gpu_tasks"],
                    max_update_error=max_error,
                    max_update_relative_l2=max_relative_l2,
                    per_element_update_parity=strict_update_parity,
                )
            )
        )
        c.submit(decoded)
        c.commit_round()
        self.assertEqual(c.version, 1)


if __name__ == "__main__":
    unittest.main()
