"""Fresh family lowering and browser admission; only metadata and fake I/O."""

import json
import subprocess
import unittest
from copy import deepcopy
from pathlib import Path

from fixtures import cases, stages

from ygaraty.backends.webgpu.distribution import (
    compile_distribution,
    gate_stride,
)
from ygaraty.backends.webgpu.operations import legacy_stage


class PlanTests(unittest.TestCase):
    def test_fresh_source_manifests_match_existing_plan_fixtures(self):
        for case in cases():
            with self.subTest(family=case["family"], config=case["config"]):
                source = stages(case)
                actual = compile_distribution(source)
                if "plan" in case:
                    self.assertEqual(actual, case["plan"])
                self.assertEqual(compile_distribution(source), actual)
                self.assertEqual(actual["version"], 3)
                for stage in source:
                    stage["manifest"]["plan"] = legacy_stage(
                        stage["manifest"]["plan"]
                    )
                self.assertEqual(compile_distribution(source), actual)

    def test_gate_rows_are_padded_and_unpadded_stored_plans_still_admit(self):
        self.assertEqual([gate_stride(v) for v in (16, 32, 48)], [64, 64, 128])
        self.assertEqual(gate_stride(16, aligned=False), 32)
        case = next(c for c in cases() if c["config"].get("hidden") == 1024)
        source = stages(case)  # 0.8B-shaped: 16 value heads
        old = compile_distribution(source, aligned_gates=False)
        self.assertIn('"yStride": 32', json.dumps(old))
        for stage, plan in zip(source, old["stages"]):
            stage["manifest"]["plan"] = plan
        # A bundle prepared before the padding admits and runs the fresh plan.
        fresh = compile_distribution(source)
        self.assertNotEqual(fresh, old)
        self.assertNotIn('"yStride": 32', json.dumps(fresh))
        self.assertEqual(
            [stage["manifest"]["plan"] for stage in source], fresh["stages"]
        )

    def test_invalid_distribution_does_not_partially_update(self):
        for change in ("assignment", "route", "task"):
            source = stages(cases()[0])
            if change == "assignment":
                source[-1]["layers"][1] += 1
            elif change == "route":
                source[-1]["fixture"]["plan_route"] = {
                    "next": 99,
                    "output": "token",
                }
            else:
                compile_distribution(source)
                source[-1]["manifest"]["plan"]["tasks"][0]["attrs"][
                    "owner"
                ] = 99
            before = deepcopy(source)
            with self.assertRaises(ValueError):
                compile_distribution(source)
            self.assertEqual(before, source)

    def test_fresh_plans_admit_in_extracted_browser_executor(self):
        source = []
        for case in cases():
            assignments = stages(case)
            compile_distribution(assignments)
            source.extend(assignments)
        module = (
            Path(__file__).resolve().parents[2]
            / "ygaraty/backends/webgpu/browser/inference_plan.mjs"
        )
        script = """
import {readFileSync} from 'node:fs';
import {strictEqual} from 'node:assert';
const {validatePlan} = await import(process.argv[1]);
for (const {manifest} of JSON.parse(readFileSync(0, 'utf8'))) {
  const plan = validatePlan(
    manifest.plan, manifest, manifest.config.family, manifest.config.hidden
  );
  strictEqual(Object.isFrozen(plan), true);
  strictEqual(plan.submit_token_layers, 64);
}
"""
        result = subprocess.run(
            ["node", "--input-type=module", "-e", script, module.as_uri()],
            input=json.dumps(source),
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
