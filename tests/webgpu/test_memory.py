"""Check estimates against the browser allocator without a GPU."""

import copy
import json
import shutil
import subprocess
import unittest
from pathlib import Path

from ygaraty.backends.webgpu.memory import estimate_memory, report_memory

SMALL = dict(  # Qwen3.5-0.8B's layout (9B defaults otherwise)
    layers=24,
    attention_heads=8,
    attention_kv_heads=2,
    linear_k_heads=16,
    linear_v_heads=16,
)


def bundle(workers=2, geometry=None):
    files = {}
    layer_count = (geometry or {}).get("layers", 32)
    value_heads = (geometry or {}).get("linear_v_heads", 32)
    conv = (
        2 * (geometry or {}).get("linear_k_heads", 16) * 128
        + value_heads * 128
    )

    def tensor(name, shape, size, kind="F32"):
        files[name] = {"length": size}
        return dict(
            file=name,
            shape=shape,
            row_bytes=size // shape[0] if len(shape) == 2 else size,
            kind=kind,
        )

    layers = []
    for i in range(layer_count):
        tensors = {"weight": tensor(f"w{i}", [256, 256], 36864, "Q4_K")}
        if i % 4 != 3:
            tensors.update(
                {
                    name: tensor(f"{name}{i}", shape, size)
                    for name, shape, size in [
                        ("ssm_a", [value_heads], 4 * value_heads),
                        ("ssm_dt.bias", [value_heads], 4 * value_heads),
                        ("ssm_conv1d.weight", [conv, 4], 16 * conv),
                    ]
                }
            )
        layers.append(
            dict(
                index=i,
                type="full" if i % 4 == 3 else "linear",
                tensors=tensors,
            )
        )
    embed = tensor("embed", [256, 2560], 537600, "Q6_K") | {"first": 0}
    output = tensor("output", [256, 2560], 537600, "Q6_K") | {"first": 0}
    norm = tensor("norm", [2560], 10240)
    stages = []
    for i in range(workers):
        start = i * layer_count // workers
        end = (i + 1) * layer_count // workers
        m = dict(
            config=dict(
                hidden=2560, intermediate=9216, layers=32, tied_embeddings=True
            )
            | (geometry or {}),
            layers=layers[start:end],
        )
        if i == 0:
            m["embed"] = {"chunks": [embed]}
        if i == workers - 1:
            m["head"] = dict(
                shared_embedding=True, chunks=[output], vocab=256, norm=norm
            )
        stages.append(
            dict(
                layers=[start, end],
                manifest=m,
                fixture=dict(max_tokens=64, max_positions=4096, kv="f16"),
            )
        )
    return dict(stages=stages, files=files)


class MemoryTests(unittest.TestCase):
    @unittest.skipUnless(
        shutil.which("node"), "Node required for browser allocation oracle"
    )
    def test_matches_browser_allocations_for_workers_context_and_slots(self):
        module = (
            Path(__file__).parents[2]
            / "ygaraty/backends/webgpu/browser/qwen35_gpu.mjs"
        ).as_uri()
        script = """import {Stage} from MODULE;
import {readFileSync} from 'node:fs';
const {data,context,slots}=JSON.parse(readFileSync(0,'utf8'));
const totals=[];
for(const s of data.stages){
 const gpu={allocated:0,repack:[],buffer(n){
   n=Math.max(16,Math.ceil(n/16)*16);this.allocated+=n;return {size:n}
 },upload(d){return this.buffer(d.byteLength)}};
 const stage=new Stage(gpu,s.manifest,s.fixture.max_tokens,
   context,{kv:s.fixture.kv,slots});
 await stage.load(async name=>{const n=data.files[name].length;
   return n<=256?new ArrayBuffer(n):{byteLength:n}});
 totals.push(gpu.allocated);
}
console.log(JSON.stringify(totals));""".replace("MODULE", json.dumps(module))
        cases = [
            (workers, context, slots, geometry)
            for geometry in (None, SMALL)
            for workers in range(1, 5)
            for context, slots in [(4096, 1), (2048, 3)]
        ]
        for workers, context, slots, geometry in cases:
            with self.subTest(
                workers=workers,
                context=context,
                slots=slots,
                small=geometry is not None,
            ):
                data = bundle(workers, geometry)
                actual = json.loads(
                    subprocess.check_output(
                        ["node", "--input-type=module", "-e", script],
                        input=json.dumps(
                            dict(data=data, context=context, slots=slots)
                        ).encode(),
                    )
                )
                self.assertEqual(
                    actual,
                    [
                        r["allocated"]
                        for r in estimate_memory(data, context, slots)
                    ],
                )

    def test_smaller_heads_need_less_memory(self):
        nine, small = (
            estimate_memory(bundle(1)),
            estimate_memory(bundle(1, SMALL)),
        )
        self.assertLess(small[0]["allocated"], nine[0]["allocated"])

    def test_tied_embeddings_count_once(self):
        data = bundle(1)
        tied = estimate_memory(data)[0]["weights"]
        data["stages"][0]["manifest"]["head"]["shared_embedding"] = False
        self.assertEqual(
            estimate_memory(data)[0]["weights"] - tied,
            data["files"]["output"]["length"],
        )

    def test_context_and_slots_change_memory(self):
        data = bundle()
        base = estimate_memory(data, 2048, 1)[0]
        self.assertGreater(
            estimate_memory(data, 4096, 1)[0]["recommended"],
            base["recommended"],
        )
        self.assertGreater(
            estimate_memory(data, 2048, 2)[0]["recommended"],
            base["recommended"],
        )
        with self.assertRaises(ValueError):
            estimate_memory(data, 8192)
        with self.assertRaises(ValueError):
            report_memory(data, 3)
        large = copy.deepcopy(data)
        large["files"]["w0"]["length"] = 4 * 1024**3
        with self.assertRaisesRegex(ValueError, "below"):
            report_memory(large, 2, memory_gb=1)
