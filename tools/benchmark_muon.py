"""Local hardware-GPU Muon parity and paired task/resident timing benchmark.

Uses the public Tape, compiler and Deno runtime. No rentals or downloads.
Timing includes host momentum/normalization and final updates. Each arm warms
once; repeated pairs alternate order on the same device. This is not training
convergence or a multi-GPU scaling measurement.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ygaraty.compiler.muon import append_muon
from ygaraty.compiler.plan import split
from ygaraty.compiler.shapes import ShapeTape
from ygaraty.runtime.local import execute_plan

RUNNER = r"""
import {ComputeSDK} from 'BROWSER/sdk_compute.mjs';
import {executeLocalPlan} from 'BROWSER/diloco.mjs';
const cases=JSON.parse(await Deno.readTextFile(Deno.args[0]));
const sdk=new ComputeSDK();await sdk.initialize(true);
if(!sdk.device || sdk.adapterInfo?.isFallbackAdapter!==false)throw Error('Hardware GPU required');
const resident=sdk.executeMuonBlock.bind(sdk), rows=[];
try {
  for(const c of cases) {
    const inputs=Object.fromEntries(Object.entries(c.inputs).map(([k,v])=>[k,{shape:v.shape,data:Float64Array.from(v.data)}]));
    const timings={tasks:[],resident:[]};let error=0,blocks=0;
    for(let repeat=-1;repeat<REPEATS;repeat++) {
      for(const arm of (repeat%2===0?['tasks','resident']:['resident','tasks'])) {
        sdk.executeMuonBlock=arm==='tasks'?async()=>null:async(...args)=>{
          const value=await resident(...args);if(value)blocks++;return value;
        };
        const start=performance.now();
        const result=await executeLocalPlan(c.plan,inputs,sdk,{precision:'webgpu'});
        const elapsed=(performance.now()-start)/1000;
        if(repeat>=0)timings[arm].push(elapsed);
        for(const [id,expected] of Object.entries(c.expected)) {
          const actual=result.values.get(id).data;
          actual.forEach((v,i)=>{error=Math.max(error,Math.abs(v-expected[i]));});
        }
      }
    }
    if(error>1e-7)throw Error(`Parity failed ${c.name}: ${error}`);
    if(!blocks)throw Error(`Resident path unused: ${c.name}`);
    const median=xs=>{const a=[...xs].sort((a,b)=>a-b);return (a[Math.floor((a.length-1)/2)]+a[Math.floor(a.length/2)])/2;};
    rows.push({name:c.name,shape:c.shape,steps:c.steps,max_error:error,resident_blocks:blocks,
      seconds:timings,median_seconds:{tasks:median(timings.tasks),resident:median(timings.resident)},
      speedup:median(timings.tasks)/median(timings.resident)});
  }
  console.log(JSON.stringify({kind:'local-muon-resident-benchmark',adapter:sdk.adapterInfo,
    runtime:Deno.version,parity_tolerance:1e-7,warmup_per_arm:1,repeats:REPEATS,rows}));
} finally {sdk.device?.destroy();}
"""


def cases():
    rng = np.random.default_rng(2026)
    specs = [
        ((8, 9), 5, False),
        ((16, 72), 5, False),
        ((16, 144), 5, False),
        ((8, 3), 5, False),
        ((4, 4), 1, False),
        ((1, 5), 10, True),
        ((3, 8), 10, False),
    ]
    result = []
    for shape, steps, zero in specs:
        tape = ShapeTape()
        p = tape.declare("weight", shape, dict(kind="zeros"), True)
        g = tape.declare("gradient", shape, dict(kind="zeros"))
        tape.gradients[p] = g
        updates = append_muon(tape, [p], 0.02, steps=steps)
        arrays = {
            p: rng.normal(size=shape),
            g: np.zeros(shape) if zero else rng.normal(size=shape),
            "muon_m_weight": np.zeros(shape)
            if zero
            else rng.normal(size=shape),
        }
        tasks = split(tape, 512)
        expected = execute_plan(tasks, arrays, list(updates.values()))
        result.append(
            dict(
                name=f"{shape[0]}x{shape[1]}-{steps}steps"
                + ("-zero" if zero else ""),
                shape=shape,
                steps=steps,
                plan=dict(
                    tasks=[t.describe() for t in tasks],
                    loss=updates[p],
                    updates=updates,
                ),
                inputs={
                    k: dict(shape=list(v.shape), data=v.ravel().tolist())
                    for k, v in arrays.items()
                },
                expected={
                    k: expected[k].ravel().tolist() for k in updates.values()
                },
            )
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 50:
        parser.error("repeats must be in 1..50")
    deno = shutil.which("deno")
    if not deno:
        parser.error("Deno is required")
    browser = (
        Path(__file__).resolve().parents[1] / "ygaraty/backends/webgpu/browser"
    )
    with tempfile.TemporaryDirectory(prefix="muon-benchmark-") as directory:
        root = Path(directory)
        fixture = root / "cases.json"
        fixture.write_text(json.dumps(cases()))
        script = root / "benchmark.mjs"
        script.write_text(
            RUNNER.replace("BROWSER", browser.as_uri()).replace(
                "REPEATS", str(args.repeats)
            )
        )
        result = subprocess.run(
            [
                deno,
                "run",
                "--unstable-webgpu",
                f"--allow-read={fixture}",
                str(script),
                str(fixture),
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
    report = json.loads(result.stdout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in report["rows"]:
        print(
            f"{row['name']}: {row['speedup']:.2f}x, max error {row['max_error']:.2g}"
        )


if __name__ == "__main__":
    main()
