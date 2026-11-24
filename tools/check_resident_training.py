"""Hardware-GPU numerical and failure gates for the resident Tape executor.

Uses independent NumPy runtime outputs, including strided inputs, loss extremes,
label smoothing, optimizer states and zero norms. No remote machines needed.
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from ygaraty.runtime.kernels import execute
from ygaraty.training.session import implementation_hash

RUNNER = r"""
import {ComputeSDK} from 'BROWSER/sdk_compute.mjs';
import {residentTrainingGPU} from 'BROWSER/resident_training.mjs';
const fixtures=JSON.parse(await Deno.readTextFile(Deno.args[0]));
const sdk=new ComputeSDK();await sdk.initialize(true);
if(!sdk.device || sdk.adapterInfo?.isFallbackAdapter!==false)throw Error('Hardware GPU required');
const unpack=v=>({shape:v.shape,data:Float64Array.from(v.data)});
const request=c=>({weights:{w:{shape:c.shape,data:new Float64Array(c.expected.length)}},moments:{m:{},v:{}},
 steps:[{plan:c.plan,constants:Object.fromEntries(Object.entries(c.constants).map(([k,v])=>[k,unpack(v)]))}]});
const rows=[];
try{
 for(const c of fixtures){
  const result=await residentTrainingGPU(sdk.device,request(c),'compensated');
  if(!result || result.stats.readbacks!==1 || result.stats.cpu_tasks!==0)throw Error('Resident path not used');
  let error=0;
  result.weights.w.data.forEach((x,i)=>{
   error=Math.max(error,Math.abs(x-c.expected[i]));
   if(Math.abs(x-c.expected[i])>1e-10+1e-12*Math.abs(c.expected[i]))throw Error(`${c.name} mismatch ${i}: ${x} vs ${c.expected[i]}`);
  });
  rows.push({name:c.name,max_error:error,stats:result.stats});
 }
 const bad=request(fixtures[0]);
 bad.steps[0].constants.bad={shape:[1,1],data:new Float64Array([-1])};
 bad.steps[0].plan={...bad.steps[0].plan,tasks:[{id:'hidden_nan',op:'inverse_norm',shape:[1,1],inputs:[{value:'bad'}],attrs:{epsilon:1e-7}},...bad.steps[0].plan.tasks]};
 let nonfinite=false;
 try{await residentTrainingGPU(sdk.device,bad,'compensated');}catch(e){if(!String(e).includes('Nonfinite'))throw e;nonfinite=true;}
 if(!nonfinite)throw Error('Hidden nonfinite intermediate was accepted');
 const controller=new AbortController(),cancelled=request(fixtures[0]);
 cancelled.steps=Array.from({length:8},(_,i)=>({...cancelled.steps[0],plan:{...cancelled.steps[0].plan,step:i+1}}));
 cancelled.signal=controller.signal;
 const promise=residentTrainingGPU(sdk.device,cancelled,'compensated');
 setTimeout(()=>controller.abort(),0);
 let aborted=false;try{await promise;}catch(e){if(e.name!=='AbortError')throw e;aborted=true;}
 if(!aborted)throw Error('Cancellation was ignored');
 await residentTrainingGPU(sdk.device,request(fixtures[0]),'compensated');
 sdk.device.destroy();let lost=false;
 try{await residentTrainingGPU(sdk.device,request(fixtures[0]),'compensated');}catch{lost=true;}
 if(!lost)throw Error('Destroyed device was accepted');
 console.log(JSON.stringify({kind:'resident-training-kernel-gates',runtime:Deno.version,adapter:sdk.adapterInfo,
 tolerance:{absolute:1e-10,relative:1e-12},rows,nonfinite_rejected:nonfinite,cancellation_rejected:aborted,reusable_after_failure:true,device_loss_rejected:lost}));
}finally{sdk.device?.destroy();}
"""


def fixtures():
    rng = np.random.default_rng(42)
    cases = []

    def add(name, op, arrays, attrs=None, regions=None):
        attrs = attrs or {}
        regions = regions or [None] * len(arrays)
        inputs = [
            a
            if region is None
            else a[region[0] : region[1], region[2] : region[3]]
            for a, region in zip(arrays, regions)
        ]
        expected = execute(op, inputs, attrs)
        shape = list(expected.shape)
        constants = {
            f"a{i}": dict(shape=list(a.shape), data=a.ravel().tolist())
            for i, a in enumerate(arrays)
        }
        constants["zero_loss"] = dict(shape=[1, 1], data=[0])
        task = dict(
            id="out",
            op=op,
            shape=shape,
            attrs=attrs,
            inputs=[
                dict(value=f"a{i}", region=region)
                for i, region in enumerate(regions)
            ],
        )
        plan = dict(
            step=1,
            parameters=dict(w="weight"),
            state_parameters=dict(m={}, v={}),
            updates=dict(weight="out"),
            loss="zero_loss",
            tasks=[task],
        )
        cases.append(
            dict(
                name=name,
                shape=shape,
                constants=constants,
                plan=plan,
                expected=expected.ravel().tolist(),
            )
        )

    a, b = rng.normal(size=(4, 6)), rng.normal(size=(6, 5))
    add(
        "strided-matmul",
        "matmul",
        [a, b],
        regions=[[1, 4, 1, 5], [1, 5, 1, 4]],
    )
    add("strided-sum", "sum", [a], regions=[[1, 4, 1, 5]])
    add("strided-transpose", "transpose", [a], regions=[[1, 4, 1, 5]])
    add("reduce-rows", "reduce_rows", [a])
    add("squared-norm", "sum_squares", [a])
    add(
        "tiled-assembly",
        "assemble",
        [a, a],
        dict(shape=[4, 4], blocks=[[0, 4, 0, 2], [0, 4, 2, 4]]),
        [[0, 4, 1, 3], [0, 4, 4, 6]],
    )
    add("multi-input-add", "add_many", [a, a / 2, -a / 3])
    for value in [0, 1e-24, 123.45]:
        add(
            f"inverse-norm-{value}",
            "inverse_norm",
            [np.array([[value]])],
            dict(epsilon=1e-7),
        )
    add(
        "zero-gradient-clipping",
        "clip_gradient",
        [a, np.zeros((1, 1))],
        dict(clip_norm=1),
    )
    add(
        "gradient-clipping",
        "clip_gradient",
        [a, np.array([[49.0]])],
        dict(clip_norm=1),
    )
    options = dict(
        beta1=0.9, beta2=0.999, epsilon=1e-8, learning_rate=0.001, step=1024
    )
    add("adam-m", "adam_m", [a, a / 10], options)
    add("adam-v", "adam_v", [a * a, a / 10], options)
    add(
        "adam-update-tiny-variance",
        "adam_update",
        [a, a * 1e-10, a * a * 1e-20],
        options,
    )
    logits = rng.normal(size=(4, 5)) * 10
    for label, x in [("ordinary", logits), ("extreme", logits * 100)]:
        attrs = dict(
            targets=[1, 0, -1, 4], ignore_index=-1, label_smoothing=0.1
        )
        add(f"ce-{label}", "cross_entropy", [x], attrs)
        add(f"ce-backward-{label}", "ce_backward", [x, np.ones((1, 1))], attrs)
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    deno = shutil.which("deno")
    if not deno:
        parser.error("Deno is required")
    browser = (
        Path(__file__).resolve().parents[1] / "ygaraty/backends/webgpu/browser"
    )
    with tempfile.TemporaryDirectory(
        prefix="resident-training-check-"
    ) as directory:
        root = Path(directory)
        data = root / "fixtures.json"
        data.write_text(json.dumps(fixtures()))
        script = root / "check.mjs"
        script.write_text(RUNNER.replace("BROWSER", browser.as_uri()))
        result = subprocess.run(
            [
                deno,
                "run",
                "--unstable-webgpu",
                f"--allow-read={data}",
                str(script),
                str(data),
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode:
            raise RuntimeError(result.stderr[-6000:])
    report = json.loads(result.stdout)
    report["source_sha256"] = implementation_hash()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Passed {len(report['rows'])} numerical cases plus nonfinite, cancellation and device-loss gates"
    )


if __name__ == "__main__":
    main()
