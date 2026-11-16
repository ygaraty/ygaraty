// WS-32: residentDecode's order of steps, read-backs, stop rule and rewind, on a fake device whose "GPU" runs
// each submitted copy at once and a fake stage whose step maps its input id to the next token.
import test from "node:test";
import assert from "node:assert/strict";
import { residentDecode } from "../../../ygaraty/backends/webgpu/browser/inference_plan.mjs";

globalThis.GPUBufferUsage ??= { MAP_READ: 1, COPY_DST: 8 };
globalThis.GPUMapMode ??= { READ: 1 };

function fakeBuffer() {
  const data = new Uint32Array(1);
  return { data, mapped: false, async mapAsync() { this.mapped = true; }, getMappedRange() { return data.slice().buffer; },
           unmap() { this.mapped = false; }, destroy() {} };
}
function setup(next) {
  const steps = [];
  const device = {
    createBuffer: () => fakeBuffer(),
    createCommandEncoder: () => { const ops = []; return { copyBufferToBuffer: (a, _o, b) => ops.push([a, b]), finish: () => ops }; },
    queue: { submit: ([ops]) => { for (const [a, b] of ops) { if (b.mapped) throw new Error("used while mapped"); b.data[0] = a.data[0]; } },
             onSubmittedWorkDone: async () => {} },
  };
  const stage = { residentDecode: true, token: fakeBuffer(), ids: fakeBuffer(), position: 10, positions: [10],
    async run({ slot }) { steps.push(this.ids.data[0]); this.token.data[0] = next(this.ids.data[0]); this.position++; this.positions[slot] = this.position; } };
  return { device, stage, steps, gpu: { device, defer: false, profiling: true } };
}

test("n tokens after the first, one step per token consumed, nothing rewound", async () => {
  const { stage, steps, gpu } = setup((t) => t + 1);
  stage.token.data[0] = 5;
  const got = await residentDecode(stage, gpu, 5, 4, 0, () => false);
  assert.deepEqual(got, [6, 7, 8, 9]);
  assert.deepEqual(steps, [5, 6, 7, 8]);  // as the usual loop: run(first) .. run(token n-1)
  assert.equal(stage.position, 14);
  assert.equal(gpu.defer, false); assert.equal(gpu.profiling, true);  // restored
});

test("a stop token ends the answer; the step run past it is rewound", async () => {
  const { stage, steps, gpu } = setup((t) => t + 1);
  stage.token.data[0] = 5;
  const seen = [];
  const got = await residentDecode(stage, gpu, 5, 10, 0, (t) => t === 7, (t) => seen.push(t));
  assert.deepEqual(got, [6, 7]);
  assert.deepEqual(seen, [6, 7]);
  assert.deepEqual(steps, [5, 6, 7]);  // the step on 7 ran ahead of the stop rule ...
  assert.equal(stage.position, 12);     // ... and is undone: positions as if run(5), run(6) only
  assert.equal(stage.positions[0], 12);
});

test("a stopping first token runs nothing; stages with recurrent state are refused", async () => {
  const { stage, steps, gpu } = setup((t) => t + 1);
  assert.deepEqual(await residentDecode(stage, gpu, 3, 5, 0, (t) => t === 3), []);
  assert.deepEqual(steps, []);
  await assert.rejects(residentDecode({ ...stage, residentDecode: false }, gpu, 3, 5, 0, () => false), /recurrent/);
});

// WS-43: a stage with recurrent state (a running sum standing in for Qwen3.5's DeltaNet state) saves it before
// each step after the first and restores it when the step ran past the stop rule.
test("recurrent state: the step run past the stop rule is undone from its snapshot", async () => {
  const { stage, steps, gpu } = setup((t) => t + 1);
  stage.residentDecode = false;
  stage.state = 0; stage.saved = null; stage.snaps = 0;
  stage.snapshot = function (slot) { this.saved = { state: this.state, position: this.positions[slot] }; this.snaps++; };
  stage.restore = function (slot) { this.state = this.saved.state; this.positions[slot] = this.saved.position; this.position = this.saved.position; };
  const run = stage.run;
  stage.run = async function (input) { this.state += this.ids.data[0]; return run.call(this, input); };
  stage.token.data[0] = 5;
  const got = await residentDecode(stage, gpu, 5, 10, 0, (t) => t === 7);
  assert.deepEqual(got, [6, 7]);
  assert.deepEqual(steps, [5, 6, 7]);
  assert.equal(stage.state, 5 + 6);       // the step on 7 is undone: as if run(5), run(6) only
  assert.equal(stage.position, 12);
  assert.equal(stage.snaps, 2);           // before the steps on 6 and 7 (the first step is never undone)
});

test("recurrent state without snapshot and restore is refused", async () => {
  const { stage, gpu } = setup((t) => t + 1);
  await assert.rejects(residentDecode({ ...stage, residentDecode: false }, gpu, 3, 5, 0, () => false), /cannot restore/);
});
