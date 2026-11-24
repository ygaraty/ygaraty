# Open research

Help make heterogeneous compute easier to use. This directory shares our
experiments, reusable reference code, measured results and limitations so anyone
can reproduce, challenge and extend the work. Negative results and small
experiments are welcome; see [contributing](../CONTRIBUTING.md) for the lightweight
report structure. Historical reports retain their original IDs and stated limits.

Our training research focuses on DiLoCo and our own distributed compute compiler
and execution engine. This is work in progress; the [public experiments below](#reports)
record what we have tried, measured and still need to verify.

Start with the [user guide](../docs/README.md) for supported current
commands. Reports retain historical commands, links and infrastructure names;
they are evidence, not a promise that each old runner works in this release.
The compiler, training recipes and browser kernels are included in the public
package; see [Python API and training](../docs/python-api.md). Legacy hosted
application integrations and production operations notes have been archived
outside this repository. Reusable references and benchmark helpers remain public;
see [research APIs](src/ygaraty_webgpu_research/README.md).

## Reports

Browse by topic; report IDs and filenames remain stable. Each report appears once
under its main topic. Read its status and limitations: this list includes proposals,
negative results and completed experiments, not just validated features.

- [DiLoCo training](#diloco-training)
- [Compiler and numerical correctness](#compiler-and-numerical-correctness)
- [Distributed engine and scheduling](#distributed-engine-and-scheduling)
- [Inference and model support](#inference-and-model-support)
- [WebGPU kernels and runtime performance](#webgpu-kernels-and-runtime-performance)
- [Training methods and model research](#training-methods-and-model-research)

Historical host addresses use documentation-only placeholders; personal paths
and private deployment details are omitted. Hardware, topology and measurements
are retained. Cloud marketplace operations are outside this research collection.

### DiLoCo training

- [Muon DiLoCo: one/four physical GPUs and a same-ConvNet upstream PyTorch reference](muon-diloco-vast.md)

- [Muon versus Adam: two-seed local one/three-worker learning and time-to-loss comparison](muon-diloco-local.md)

- [RunPod native preflight: RTX 4090 parity passed; four-GPU comparison blocked by capacity](runpod-native-preflight.md)

- [Native wgpu peer tree: binary transport and 1/2/4/10-process hardware-GPU checks](native-peer-tree.md)

- [Tree gradient reduction: matched global batches preserve learning quality, with measured same-GPU overhead](tree-gradient-reduction.md)

- [One versus three browser workers: measured Fashion-MNIST learning curves and time to loss](convnet-one-vs-three.md)

- [Browser DiLoCo: public training service, recovery and one/two-contributor WebGPU checks](browser-diloco-recovery.md)

Local steps, replica averaging, communication precision and learning across machines. Start with experiments 040, 043 and 057.

- [040: DiLoCo over HTTP tracks the PyTorch reference to within 0.004](040-diloco-over-http.md)
- [043: DiLoCo across 8 machines: CIFAR-10 trained over HTTP, identical to a single host and 7.7 times sooner to its best accuracy than one machine](043-diloco-cifar-across-machines.md)
- [044: The CIFAR ConvNet on full data, one host: 4 DiLoCo replicas end where plain Adam does, 8 fall short in 10 epochs](044-diloco-convnet-cifar-one-host.md)
- [045: Experiment 45: ConvNet remote scaling — WebGPU versus native PyTorch](045-convnet-remote-scaling.md)
- [052: Does the float32 DiLoCo wire shorten the round without costing accuracy? 27% faster on home links, but 0.6 points lower in all three seeds](052-diloco-float32-wire-matched-seeds.md)
- [057: Float32 for the pseudo-gradients only: 23% less time on home links and no accuracy loss, now the default](057-diloco-upload-only-float32.md)
- [DO-12: Hybrid islands: sharded islands averaged rarely, against one synchronous pool, in time to loss](DO-12.md)
- [DO-13: Eager outer updates: hide the islands' averaging behind the next inner steps](DO-13.md)
- [DO-1T: Do float32 uploads shorten Transformer DiLoCo rounds without accuracy loss?](DO-1T.md)
- [DO-3: Does more frequent sync (H) recover 8-replica ConvNet learning efficiency?](DO-3.md)

### Compiler and numerical correctness

Graph transformations, task construction, numerical parity and compiled plans. Start with 020, 030 and DO-47. Source: [compiler](../ygaraty/compiler/README.md) and [analysis](../ygaraty/analysis/README.md).

- [002: Larger-network tile, cache and grouping sweep](002-large-mlp-tile-sweep.md)
- [012: English → Brazilian Portuguese Transformer: one-minibatch parity](012-transformer-one-minibatch-parity.md)
- [014: Browser precision comparison: WebGPU versus float64](014-webgpu-vs-float64-precision.md)
- [015: WebGPU precision fix: compensated multiplication and accumulation](015-webgpu-compensated-precision-fix.md)
- [016: Repeated browser precision benchmark (Apple M1)](016-repeated-browser-precision-benchmark.md)
- [017: Full attention-LSTM translation: local parity](017-attention-lstm-translation-parity.md)
- [019: Regression check after the package reorganization](019-package-reorganization-check.md)
- [020: PyTorch parity baseline across seven executors, including real WebGPU](020-parity-baseline-webgpu.md)
- [021: Tile size: kernel speed versus whole-update speed](021-tile-size-sweep.md)
- [030: A Transformer with fused layer tasks: the model built for the round-trip rule](030-transformer-fused-layers.md)
- [032: The output layer split by vocabulary: owners for the tied embedding](032-owned-vocabulary.md)
- [034: bf16 activations on the wire, the embedding fused and the final norms folded: the simulator's first two predictions, built](034-bf16-activations-fused-embedding.md)
- [054: Dropping the tasks nobody reads: 24% fewer tasks, 4% less time per update on the tiny Transformer, on loopback and on home links alike](054-prune-unread-tasks-tiny-transformer.md)
- [072: Owners' weight gradients off the chain don't shorten the update: +1.4% at band 16, −2.1% at band 64](072-split-backward.md)
- [076: Parallel layers 2–3 on both sides remove 4 of 12 hops for 0.025 of held-out loss](076-parallel-layers.md)
- [078: Parallel layers 1–2 shorten the distributed update 9.5%, just short of the gate, and learn better](078-parallel-layers-graph.md)
- [DO-37: Does the Torch-free ConvNet preserve gradients and learning?](DO-37.md)
- [DO-4: Does the pinned MLP tile-epilogue bundle pass numerical validation on a server?](DO-4.md)
- [DO-47: Compiled inference plans preserve current outputs, memory and latency](DO-47.md)
- [OFF-2: Does the MLP tile-epilogue rewrite still gain under pessimistic overhead?](OFF-2.md)

### Distributed engine and scheduling

- [Native peer-tree training on one versus four physical GPUs: close learning quality, slower execution over internet links](vast-native-one-vs-four.md)

Placement, memory, storage, transport, relays, handoffs and recovery across heterogeneous workers. Source: [runtime](../ygaraty/runtime/README.md).

- [001: Caching and task grouping on the tiny MLP](001-zeromq-cache-grouping-tiny-mlp.md)
- [003: Redis-backed, memory-bounded training path](003-redis-memory-bounded-training.md)
- [004: Streaming import and verification under 128 MiB container limits (MLP)](004-streaming-capacity-mlp-128mib.md)
- [007: PostgreSQL tensor storage and Redis/PostgreSQL comparison](007-storage-backend-comparison.md)
- [008: PostgreSQL larger than the database memory cap](008-postgres-out-of-cache-capacity.md)
- [009: Reducing storage traffic while preserving exact results](009-storage-traffic-reduction.md)
- [011: First HTTP and real-browser contributor runs](011-http-browser-path-validation.md)
- [018: Tiny LSTM: 100 distributed updates with worker and coordinator failures](018-tiny-lstm-100-updates-recovery.md)
- [023: Where does the coordinator's time go?](023-coordinator-overhead.md)
- [025: Surviving a long run: contributors leaving and a coordinator restart](025-long-run-churn-restart.md)
- [026: Where does the HTTP path's time go?](026-http-path-profile.md)
- [027: Several LSTM steps per task: does cutting the chain's hops shorten an update?](027-chain-steps-per-task.md)
- [028: A block relay: taking the coordinator out of the data path](028-block-relay.md)
- [029: Round trips on the critical path: network latency, and three changes that remove them](029-round-trips.md)
- [031: Owned layers on emulated home links: what a volunteer's connection does to an update](031-owned-layers-home-links.md)
- [033: A simulator of the update: the real plan and scheduler on modeled links](033-update-simulator.md)
- [036: Eager claims and the embedding gradient at its owners: 17.6 s per update with 16 workers](036-eager-claims-embedding-tail.md)
- [038: What the next changes could buy: placement on mixed pools, wider models, 8-bit activations (simulated)](038-what-next-changes-buy.md)
- [039: Capacity placement on a mixed pool over DigitalOcean: 25% faster with 6 fast uplinks among 16](039-capacity-placement-mixed-pool.md)
- [042: One layer per fast worker beats pairs on real links: 7 to 16% faster, where the simulator said pairs pay](042-spread-layers-over-fast-workers.md)
- [047: The sharded factorization over HTTP: 1, 2 and 4 worker cores on loopback and on home links](047-factorization-over-http-cores-links.md)
- [049: Does gzip on the wire pay for the Transformer on home links? Yes on the weak workers, 4% faster; no on the strong ones](049-wire-gzip-transformer-digitalocean.md)
- [050: Browser workers on the block relay](050-browser-workers-on-the-relay.md)
- [051: Where does a hop go on home links? In bytes, not round trips; sending small results inline saves at most 4%](051-where-a-hop-goes-on-home-links.md)
- [053: With 2 fast workers among 16, pairs and one layer each tie: a 20 Mb/s worker owning a decoder layer paces both](053-two-strong-workers-pairs-or-one-each.md)
- [056: Batched relay calls: browser workers on the relay go from 18% slower to 21% faster than direct on home links](056-batched-relay-calls-on-home-links.md)
- [058: Handoffs: with the coordinator off the hop, the tiny Transformer's update on home links is 20% shorter](058-coordinator-off-the-hop.md)
- [059: Non-blocking handoffs: late handoffs were workers stuck on an earlier one; removing that gives another 6%](059-non-blocking-handoffs.md)
- [060: Handoff reads by region: the extra download is gone; the lost cache hits never cost anything](060-handoff-reads-by-region.md)
- [061: Handoffs on the 30.3M Transformer: 22.5% shorter updates on DigitalOcean home links, now the default](061-handoffs-30m-digitalocean.md)
- [062: Pairs of layers lose 10%, but not to upload slots: the queue is after the upload](062-pair-holder-upload-slots.md)
- [063: With handoffs on, pairs of layers lose 3.9% instead of 10.3%](063-pairs-penalty-with-handoffs.md)
- [064: Removing the result-post queue recovers only 17% of the pairs penalty; the claim path is what is left](064-concurrent-result-posts.md)
- [065: 20 Mb/s workers get their full link rate; each upload costs one round trip more, nothing else](065-weak-uplink-delivered-rate.md)
- [071: Corrected handoff timing gets experiment 59 right but still overstates region reads: 4.3 points off on experiment 60](071-handoff-timing-holdouts.md)
- [089: Pricing region reads as the relay serves them cuts the simulator's handoff error to 3.1 points, just short of 3](089-handoff-region-reads-as-blocks.md)
- [097: Direct WebRTC links between Qwen WebGPU stages cut time per token 61% against the coordinator relay](097-qwen35-webrtc-direct-links.md)
- [098: User sessions over direct and private WebRTC links keep the relay's tokens, but are only 11% faster between Poland and Sweden](098-user-sessions-webrtc-routes.md)
- [099: With close stages, direct links cut user-session time per token 60%; one private session in eight failed to link](099-user-sessions-webrtc-close-stages.md)
- [151: Each chat decode step costs two stage-link round trips, not one: 91 ms + 2.03 × RTT per token, which explains DO-50 round 2](151-turn-rtt-decode-local.md)
- [153: Sending the start of each link message in small pieces removes the extra round trip per decode step: 583 to 341 ms per token at 243 ms, tokens identical](153-link-pieces-local.md)
- [DO-11: GPU owners: how much raw compute does the pool gain, and when does it show?](DO-11.md)
- [DO-2: Does the batched relay scale over 1, 2 and 4 physical WebGPU workers and group sizes?](DO-2.md)
- [DO-24: Does adaptive stage grouping beat fixed placement on small mixed fleets?](DO-24.md)
- [DO-54: Does WS-26's piece schedule remove the extra round trip on a real transatlantic relayed link?](DO-54.md)
- [DO-58: KV slices on memory contributors: exact long context by adding contributors, not compute](DO-58.md)
- [DO-59: Moving a live session to another contributor: faster than replaying the conversation?](DO-59.md)
- [DO-62: Do parallel piece downloads with a stall timeout make a contributor's cold load fast and unstallable?](DO-62.md)
- [DO-72: Can a one-worker Vast VM serve WebGPU over an isolated overlay?](DO-72.md)
- [LOCAL-2: Diagnose the remaining scheduling opportunity](LOCAL-2.md)
- [OFF-3: Are the consumers of a pair holder's results busy when they claim?](OFF-3.md)
- [OFF-4: What is left of the pairs penalty with handoffs on?](OFF-4.md)
- [OFF-7: Lossless coding of bf16 activations: how much smaller than gzip, at what cost?](OFF-7.md)
- [OFF-8: Spreading one layer over several owners: what does it cost on the chain?](OFF-8.md)
- [OFF-9: With handoffs, should strong workers own contiguous layers again? An interval planner priced by the simulator](OFF-9.md)

### Inference and model support

Model bring-up, quantization, sessions, context, concurrent chats and end-to-end correctness. Current validated product behavior: [TESTED.md](../TESTED.md).

- [079: Phi-4 runs on Ygaraty locally: kernels, KV cache and the two-worker loop give the reference's tokens](079-phi4-local-engine.md)
- [080: Real Phi-4 matches transformers on the workstation CPU, layer by layer and token by token](080-phi4-real-layers-cpu.md)
- [081: Phi-4 split over two H100 Droplets gives the reference's tokens at 44 ms per token](081-phi4-two-h100-pipeline.md)
- [082: 4-bit Phi-4 runs on two 8 GB GPUs across the internet, at 1.35 s per token](082-phi4-int4-vast-8gb.md)
- [083: 4-bit TwIL-8B runs on four GPUs capped at 3.5 GB, at 0.95 s per token, but runs out of memory near 30k tokens](083-twil-int4-four-capped-gpus.md)
- [084: Persistent sessions halve Phi-4's time between tokens on emulated links, with the same tokens](084-persistent-phi4-sessions.md)
- [085: TwIL-8B in bf16 on one RTX 3090 runs at 50 ms per token, exactly as transformers on that GPU; one prompt departs from the CPU reference](085-twil-bf16-one-3090.md)
- [086: Splitting Phi-4 over two owners on one A100 adds 1–3 ms per token, and persistent sessions win it back](086-phi4-split-one-a100.md)
- [087: Over close internet links, persistent sessions cut TwIL-8B's time between tokens 18%, short of the 30% bar](087-persistent-sessions-internet.md)
- [092: Qwen3.5-9B Q4 runs on four rented WebGPU contributors: 580 ms per token, first token after 37 s at 8,192 tokens](092-qwen35-webgpu-four-contributors.md)
- [093: Eight interleaved conversations get 9.3x the Qwen WebGPU pipeline's throughput on the same four GPUs, every token unchanged](093-qwen35-interleaved-conversations.md)
- [094: With the key/value cache preallocated, 4-bit TwIL-8B serves a 30,000-token session on four 3.5 GB-capped GPUs](094-twil-int4-30k-fits-capped-gpus.md)
- [095: Two 16-layer stages cut the Qwen WebGPU pipeline's time per token 57%, from 580 to 250 ms, same tokens](095-qwen35-two-stages.md)
- [096: Two two-stage Qwen copies on one coordinator give 1.86× one copy's throughput at the same speed per user](096-qwen35-copies-on-one-coordinator.md)
- [100: Fast (direct) rollout check did not complete: the relay baseline hung on a Poland–Germany pair, while Fast answered 7 of 8 sessions](100-fast-sessions-rollout-check.md)
- [101: TwIL-8B runs on the new dense WebGPU family with reference parity and serves a user-panel chat](101-twil-8b-dense-webgpu-family.md)
- [102: The 3- and 4-contributor plans for TwIL-8B and Qwen3.5-9B keep parity, fit their memory and serve](102-three-four-worker-plans.md)
- [103: Fast (direct) passes the rollout gate: same answers, links open at go, 63% less time per token than the relay](103-fast-sessions-rollout-rerun.md)
- [106: Phi-4 runs on the dense WebGPU family with reference parity and serves a panel chat](106-phi4-dense-webgpu-family.md)
- [107: A faster draft back-off brings summaries to +4.6% but costs edits a third of their gain; every logit identical in 27 runs](107-speculative-backoff-rerun.md)
- [109: Cost-aware drafting cuts edits 64% and a held-out code edit 73%, bit for bit, but still loses 15% on a summary](109-cost-aware-drafting.md)
- [110: On two RTX 3060s with direct links, the rows-per-thread kernel raises eight-conversation throughput 1.86×, tokens unchanged](110-batched-decode-rented.md)
- [111: Cautious cost-aware drafting keeps edits 61% and code 69% faster, bit for bit, and stops losing on summaries](111-cautious-cost-aware-drafting.md)
- [112: On rented GPUs, exactly checked speculative decoding cuts edits 65% and code 73%, bit for bit; a summary ran 4% slower without drafting](112-speculative-decoding-rented.md)
- [113: One rented A100 80 GB serves all of Kolibri-1 as a single WebGPU contributor, reference-exact, at 125–174 ms per token](113-kolibri-single-a100-contributor.md)
- [133: Chat sessions recover from a cancelled turn and a lost contributor with the resident loop on, with the same tokens](133-resident-recovery.md)
- [148: Many and long chat turns run without a stall on the workstation once a second-turn stall is fixed: 135 of 135 turns](148-chat-turns-local-screen.md)
- [149: 2048-token replies and a full 16,384-position conversation run without a stall on the workstation: 32 of 32 turns, capacity reset completes](149-long-replies-full-context-local.md)
- [150: One S5-nano contributor serves both relay and direct chat on the workstation: 60 of 60 turns, 2.91 GB allocated](150-one-worker-chat-local.md)
- [152: Greedy WebGPU decode is bitwise repeatable run to run: 5 of 5 runs identical in tokens and logits, with either matvec kernel](152-decode-repeatability-local.md)
- [154: Chat turns are bitwise repeatable; WS-23's "mismatches" were the test driver hashing the page's timing line](154-chat-repeatability-local.md)
- [155: A chat session recovers from a cancelled turn, a killed contributor and a broken direct link, with turns identical to an undisturbed session](155-chat-recovery-local.md)
- [DO-15: Phi-4 on two RunPod RTX 4000 Ada pods, in different data centers](DO-15.md)
- [DO-20: Does Q4 make small-GPU Phi-4 pipelines economically useful?](DO-20.md)
- [DO-21: Does staged batching increase useful concurrent Phi-4 capacity?](DO-21.md)
- [DO-22: Where does Ygaraty outperform exo or Helix under matched constraints?](DO-22.md)
- [DO-23: Can WebGPU owners execute real Phi-4 layers correctly and usefully?](DO-23.md)
- [DO-28: Smaller prefill crossings: does a bf16 wire plus byte-split coding cut Phi-4's time to first token?](DO-28.md)
- [DO-29: KV cache in CPU memory: longer prompts or more sessions on 8 GB GPUs, at what cost per token?](DO-29.md)
- [DO-30: Can Ygaraty serve Phi-4 without a PyTorch runtime dependency?](DO-30.md)
- [DO-34: Persistent sessions on real GPUs in one VPC: TwIL-8B on two Vultr A16s](DO-34.md)
- [DO-42: S5-nano Q4 WebGPU: parity, memory and distribution cost](DO-42.md)
- [DO-48: Kolibri WebGPU inference within contributor-selected resource commitments](DO-48.md)
- [DO-50: Do repeated chat cycles stall despite connected contributors?](DO-50.md)
- [DO-51: Can one 8 GiB S5-nano worker serve relay and direct chat reliably?](DO-51.md)
- [DO-52: Can Qwen3.8-27B reuse the Qwen WebGPU backend with Q4 parity and bounded memory?](DO-52.md)
- [DO-56: A fixed window for Qwen3.5's full-attention layers: how much quality does a capped KV cache cost?](DO-56.md)
- [DO-57: A quantized KV cache for Qwen3.5's full-attention layers: 2–4× the context in the same memory?](DO-57.md)
- [DO-60: Conversations per 4 GB contributor at long context, with fixed memory per conversation](DO-60.md)
- [DO-61: Does keeping a one-contributor session's token loop on the contributor bring production decode to its on-machine time?](DO-61.md)
- [DO-80: Qwen3-Coder-30B-A3B through shared kernels and compiled plans](DO-80.md)
- [DO-85: Do batched token reports let a one-contributor production chat stream at the GPU's speed instead of one token per round trip?](DO-85.md)
- [DO-86: Does exact n-gram drafting speed up one-contributor production chats on code?](DO-86.md)
- [DO-87: Does the resident token loop speed up a production chat when only one chat is active?](DO-87.md)
- [OFF-10: What workers need to serve a session at 4k, 8k and 16k context](OFF-10.md)
- [WS-11: Does 4-bit Phi-4 (`--weights int4`) keep the reference's tokens?](WS-11.md)
- [WS-29: How soon can a contributor that leaves or crashes mid-load be replaced in a panel session?](WS-29.md)

### WebGPU kernels and runtime performance

GPU kernels, browser and native runtimes, profiling and hardware comparisons.

- [088: Four owner-kernel changes make a Phi-4 decode step 43–50% faster on an A100, with the same tokens](088-phi4-owner-kernels.md)
- [104: Rows per thread make 8 decode rows cost 1.3–1.7× one, bit for bit, but no single variant passes: R = 4 slows single tokens](104-matvec-rows-per-thread.md)
- [105: Rows per thread by token count halve edit time with speculative decoding, bit for bit, but plain decode gains only 4% and summaries lose 11%](105-matvec-policy-in-stages.md)
- [108: The rows-per-thread kernel doubles batched throughput: eight conversations at 19.4 tokens/s against 9.8, tokens unchanged](108-batched-decode-rows-per-thread.md)
- [114: On one A100, a Kolibri-1 token's GPU work is 28–38 ms of about 100 ms; merging expert chunks saves 9.5 ms, bitwise identical](114-kolibri-a100-decode-profile.md)
- [115: Per operation on one A100, Kolibri-1's GPU time is Q6_K matvecs 29%, attention 24%, norms 13%; variant timings confounded by context growth](115-kolibri-a100-operation-bench.md)
- [116: Caching plan checks and bind groups halves a Kolibri-1 token on one A100, 101 → 52 ms, bitwise identical](116-webgpu-host-caches.md)
- [117: Fusing sandwich norms with their residual adds and the QK-norm pair cuts a Kolibri-1 token 52 → 47 ms on one A100, bitwise identical](117-webgpu-fused-norms.md)
- [118: On the Radeon 680M our Qwen3.5-9B decode runs at 65% of native llama.cpp and WebLLM at 72%; per byte of weights the two browsers tie](118-webgpu-gap-webllm-llamacpp-680m.md)
- [119: On the 680M, matrix-vector products are 94% of decode GPU time and run at 71% of the copy roof; attention stays under 3 ms at 4k](119-webgpu-roofline-680m.md)
- [120: Keeping the token on the GPU halves a Kolibri-1 decode step on an A100 at short context, 73 → 35 ms, same tokens](120-webgpu-resident-decode-a100.md)
- [121: A coalesced one-token matrix-vector kernel reads K-quants at 81–99% of the 680M's copy roof; Qwen3.5-9B decode 23% faster, reference-exact](121-coalesced-matvec-kernel.md)
- [122: Smaller attention splits make Kolibri-1 decode 3.6× faster at 1k context on an A100, but the long-prompt reference check fails in both arms](122-attention-decode-splits-a100.md)
- [123: A100 WebGPU and WebLLM results (experiment 123, public projection)](123-four-engines-a100.md)
- [124: Kolibri-1's long-prompt gap is f16 cache rounding flipping near-tied experts, not a window or chunking bug](124-kolibri-long-prompt-drift.md)
- [125: A batched matvec kernel doubles two to four conversations, exactly, but both batch paths stall near 107 tokens per second](125-batched-matvec-a100.md)
- [126: A 32-conversation decode step is arithmetic-bound in the matrix products, not in per-conversation work](126-batch-step-profile-680m.md)
- [127: The prefill GEMM decodes 32 conversations 1.34× faster than the batched matvec, but loses at 8 and 16](127-batch-kernels-680m.md)
- [128: Paid contributors can run our engine without a browser: Node with Dawn is exact and 6% faster than Chrome; wgpu offers f16 but is 2× slower today](128-native-webgpu-a100.md)
- [129: Keeping the token on the GPU now works for Qwen3.5's recurrent layers: 6.5% faster on the 680M, tokens identical](129-resident-decode-qwen35.md)
- [130: A vectorized Q8_0 decoder makes Qwen3.5's small Q8_0 products 2.3× faster on the 680M, short of the 3× gate](130-q8-decoder-680m.md)
- [131: Deno with wgpu loses its 37 ms per step waiting for completions: about 11 ms per mapAsync and per wait, a polling delay](131-wgpu-overhead-a100.md)
- [132: Used for decode-sized calls only, the Q8_0 decoder is 1.2–2.8× faster and never slower: now the default](132-q8-decode-only-680m.md)
- [134: Shapes picked by a single-matrix sweep make A100 decode 18.5% slower: the microbenchmark measured L2-cached reads](134-matvec-retune-a100.md)
- [135: Products that read the same input now share one dispatch: 29% fewer launches per token, tokens identical, no loss on the 680M](135-matvec-group-680m.md)
- [136: Grouped one-token products shorten Qwen3.5-9B's A100 decode step 11.5%, tokens identical: now the default](136-matvec-group-a100.md)
- [137: Folding the RMS norm into the grouped product is exact but slower: +6.5% on an A100, +17% on the 680M](137-norm-fold.md)
- [138: Several rows per thread make the one-token products slower on an A100 (+17% with 2, +65% with 4), tokens identical](138-matvec-rows-a100.md)
- [139: Grouped one-token products shorten Kolibri-1's A100 step 4.2%, tokens identical: kept opt-in](139-matvec-group-kolibri-a100.md)
- [140: On an A100 the one-token products are decoding-bound: the same reads without decoding run 1.71× faster](140-decode-roof-a100.md)
- [141: Unpacking Q4_K's scales at load cuts the A100 decode step 16.7% despite a third more bytes, tokens identical](141-q4k-repack-a100.md)
- [142: Repacking Q5_K and Q6_K as well cuts the A100 step another 7.8% (73 tokens per second), tokens identical](142-k-repack-a100.md)
- [143: The exact repack does not speed up Kolibri-1 on an A100 (−0.7%): its step is not bound by the dense products](143-kolibri-repack-a100.md)
- [144: Measuring the repack at load picks it correctly on the 680M but not reliably on an A100: the measurement is too short there](144-repack-calibration.md)
- [145: Caching shader sources cuts host encoding only 10% on Kolibri (24–30% on Qwen3-Coder and Qwen3.5): JavaScript is not most of the host time](145-wgsl-memo.md)
- [146: On an A100 host Kolibri-1's resident decode step is GPU-bound (33.1 ms against 32.5 of GPU work): host time is already hidden](146-kolibri-host-breakdown-a100.md)
- [147: Neither vec4 staging, larger register tiles nor repacked weights lift the batch GEMM much: +9% at 32 conversations on an A100](147-batch-gemm.md)
- [DO-53: Where does a Qwen WebGPU decode step spend its time? (kernel-family breakdown)](DO-53.md)
- [DO-70: Is WebGPU's 725 GB/s copy limit on an A100 a browser limit or a setup limit?](DO-70.md)
- [DO_webgpu_speed: WebGPU worker speed: closing the gap with WebLLM and native](DO_webgpu_speed.md)
- [WS-37: Does one gathered dispatch for the top-k experts make Kolibri's expert path bandwidth-bound?](WS-37.md)
- [WS-38: Do subgroup GEMM kernels cut prefill time (first token) by half?](WS-38.md)
- [WS-39: How does ExecuTorch's WebGPU backend compare with our kernels on the same GPU and model?](WS-39.md)
- [WS-40: Does repacking K-quants into a simple 4-bit groupwise layout cut bytes per token and decode time without losing accuracy?](WS-40.md)
- [WS-42: Can 64-key decode attention splits become the default, checked on short prompts and against the old splits on long ones?](WS-42.md)

### Training methods and model research

Optimizers, architectures, factorization and learning efficiency beyond the DiLoCo protocol. Source: [training](../ygaraty/training/README.md).

- [005: Single-layer LSTM with short windows under 128 MiB](005-lstm-short-window-capacity.md)
- [006: Byte-level language model with embeddings and Adam under 128 MiB](006-byte-language-model-adam.md)
- [010: SQL-fed attention LSTM in two browser tabs, eight updates](010-sql-browser-lstm-eight-updates.md)
- [013: Transformer minibatch computed by two browser tabs](013-transformer-browser-job.md)
- [022: How much training does a useful translator need?](022-translation-quality-reference.md)
- [024: The experiment-22 recipe, trained by Ygaraty](024-distributed-translation-real-data.md)
- [035: Batch 128 against 64 at equal pairs: twice the pairs per update, little more progress per update](035-batch-128-equal-pairs.md)
- [037: Batches by length, and nanochat training time as the KPI: 1.4 times faster to the same loss](037-batches-by-length.md)
- [041: Wider models learn no faster here: width 512 reaches the same loss 1.5 to 2.2 times sooner](041-wider-models-learning.md)
- [046: A factorization larger than any machine's memory](046-factorization-larger-than-memory.md)
- [048: A protein-space factorization on WebGPU workers with a memory guarantee](048-factorization-protein-map-webgpu.md)
- [055: N-gram hints: the translator reaches the same held-out loss with about 35% fewer pairs](055-ngram-hints-translation.md)
- [066: Growing the ConvNet from the head stalls at 33%; channel-copy layers are 3× slower even at full depth](066-progressive-convnet-growth.md)
- [067: Muon learns far more per pair, but its optimizer step makes CPU updates 1.77× slower](067-muon-owned-layers.md)
- [068: Cheaper Muon doesn't help: lower rates and fewer Newton–Schulz steps don't beat 80% of Adam's time](068-cheaper-muon.md)
- [069: Raising Muon's rate at 1–2 Newton–Schulz steps makes it worse: update size wasn't the problem](069-muon-fewer-steps-higher-rate.md)
- [070: A batch ramp saves about 5% of wall time, not 25–45%: no bigger batch reaches the 3,072-token loss by 131,072 pairs](070-batch-ramp.md)
- [073: N-gram hints make the pool's updates 58% slower, which cancels their learning gain](073-hints-timing-pool.md)
- [074: Muon's owner step adds 0.28 s to the pool's update, small enough to keep its gain](074-owned-muon-step.md)
- [075: An open teacher's translations as targets don't help a student trained from scratch at 32,768 pairs](075-teacher-targets.md)
- [077: Owners one update ahead lose 0.17 of held-out loss even with a look-ahead correction](077-stale-layer-weights.md)
- [090: Parallel layers 1–2 shorten the 30.3M update on the pool by 5.8%, short of 10%](090-parallel-layers-pool.md)
- [091: On the pool, Muon reaches Adam's final held-out loss in 46% of Adam's training time](091-muon-pool-time-to-loss.md)
- [DO-27: Fine-tuning Phi-4 over the pipeline: LoRA adapters on their owners, exact against PyTorch?](DO-27.md)
- [DO-5: Do n-gram hints cut training time on the distributed 30.3M Transformer?](DO-5.md)
- [M-1: Can learning proceed through useful successive approximations?](M-1.md)
- [M-3: Factorized vocabulary layer: how much loss does a rank-r embedding cost?](M-3.md)
- [M-4: Mixture-of-experts FFN: does width help the translator learn per pair?](M-4.md)
- [M-6: Starting from an open translator's layers: how many fewer pairs to our loss?](M-6.md)
- [WS-7: Evolution strategies on home links: does a workload with no chain scale with workers?](WS-7.md)
