# 4-bit Phi-4 runs on two 8 GB GPUs across the internet, at 1.35 s per token

| | |
|---|---|
| **Conclusion** | **Works, slowly:** 4-bit Phi-4 on two 8 GB Vast.ai GPUs, 1.35 s per token (measured, Quebec–British Columbia). |
| **Status** | executed |
| **ID** | DO-16 |
| **Category** | correctness |
| **Where** | remote |
| **Date** | Added 2026-11-04, run 2026-11-04 |
| **Path** | Experiment 81's two-owner Phi-4 pipeline with `--weights int4` (round to nearest, groups of 128), on 2 Vast.ai containers with one 8 GB GPU each, on separate hosts linked only over the public internet: an RTX 3060 Ti in Quebec (coordinator, relay, owner w0) and an RTX 4060 in British Columbia (owner w1). Torch bf16, float32 head, float32 wire, relay and handoffs, prefill chunks of 512 |
| **Result** | Gate passed. Every job completed with 0 retries and 0 rejected, each prompt got its 16 tokens, and each GPU peaked at 7.4 GB of 8. 4-bit tokens equal experiment 80's reference up to a near-tie: positions 12 and 13 of the chat prompts (reference margins 0.45 and 0.06), and position 3 of the random one. Warm, for prompts of 128 / 1,024 / 3,968 tokens: first token 2.2 / 5.2 / 28.1 s, 1.35 s per token (p95 1.41 s). Compute takes 1.1 s of that (510–535 ms on the 3060 Ti, 590–600 ms on the 4060), because each 4-bit matrix is unpacked to full size before its product; the internet adds about 75 ms per crossing. Charged $0.47 of a $2 budget, about $0.30 of it most likely Vast.ai's per-GB download charge on the 29 GB checkpoints |
| **Evidence** | Measured. `public/research/evidence/correctness/do16-phi4-int4-vast-8gb-2026-11-04.json` (tokens and texts, per-session times, span medians, peak memory, hosts, drivers, cost); raw run folders in `runs/pods-data/do16/` (not in git). Code commits `2fae130` (server) and `ffdf9cc` (replacement worker); 4-bit from `13d74e6` |

## Question

**Can the smallest rentable GPUs (8 GB) run Phi-4 at all, split over two hosts on the real internet, and at what time to first token and inter-token latency?** It is also the first live test of the pods tool's Vast.ai provider.

## Prediction

- **Memory:** each owner holds about 3.6 GB of 4-bit layers plus 1 GB for the embedding or the head. With the cache, one unpacked matrix and torch's overhead, each GPU should peak at 6–7 GB of its 8 GB.
- **Tokens:** 4-bit weights change the logits, so tokens are expected to match experiment 80's reference for the first few positions, then may diverge. WS-11 measures that rigorously; here it is reported, not gated.
- **Compute:** unpacking each matrix just before its product makes each owner read and write about 14 GB per token, at an RTX 3060 Ti's or 3070's 450 GB/s: about 30–60 ms per owner.
- **Network:** a North American internet round trip of 20–80 ms, crossed about twice per token.
- **Expected inter-token latency:** 150–300 ms.

## Gate (frozen 2026-11-04, before any machine was made)

This is a feasibility trial; token quality belongs to WS-11.

1. **Works** if every job completes with 0 rejected results, each of experiment 80's 3 prompts gets its 16 tokens, and no owner runs out of GPU memory. Otherwise **Failed** (or **Did not complete** for provider trouble), with the cause named.
2. **Reported:**
   - each prompt's tokens and decoded text against experiment 80's reference, the first divergence and the reference margin there;
   - per prompt length (128, 1,024 and 3,968, seed 2026, 128 new tokens, 1 warmup plus 5 measured): median time to first token, median and p95 inter-token latency, and tokens/s;
   - peak GPU memory per owner, weight-load time, span medians, the hosts' locations and GPUs, and the round trip between them.

## Protocol

- **Machines:**
  ```bash
  pods up do16 --provider vast --location US,CA --server any-8to9gb-sm80 --workers 1 \
    --worker-size any-8to9gb-sm80 --minutes 90 --max-hourly 0.5
  ```
  - The cheapest verified on-demand offers: 1 GPU, CUDA 12.6 or newer, at least 4 direct ports, 80 GB of disk, 200 Mb/s down, reliability 0.95, never two on one host.
  - Image `ubuntu:22.04`; ports 8000 and 8101 mapped on A.
- **Setup:** `pods ship do16`, then `phi4-setup server` on A and `phi4-setup` on B, in parallel. Shard hashes are checked against experiment 80's.
- **Driver:** `runs/pods-data/do16/drive.py correctness`, then `timing 128 1024 3968`, then `analyze.py`.
- **Round trip:** 10 curl requests from B to A's mapped coordinator port.
- **Teardown:** `pods down do16` when the jobs finish, then `pods ls` must show 0. The watchdog deletes both machines at 90 min regardless.
- **Cost:** about $0.10/h per machine, about 1 h, so about $0.20–0.30. The user's budget is $2.

## Log

- 2026-11-04: Added. The user asked for Vast.ai first, on the smallest GPUs, and chose a $2 budget, a 90-minute deadline, and both machines in the US or Canada. Prepared at no cost:
  - Vast.ai lists instances through its v1 API;
  - `-smXY` and `any-NtoMgb` sizes;
  - the `do16` driver.
- 2026-11-04: Waiting for Vast.ai credit: the account had $0.00 and no billing.
- 2026-11-04 20:47–20:54: Free rehearsal on the workstation CPU: the real checkpoint, `--weights int4`, two local workers, bf16, prompt 0, 8 new tokens. All 8 tokens equal experiment 80's reference ("The capital of France is Paris.<|im_end|>"). Each worker held 4.65 GB. The first token took 317 s, including the load, then 11.8 s per token; 2 retries (the CPU load outlasting a 300 s lease) and 0 rejected.
- 2026-11-04 20:51: Vast.ai credit $10 added by the user.
- 2026-11-04 20:55: `pods up`: server RTX 3060 Ti 8 GB in Quebec, CA ($0.096/h); w0 RTX 3060 Ti 8 GB in Delaware, US ($0.097/h). The server's setup took 11 min (torch 2.15.0+cu132, driver 595), and its shard hashes match experiment 80's. The Delaware host's Hugging Face downloads kept being cut off: 938 MB after 20 min, then its SSH session was closed by the host. That worker was deleted (`down do16 w0`, its host now avoided for the run), and its replacement is an RTX 4060 8 GB in British Columbia, CA ($0.101/h). Its setup took 9 min (driver 570, so torch 2.14.1+cu126), and its hashes match.
- 2026-11-04 21:28: Correctness job: 51 steps, 0 retries, 0 rejected; 16 tokens per prompt; peak GPU memory 7.4 GB per owner. About 1.34 s per token: each owner's stage computes in 518 ms (3060 Ti) and 587 ms (4060); the relay and posts take about 75 ms each.
- 2026-11-04 21:33: The timing plan shrinks to fit the deadline: 1 warmup and 2 measured sessions of 32 new tokens per length, not 5 of 128. At 1.34 s per token, the frozen plan (2,304 tokens) would take about 51 min, with 51 min left. The measurement is reported, not gated.
- 2026-11-04 21:31–21:44: Timing jobs at 128, 1,024 and 3,968 tokens, 0 retries, 0 rejected.
- 2026-11-04 21:44: Live test of `phi4_chat --attach do16`: the chat command served the job and opened its tunnel, but its first request sent the word `input_ids` as the prompt (transformers 5 returns a BatchEncoding from `apply_chat_template`). The program crashed, and the cluster was destroyed on the way out (`pods ls` shows 0). Fixed in `39e77b1`, with tests against the real tokenizer and a local serving job.
- 2026-11-04 21:46: Both machines deleted. The account went from $10.00 to $9.53 of credit.

## Results

### Correctness (gate)

| Prompt | Tokens | First divergence | Reference margin there | 4-bit text | Reference text |
|---|---|---|---|---|---|
| 0 (chat, 19 tokens) | 16 | 12 | 0.45 | "The capital of France is Paris." then the end marker; the free run that follows differs | "The capital of France is Paris." then the end marker |
| 1 (chat, 95 tokens) | 16 | 13 | 0.06 | "In Ygaraty's distributed neural network system, the upload link often becomes the" | "... the upload link typically limits performan(ce)" |
| 2 (random, 128 tokens) | 16 | 3 | 0.17 | "IC)  1. **Define the problem...**" | "IC) 0.0.0.0:8080/ 0" |

- **Gate:** every job completed (51 steps, 0 retries, 0 rejected), each prompt got its 16 tokens, and peak memory was 7.4 GB per GPU. All three divergences come at reference margins below 0.5, so experiment 81's tie rule would have excused them too.
- **Rehearsal:** the free CPU run of prompt 0 gave the same 8 tokens as the reference.

### Timing (reported; shortened from the frozen plan to fit the deadline)

1 warmup plus 2 measured sessions of 32 new tokens per length:

| Prompt tokens | Median time to first token | Inter-token median | Inter-token p95 | Tokens/s | Peak GPU memory w0 / w1 |
|---|---|---|---|---|---|
| 128 | 2.2 s | 1,346 ms | 1,411 ms | 0.73 | 7.4 / 7.4 GB |
| 1,024 | 5.2 s | 1,345 ms | 1,391 ms | 0.68 | 7.4 / 7.3 GB |
| 3,968 | 28.1 s | 1,365 ms | 1,390 ms | 0.46 | 7.4 / 7.3 GB |

- **Where a token's 1.35 s go** (span medians): w0's stage takes 510–535 ms and w1's 590–600 ms. On w1, the relay fetch and the post take 74–86 ms each, a round trip between Quebec and British Columbia. The rest is claims and scheduling.
- **Weight load:** 45 s on w0 and 79 s on w1, which quantize from bf16 as they load; the warmup's first token was 76–104 s.

## What it taught

- **The smallest rentable GPUs can run Phi-4.** 4-bit weights fit two 8 GB cards with 0.6 GB to spare, at about $0.10/h each, over a plain internet link.
- **4-bit tokens hold up to the first near-tie.** On these 3 prompts, the 4-bit tokens matched until a position where the reference itself was within 0.5 of its second choice. WS-11 is still the real measure.
- **Unpacking is the bottleneck, by about 10×.** Each owner reads 3.6 GB of 4-bit weights per token. The prediction counted the unpacked full-size copies at the GPUs' memory bandwidth, which gives 30–60 ms per owner; the measured time is about 550 ms. A fused 4-bit kernel, which multiplies without writing the unpacked matrix, is the next lever. Until then, 4-bit trades a lot of speed for memory.
- **The internet costs little here.** About 75 ms per crossing against about 1.1 s of compute.
- **Vast.ai works end to end with the pods tool:** offers, SSH, the mapped ports, the private-repository clone, the torch build per driver (CUDA 13.2 and 12.6 here) and deletion by recorded ID. Hosts vary, though: one of three had a link too poor to download the checkpoint, and replacing it cost 25 minutes.
- **Bandwidth is billed.** About 60% of the charge was most likely the per-GB download charge on the checkpoints, not GPU time. The chat command's budget counts only hourly prices, so it underestimates Vast.ai's cost.

## Limits

- One pool, one pair of hosts, and a shortened timing plan (2 measured sessions of 32 tokens per length).
- Mixed GPUs (RTX 3060 Ti and RTX 4060) and mixed torch builds (2.15.0+cu132 and 2.14.1+cu126).
- Token quality from 3 prompts only; WS-11 measures it properly.
- Cost is read from the account's credit, which doesn't break down GPU time, disk and bandwidth.
- The chat command's live test didn't reach a reply; the bug it found is fixed and covered by tests, but it hasn't run live again.

## Reproduce

```bash
python -m ygaraty.experiments.pods up do16 --provider vast --location US,CA --server any-8to9gb-sm80 \
  --workers 1 --worker-size any-8to9gb-sm80 --minutes 90 --max-hourly 0.5
python -m ygaraty.experiments.pods ship do16
# in parallel: server `bash /root/remote.sh phi4-setup server`, w0 `bash /root/remote.sh phi4-setup`
.venv/bin/python runs/pods-data/do16/drive.py correctness
REPEATS=3 MAX_NEW=32 .venv/bin/python runs/pods-data/do16/drive.py timing 128 1024 3968
.venv/bin/python runs/pods-data/do16/analyze.py
python -m ygaraty.experiments.pods down do16 && python -m ygaraty.experiments.pods ls
```
