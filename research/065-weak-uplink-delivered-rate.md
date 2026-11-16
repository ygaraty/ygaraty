# 20 Mb/s workers get their full link rate; each upload costs one round trip more, nothing else

| | |
|---|---|
| **Conclusion** | **Found:** shaped 20 Mb/s links deliver 19.9 Mb/s; uploads cost one 52 ms round trip each, not F10's 1.6× (measured, DigitalOcean). |
| **Status** | executed |
| **ID** | DO-6 → experiment 65 |
| **Category** | transport |
| **Where** | remote |
| **Date** | Added, started and finished 2026-11-01 |
| **Path** | DigitalOcean nyc3, 3 c-2 machines: a relay host and a prober whose link goes through the pods' `latency-proxy` at 20 Mb/s up, 100 Mb/s down, 50 ms (49.3 ms added to a 0.7 ms private round trip), exactly as `scripts/remote.sh worker` shapes it. Incompressible payloads, relay `PUT /blocks/<sha>`, gzip off |
| **Result** | The relay client's sequential uploads fit time = **52.2 ms + bytes / 19.86 Mb/s** from 64 KB to 4 MB, and one 32 MB upload delivers 19.79 Mb/s (99%). So the link and our client deliver the full rate, and the only per-request cost is one round trip: 1.50× the bytes time at 256 KB, 1.13× at 1 MB, 1.04× at 4 MB. By the frozen rule this is outcome B (256 KB at 13.3 Mb/s, under 16), but the cost is the request/response round trip itself, not slow start or client overhead: `curl` with a new connection per request is no faster (1.58×, 1.15×). Two or four uploads in flight hide it (256 KB: 1.27× and 1.14×). Nothing here explains F10's 1.6× at result sizes or its 2.5 s per piece: that time is outside the link and the client. Cost about $0.10 |
| **Evidence** | [`public/research/evidence/transport/do6-weak-uplink-delivered-rate-2026-11-01.json`](evidence/transport/do6-weak-uplink-delivered-rate-2026-11-01.json) (every request's seconds, per-group medians); raw `shaped.jsonl`, `direct.jsonl`, `probe.py`, `drive.sh` in `runs/pods-data/do6/` (local, not in git). Code at `c2a01ae` |

## Question

F10 in [NEXT_2026-10-31.md](../../papers/uplink-bound/NEXT_2026-10-31.md): over 21 owner spans (experiments 42 and 53), the bytes-over-uplink model is within ±5% on 100 Mb/s owners but 18–26% low on every 20 Mb/s owner (12 of 12); the in-sample fit is upload ≈ 1.6 × bytes/uplink + 2.5 s per piece. With handoffs on (experiment 61), the upload is most of what is left of a hop (experiment 59: about 100 of 171 ms). Is the missing time in the link emulation, in TCP, or in our client? Only the last is ours to fix, and it decides how much of a weak worker's **distribution overhead** is left to remove there.

## Prediction

If one large upload reaches about 20 Mb/s but many result-sized uploads do not, the cost is per request (TCP slow start or the proxy); if neither reaches it, the shaping itself is low.

## Gate

Frozen 2026-11-01, before the run. Delivered rate = payload bits / wall time of the request, median over repeats, through the shaped link (20 Mb/s up, 50 ms).
- **Link calibration:** iperf3 through the proxy (30 s) and one 32 MB relay upload must reach at least 90% of 20 Mb/s (18 Mb/s); otherwise the emulation is miscalibrated (outcome C) and F10's gap is at least partly an artifact.
- **Per-request cost:** at 256 KB and 1 MB (result-sized), the relay client's sequential uploads reaching under 80% of 20 Mb/s, while the 32 MB upload passes, mean a per-request cost (outcome B). Its size is the fixed time per request from a fit of time = a + bytes/rate over 64 KB to 4 MB.
- **Neither:** all sizes from 256 KB up within 80% and the calibration passing: the link and the client deliver; F10's gap is elsewhere (queuing or measurement), outcome A.
- Each outcome is reported against the 1.6× of F10: the measured ratio of time to bytes/uplink at each size.

## Results

Shaped link, medians (ratio = time ÷ bytes at 20 Mb/s):

| Size | Relay client, 1 in flight | `curl`, new connection | Client, 2 in flight | Client, 4 in flight |
|---|---|---|---|---|
| 64 KB | 78.3 ms, 6.7 Mb/s, 2.99× | 86.0 ms, 3.28× | | |
| 256 KB | 157.7 ms, 13.3 Mb/s, **1.50×** | 165.9 ms, 1.58× | 1.27× | 1.14× |
| 1 MB | 475 ms, 17.7 Mb/s, **1.13×** | 484 ms, 1.15× | 1.08× | 1.04× |
| 4 MB | 1.74 s, 19.3 Mb/s, 1.04× | 1.81 s, 1.08× | | |
| 32 MB | 13.56 s, **19.8 Mb/s**, 1.01× | 13.66 s, 1.02× | | |

- **Calibration passes on the relay upload** (19.79 Mb/s, 99%). The iperf3 half did not run: the proxy reset iperf3's control connection. The 32 MB upload carries the calibration alone.
- **Fit:** 52.2 ms per request + 19.86 Mb/s. The fixed part equals the 50 ms round trip: a request's last byte crosses the link and its reply comes back. It does not grow with size, so it is not slow start on the kept-alive connection.
- **Ceiling without the proxy:** 0.6–1.2 Gb/s; the private network adds nothing measurable.

## What it taught

- **Neither the emulation nor our client is slow.** A weak worker's upload costs one round trip plus its bytes at the full 20 Mb/s.
- **F10's 1.6× and 2.5 s per piece are not upload cost.** At result sizes of 1 MB and up the link is within 13% of the bytes model, and no per-request cost comes near seconds. That time is spent elsewhere — queuing for the claim and post paths, which experiments 62 to 64 pointed at — and belongs to the local session's capture analyses (OFF-3, OFF-4).
- **What is left per upload is the round trip.** On experiment 59's hop (upload about 100 ms) roughly half is that round trip and half is bytes. Uploads in flight together already hide it; a hop on the critical chain cannot, so the remaining levers on a hop are fewer bytes, or fewer sequential uploads on the chain.

## Limits

- One link shape (20/100 Mb/s, 50 ms), emulated: no jitter, loss or real home-router buffering.
- The proxy adds no handshake delay by default (`-handshake-rtts 0`), so `curl`'s new connections were nearly free here; on a real link each new connection costs at least one more round trip. The workers keep their connections alive, so this matters only for the comparison with `curl`.
- iperf3 through the proxy failed, so raw TCP was not measured on its own.
- Payload sizes were fixed steps, not experiment 61's exact result sizes.

## Reproduce

```bash
ygaraty experiment pods up do6 --provider digitalocean --location nyc3 --server c-2 --workers 2 --worker-size c-2 --minutes 75 --max-hourly 1
ygaraty experiment pods ship do6
# every machine: bash /root/remote.sh worker-setup; apt-get install -y iperf3
pods ssh do6 w0 -- 'cat > /root/probe.py' < runs/pods-data/do6/probe.py
bash runs/pods-data/do6/drive.sh     # relay and iperf3 on the server; proxy, iperf3 and probe.py (shaped, then direct) on w0; fetches results
ygaraty experiment pods down do6 --no-fetch
```
