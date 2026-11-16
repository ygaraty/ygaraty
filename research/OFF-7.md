# Lossless coding of bf16 activations: how much smaller than gzip, at what cost?

| | |
|---|---|
| **Conclusion** | Pending |
| **Status** | open |
| **ID** | OFF-7 (number added on finish) |
| **Category** | transport |
| **Where** | this machine |
| **Date** | Added 2026-11-01 |
| **Path** | Offline analysis of captured activation and gradient blocks (bf16) from the 1.06M and 30.3M runs; no training, no network |

## Question

Experiment 49 found gzip on the weak workers' frames worth 4.3% of an update. gzip is a general byte coder. bf16 values are more compressible field by field: their exponents take few distinct values, and a coder that separates exponent and sign bytes from mantissa bytes and codes each stream on its own does better on model weights (the ZipNN line of work). The coding is lossless, so losses stay bit-identical, which keeps it inside the paper's invariant.

**On the blocks our workers actually upload, how much smaller than gzip does a float-aware lossless coder make them, and at what encode and decode time on a c-2 worker's core?**

## Prediction

- **Size:** a float-aware coder (byte-plane split plus a fast entropy coder such as zstd or Huffman per plane) makes bf16 activation blocks 15–30% smaller than gzip.
- **Time:** it encodes faster than gzip level 6 on one core.
- **Gradients** compress less than activations.

## Gate

TBD, to be frozen before the analysis. Proposed: it counts as worth a timed run if it saves at least 15% of the weak workers' upload bytes against gzip, and encoding at most 20 ms per MB more than gzip on one core. Either one alone is not enough.

## Protocol

1. **Inputs:**
   - relay blocks from experiment 58's or 61's captures, kept or regenerated with `YGARATY_CAPTURE_DIR` and block payloads saved for a few updates;
   - a short local run of the 1.06M recipe for activation blocks.

   If no payloads were kept, this needs one capture run on the workstation first. That is the only non-offline step.
2. **Coders:**
   - gzip at levels 1 and 6 (today's setting, as in experiment 49);
   - zstd at levels 1 and 3;
   - byte-plane split (sign and exponent apart from mantissa) followed by zstd;
   - the same split followed by Huffman per plane.
3. **Measures**, per block kind (forward activation, backward gradient, slice partials):
   - compressed size against raw and against gzip;
   - encode and decode MB/s on one core, pinned, in Python with the same libraries a worker would ship.
4. **Output:** a table per block kind, and the predicted change in the weak workers' busy uplink time, using L1's decomposition.

## Log

- 2026-11-01: Added, from a literature review of distribution overhead (float-aware lossless compression).
