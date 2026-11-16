# Spreading one layer over several owners: what does it cost on the chain?

| | |
|---|---|
| **Conclusion** | Pending |
| **Status** | open |
| **ID** | OFF-8 (number added on finish) |
| **Category** | simulation |
| **Where** | this machine |
| **Date** | Added 2026-11-03 |
| **Path** | Offline simulator with L5's phase-priced hop costs, experiment 61's pool (8 c-2 weak and 8 c-8 strong workers, emulated 100/20 Mb/s, 50 ms links, handoffs on), the 30.3M model. No training, no network |

## Question

Today one worker owns a whole layer, attention plus FFN, and keeps its weights and Adam moments. Spreading a layer over more contributors fits more and smaller contributors, or more parameters, into the pool. There are two ways to do it:

- **Tensor split (Megatron-style).** K owners each hold 1/K of the FFN's hidden units: a column block of the first matrix and the matching row block of the second. Each produces a full 512-dim partial, and the consumer adds the K partials. Each owner's upload is as large as today's, and the consumer downloads K of them.
- **Mixture of experts.** E experts, top-1 routing. Each expert gets about 1/E of the tokens and uploads only their outputs. M-4 asks whether this helps learning.

Either way, the FFN leaves the layer owner, so each split layer gains **one hop per direction**: attention owner → FFN owners → next layer. At L5's price of about 230 ms per hop, that is not free.

**On experiment 61's pool, how much longer is the update when one layer, or every layer, is split over K = 2 or 4 owners, by tensor split or by experts, at today's batch and at 12,288 tokens?**

## Prediction

- **Each split layer adds about 0.4–0.5 s per update:** two extra hops (forward and backward) at about 230 ms. On one layer that is about 10% of 4.6 s, so the ±5% gate below fails at today's batch, for both splits.
- **Tensor split** costs more than MoE: the consumer's download is K times larger and it waits for the slowest of K uploads.
- **MoE** uploads shrink about E times per expert, so weak 20 Mb/s uplinks gain on bytes, but the extra hop's round trips dominate.
- **At 12,288 tokens,** FFN compute and bytes grow while the round trips don't, so the relative cost of the split falls.
- **Conclusion expected:** at 30M, splitting layers costs more than it gives. It becomes worth it only when a layer doesn't fit one contributor, at nanochat scale.

## Gate

TBD, to be frozen before the simulation. Proposed:
- **Viable at today's scale:** MoE E = 4 on one decoder layer costs at most **+5%** in simulated seconds per update against dense, at 3,072 tokens.
- **Report also:** the full grid below, the added hops on the critical path, and the per-uplink busy time (L1's decomposition).
- **Evidence:** simulated only. It predicts scheduling time and establishes neither real speed nor learning.

## Protocol

1. **What has to be built (offline only):**
   - task-graph builders for a split FFN: attention task → K FFN-part tasks → a sum in the consumer (tensor split), or attention plus router → E expert tasks → a gather in the consumer (MoE);
   - placement giving FFN parts and experts to distinct workers, weak workers first, as capacity placement would.
2. **Routing load (MoE):** balanced (each expert gets 1/E of the tokens), plus one imbalanced arm with the busiest expert at 1.5× the mean, M-4's limit.
3. **Grid:**

   | Factor | Values |
   |---|---|
   | Split | dense; tensor split K = 2, 4; MoE E = 2, 4 |
   | Layers split | one decoder layer; all layers |
   | Batch | 3,072 and 12,288 tokens |

4. **Costs:** L5's calibrated hop phases and experiment 61's measured kernel times. FFN-part compute scales by 1/K; expert compute scales with token share.
5. **Check:** the dense arm reproduces L5's prediction for experiment 61's arm B within 1%.

## Log

- 2026-11-03: Added, from a discussion of factorizing and splitting layers across more contributors. Pairs with M-4. Note: an earlier answer in that conversation said a split uploads 1/K of the bytes. That holds for a single matrix split by output columns (the vocabulary slices), not for a whole FFN, and either split adds a hop per layer.
