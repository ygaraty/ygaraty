# The output layer split by vocabulary: owners for the tied embedding

| | |
|---|---|
| **Date** | 2026-10-26 |
| **Path** | Experiment 31's setup: the 30.3M Transformer, 8 NumPy workers on emulated 100/20 Mb/s links at 50 ms, float32 wire, owned layers |
| **Result** | With the tied embedding and output layer split by vocabulary rows over 8 owners, each keeping its 2,000 rows of the table and the bias with their Adam moments, the update takes **90.5 and 95.6 s** in two runs (111 to 123 s on the checkpoint update, which publishes the owners' state), against 178 s with owned layers alone and 277 s for today's design. The relay now moves 0.78 GB out and 0.46 GB in per update (today's design: 3.9 and 1.9 GB). Losses are unchanged. The output layer, about 100 s of the owned-layers update, now costs about 8 s; what remains is the owners' uplinks, one per layer, carrying every band's activations, plus the small tasks around the embeddings. |
| **Evidence** | Profiles in the session's scratch directory (`exp32/homevocab*`), not synced |

## Question

After experiment 31 the shared embedding/output layer was the whole problem: the fused output task reads the 16,000 × 512 table and uploads a 29 MB logit gradient per band, the Adam step rewrites the table and its moments, and all of it goes through the relay. Can that layer be owned too, when no single worker should hold it?

## What was built

The layer is split by vocabulary rows (`YGARATY_OWNED_VOCAB=N`, `--owned-vocab N`): owner s holds rows [sV/N, (s+1)V/N) of the table and of the output bias, with their moments, as an `OwnedLayers` shard. Per update:

- **Embedding rows.** Each owner serves the rows this batch uses from its slice (`tf_vocab_gather`); the planner concatenates them into a small table the embedding lookups read, with the batch's token ids remapped.
- **Logit statistics.** Per band of decoder rows and per owner, `tf_vocab_stats` returns four numbers per row: the slice's maximum logit, its sum of exponentials, the target's logit when the target falls in the slice, and the logits' sum. A `vocab_loss` task per band combines every owner's statistics into the label-smoothed loss with `cross_entropy`'s arithmetic. The logits never leave the owners.
- **Gradient.** `tf_vocab_grad`, per band and owner, recomputes the slice's logits, forms its part of the logit gradient from the combined statistics (`ce_backward`'s arithmetic), keeps the slice's weight and bias gradients, and returns its part of the decoder output's gradient, a (rows, 512) partial the band's readers sum. One marker row makes the owner's step depend on it.
- **Embedding gradient and step.** `tf_vocab_embed_grad` hands the owner the gradient rows of the embeddings it served; `tf_vocab_step` adds them to the band gradients and applies Adam to the slice. On checkpoint updates `tf_vocab_publish` tasks return the slices, assembled by region into the whole table, bias and moments.

The tape has `vocab_gather`, `vocab_output`, `vocab_grad` and `vocab_embed_grad` nodes so its meaning is unchanged (the table and bias remain inputs of those nodes and get no tape gradient); the planner sends none of them to a worker. The owner routing, load, publish and failure rules are experiment 31's, with `vocab.s` as the owner keys, so a worker may own a layer and a shard.

**Checks.** Two updates with 3 shards plus owned layers, and with one shard per vocabulary row (shards that serve no row and get no embedding gradient), match the plan that ships the table to rtol 1e-9 (summation order differs). The full job in process, with a checkpoint publish, gives the same losses as the unowned job. Over HTTP and the relay, the small model's three losses equal the float64 reference to all printed digits. At full size the losses are experiment 30's.

## Result

Seconds per update on the emulated home links (8 workers, 100/20 Mb/s, 50 ms, float32 wire):

| Design | Update 2 | Update 3 (publish) | Relay fetched / stored | Tasks | Critical path |
|---|---:|---:|---:|---:|---:|
| Today's design | 277 | 295 | 3.9 GB / 1.9 GB | 1,393 | 28 hops |
| Owned layers | 178 | 188 | 2.3 GB / 0.93 GB | 1,000 | 28 hops |
| **Owned layers and vocabulary** | **90.5**, 95.6 | 123, 111 | 0.78 GB / 0.46 GB | 812 | 32 hops |

Per worker and update (first run, median of updates 2 and 3): 2.3 s of compute, 69 s in claims held open waiting for a ready task, 23 s posting results, 22 s fetching, 12 s in empty claims. By family: owner load and publish 29 s (the checkpoint update), layers forward 18 s and backward 17 s, layer norms 12 s, elementwise 9 s, the vocabulary gradient 7 s, placements 7 s, everything else under 3 s each. The two final layer norms alone, 16 small tasks on the critical path, cost 12 s. The measured timeline of the second run shows the encoder's three layers taking 12 s, the decoder's 8 s, each final norm 3.5 to 4 s, and the output statistics of all 8 owners 1 s.

## What it taught

- **Splitting a layer by its output rows is enough for an output layer**: the owners exchange 4 numbers per row and a 0.9 MB partial per band, instead of 29 MB of logit gradient and 33 MB of table per band.
- **The remaining time is the owners' uplinks and the round trips of small tasks**, in that order. Each layer's 8 bands of activations (0.9 MB each) leave through one 20 Mb/s link, 2.9 s per layer and direction before any round trip, 35 s per update across the 12 layer directions. The embeddings' lookups, scalings, position adds and placements are about 60 one-megabyte tasks, each a round trip and a 0.4 s upload.
- Experiment 33's simulator, calibrated on these runs, says that more workers or smaller bands do not shorten this design on 20 Mb/s uplinks (65 to 85 s with 8 to 32 workers); a 50 Mb/s uplink would (45 to 49 s). Cutting activation bytes or spreading a layer's bands over several owners is what reaches lower.

## Limits

Two runs on one machine, one batch; a worker may own a layer and a vocabulary shard, which puts two owners' uplink traffic on one link. The policy spreads 14 owner keys over 8 workers as ceil(14/8) = 2 per worker, so some workers own two consecutive layers; that is not a loss in itself (the activation still goes through the relay), but it is not chosen for balance. The publish update is 20 to 30 s longer; publishing could run in the background of the next update. Owners' failure handling is experiment 31's.

## Running it

```bash
# experiment 31's servers and proxies, then:
ygaraty experiment http_profile --recipe transformer --embedding 512 --hidden 2048 --layers 3 --heads 8 --band 8 \
  --workers 8 --workers-kind python --tiles 512 --max-tasks 16 --max-task-mib 512 --layer-task-mib 512 --output-task-mib 1024 \
  --steps 3 --repeats 1 --learning-rate 0.0005 --relay-url http://127.0.0.1:8101 --relay-token T \
  --rtt-ms 50 --down-mbps 100 --up-mbps 20 --float32-wire --owned-layers --owned-vocab 8     # about 6 minutes
```
