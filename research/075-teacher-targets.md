# An open teacher's translations as targets don't help a student trained from scratch at 32,768 pairs

| | |
|---|---|
| **Conclusion** | **Failed gate:** teacher targets add +0.24 chrF, against +2 required, and cost 0.15 of held-out loss (measured, two seeds). |
| **Status** | executed |
| **ID** | M-5 (experiment 75) |
| **Category** | training |
| **Where** | workstation (omarchy) |
| **Date** | 2026-11-03 |
| **Path** | Teacher `Helsinki-NLP/opus-mt-en-ROMANCE` with `>>pt_br<<`, greedy, run once on the workstation over the 32,768 training sources. Student: the 30.3M translator in the PyTorch reference (`batch_learning`, experiment 67's Adam recipe), from scratch, seeds 2026 and 2027. Arms: references R, teacher targets T, a seeded half-and-half mix RT |
| **Result** | Mean greedy chrF on the 1,024 deduplicated held-out pairs: R 5.80, T 6.04 (+0.24), RT 5.77 (−0.04), against +2 required. Held-out loss: R 5.717, T +0.147 (limit 0.05), RT +0.034. The teacher itself scores chrF 57.3 on the same split, which may be inflated by contamination. It translated the 32,768 sources in 1,003 s on 8 threads while sharing the CPU. Free; about 1 hour |
| **Evidence** | [`evidence/training/m5-teacher-targets-2026-11-03.json`](evidence/training/m5-teacher-targets-2026-11-03.json) (curves, chrF, the teacher's metadata and held-out chrF, gate summary); raw runs, the targets, `queue.sh`, `teacher_chrf.py` and `evidence.py` in `runs/experiments/queue-20261103-185345/` (local, not in git). Code at 524853b: `ygaraty/experiments/distill_targets.py`, `batch_learning --teacher-targets/--teacher-mix` |

## Question

The paper's budget gap lies in learning per pair as much as in seconds per update. Sequence-level knowledge distillation (Kim & Rush, EMNLP 2016) trains a small translator on a large teacher's translations of the training sources, which are more consistent and simpler than human references. The teacher runs once and offline, and translates each source independently. That is the wide, stateless work a volunteer pool does well.

**Trained from scratch on teacher targets, or on a mix of teacher and reference targets, does the student translate the real references better at 32,768 pairs?**

## Prediction and gate (frozen before any run)

**Prediction:** on chrF, T is +2 to +5 over R, with the mix in between or best. Held-out loss against the references is about the same or slightly worse for T.

**Gate:**
- **Primary:** mean greedy chrF over seeds 2026 and 2027 on the deduplicated held-out split (1,024 pairs), against the human references, at 32,768 pairs. T or RT must reach at least **+2 chrF** over R.
- **Required alongside:** the same arm's held-out loss against the references must be no more than **0.05** worse than R's.
- **Report also:** the teacher's own chrF on the same split, and its translation time.

**Deviations, decided before running:**
- The teacher translates only the 32,768 training pairs, greedy only.
- Its targets replace the references in place (`--teacher-targets`, `--teacher-mix`, seeded mix) rather than as a new dataset version. The pairs match R's, but length-sorted batches are recut by the new lengths.
- Teacher targets longer than 63 tokens keep their reference: 47 of 32,768 in T, 26 in RT.

## Protocol

- **Targets.** `distill_targets` selects the same usable pairs as `batch_learning` and translates them, sorted by length, in batches of 64. It writes `{example_id, source, reference, teacher}` per pair. None came back empty. Example: 'I did love her.' has the reference 'Eu amava-a.' and the teacher's 'Eu a amava.'
- **Runs.** The three arms ran in the shared 22-run queue described in [experiment 76](076-parallel-layers.md). R is that queue's baseline, which reproduces experiment 67's Adam arm (5.717).
- **Teacher's own chrF.** `teacher_chrf.py` translates the 1,024 deduplicated held-out sources the same way and scores them against the references the student is scored against, in 49 s.

## Log

- 2026-11-03: Added. Gate frozen with the user's yes.
- 2026-11-03: Teacher targets done under watch.py (`finished`): 1,003 s on 8 threads, while the queue's baseline ran beside them. That contention is why the baseline's training took 729 s against about 340 s for the other arms; losses are unaffected.
- 2026-11-03: Queue `finished`, 22 of 22 runs exited 0. M-5's arms ran 19:21–19:55. Teacher chrF scored afterwards.

## Results

| Arm | chrF 2026 | chrF 2027 | Mean chrF | Against R | Loss mean | Against R | Gate |
|---|---|---|---|---|---|---|---|
| R (references) | 6.29 | 5.32 | 5.80 | — | 5.717 | — | — |
| T (teacher) | 5.01 | 7.07 | 6.04 | +0.24 | 5.863 | +0.147 | fails both |
| RT (mix ½) | 6.32 | 5.22 | 5.77 | −0.04 | 5.750 | +0.034 | fails chrF |
| Teacher itself | | | 57.3 (BLEU 33.0) | | | | upper reference |

T's held-out loss is above R's at every evaluation point, by 0.06–0.18 in the mean of the two seeds.

## What it taught

- **At this budget the student is too early for distillation to show.** At a loss around 5.7, every arm's greedy decoder still loops on repeated words, and chrF sits near 6 against the teacher's 57. Kim & Rush's gain comes from a student that can already fit the data, and this one can't yet. The seed-to-seed spread in chrF (up to 2 points) is larger than any effect between arms.
- **Teacher targets are harder for this student to transfer to the references, not easier.** T loses 0.15 against the references, three times the limit. This is the expected style gap, but it is not offset by any chrF gain here.
- **The mix is close to neutral** (+0.03 loss, −0.04 chrF), so half the pairs can come from the teacher without much cost.

## Limits

- **One budget and two seeds.** Distillation may help later in training, at the 131,072 pairs of experiment 70 or beyond, where chrF starts to measure translation. That is a new experiment with its own gate.
- **Greedy teacher only;** beam 4 targets were not made.
- **Contamination:** OPUS-MT was trained on OPUS corpora, and OPUS-100 is drawn from them, so the teacher's 57.3 may be flattered. The student saw only translations of training sources.
- **Teacher loss on held-out teacher translations** (the predicted "easier targets" check) was not measured.
- **Licence:** OPUS-MT models are CC-BY 4.0, by the Helsinki-NLP group (Tiedemann and Thottingal, 2020).

## Reproduce

```bash
export YGARATY_DATASET_URL=sqlite:///$PWD/runs/datasets/opus100-en-pt/opus100-en-pt.sqlite
.venv/bin/python -m ygaraty.experiments.distill_targets --version 1m-seed2026 --pairs 32768 --output runs/experiments/M-5/teacher/targets.jsonl
COMMON="--embedding 512 --hidden 2048 --heads 8 --layers 3 --version 1m-seed2026 --batch 64 --length-window 64 --batch-tokens 3072 --pairs 32768 --eval-every 4096 --optimizer adam --learning-rate 0.0005 --hint-index runs/experiments/055-hints/index --eval-dedup --decode"
for seed in 2026 2027; do
  .venv/bin/python -m ygaraty.experiments.batch_learning $COMMON --seed $seed --output runs/experiments/M-5/R-$seed
  .venv/bin/python -m ygaraty.experiments.batch_learning $COMMON --seed $seed --teacher-targets runs/experiments/M-5/teacher/targets.jsonl --teacher-mix 1.0 --output runs/experiments/M-5/T-$seed
  .venv/bin/python -m ygaraty.experiments.batch_learning $COMMON --seed $seed --teacher-targets runs/experiments/M-5/teacher/targets.jsonl --teacher-mix 0.5 --output runs/experiments/M-5/RT-$seed
done
```

The teacher's own held-out chrF: copy `teacher_chrf.py` from the original run directory and run `.venv/bin/python teacher_chrf.py <dir with runs/R-2026/decode.jsonl and teacher/>`. Needs `transformers` and `sentencepiece`; the teacher is downloaded once (about 300 MB).
