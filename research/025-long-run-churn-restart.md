# Surviving a long run: contributors leaving and a coordinator restart

| | |
|---|---|
| **Date** | Planned |
| **Path** | Ygaraty HTTP coordinator, browser and Node contributors |
| **Result** | Not run yet |
| **Evidence** | — |

## Question

Can a multi-hour run of the experiment-24 setup survive contributors joining and leaving and a deliberate coordinator restart, without losing or repeating updates?

## Needs first

HTTP restart recovery (roadmap step 2): persisting the job and every committed update, and resuming on the right batch.

## Plan

Run for several hours with contributors joining and leaving on a schedule. Kill the coordinator once just before and once just after a checkpoint write, then restart it.

## Success criteria

Committed updates are contiguous, with none skipped or repeated. Weights after resume match an uninterrupted reference run of the same batches. Disk use stays within a budget set beforehand.
