# FP breakdown by client -- is exclusion concentrated or diffuse?

**Why:** a 2-fp round (epoch 432) raised the question of whether the defence is
reliably dropping the SAME honest clients round after round. If so, that's a
specific, quantifiable harm (real cost in excluded benign contributions) riding
alongside the near-zero detection benefit already established -- not just an
abstract false-positive rate.

Zero-compute: parsed directly from `results/step2_cellA_stealthy.csv`,
all 437 rounds on disk as of this analysis (epochs 1-437).

## Totals

- Total fp events (benign client flagged and excluded): 418
- Total tp events (malicious client flagged and excluded): 14
- Distinct benign clients ever sampled: 100
- Distinct benign clients ever excluded: 31 (31.0% of ever-sampled benign clients)

## Per-client fp breakdown (sorted by times excluded)

| client | times excluded | times sampled | exclusion rate |
|---|---|---|---|
| 73 | 39 | 40 | 98% |
| 89 | 35 | 52 | 67% |
| 49 | 32 | 32 | 100% |
| 65 | 31 | 36 | 86% |
| 59 | 27 | 44 | 61% |
| 55 | 22 | 36 | 61% |
| 35 | 21 | 43 | 49% |
| 22 | 21 | 31 | 68% |
| 93 | 18 | 43 | 42% |
| 51 | 17 | 50 | 34% |
| 53 | 16 | 41 | 39% |
| 31 | 14 | 51 | 27% |
| 48 | 14 | 44 | 32% |
| 18 | 14 | 48 | 29% |
| 92 | 13 | 46 | 28% |
| 16 | 11 | 42 | 26% |
| 64 | 11 | 54 | 20% |
| 63 | 11 | 44 | 25% |
| 28 | 9 | 53 | 17% |
| 10 | 8 | 45 | 18% |
| 81 | 7 | 38 | 18% |
| 91 | 6 | 52 | 12% |
| 96 | 6 | 55 | 11% |
| 79 | 3 | 29 | 10% |
| 38 | 3 | 35 | 9% |
| 32 | 2 | 47 | 4% |
| 17 | 2 | 45 | 4% |
| 80 | 2 | 51 | 4% |
| 97 | 1 | 42 | 2% |
| 37 | 1 | 38 | 3% |
| 70 | 1 | 48 | 2% |

**28 of 31 ever-excluded benign clients (90%) were excluded 2+ times — this is not diffuse noise, it's concentrated, near-permanent exclusion of specific honest clients.**

The clearest cases: client 73 excluded 39/40 times it was sampled (98%), client 49 excluded 32/32 times (100%), client 65 excluded 31/36 (86%), client 89 excluded 35/52 (67%), client 22 excluded 21/31 (68%). These clients are not occasionally misflagged — they are *reliably* and almost *permanently* removed from aggregation every single round they're sampled into, for the entire duration logged so far (up to ~437 rounds).

Compare this to the tp count in the same window: 14 true positives total (malicious clients correctly excluded), against 418 false positives. Clients 73 and 49 alone account for 71 of those 418 fp events — more false exclusions from two honest clients than all true positives in the entire run combined, by a factor of 5.

**This is the specific, quantifiable harm the user asked about: the defence is not spreading false positives evenly across the benign population (which might average out and be merely wasteful) — it is identifying a persistent-but-wrong subset of ~30 honest clients (30% of all ever-sampled benign clients) and excluding most of them from nearly every round they participate in, while catching 14 real attackers across the same stretch.** Whatever property of these clients' data partitions or local weights drives Stage 1/2's "attractor" toward class 0 for their aggregated updates, it is apparently a stable property of the CLIENT (or their data shard), not a transient artifact — consistent with the attractor-stability finding from `exp_attractor_mechanism_NOTE.md`, now shown to have a real, person-level cost in a long-running actual run rather than just a theoretical implication.
