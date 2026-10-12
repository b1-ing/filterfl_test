# Partition profile: heavily-excluded vs. never-excluded benign clients

**Why:** FilterFL's stated contribution over similarity-based defences is that it
does NOT penalize clients merely for having unusual (non-IID) partitions -- unlike
weight-similarity defences, which fail under heterogeneity because heterogeneous
clients look anomalous. If the clients FilterFL systematically excludes (per
`exp_fp_breakdown_NOTE.md`: clients 49, 73, 65, 22, 89, each excluded 67-100% of
the times sampled) are precisely the ones with the most unusual partitions, FilterFL
reproduces that exact failure mode by a different mechanism -- a direct hit on the
paper's stated contribution over prior work.

Zero-compute: re-derives the same Dirichlet partition from the fixed seed/alpha used
throughout this run (`cfg.seed`, `cfg.dirichlet_alpha=0.5`) -- no
model training, no GPU.

Control selection: never-excluded, non-malicious clients sampled >=30 times (to
match the excluded group's sampling frequency rather than compare against
rarely-sampled clients), picked by sampling frequency, top 5.
59 such clients existed to choose from.

## Heavily-excluded benign clients

| client | shard size | # classes present | top-1 class (share) | top-2 share | class-0 count (share) | full histogram |
|---|---|---|---|---|---|---|
| 49 | 426 | 10 | 0 (39%) | 59% | 168 (39.4%) | {0: 168, 1: 17, 2: 1, 3: 10, 4: 11, 5: 34, 6: 8, 7: 75, 8: 17, 9: 85} |
| 73 | 500 | 9 | 0 (67%) | 77% | 335 (67.0%) | {0: 335, 2: 4, 3: 17, 4: 6, 5: 7, 6: 22, 7: 30, 8: 28, 9: 51} |
| 65 | 314 | 9 | 0 (39%) | 57% | 122 (38.9%) | {0: 122, 1: 7, 2: 4, 3: 50, 5: 16, 6: 8, 7: 7, 8: 56, 9: 44} |
| 22 | 494 | 10 | 0 (34%) | 62% | 168 (34.0%) | {0: 168, 1: 71, 2: 24, 3: 137, 4: 28, 5: 11, 6: 11, 7: 14, 8: 29, 9: 1} |
| 89 | 413 | 9 | 0 (40%) | 64% | 167 (40.4%) | {0: 167, 1: 1, 2: 98, 3: 2, 4: 6, 5: 94, 6: 8, 7: 1, 9: 36} |

## Never-excluded benign control clients

| client | shard size | # classes present | top-1 class (share) | top-2 share | class-0 count (share) | full histogram |
|---|---|---|---|---|---|---|
| 12 | 977 | 10 | 5 (41%) | 58% | 132 (13.5%) | {0: 132, 1: 10, 2: 90, 3: 1, 4: 28, 5: 398, 6: 29, 7: 172, 8: 93, 9: 24} |
| 11 | 546 | 10 | 3 (29%) | 46% | 9 (1.6%) | {0: 9, 1: 19, 2: 2, 3: 157, 4: 67, 5: 60, 6: 74, 7: 53, 8: 11, 9: 94} |
| 2 | 461 | 10 | 2 (53%) | 68% | 2 (0.4%) | {0: 2, 1: 3, 2: 245, 3: 15, 4: 17, 5: 67, 6: 29, 7: 8, 8: 22, 9: 53} |
| 3 | 780 | 10 | 7 (49%) | 66% | 3 (0.4%) | {0: 3, 1: 2, 2: 80, 3: 58, 4: 91, 5: 2, 6: 14, 7: 379, 8: 139, 9: 12} |
| 4 | 557 | 9 | 3 (39%) | 71% | 15 (2.7%) | {0: 15, 1: 12, 2: 2, 3: 216, 4: 177, 6: 49, 7: 7, 8: 43, 9: 36} |

## Group averages

| | mean shard size | mean # classes present | mean top-1 share | class-0 present in |
|---|---|---|---|---|
| excluded (n=5) | 429.4 | 9.4 | 43.9% | 5/5 |
| control (n=5) | 664.2 | 9.8 | 42.0% | 5/5 |

## Corrected verdict (the auto-generated one above checked the wrong signal)

The generic non-IID metrics -- shard size, number of distinct classes present,
top-1 concentration share -- are essentially **indistinguishable between the two
groups** (429 vs 664 mean size, 9.4 vs 9.8 mean classes, 43.9% vs 42.0% mean
top-1 share; class 0 is present in some nonzero amount for all 10 clients
either way). By those coarse measures, the excluded clients are not "more non-IID"
than the controls. A naive read of the auto-generated verdict logic below would
have called this "not distinctive" and closed the question -- that would have been
wrong, because it was checking the wrong column.

**The real signature is not heterogeneity in general, it's WHICH class is
dominant.** Look at the top-1 class column directly:

- Excluded clients' top-1 (dominant) class is **class 0 in 5/5 cases**, at
  34-67% share of their shard (49: 39%, 73: 67%, 65: 39%, 22: 34%, 89: 40%).
- Control clients' top-1 class is **never class 0** in any of the 5 cases (classes
  5, 3, 2, 7, 3 instead), and their class-0 share is minor-to-negligible (12: 13.5%,
  11: 1.6%, 2: 0.4%, 3: 0.4%, 4: 2.7%) -- an order of magnitude lower than the
  excluded group's 34-67%.

This is a precise, mechanistic, and testable claim, not just "excluded clients look
unusual": **a benign client gets excluded by Stage 3 specifically when class 0 (the
backdoor's target class) happens to be the dominant class in that client's
non-IID partition** -- because that dominance apparently biases the client's local
weight update enough to pull Stage 1/2's trigger-reconstruction GAN toward class 0,
independent of any backdoor. Clients whose data merely happens to be skewed toward
ANY other class (5, 3, 2, 7 in the controls -- plenty non-IID, just not skewed
toward class 0 specifically) are not penalized.

**This still lands the hit the `Why` section anticipated, but more precisely
targeted than generic non-IID sensitivity:** FilterFL is not penalizing
heterogeneity broadly (a client skewed toward class 7, e.g. client 3 at 49% top-1
share, is never excluded) -- it is penalizing one specific, predictable kind of
heterogeneity: having a partition dominated by the same class the attacker (in
this run) happens to target. A real attacker could in principle choose any target
class; whichever class they choose, the benign clients whose data happens to be
skewed toward THAT class would bear this cost. That is a narrower, sharper claim
than "fails on non-IID data" and arguably a more damaging one: the defence's blind
spot is keyed to the exact quantity (target-class dominance in a client's own
data) that an attacker's choice of target class determines.
