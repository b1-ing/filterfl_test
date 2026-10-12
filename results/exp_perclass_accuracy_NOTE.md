# Does excluding class-0-dominant clients degrade class-0 accuracy? (conjecture step 3)

**Why:** clients 49, 73, 65, 22, 89 (class-0-dominant, per `exp_partition_profile_NOTE.md`)
are excluded from aggregation 67-100% of the time they're sampled
(`exp_fp_breakdown_NOTE.md`). If that visibly degrades class-0 test accuracy
relative to the other nine classes, it's evidence for a feedback loop where the
detector's miscalibration actively harms the class it's nominally protecting.
If class 0 ISN'T degraded, this step of the conjecture chain is false.

Method: loaded all 171 available snapshot checkpoints (epochs
65-915, every 5 epochs) and computed per-class test accuracy
for each -- pure inference, no training. target_class=0.

## Summary

- Mean class-0 accuracy across all snapshots: **65.60**
- Mean other-9-class accuracy: **72.18**
- Mean diff (class0 - others): **-6.58**
- Class 0 worse than the other-class mean in **103/171 (60%)** of snapshots

## Per-class overall mean accuracy, ranked worst to best (1 = worst)

| rank | class | mean accuracy | |
|---|---|---|---|
| 1 | 3 | 54.82 |
| 2 | 2 | 63.88 |
| 3 | 5 | 64.05 |
| 4 | 0 | 65.60 | <-- TARGET CLASS
| 5 | 4 | 69.27 |
| 6 | 7 | 72.22 |
| 7 | 9 | 77.65 |
| 8 | 1 | 77.77 |
| 9 | 6 | 82.10 |
| 10 | 8 | 87.86 |

**Class 0's rank among all 10 classes: 4/10** (1 = worst)

## Verdict

**Class 0 is NOT clearly degraded relative to the other nine classes.** Mean class-0 accuracy (65.60) is 6.58 points below the other-9-class mean (72.18), worse than the other-class-mean in only 103/171 (60%) of snapshots, and ranks 4/10 overall (1=worst) -- not among the worst classes. **This breaks conjecture step 3 as stated**: despite clients 49/73/65/22/89 being excluded 67-100% of the time, class 0 does not show a correspondingly large accuracy deficit. Possible reasons worth separate investigation: class 0 ('airplane') may simply be an easier class for this architecture regardless of training signal volume, non-excluded class-0-dominant clients' partial contribution may be enough, or the excluded clients' class-0 SHARE (30-67% of their own shard) may not translate to a large fraction of all class-0 training examples globally (100 participants, Dirichlet-partitioned -- need to check what fraction of ALL class-0 examples these 5 clients actually hold).
