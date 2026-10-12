# Full-budget (paper Table 2) re-run of the benign-UAP test

**Why:** companion to `exp_fullbudget_detection_baseline_NOTE.md` -- tests whether
the class-4 attractor found against a known backdoor at reference-repo GAN budget
(stage1=10, stage2=5, images_per_class=32) also persists at the paper's Table 2
budget (stage1=12, stage2=15, images_per_class=1) when there is NO backdoor
anywhere in the system at all (G_agg = epoch-895 checkpoint aggregated
with 10 benign clients' updates only, `cfg.is_poison=False`).
G_agg MA=54.86. rho=0.4, target_class=0.

## Results

| run | raw-T stage3 pred_class | confidence |
|---|---|---|
| 0 | 8 | 0.9459 |
| 1 | 8 | 0.9418 |
| 2 | 8 | 0.9569 |
| 3 | 8 | 0.9271 |
| 4 | 8 | 0.9443 |

pred_class distribution across 5 runs: {8: 5}

## Verdict

pred_class distribution at full budget ({8: 5}) does not reproduce the reference-budget class-4 lock as cleanly -- see table.
