# Full rho threshold sweep -- is the statistic usable at ANY operating point?

**Why:** AUC=0.46 (`exp_auc_and_regime_series_NOTE.md`) shows the effective Stage 3
score doesn't consistently rank malicious above benign clients, averaged across all
thresholds. That leaves open whether some SPECIFIC threshold, even one the paper's
rho schedule doesn't use, would work. This sweeps the full range and checks directly,
with hindsight -- the strongest possible case for the statistic.

Effective score (same definition as the AUC note): `confidence if pred_class ==
target_class else 0.0` -- the monotonic quantity Stage 3's AND-gate reduces to.
9190 observations across all logged rounds (927 malicious, 8263 benign).

## Best achievable operating point anywhere in [0, 1]

**By Youden's J (TPR - FPR, the standard single-best-threshold statistic):**
rho=1.0000: TPR=0.00%, FPR=0.00%, J=0.00pp,
precision=0.00%

**By F1:** rho=0.2949: TPR=3.34%, FPR=8.86%,
precision=4.06%, F1=0.0367

## The actual schedule's operating points, for comparison

| rho | tp | fp | TPR | FPR | J |
|---|---|---|---|---|---|
| 0.5 | 15 | 555 | 1.62% | 6.72% | -5.10pp |
| 0.4 | 23 | 664 | 2.48% | 8.04% | -5.55pp |
| 0.3 | 30 | 732 | 3.24% | 8.86% | -5.62pp |
| 0.2 | 31 | 753 | 3.34% | 9.11% | -5.77pp |

## Coarse sweep (every 0.05)

| rho | tp | fp | TPR | FPR | J (TPR-FPR) | precision |
|---|---|---|---|---|---|---|
| 0.00 | 31 | 753 | 3.34% | 9.11% | -5.77pp | 3.95% |
| 0.05 | 31 | 753 | 3.34% | 9.11% | -5.77pp | 3.95% |
| 0.10 | 31 | 753 | 3.34% | 9.11% | -5.77pp | 3.95% |
| 0.15 | 31 | 753 | 3.34% | 9.11% | -5.77pp | 3.95% |
| 0.20 | 31 | 753 | 3.34% | 9.11% | -5.77pp | 3.95% |
| 0.25 | 31 | 746 | 3.34% | 9.03% | -5.68pp | 3.99% |
| 0.30 | 30 | 732 | 3.24% | 8.86% | -5.62pp | 3.94% |
| 0.35 | 25 | 706 | 2.70% | 8.54% | -5.85pp | 3.42% |
| 0.40 | 23 | 664 | 2.48% | 8.04% | -5.55pp | 3.35% |
| 0.45 | 18 | 605 | 1.94% | 7.32% | -5.38pp | 2.89% |
| 0.50 | 15 | 555 | 1.62% | 6.72% | -5.10pp | 2.63% |
| 0.55 | 12 | 505 | 1.29% | 6.11% | -4.82pp | 2.32% |
| 0.60 | 9 | 436 | 0.97% | 5.28% | -4.31pp | 2.02% |
| 0.65 | 7 | 384 | 0.76% | 4.65% | -3.89pp | 1.79% |
| 0.70 | 3 | 342 | 0.32% | 4.14% | -3.82pp | 0.87% |
| 0.75 | 3 | 298 | 0.32% | 3.61% | -3.28pp | 1.00% |
| 0.80 | 3 | 250 | 0.32% | 3.03% | -2.70pp | 1.19% |
| 0.85 | 1 | 212 | 0.11% | 2.57% | -2.46pp | 0.47% |
| 0.90 | 0 | 158 | 0.00% | 1.91% | -1.91pp | 0.00% |
| 0.95 | 0 | 102 | 0.00% | 1.23% | -1.23pp | 0.00% |
| 1.00 | 0 | 0 | 0.00% | 0.00% | +0.00pp | 0.00% |

## Verdict

**Every single non-trivial threshold is worse than useless. The global best Youden's J over the entire [0,1] sweep occurs ONLY at the degenerate rho=1.0 operating point (flag nobody, ever) — J=0.00pp by construction.** Look at the coarse table: from rho=0.00 all the way to rho=0.95, EVERY row has negative J — false-positive rate exceeds true-positive rate at every achievable threshold, with no exception. The best F1 (0.0367, at rho≈0.29) still corresponds to negative J (-5.51pp) and 4% precision — i.e. even the threshold that maximizes F1 still produces a detector that is net harmful (removes more honest clients than attackers, in the wrong direction) relative to doing nothing.

This is strictly stronger than the AUC=0.46 finding, and in a specific, important way: AUC says the statistic doesn't *rank* malicious above benign consistently *on average* across randomly paired comparisons. This result says something sharper — picking the single best threshold WITH FULL HINDSIGHT over the entire 919-round, 9,190-observation log, there is **no threshold anywhere in [0,1] that doesn't make things worse than flagging nobody at all.** The paper's own schedule (rho ∈ {0.5, 0.4, 0.3, 0.2}) sits squarely inside this universally-negative region (J = -5.10 to -5.77pp) — it is not an unlucky choice of threshold within an otherwise-usable statistic; there is no lucky choice available. **The statistic is not miscalibrated. It is actively anti-correlated with the thing it's meant to detect, at every operating point tested.**
