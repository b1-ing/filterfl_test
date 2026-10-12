# AUC discrimination measure + per-regime fp/tp series

**Correction carried over from the previous note:** the rho schedule DECREASES
over training (0.5 -> 0.4 -> 0.3 -> 0.2), making the confidence threshold
progressively MORE PERMISSIVE as training proceeds, not stricter. All framing
below uses this corrected direction.

Zero-compute, parsed directly from `results/step2_cellA_stealthy.csv`
(534 rounds on disk as of this analysis).

## 1. AUC: does the effective Stage 3 score discriminate malicious from benign clients?

**Effective score definition:** Stage 3 flags a sampled client iff
`pred_class == target_class AND confidence > active_rho`. This AND-gate reduces
to a single monotonic score for ranking purposes: `score = confidence if
pred_class == target_class else 0.0` -- a client whose reconstructed class
doesn't match the target can NEVER be flagged regardless of its confidence, so it
belongs below every attainable flagged-confidence value (which lie in (0, 1]).
AUC = P(a randomly chosen malicious client's score > a randomly chosen benign
client's score), via Mann-Whitney U (ties count as 0.5). 0.5 = no discrimination,
1.0 = perfect separation.

**AUC = 0.4613 over all 5340 observations to date.** This sits essentially at chance (0.5) -- the single cleanest number in the project: Stage 3's effective score (confidence, gated by target-class match) has NO discriminative power between malicious and benign clients in this run, stated as a direct measurement rather than an inference from distribution overlap.

### AUC by rho regime

| rho regime | AUC | n malicious obs | n benign obs |
|---|---|---|---|
| 0.5 | 0.4619 | 518 | 4482 |
| 0.4 | 0.4510 | 34 | 306 |

## 2. Per-regime fp/tp rate series

| rho regime | epoch range | n rounds | tp total | fp total | tp/round | fp/round |
|---|---|---|---|---|---|---|
| 0.5 | 1-500 | 500 | 14 | 446 | 0.028 | 0.892 |
| 0.4 | 501-534 | 34 | 0 | 28 | 0.000 | 0.824 |

**fp/round is NOT monotonic as rho becomes more permissive across regimes seen so far (rho=0.5: 0.892/round -> rho=0.4: 0.824/round), while tp stays at nonzero in at least one regime throughout.** This does not (yet) cleanly show the strongest version of the pattern -- see the per-regime table for the actual numbers.

**Caveat:** only regimes actually reached in the run so far are shown above (rho=0.3
and rho=0.2 will be added once epochs 1000 and 1500 respectively are crossed, if
the run is extended that far). Early regimes also have far more data (rho=0.5 spans
500 rounds, the current regime spans far fewer) -- rates are still directly
comparable per-round, but confidence in the later regime's rate is lower until more
rounds accumulate.

## Correction to the earlier before/after report

`exp_rho_crossing_before_after_NOTE.md` compared two SHORT windows (epochs 450-499
vs 501-510: 50 and 10 rounds) and reported fp/round roughly doubling (0.40 -> 0.90).
With the FULL rho=0.5 regime now available (all 500 rounds), that headline doesn't
hold up: **fp/round over the complete rho=0.5 regime is 0.892 -- already close to
the rho=0.4 regime's 0.824, not the 0.40 baseline the small window suggested.** The
450-499 window was evidently an atypically quiet 50-round stretch within the
rho=0.5 regime, not representative of it. Taking the full-regime numbers as the
more reliable estimate: **fp/round did not rise when the gate became more
permissive -- it was already elevated throughout the rho=0.5 regime and stayed
essentially flat (if anything, slightly lower) into rho=0.4.** Similarly, tp was
NOT zero throughout rho=0.5 (14 total, 0.028/round) -- the earlier window's "0 tp
in 50 rounds" was also a locally quiet patch, not representative of the full
regime's (still very low, but nonzero) tp rate.

**This supersedes the earlier before/after headline. The AUC result above is the
reliable one** (computed over the complete regimes, not short hand-picked windows):
discrimination sits at chance (0.46) in both regimes measured so far, and the
elevated, roughly-flat fp rate alongside a near-zero tp rate holds across the full
rho=0.5 and rho=0.4 regimes alike -- not because the gate got more permissive, but
because the underlying confidence statistic doesn't separate malicious from benign
clients in the first place, at either threshold. The monotonic-rise pattern asked
about in point 2 doesn't appear (at least not yet, across these two regimes) --
but the reason the strongest version of the story doesn't show up is reassuring
rather than concerning: the fp rate was already as bad as it was going to get from
round 1, independent of where rho sits.
