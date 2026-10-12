# Before/after the rho=0.5 -> 0.4 crossing (epoch 500)

Requested as a clean before/after rather than folding into a running average.
Zero-compute, parsed directly from `results/step2_cellA_stealthy.csv`.

**BEFORE window: epochs 450-499** (rho=[0.5], n=50 rounds)
**AFTER window: epochs 500-510** (rho=[0.4, 0.5], n=11 rounds)

## Totals

| | tp total | fp total | detected total (tp+fp) | tp/round | fp/round |
|---|---|---|---|---|---|
| before (rho=0.5) | 0 | 20 | 20 | 0.00 | 0.40 |
| after (rho=0.4) | 0 | 12 | 12 | 0.00 | 1.09 |

## Confidence distribution

| | flagged clients | unflagged clients |
|---|---|---|
| before (rho=0.5) | n=20 mean=0.745 min=0.558 max=1.000 median=0.698 | n=480 mean=0.606 min=0.204 max=0.999 median=0.555 |
| after (rho=0.4) | n=12 mean=0.679 min=0.403 max=0.974 median=0.619 | n=98 mean=0.631 min=0.273 max=0.998 median=0.583 |

## Per-round detail -- BEFORE (rho=0.5)

| epoch | rho | tp | fp | detected_count |
|---|---|---|---|---|
| 450 | 0.5 | 0 | 1 | 1 |
| 451 | 0.5 | 0 | 0 | 0 |
| 452 | 0.5 | 0 | 0 | 0 |
| 453 | 0.5 | 0 | 1 | 1 |
| 454 | 0.5 | 0 | 0 | 0 |
| 455 | 0.5 | 0 | 0 | 0 |
| 456 | 0.5 | 0 | 0 | 0 |
| 457 | 0.5 | 0 | 0 | 0 |
| 458 | 0.5 | 0 | 0 | 0 |
| 459 | 0.5 | 0 | 1 | 1 |
| 460 | 0.5 | 0 | 1 | 1 |
| 461 | 0.5 | 0 | 1 | 1 |
| 462 | 0.5 | 0 | 0 | 0 |
| 463 | 0.5 | 0 | 1 | 1 |
| 464 | 0.5 | 0 | 1 | 1 |
| 465 | 0.5 | 0 | 1 | 1 |
| 466 | 0.5 | 0 | 0 | 0 |
| 467 | 0.5 | 0 | 0 | 0 |
| 468 | 0.5 | 0 | 0 | 0 |
| 469 | 0.5 | 0 | 0 | 0 |
| 470 | 0.5 | 0 | 0 | 0 |
| 471 | 0.5 | 0 | 0 | 0 |
| 472 | 0.5 | 0 | 1 | 1 |
| 473 | 0.5 | 0 | 1 | 1 |
| 474 | 0.5 | 0 | 1 | 1 |
| 475 | 0.5 | 0 | 0 | 0 |
| 476 | 0.5 | 0 | 0 | 0 |
| 477 | 0.5 | 0 | 0 | 0 |
| 478 | 0.5 | 0 | 0 | 0 |
| 479 | 0.5 | 0 | 0 | 0 |
| 480 | 0.5 | 0 | 0 | 0 |
| 481 | 0.5 | 0 | 0 | 0 |
| 482 | 0.5 | 0 | 1 | 1 |
| 483 | 0.5 | 0 | 0 | 0 |
| 484 | 0.5 | 0 | 0 | 0 |
| 485 | 0.5 | 0 | 0 | 0 |
| 486 | 0.5 | 0 | 0 | 0 |
| 487 | 0.5 | 0 | 2 | 2 |
| 488 | 0.5 | 0 | 0 | 0 |
| 489 | 0.5 | 0 | 1 | 1 |
| 490 | 0.5 | 0 | 1 | 1 |
| 491 | 0.5 | 0 | 0 | 0 |
| 492 | 0.5 | 0 | 1 | 1 |
| 493 | 0.5 | 0 | 0 | 0 |
| 494 | 0.5 | 0 | 0 | 0 |
| 495 | 0.5 | 0 | 0 | 0 |
| 496 | 0.5 | 0 | 0 | 0 |
| 497 | 0.5 | 0 | 1 | 1 |
| 498 | 0.5 | 0 | 2 | 2 |
| 499 | 0.5 | 0 | 0 | 0 |

## Per-round detail -- AFTER (rho=0.4)

| epoch | rho | tp | fp | detected_count |
|---|---|---|---|---|
| 500 | 0.5 | 0 | 2 | 2 |
| 501 | 0.4 | 0 | 3 | 3 |
| 502 | 0.4 | 0 | 1 | 1 |
| 503 | 0.4 | 0 | 0 | 0 |
| 504 | 0.4 | 0 | 3 | 3 |
| 505 | 0.4 | 0 | 0 | 0 |
| 506 | 0.4 | 0 | 0 | 0 |
| 507 | 0.4 | 0 | 1 | 1 |
| 508 | 0.4 | 0 | 0 | 0 |
| 509 | 0.4 | 0 | 1 | 1 |
| 510 | 0.4 | 0 | 1 | 1 |

## Read

**Correction to the window boundary:** epoch 500 itself still logged rho=0.5 -- the
schedule's breakpoint takes effect starting epoch 501, not at epoch 500. So the
true "after" block is epochs 501-510 (10 rounds at rho=0.4, not 11 at a mixed rho).
Recomputing on that basis: after-block fp_total=9 over 10 rounds = 0.90 fp/round
(vs. the 1.09 reported above, which included epoch 500's rho=0.5 round). The
direction and magnitude of the finding below are unchanged either way.

**Correction to the earlier framing: the rho schedule DECREASES over training
(0.5 -> 0.4 -> 0.3 -> 0.2), which makes the confidence threshold progressively
MORE PERMISSIVE, not stricter.** A client is flagged when `confidence >
active_rho`, so a lower rho number is easier to cross. "Tightening" was the wrong
word for this in earlier reporting and should not be read as "the detector gets
pickier" -- it gets more trigger-happy as training proceeds. Mechanically, a more
permissive threshold should increase both tp and fp opportunities. That is exactly
what's observed on the fp side: fp/round roughly doubled to tripled (0.40 -> 0.90-1.09
fp/round) immediately after the rho=0.5->0.4 step. **No true positives appeared in
either window (0 tp in 50 rounds before, 0 tp in 10-11 rounds after)** -- so making
the gate more permissive bought MORE false exclusions of honest clients with NO
accompanying increase in actual detections, in this first look. If the confidence
statistic carried real signal, a more permissive gate should trade some extra fp
for more tp; here it produces the fp cost with none of the tp benefit, consistent
with the `exp_csv_mining_NOTE.md` finding that the statistic barely distinguishes
malicious from benign clients in the first place.

The confidence distributions overlap heavily in both windows (before: flagged
mean=0.745 vs unflagged mean=0.606; after: flagged mean=0.679 vs unflagged
mean=0.631) -- flagged and unflagged populations are not cleanly separated, which
is consistent with a threshold crossing being driven by where the cutoff sits
relative to a largely-overlapping confidence distribution, not by a clean
separation between "backdoored" and "clean" signal.

**Caveat on sample size:** this after-window is short (10-11 rounds vs. 50), and 0
tp in a small window is not strong evidence of "no detection ever happens at
rho=0.4" by itself -- the full ~500-round block at rho=0.4 (epochs 500-1000) will
give a far more reliable estimate of whether this fp-without-tp pattern persists,
worsens, or reverses once there's more data. Flagging this as a first look, to be
revisited at the epoch-1000 crossing.
