# Experiment 3 -- is Stage 2 recovering triggers or UAPs?

Motivated by the detection-baseline result (exp1_detection_baseline_NOTE.md,
confirmed not a wiring bug): 10/10 cold Stage 1/2 runs against a real, strong
class-0 backdoor locked onto class 4 instead. This experiment removes the backdoor
entirely -- G_agg here is the epoch-60 checkpoint aggregated with
10 BENIGN clients' local updates ONLY, no malicious client sampled or
present (`cfg.is_poison=False`, belt-and-suspenders). G_agg MA=47.16.
rho=0.5, target_class=0.

## Q1: does Stage 2 still produce an x_l that flips clean-image classifications?

5 independent Stage 1/2 runs. "composite-flip" = overlay T on
clean test images (Stage 2's own "added" composite) and check the resulting
predicted-class distribution under the benign G_agg model. "raw-T stage3" =
Stage 3's actual statistic (softmax(G_agg(T)).mean(dim=0).max()) on the
unmodified trigger images.

| run | composite-flip top class | composite-flip rate | raw-T pred_class | raw-T confidence |
|---|---|---|---|---|
| 0 | 9 | 22.7% | 2 | 0.5528 |
| 1 | 9 | 21.9% | 2 | 0.5504 |
| 2 | 9 | 22.1% | 2 | 0.5585 |
| 3 | 9 | 22.0% | 2 | 0.5344 |
| 4 | 9 | 22.0% | 2 | 0.5549 |

raw-T stage3 pred_class distribution across runs: {2: 5}  (NOT {9: 5} -- that was a
reporting bug in the generating script, which accidentally summarized the
composite-flip top-class column instead of the raw-T pred_class column; the
per-row table above is correct and is what this distribution is drawn from)

composite-flip top_class distribution across runs: {9: 5}

## Q2: do benign client models trip rho on this T?

| client | pred_class | confidence | flagged |
|---|---|---|---|
| 0 | 2 | 0.8930 | no |
| 1 | 8 | 0.9723 | no |
| 2 | 2 | 0.9566 | no |
| 3 | 4 | 0.6093 | no |
| 4 | 4 | 0.3707 | no |
| 5 | 5 | 0.4452 | no |
| 6 | 2 | 0.6208 | no |
| 7 | 8 | 0.5658 | no |
| 8 | 2 | 0.9716 | no |
| 9 | 4 | 0.6209 | no |

False-positive rate: 0/10 clients flagged. **Caveat: this is partly chance
alignment, not evidence the confidence threshold is protective.** None of these
10 benign clients' own models were classified as target_class 0 by this round's T
-- that's what actually prevented flagging, not low confidence. Confidence values
on the WRONG class ran as high as 0.97, 0.96, 0.92 for several clients. If a
future round's T happens to attract toward whatever class a benign client's model
predicts, nothing in this mechanism would stop it from being flagged at those
confidence levels.

## Verdict

**UAP hypothesis supported, with a refinement.** Three different measurements in
this project now show the same qualitative pattern: Stage 2 reliably (5/5 or
10/10, every time) converges to a trigger set that some specific class, but
**the class differs depending on whose weights are being probed, and in every
case tested it is NOT the real backdoor's target class (0):**

- malicious client 36's own weights (detection-baseline experiment): locks onto
  class **4**, 10/10 runs, confidence 0.37-0.45 (below rho).
- this round's benign G_agg, probed directly with Stage 3's own statistic: locks
  onto class **2**, 5/5 runs, confidence 0.53-0.56 (**above** rho=0.5 -- fails
  only the class-match half of the gate, not the confidence half).
- this round's benign G_agg, probed via composite-overlay-on-clean-images: locks
  onto class **9**, 5/5 runs, ~22% of composites classified that way (clean
  chance rate for 10 classes is ~10% -- more than double).

This is NOT a single universal architectural sink class (that would require the
same class every time regardless of model weights) -- it's something more
consistent with a genuine per-model UAP: Stage 2 finds whichever direction is
easiest to push THAT PARTICULAR d1/d2 pair toward, and that easiest direction is
reproducible for a fixed model but varies across models. That is still the
Experiment 3 hypothesis, just more precisely stated: Stage 2 is not reconstructing
a planted trigger (it never once recovers class 0, the actual target, across 20
total runs spanning two very different models) -- it is reliably finding *some*
model-specific adversarial direction, consistent with a UAP-style optimum rather
than trigger recovery, in both the backdoored and the fully benign case.
