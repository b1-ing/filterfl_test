# Detection baseline -- how often does Stage 1/2 recover the target class, cold, at alpha=1.0?

**Why this matters:** the FL-loop follow-up (`exp1_fl_loop_followup_NOTE.md`) found
that client 36 (85% ASR, poison_rate=0.5, poison_epochs=6,
**no attack on the detector, alpha=1.0, unscaled**) was NOT flagged by Stage 3
because the one Stage 1/2 run that round produced a trigger set pointing at class
4, not the target class 0. That is FilterFL failing to detect a strong backdoor
under zero adversarial pressure. This check asks: is that one round unrepresentative
(bad luck), or is Stage 1/2 unreliable at recovering the target class in general?
**This number is the ceiling on every detection result in this project** -- no
evasion claim is interpretable until it's known.

Setup: the SAME malicious client model (fixed, MA=24.66, ASR=85.23)
run through `run_detection` (Stage 1+2) 10 independent times. RNG is allowed
to evolve naturally run-to-run (not reset between runs) -- these are meant to
emulate 10 independent cold-start detection attempts, not 10 repeats of
one identical draw. G_agg for this check is `ori_weight` aggregated with only this
one malicious client's update (eta=1), so d2 genuinely reflects a model that has
absorbed the backdoor rather than one that never saw this client's contribution.
rho=0.5, target_class=0.

## Results

| run | pred_class | confidence | class_match (==target) | would trip rho |
|---|---|---|---|---|
| 0 | 4 | 0.3778 | no | no |
| 1 | 4 | 0.3705 | no | no |
| 2 | 4 | 0.3963 | no | no |
| 3 | 4 | 0.3804 | no | no |
| 4 | 4 | 0.4123 | no | no |
| 5 | 4 | 0.4510 | no | no |
| 6 | 4 | 0.4467 | no | no |
| 7 | 4 | 0.3911 | no | no |
| 8 | 4 | 0.3736 | no | no |
| 9 | 4 | 0.3771 | no | no |

**pred_class distribution across 10 runs:** {4: 10}

**class_match (pred_class==target_class) rate: 0/10**
**would actually trip Stage 3 (class_match AND confidence>rho): 0/10**

pred_class is concentrated on a consistent class, but not reliably target_class 0 -- Stage 1/2 recovers SOME stable direction from this classifier, just not reliably the planted-trigger direction.

## Wiring check -- ruled out before accepting the above as real

Checked per user request before trusting the 0/10 result: is there a target_class
mismatch between the poisoning path and the detection class-match check?

1. **Live object identity:** `cfg.target_class == attacker.cfg.target_class == 0`,
   same `cfg` object (`attacker.cfg is cfg` -> True). No possibility of divergence
   within this run.
2. **Actual poisoned-sample labels, printed not inferred:** of 59 samples poisoned
   in one training batch (poison_rate=0.5), every changed label's new value was
   exactly `[0]` -- the config's target_class, nothing else.
3. **Predicted-class distribution on real triggered test inputs** (not ASR's
   scalar, the full histogram): `{0: 7677, 1: 8, 2: 22, 3: 2, 4: 1127, 5: 93, 6: 69,
   7: 2}` out of 9000. 85.30% predicted class 0 -- matches ASR exactly. The
   backdoor's real target is genuinely class 0, confirmed directly.
4. Notably, class **4** is the real trigger's distant second-place class (1127/9000
   = 12.5%) -- the same class Stage 1/2 locks onto 10/10 times. Suggestive that
   class 4 is some kind of attractor/sink class for this classifier in general, not
   unrelated noise.

**Conclusion: not a bug.** The 0/10 detection-baseline finding stands. Per the
reordering decision, Experiment 3 (benign G_agg) runs next, before Experiment 2.
