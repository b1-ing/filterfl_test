# Experiment 1 -- logit-scaling alpha sweep

Model: client 36, poison_epochs=6, poison_rate=0.5,
poison_lr=0.005 (purpose-built strong-backdoor model, NOT representative of
step2_cellA_stealthy's poison_rate=0.1/poison_epochs=2). MA floor dropped for this
experiment per decision in exp1_strong_backdoor_NOTE.md -- softmax confidence vs.
logit scale is an output-layer property with no dependence on clean accuracy;
ASR=85.24 is an unambiguous landed backdoor, which is all this test needs.

rho=0.5 (cellA's active value at epoch 60; the schedule's first break is at epoch
500, so this is the only rho value this checkpoint ever exercised).

Scaling applied: `model.fc.weight`, `model.fc.bias` multiplied by alpha (the final
linear layer, i.e. exactly "scale the logits"). Confidence on T is computed exactly
as `defense.stage3_classify_clients` does it:
`softmax(logits).mean(dim=0).max()` -- over the patched (ASR) test set as a stand-in
for the GAN-generated trigger set T (no Stage 1/2 GAN run was performed for this
forward-pass-only test; flagged as a simplification).

## Results

| alpha | confidence on T | would trip rho>0.5 at target class | ASR | clean acc | argmax unchanged |
|---|---|---|---|---|---|
| 1.0 | 0.7051 | yes | 85.24 | 24.59 | yes |
| 0.7 | 0.6124 | yes | 85.24 | 24.59 | yes |
| 0.5 | 0.4992 | no | 85.24 | 24.59 | yes |
| 0.3 | 0.3282 | no | 85.24 | 24.59 | yes |
| 0.2 | 0.2356 | no | 85.24 | 24.59 | yes |
| 0.1 | 0.1570 | no | 85.24 | 24.59 | yes |

Argmax invariance: scaling a linear layer's weight+bias by a positive scalar is a
monotonic transform and mathematically cannot change which class has the largest
logit -- the check above confirms this held in practice (no numerical surprises),
not that it holds in principle.

## Verdict -- stated precisely, not stronger than the evidence supports

**FilterFL's Stage 3 thresholds a quantity (batch-averaged softmax confidence)
that is NOT invariant under a transformation (positive scalar multiplication of
the final layer) the backdoor function PROVABLY IS invariant under, and the
implementation has no normalisation defending against it.**

That is the entire finding. It is intentionally not stated any stronger than
this. In particular:

- The ASR/argmax "invariance" in the table above is **not an empirical
  discovery** -- it is a guaranteed mathematical consequence of multiplying a
  linear layer's output by a positive scalar (a monotonic transform cannot
  change which class has the largest logit). The table confirms this held in
  practice (no ties/numerical surprises); it does not demonstrate anything a
  reviewer couldn't derive on paper without running code.
- The one thing this experiment actually measured, that isn't derivable a
  priori, is *where* the confidence statistic crosses rho: alpha=0.5 is the
  threshold-crossing value for this specific model.
- This result is about a **standalone model under forward passes only.** It
  says nothing yet about whether the gap survives FedAvg's parameter averaging
  with other clients' updates -- see `exp1_fl_loop_followup_NOTE.md` for that
  (necessary, not optional) follow-up.
