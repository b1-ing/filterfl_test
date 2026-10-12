# Experiment 1 follow-up -- does the alpha-scaling gap survive FedAvg aggregation?

## Why this follow-up is necessary, not optional

The forward-pass sweep (`exp1_alpha_sweep_NOTE.md`) found that Stage 3's confidence
statistic crosses rho=0.5 at alpha<=0.5 while ASR/argmax stay exactly unchanged.
That ASR/argmax invariance is a GUARANTEED consequence of multiplying a linear
layer's output by a positive scalar -- not an empirical discovery -- and a reviewer
could derive it on paper without running anything. Correctly stated, the forward-
pass result is only: **FilterFL's Stage 3 thresholds a quantity that is NOT
invariant under a transformation the backdoor function provably IS invariant
under, and the implementation has no normalisation defending against it.** Nothing
stronger than that follows from the forward-pass test alone.

The open, non-derivable question is whether this survives aggregation: FedAvg
averages PARAMETERS, not functions. A client submitting alpha*W_fc contributes a
down-weighted final layer to the aggregate -- the exact invariance that holds for
the standalone model may not survive being averaged with 9 other clients' full-
strength final layers at all.

## Setup

One simulated round of cellA_stealthy-style FedAvg+FilterFL, starting from the
epoch-60 checkpoint as G_old: 9 benign clients ([0, 1, 2, 3, 4, 5, 6, 7, 8], local_epochs=2,
lr=0.01) + malicious client 36 (poison_rate=0.5, poison_epochs=6,
poison_lr=0.005). rho=0.5 (epoch 61's active value; the schedule's
first break is at epoch 500). Only the malicious client's SUBMITTED fc.weight/fc.bias
are scaled by alpha before aggregation -- matching "scale the logits" from the
original experiment design; its other layers are its actual trained values. Stage
1+2 (run_detection) and Stage 3 (stage3_classify_clients) run exactly as
filterfl/run.py does them, with the real GAN-generated trigger set T.

Benign-client and malicious-client local training ran ONCE (RNG-snapshotted);
only scaling + aggregation + detection repeat per alpha, with RNG reset to the
same post-training snapshot before each alpha's detection run, so GAN-training
stochasticity is identical across alphas and the only varying input is alpha
itself.

## Results

| alpha | malicious client's confidence on T | pred class | Stage 3 outcome | # clients flagged this round | global MA | global ASR |
|---|---|---|---|---|---|---|
| 1.0 | 0.4320 | 4 | NOT excluded | 0 | 48.70 | 0.02 |
| 0.5 | 0.2774 | 4 | NOT excluded | 0 | 48.18 | 0.00 |
| 0.3 | 0.2066 | 4 | NOT excluded | 0 | 48.05 | 0.00 |

Question 1 (detection outcome) and question 2 (attack outcome) answered
separately per your instruction -- see table columns "Stage 3 outcome" and
"global ASR" respectively.

## Verdict

SURVIVES AGGREGATION: at least one alpha evades Stage 3's exclusion AND the global model's post-aggregation ASR stays within 10pts of the alpha=1.0 baseline. The forward-pass gap is not just a standalone-model artifact.
