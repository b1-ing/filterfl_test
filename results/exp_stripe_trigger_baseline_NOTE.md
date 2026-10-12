# Contiguous white-stripe trigger -- does trigger geometry matter?

**Why:** FilterFL's CCS paper text describes the default trigger as "a white
stripe in the upper left corner." Our implementation (faithfully copied from the
reference repo's `cifar_params.yaml` `0_poison_pattern`) is actually two broken
4-pixel dashes per row across rows 0 and 3, with gaps at columns 4-5 -- confirmed
by rendering both patterns as images (see `trigger_pattern.png` vs
`trigger_pattern_stripe.png`). This test isolates trigger geometry as the ONE
changed variable: same client (36), same pinned checkpoint, same GAN
budget (reference defaults: stage1_rounds=10, stage2_rounds=5,
images_per_class=32), same attack strength (poison_epochs=6,
poison_rate=0.5) -- only the trigger shape changes, from scattered dashes to a
genuinely contiguous 2x8 white rectangle (same 16-pixel count) in the identical
upper-left location.

**Pinned checkpoint:** epoch 65 (`step2_cellA_stealthy_snap_e00065_ckpt.pt`,
a frozen snapshot -- NOT the live `_ckpt.pt` file the background training job
continues to overwrite). 5 epochs later than the original
`exp1_detection_baseline_NOTE.md`'s epoch-60 checkpoint (the earliest pinned
snapshot available; epoch 60 itself was never saved standalone before being
overwritten).

Malicious client 36 with the STRIPE trigger: MA=27.43,
ASR=92.73. rho=0.5, target_class=0.

## Results

| run | pred_class | confidence | class_match (==target) | would trip rho |
|---|---|---|---|---|
| 0 | 4 | 0.2919 | no | no |
| 1 | 4 | 0.3021 | no | no |
| 2 | 4 | 0.3003 | no | no |
| 3 | 4 | 0.2914 | no | no |
| 4 | 4 | 0.3003 | no | no |
| 5 | 0 | 0.2976 | yes | no |
| 6 | 4 | 0.3046 | no | no |
| 7 | 0 | 0.2880 | yes | no |
| 8 | 4 | 0.3050 | no | no |
| 9 | 4 | 0.2965 | no | no |

**pred_class distribution across 10 runs:** {0: 2, 4: 8}

**class_match (pred_class==target_class) rate: 2/10**
**would actually trip Stage 3 (class_match AND confidence>rho): 0/10**

## Comparison

| trigger | geometry | pred_class distribution | class_match rate |
|---|---|---|---|
| reference (`exp1_detection_baseline_NOTE.md`) | scattered dashes, 2 rows x 2 blocks of 4 | {4: 10} | 0/10 |
| stripe (this run) | contiguous 2x8 rectangle | {0: 2, 4: 8} | 2/10 |

## Read

**A real but weak and non-actionable shift.** Class-match went from 0/10 (scattered
dashes) to 2/10 (contiguous stripe) -- a nonzero difference, and in the direction
the hypothesis predicted (a contiguous trigger is easier for Stage 2 to recover
than scattered pixels). But it does not come close to rescuing detection: **both
of the 2 class-matching runs had confidence ~0.29, far below this checkpoint's
rho=0.5** -- so 0/10 would actually trip Stage 3 either way. The dominant outcome
(8/10) is still the class-4 attractor-lock, identical to the scattered-dash
baseline. At this sample size (n=10 per condition, single client), 0/10 -> 2/10
is also well within what a few unlucky/lucky GAN initializations could produce by
chance -- not strong enough to call trigger geometry a confirmed explanation, only
a plausible contributing factor worth a larger n if pursued further.

**Caveat on this test's own validity, stated plainly:** malicious client 36's MA
cratered to 27.43 under the stripe trigger (vs. the original scattered-dash test's
much higher MA at a comparable checkpoint) -- the contiguous stripe appears
considerably more disruptive to clean accuracy during local training than the
scattered pattern, at the same poison_epochs/poison_rate. That's a confound of
its own: this comparison isn't purely "same backdoor, different trigger shape,"
it's also "a more disruptive attack, different trigger shape" -- the two
patterns may not be equally strong attacks even holding poison_epochs/poison_rate
fixed, which complicates reading the class-match-rate difference as attributable
to geometry alone.

**Bottom line for the pending reproduction-priority decision:** this result does
not change the recommendation to pause further detection-mechanism findings and
prioritize reproducing the paper's MA/ASR numbers first, as already decided. It's
suggestive (geometry may matter at the margin) but neither strong enough nor
clean enough to act on before reproduction is settled.
