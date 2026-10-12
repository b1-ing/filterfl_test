# C-dependent normalization: does Stage 2's loss balance survive moving off CIFAR-10?

**Why:** FilterFL's Stage 2 loss (`L2`) mixes two structurally different kinds of
terms: a "naturalness" term (std of a softmax distribution across C classes) and
one or more discriminator-probability terms (a single class's softmax probability,
always in [0,1]). If these two kinds of terms have ceilings that scale differently
with the number of classes C, then fixed weights (γ, λ) tuned against CIFAR-10
(C=10) produce a DIFFERENT effective loss balance on GTSRB (C=43) or CIFAR-100
(C=100) -- even with zero code changes, zero retraining of the weights themselves.
Pure arithmetic, no model or GPU needed.

## What is NOT the issue (checked and ruled out)

The empirical-mean normalization (dividing by the number of samples in a batch) is
already handled correctly in both the reference's `gan.py` and this project's
`defense.py`: every "naturalness" and "discriminator probability" term is computed
via `.mean(dim=0)` over the ACTUAL batch size, whether that's `(C-1) * images_per_class`
samples (the "added to other classes" branch) or `images_per_class` samples (the
"subtracted from own class" branch). There is no raw, unnormalized sum anywhere that
would blow up proportionally to C from sample-count alone. This specific concern
(1/C vs 1/(C-1) vs unnormalized SAMPLE COUNTS) is a non-issue in the actual code.

## What IS the issue: the naturalness term's CEILING shrinks with C, the probability terms' CEILING does not

The naturalness term is `std(softmax(D(x)), dim=1)` -- the standard deviation of a
C-way softmax output, across its C class-probabilities, for a given sample. This
statistic has a hard mathematical ceiling that depends on C alone, achieved in the
limit of a maximally confident (near one-hot) prediction:

```
max_std(C) = sqrt(C - 1) / C
```

Derivation: for the one-hot limit [1, 0, 0, ..., 0] (C entries), mean = 1/C, and
variance = (1/C)(1-1/C)^2 + ((C-1)/C)(0-1/C)^2 = (C-1)/C^2, so std = sqrt(C-1)/C.

| C | max_std(C) |
|---|---|
| 10 (CIFAR-10) | sqrt(9)/10 = **0.3000** |
| 43 (GTSRB) | sqrt(42)/43 = **0.1507** |
| 100 (CIFAR-100) | sqrt(99)/100 = **0.0995** |

**The naturalness term's achievable ceiling drops by half going from CIFAR-10 to
GTSRB, and by two-thirds going from CIFAR-10 to CIFAR-100** -- purely from the
mechanics of a softmax over more classes, regardless of how "natural" or
"unnatural" the actual image looks, and regardless of any retraining.

The probability-based terms (`d2_target_prob` and similar) are themselves a single
class's softmax output -- always bounded in [0, 1] **independent of C**. Their
ceiling does not shrink as C grows (a well- or poorly-trained discriminator can in
principle push a target-class probability arbitrarily close to 0 or 1 regardless of
how many total classes exist).

So: fixed weights (γ, λ) chosen to balance a C=10 naturalness ceiling of 0.30 against
probability terms with ceiling 1.0 will OVER-weight a far smaller naturalness
contribution at higher C, without the loss function's author having to change
anything in the code.

## Effective term balance at C = 10, 43, 100

Using this project's paper-faithful 3-term formulation (γ=0.75, λ=0.125,
1-γ-λ=0.125 -- `filterfl/defense.py`'s Stage 2 loss:
`γ·(naturalness_added + naturalness_subbed)/2 + λ·(1 - d2_added_target_prob) +
(1-γ-λ)·d2_subbed_target_prob`), computing each term's maximum achievable
contribution (its ceiling) and its share of the total ceiling:

| C | naturalness ceiling | λ-term ceiling | (1-γ-λ)-term ceiling | total ceiling | naturalness share |
|---|---|---|---|---|---|
| 10 | 0.2250 | 0.1250 | 0.1250 | 0.4750 | **47.4%** |
| 43 | 0.1130 | 0.1250 | 0.1250 | 0.3630 | **31.1%** |
| 100 | 0.0746 | 0.1250 | 0.1250 | 0.3246 | **23.0%** |

Using the reference's ACTUAL shipped formula instead (no independent λ -- a fixed
0.75/0.25 split applied across two naturalness terms and three probability-shaped
terms per `gan.py:149-178`, CODE_NOTES.md discrepancy #1):

| C | naturalness ceiling (2 terms) | probability-terms ceiling (3 terms) | total ceiling | naturalness share |
|---|---|---|---|---|
| 10 | 0.4500 | 0.7500 | 1.2000 | **37.5%** |
| 43 | 0.2261 | 0.7500 | 0.9761 | **23.2%** |
| 100 | 0.1492 | 0.7500 | 0.8992 | **16.6%** |

Both formulations show the same qualitative collapse: **naturalness's share of the
total achievable loss roughly halves from CIFAR-10 to GTSRB, and drops by
50-60% further from GTSRB to CIFAR-100.**

## What this predicts

If FilterFL's γ (and λ, in the paper-faithful reading) were tuned -- by hand or by
sweep -- to produce good trigger reconstructions on CIFAR-10, the SAME weights
applied to GTSRB or CIFAR-100 are optimizing a loss that has quietly shifted most of
its achievable range onto the discriminator-probability terms and away from the
naturalness constraint, without anyone changing a hyperparameter. Concretely: Stage
2's search for a trigger becomes progressively LESS constrained by "does this look
like a plausible image" and progressively MORE driven by "does this fool the
discriminator into predicting the target class," as C grows -- which is exactly the
kind of shift that could produce a less visually-coherent, more adversarial-noise-like
reconstructed trigger on higher-class-count datasets, independent of any other
difference between CIFAR-10, GTSRB, and CIFAR-100 (image size, task difficulty,
sample count per class, etc.).

**This is a testable, falsifiable prediction stated without running anything**:
FilterFL's own reported GTSRB/CIFAR-100 trigger-reconstruction quality (if visually
inspectable in their paper's figures) should look less constrained/more
noise-like than their CIFAR-10 reconstructions, specifically because of this
mechanical loss-balance shift -- not because GTSRB/CIFAR-100 are simply "harder"
datasets in some generic sense.
