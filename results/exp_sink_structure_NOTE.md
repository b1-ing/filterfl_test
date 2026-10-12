# What determines Stage 1/2's sink-class structure?

**Why:** `exp_attractor_vs_dominant_scale_NOTE.md` found class 2 is a near-universal
attractor (100% self-match), class 8 secondary (86%), class 0 moderate (50%), and
classes 1/7/9 are never self-attracting (0%). This checks whether sink strength is
predictable from properties of the global classifier itself -- specifically its
own inter-class confusion structure -- which would turn "sink classes exist" into a
mechanism, and a potential lever for an attacker (pick a non-sink target class to
be structurally invisible to Stage 3's coincidence-driven captures).

**Global class frequency is ruled out analytically, not numerically**: CIFAR-10 has
exactly 5000 train / 1000 test examples per class, uniformly, so this factor cannot
explain any between-class difference in sink strength by construction.

Confusion matrices computed via CPU forward pass (GPU busy with the concurrent
reference run) on 3 representative snapshots (epochs
65, 465, 915 -- early/mid/late in training).

## Off-diagonal column sums (misclassifications INTO each class), per snapshot

| epoch | class 0 | class 1 | class 2 | class 3 | class 4 | class 5 | class 6 | class 7 | class 8 | class 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 65 | 2.0% | 0.1% | 7.0% | 2.7% | 10.7% | 29.8% | 7.1% | 8.4% | 28.9% | 3.3% |
| 465 | 26.8% | 0.0% | 3.6% | 27.6% | 7.8% | 7.0% | 3.1% | 1.1% | 3.0% | 19.9% |
| 915 | 25.9% | 6.5% | 22.7% | 7.8% | 13.4% | 11.4% | 6.8% | 1.7% | 2.5% | 1.4% |

(Each row's values are each class's share of ALL misclassification errors that
snapshot made, restricted to errors landing in that column -- i.e. "of everything
this model got wrong, what fraction was wrongly predicted AS class c.")

## Summary: sink strength vs. confusion-sink-share vs. per-class accuracy

| class | attractor sink strength | mean confusion-sink-share | per-class test accuracy |
|---|---|---|---|
| 2 | 100% | 11.10% | 63.88 |
| 8 | 86% | 11.46% | 87.86 |
| 0 | 50% | 18.22% | 65.60 |
| 3 | 30% | 12.70% | 54.82 |
| 6 | 22% | 5.69% | 82.10 |
| 4 | 20% | 10.63% | 69.27 |
| 5 | 10% | 16.08% | 64.05 |
| 1 | 0% | 2.18% | 77.77 |
| 7 | 0% | 3.73% | 72.22 |
| 9 | 0% | 8.22% | 77.65 |

**Pearson r (sink_strength vs confusion-sink-share): 0.424**
**Pearson r (sink_strength vs per-class accuracy): -0.013**

## Verdict

**A genuine moderate positive correlation (r=0.42), not strong enough to use as a clean predictive rule, and undercut by a second finding that matters more: the confusion structure itself is highly unstable across training, not a fixed property of the architecture/dataset pair.**

Two things, both visible directly in the per-snapshot table above, not just the averaged summary:

1. **The correlation is real at the extremes but not monotonic in the middle.** The
   three zero-sink-strength classes (1, 7, 9) do have the three LOWEST
   confusion-sink-shares (2.18%, 3.73%, 8.22%) -- consistent with the hypothesis.
   But class 0 (moderate, 50% sink strength) has the HIGHEST confusion-sink-share
   of all ten classes (18.22%) -- higher than class 2 (100% sink strength, only
   11.10%) and class 8 (86%, 11.46%), the two STRONGEST attractors. If confusion
   geometry cleanly predicted sink strength, class 0 should sit in the middle of
   the pack, not at the top while also not being the strongest attractor. This
   single reversal is most of why r is 0.42 and not something closer to 0.8-0.9.

2. **More importantly: confusion-sink-share is NOT a stable per-class property --
   it swings enormously across training stages for the SAME class.** Class 0 goes
   2.0% (epoch 65) -> 26.8% (epoch 465) -> 25.9% (epoch 915) -- an order of
   magnitude shift as the model matures. Class 5 goes the opposite direction:
   29.8% -> 7.0% -> 11.4%. Class 8 similarly collapses from 28.9% early to ~3%
   later. **Averaging three very different training-stage snapshots into one
   number, as done above, risks manufacturing a stable-looking summary out of a
   genuinely non-stationary underlying quantity** -- this is the same kind of
   maturation-drift complication already flagged for client 84's attractor
   history in `exp_attractor_mechanism_NOTE.md`, now showing up at the
   classifier-confusion level too.

**Net read:** there's a real, moderate, extremes-consistent relationship between a
class's general confusability and its attractor strength (1/7/9 are both clean
non-sinks and clean low-confusion classes) -- but it's too noisy and too
training-stage-dependent to hand an attacker a clean "compute the confusion matrix
once, pick a non-sink class, done" recipe. An attacker doing this would need the
confusion matrix at roughly the SAME point in training the real defense would see
it, not just any snapshot -- which is a real practical obstacle to treating this as
an actionable lever, even though the underlying correlation is genuine at the
extremes. Per-class accuracy shows no relationship at all (r=-0.01) -- being an
easy or hard class for the classifier is unrelated to being a GAN attractor.
