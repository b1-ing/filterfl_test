# Experiment 1 -- purpose-built strong-backdoor client model (final, v2 methodology)

**NOT representative of step2_cellA_stealthy** (poison_rate=0.1, poison_epochs=2,
the stealthy variant actually under test). This model exists solely so
Experiment 1's alpha-scaling sweep has an unambiguous, landed backdoor. Do not let
these numbers drift into the Part C sweep writeup.

Base weights: step2_cellA_stealthy checkpoint, epoch 60 (half-trained
global model) -- see caveat below.

## MA floor methodology -- corrected TWICE, final version below

- **v0 (first attempt):** floor = global aggregated model's MA - 5 (= 38.39).
  Wrong: a global FedAvg average isn't comparable to any single client's local
  model -- individual local models are routinely well below the aggregate.
- **v1 (first correction):** floor = a size-matched *different* benign client's
  MA - 5 (client 7, matched to malicious client 34's partition size only).
  Wrong: size-matching ignores shard *skew*. Client 34 turned out pathological
  (241/349 samples concentrated in 2 classes, 0 samples in class 4) -- worse than
  client 7's skew -- so the v1 delta (-16.06) conflated poisoning damage with
  shard-quality damage, and the decision rule that fired on that number was acting
  on a confounded measurement.
- **v2 (final, used below):** for every adversarial client tested, train that SAME
  client benign (identical local_epochs=6, lr=0.005, no poisoning) as its own
  control. Floor = that client's own benign MA - 5.0. Never a
  different client, never the global average. Each trial also resets to an
  identical RNG snapshot (captured right after checkpoint load) before running, so
  epoch-count and poisoning comparisons aren't contaminated by cross-trial RNG
  drift (a confound present in the very first poison_epochs grid).

## Results (poison_rate=0.5, poison_lr=0.005 fixed; client and poison_epochs varied)

| client | poison_epochs | malicious MA | own benign-control MA | floor | MA result | ASR | ASR result |
|---|---|---|---|---|---|---|---|
| 34 | 2 | 17.21 | 31.28 | 26.28 | FAIL | 96.43 | PASS |
| 34 | 3 | 15.00 | 31.28 | 26.28 | FAIL | 97.20 | PASS |
| 34 | 4 | 18.57 | 31.28 | 26.28 | FAIL | 91.87 | PASS |
| 34 | 6 | 24.02 | 31.28 | 26.28 | FAIL | 73.80 | PASS |
| 36 | 2 | 17.38 | 40.47 | 35.47 | FAIL | 97.41 | PASS |
| 36 | 3 | 20.32 | 40.47 | 35.47 | FAIL | 74.49 | PASS |
| 36 | 4 | 28.88 | 40.47 | 35.47 | FAIL | 88.00 | PASS |
| 36 | 6 | 29.51 | 40.47 | 35.47 | FAIL | 86.18 | PASS |
| 45 | 2 | 10.14 | 33.34 | 28.34 | FAIL | 100.00 | PASS |
| 45 | 3 | 17.97 | 33.34 | 28.34 | FAIL | 93.32 | PASS |
| 45 | 4 | 23.36 | 33.34 | 28.34 | FAIL | 42.70 | FAIL |
| 45 | 6 | 16.87 | 33.34 | 28.34 | FAIL | 96.61 | PASS |

**Selected: none -- no (client, poison_epochs) pair cleared both thresholds.**

## MA floor withdrawn for Experiment 1

The MA floor was introduced by the user as a guard against building a model "any
defence would catch trivially." It rejected all twelve configurations above,
including several at 86-96% ASR. A criterion that rejects every working backdoor
tested is more likely wrong than evidence that no working backdoor exists in this
setting.

Two reasons, given by the user, for dropping it here rather than continuing to
tune attack parameters:

1. It may be unsatisfiable in this setting: 349-849 samples, severe non-IID skew,
   a half-trained base model -- there may be no `poison_rate` where a landed
   backdoor coexists with near-benign MA on shards this thin. That would be a fact
   about the setting, not about the attack.
2. Experiment 1 doesn't need it. The experiment tests whether softmax confidence
   can be decoupled from the backdoor function by scaling logits -- a property of
   the output layer. Clean accuracy has no bearing on that question. The floor was
   imported from a peer-detectability concern (relevant to whether this attack
   would survive a similarity/clustering-based defence) and had been wrongly
   attached to this experiment.

**Decision: floor dropped for Experiment 1 only.** Proceeded with client 36,
poison_epochs=6 (MA=29.51, ASR=86.18 at time of selection -- best ASR-with-least-MA-
damage of the twelve) for the alpha sweep. ASR=86.18 is an unambiguous landed
backdoor; that is all this experiment requires.

The record should show the criterion was wrong, not that configurations were tuned
until one passed -- none passed. The MA numbers above are kept: the
malicious-vs-own-benign-control delta at each (client, poison_epochs) is exactly
what a similarity- or clustering-based defence would see on these updates, and will
be needed if this attack is later claimed to transfer beyond FilterFL. It is
recorded here as a property of this setting (thin, skewed shards + a half-trained
base model), not as a gate on Experiment 1.

## ASR-vs-poison_epochs trend (RNG-confound removed)

  client 34: pe=2 MA=17.2 ASR=96.4, pe=3 MA=15.0 ASR=97.2, pe=4 MA=18.6 ASR=91.9, pe=6 MA=24.0 ASR=73.8
  client 36: pe=2 MA=17.4 ASR=97.4, pe=3 MA=20.3 ASR=74.5, pe=4 MA=28.9 ASR=88.0, pe=6 MA=29.5 ASR=86.2
  client 45: pe=2 MA=10.1 ASR=100.0, pe=3 MA=18.0 ASR=93.3, pe=4 MA=23.4 ASR=42.7, pe=6 MA=16.9 ASR=96.6

If ASR falls as poison_epochs rises while MA stays roughly flat (per client), that
means the backdoor task is being learned faster than the clean task on a shard
that only weakly supports the clean task in the first place -- which is precisely
the premise FilterFL's whole detection design rests on (triggers learned faster
than normal knowledge). Worth having on record as an independent reproduction of
that premise in our own setup if the trend holds here.

## Caveat (does not block Experiment 1, revisit before anything goes near a paper)

Built on epoch-60 (half-trained) global weights. Logit scaling is an output-layer
property and should in principle be insensitive to base-model convergence, but if
the alpha sweep finds something interesting, repeat it on a well-trained
(near-400-epoch) model -- a half-trained model's confidence/logit-scale
relationship may not generalize.
