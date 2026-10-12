# SiftFL — assumptions made in the absence of the paper

The SiftFL paper (IEEE Xplore, ~Aug 2026) was not accessible to read directly.
Everything below is implemented from the one-line description in the task
brief ("a scheduled, trust-history variant of FilterFL") plus ordinary
engineering judgment, and is called out explicitly rather than silently
assumed. Implementation: `filterfl/config.py` (`ScheduleConfig`,
`TrustHistoryConfig`), `filterfl/trust.py`, orchestration in `filterfl/run.py`.

## Assumption 1 — what happens on a non-detection round

SiftFL claims up to 99% cost reduction by running Stages 1–2 only every `k`
rounds, which implies `k` is large. Two readings are equally plausible for
what happens on the other `k-1` rounds:

- **`last_filter` (chosen default)**: re-use the most recently computed
  trigger set T to classify this round's sampled clients, at full ρ-threshold
  cost of Stage 3 only (cheap — no CGAN retraining). Filtering still happens
  every round; only the expensive trigger-regeneration (Stages 1–2) is
  skipped.
- **`fedavg`**: do nothing — plain FedAvg with no filtering at all on
  non-detection rounds.

Both are implemented (`schedule.non_detection_round_policy` config flag).
`last_filter` is the default because it's the more defensible reading of "cost
reduction" (99% of the cost is in the CGAN training, not the Stage-3
classification pass) and because a defence that does *nothing* for up to
k-1 consecutive rounds would be a strange thing to publish as a security
improvement. This is a guess, not a recovered fact — flip the flag to compare.

## Assumption 2 — schedule shape

`fixed` (every k-th round) and `randomized` (per-round Bernoulli with mean
rate 1/k, seeded off `(seed, epoch)` for reproducibility) are both
implemented (`schedule.randomized`). No basis in the (unread) paper for
preferring one — exposed as a flag because the brief specifically asked for
both and said the difference matters.

## Assumption 3 — trust update rule

`trust.py`: trust is a per-client scalar, starting at `initial_trust` (0.0),
moving by a fixed additive step per round a client is *sampled*:
`-decay_rate` if flagged that round, `+accumulation_rate` if clean. A client
is excluded outright if trust drops below `exclusion_threshold`, regardless
of that round's Stage-3 confidence on T.

Assumptions baked into this, any of which could be wrong relative to the
actual paper:
- **Unbounded, not EMA.** Trust can grow or shrink without limit. An
  exponential-moving-average formulation (`trust = β·trust + (1-β)·signal`)
  is an equally plausible reading of "trust history" and would behave very
  differently under a patient attacker (EMA forgets old evidence, this
  doesn't).
- **Updates on every sampled round, not just detection rounds.** On a
  non-detection round under the `fedavg` policy there is no Stage-3 signal to
  update trust from at all — in that case trust silently doesn't move for
  that client that round. Under `last_filter`, trust updates off the
  possibly-stale T.
- **Symmetric accumulation/decay rates are NOT assumed equal.** They're
  independent config knobs (`accumulation_rate`, `decay_rate`) precisely
  because an attacker's optimal strategy depends heavily on their relative
  size (slow accumulation + fast decay favors the defender; the reverse
  rewards patience).

## Why this matters for the actual research question

Per CODE_NOTES.md's "Design assumptions" section, FilterFL's whole trust
chain rests on G_old being benign, with nothing to check that beyond "Stage 3
didn't flag it last round." SiftFL's scheduling (Assumption 1/2 above) means
that on non-detection rounds, under the `fedavg` policy, G_old can go
*multiple rounds* without any Stage-3 check at all — the exact widened window
the generalization work is aimed at. Under `last_filter`, G_old is still
checked every round, but against an increasingly stale trigger set, which is
a softer version of the same widening. Either reading of this assumption
plausibly make the low-and-slow attack easier, which is why both are kept
live as config flags rather than collapsed to one default without comment.
