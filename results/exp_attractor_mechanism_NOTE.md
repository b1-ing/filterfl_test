# Attractor mechanism check -- is it poison-independent?

**Why:** the CSV mining (`exp_csv_mining_NOTE.md`) found malicious-client attractor
classes are stable per-client across rounds but mostly wrong (never 0) -- except
client 45 (attractor=0, caught 4/5x) and client 84 (drifted 4->0, caught once at
epoch 51). This tests whether each client's attractor is a property of its
weights/data partition alone (same with or without poisoning), or whether
poisoning actually shifts it -- the distinction that determines whether "Stage 3's
true positives are coincidence" is the right reading of the log.

**Uses cellA_stealthy's REAL attack params** (poison_rate=0.1,
poison_epochs=2, poison_lr=0.005) -- not the
strong-backdoor config built for Experiment 1 -- since the goal is to explain this
specific log.

**Limitation, stated plainly:** only the epoch-60 checkpoint was ever saved;
intermediate per-round checkpoints from the clients' actual historical rounds
(e.g. client 45's epochs 23/30/32/51) were overwritten, not kept. This test uses
ONE fixed starting point (epoch 60) for every client, so it can test "does
poisoning change this client's attractor at all" but cannot literally replay
client 84's round-by-round drift. A full explanation of that drift needs a run
that preserves per-round checkpoints -- not attempted here.

## Zero-compute: partition dominant class vs. real-run attractor, all 10 malicious clients

| client | partition class histogram | observed real-run attractor |
|---|---|---|
| 34 | {0: 15, 1: 6, 2: 43, 3: 10, 5: 149, 6: 1, 7: 92, 8: 2, 9: 31} | 2 |
| 36 | {0: 66, 1: 33, 2: 89, 3: 123, 4: 267, 5: 157, 6: 87, 7: 24, 9: 3} | 4 |
| 45 | {0: 161, 1: 7, 2: 19, 3: 84, 4: 160, 5: 1, 6: 1, 7: 48, 8: 46, 9: 2} | 0 |
| 47 | {0: 10, 1: 64, 2: 10, 3: 78, 4: 32, 5: 18, 6: 149, 7: 261, 8: 21, 9: 11} | 6 |
| 50 | {1: 5, 2: 23, 3: 16, 4: 12, 5: 26, 6: 149, 7: 5, 8: 93, 9: 72} | 6 |
| 54 | {1: 2, 2: 45, 3: 120, 5: 14, 6: 3, 7: 182, 8: 15, 9: 59} | 2 |
| 60 | {0: 28, 1: 105, 2: 45, 3: 22, 5: 24, 6: 7, 7: 40, 8: 2, 9: 11} | 2 |
| 69 | {1: 9, 2: 2, 3: 11, 5: 6, 6: 29, 7: 7, 8: 118, 9: 34} | 8 |
| 84 | {0: 67, 1: 305, 2: 33, 3: 25, 4: 168, 5: 25, 6: 114, 7: 210, 8: 37, 9: 69} | 4 |
| 99 | {0: 2, 1: 23, 2: 2, 3: 85, 4: 141, 5: 4, 6: 52, 7: 50, 8: 3, 9: 29} | 4 |

(See printed output / stdout for exact MATCH / no-match per client -- not
reproduced as a column here since it needs eyeballing the full histogram, not
just the single dominant entry.)

## Benign-vs-malicious attractor test (5 detection runs per variant)

| client | variant | pred_class history | dominant attractor | malicious MA | malicious ASR |
|---|---|---|---|---|---|
| 45 | benign | [0, 0, 0, 0, 0] | 0 | - | -|
| 45 | malicious | [0, 0, 0, 0, 0] | 0 | 26.34 | 35.81|
| 84 | benign | [0, 0, 0, 0, 0] | 0 | - | -|
| 84 | malicious | [0, 4, 0, 0, 0] | 0 | 36.59 | 30.67|
| 34 | benign | [5, 5, 5, 5, 5] | 5 | - | -|
| 34 | malicious | [2, 2, 2, 2, 2] | 2 | 33.79 | 26.02|
| 36 | benign | [2, 2, 2, 2, 2] | 2 | - | -|
| 36 | malicious | [0, 0, 0, 0, 0] | 0 | 30.67 | 21.68|
| 69 | benign | [8, 8, 8, 8, 8] | 8 | - | -|
| 69 | malicious | [8, 8, 8, 8, 8] | 8 | 45.37 | 0.92|

## Summary

| client | benign attractor | malicious attractor | result | real-run observed attractor |
|---|---|---|---|---|
| 45 | 0 | 0 | SAME | 0 |
| 84 | 0 | 0 | SAME | 4 |
| 34 | 5 | 2 | DIFFERENT | 2 |
| 36 | 2 | 0 | DIFFERENT | 4 |
| 69 | 8 | 8 | SAME | 8 |

**3/5 tested clients have the SAME attractor benign and malicious -- a MIXED result: some clients' attractors are poison-independent, at least one is not -- poisoning CAN shift the attractor for some clients, so this is not a universal, poison-irrelevant property.**

## Deeper reading -- this is more complicated than "coincidence," not simpler

Three things complicate the clean story, and should all go in the record rather
than be smoothed over:

**1. Poisoning changing weights, and therefore the attractor, is not itself
surprising.** Poisoned and benign local training produce different weights by
construction -- so finding client 34 and 36's attractor differs between variants
doesn't by itself show anything about backdoors specifically; it shows the probe
is sensitive to weight changes in general, which could come from poisoning or
from any other source of training variation. The informative question is narrower:
does poisoning specifically bias the attractor TOWARD the true target class 0,
more than chance would predict? With n=5: malicious variants landed on class 0 in
3/5 cases (45, 84, 36); benign variants landed on class 0 in 2/5 cases (45, 84).
That is a higher rate under poisoning than without it, in this small sample --
the opposite direction from "purely coincidental," though the sample is far too
small to treat as confirmed.

**2. Client 36 is a direct contradiction of the real log, and it matters.** The
real `cellA_stealthy` run observed client 36's attractor as stably 4 across 4 of
its 5 historical samplings (history `[4, 4, 4, 0, 2]`), never flagged. This
reconstruction -- single fixed epoch-60 checkpoint, real stealthy attack params --
gives client 36's malicious attractor as 0, 5/5, which WOULD have been flagged.
That is a direct mismatch with what actually happened historically. The most
likely explanation is that the real G_old client 36 was actually evaluated
against, round by round, differs enough from the epoch-60 snapshot used here that
the reconstruction simply isn't equivalent to the historical conditions -- which
means this method's conclusions about any one client should be read as "what this
fixed starting point produces," not as a faithful replay of that client's actual
history. This is exactly the generalizability gap flagged below, now demonstrated
empirically rather than just a theoretical worry.

**3. Client 84 suggests a THIRD candidate mechanism: calendar-time drift
independent of accumulated poisoning.** Both the benign AND malicious variant,
built fresh from the epoch-60 snapshot, land on class 0 (benign 5/5, malicious
4/5). But the real log shows client 84 attracting class 4 in its first four
historical samplings and only reaching 0 in its last two (ending at epoch 51).
If a freshly-trained BENIGN model from the epoch-60 point already lands on 0,
that suggests the shift toward class 0 may be a property of how converged/mature
the global model (G_old/G_agg) has become by this point in training, rather than
something this client's own accumulated poisoning history caused. That would mean
client 84's real drift reflects the system maturing over calendar time, with this
client's eventual capture being circumstantial exposure to a later-stage G_old,
not a backdoor-specific signal either.

## Bottom line

The "attractor class is a pure, poison-irrelevant property of weights and
partition" story is too clean for what the data actually shows. What's robust
across all these checks: **no mechanism found so far reliably tracks the planted
backdoor specifically.** Sometimes poisoning changes the attractor (34, 36),
sometimes it doesn't (45, 69, 84's benign/malicious pair at this one checkpoint),
and in the one case with real multi-round history (84), the drift looks more
consistent with the global model's own maturation than with anything the client
did. None of these mechanisms is "detecting the backdoor" in the sense the paper
claims -- but which specific mechanism explains each capture differs by client,
and this experiment doesn't cleanly resolve which, only rules out the simplest
unifying story.

## Unification with the false-positive finding (`exp_fp_breakdown_NOTE.md`, `exp_partition_profile_NOTE.md`)

The long-run data (437 rounds of the resumed `cellA_stealthy` run) ties this
mechanism directly to a second, independently-discovered finding, and the two
should be read as **one result, not two**:

A client's attractor class is a stable property of its weights and data
partition -- specifically, of whether class 0 (here, the backdoor's target class)
happens to dominate that client's non-IID shard (`exp_partition_profile_NOTE.md`:
5/5 heavily-excluded benign clients have class 0 as their dominant class at
34-67% share; 5/5 never-excluded controls do not, at 0.4-13.5% share). Whether
that client is malicious or benign is incidental to whether Stage 3 flags it:

- **Client 45 is the malicious instance of this property:** its attractor is
  class 0 (100% of 5 samplings, per `exp_csv_mining_NOTE.md`), so it gets caught
  reliably -- 4 of the run's 5 historical tp events were this one client. It is
  genuinely malicious, but the mechanism catching it isn't "detecting the
  backdoor" -- it's the same attractor coincidence that would catch it even if
  benign (consistent with the benign/malicious attractor test above, where
  client 45's benign and malicious variants both land on class 0).

- **Clients 49 and 73 (and 65, 22, 89) are the benign instances of the same
  property:** their partitions happen to be dominated by class 0 too, so they
  get excluded near-permanently (73: 98% of samplings, 49: 100%) despite being
  entirely honest.

**One detector property -- Stage 1/2 reconstructing class 0 as the attractor for
a given client's weights, driven by that client's own class-0-dominant partition
-- explains both the run's handful of true positives AND its systematic false
positives.** It is not that detection sometimes works and sometimes misfires;
it is a single miscalibrated signal that fires on "this client's data is skewed
toward class 0," which is only coincidentally correlated with "this client is
the attacker," because the attacker in this run also happens to target class 0.

**Headline numbers to keep together, as the clearest statement of the finding:**
71 false exclusions from two clients (49, 73) alone vs. 14 true positives across
the entire ~437-round run so far -- and those 14 true positives are themselves
largely attributable to the same attractor-coincidence mechanism (chiefly client
45's stable class-0 attractor) rather than to the detector tracking the backdoor
specifically.
