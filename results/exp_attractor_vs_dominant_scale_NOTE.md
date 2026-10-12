# Does attractor track dominant partition class IN GENERAL, not just for class 0?

**Why:** `exp_partition_profile_NOTE.md` found, at n=5 vs n=5, that heavily-excluded
benign clients are class-0-dominant and never-excluded controls are not. The obvious
confound: class-0-dominant clients might just be "good at predicting class 0" for
mundane reasons unrelated to any detector mechanism, making this close to tautological.
The real test is whether a client dominant in some OTHER class (e.g. class 7) also
reliably attracts to ITS OWN dominant class -- if so, the mechanism is general
("Stage 1/2 recovers a client's dominant class, period"), not a class-0-specific
artifact.

Zero-compute: re-derives the fixed-seed partition and parses all 919 rounds
of `results/step2_cellA_stealthy.csv` directly. "Attractor" here = the MODE of a
client's pred_class history across however many times it was sampled (clients with
only 1 sampling excluded -- can't assess stability from a single observation).

## Scale-up result: all clients with >=2 samplings (n=100)

Overall match rate (attractor == dominant partition class): **31/100 = 31.0%**

### Confusion: dominant partition class -> observed attractor class

| dominant class | n clients | attractor distribution | self-match rate |
|---|---|---|---|
| 0 | 14 | {0: 7, 2: 3, 3: 1, 4: 2, 8: 1} | 7/14 = 50% |
| 1 | 10 | {2: 4, 3: 3, 4: 3} | 0/10 = 0% |
| 2 | 10 | {2: 10} | 10/10 = 100% |
| 3 | 10 | {0: 1, 2: 1, 3: 3, 4: 3, 6: 1, 8: 1} | 3/10 = 30% |
| 4 | 10 | {2: 1, 3: 5, 4: 2, 5: 2} | 2/10 = 20% |
| 5 | 10 | {0: 1, 2: 5, 3: 1, 4: 1, 5: 1, 8: 1} | 1/10 = 10% |
| 6 | 9 | {2: 1, 3: 4, 4: 2, 6: 2} | 2/9 = 22% |
| 7 | 11 | {0: 1, 2: 3, 3: 3, 4: 2, 6: 1, 8: 1} | 0/11 = 0% |
| 8 | 7 | {2: 1, 8: 6} | 6/7 = 86% |
| 9 | 9 | {2: 1, 3: 4, 4: 3, 6: 1} | 0/9 = 0% |

### Does it hold outside class 0?

| group | n clients | match rate (attractor == own dominant class) |
|---|---|---|
| dominant class == 0 | 14 | 7/14 = 50.0% |
| dominant class != 0 | 86 | 24/86 = 27.9% |

**Updated at n=919 rounds (previously n=469) -- same structure, slightly weaker numbers.** Non-class-0-dominant clients match their own dominant class at only 24/86 = 27.9% (was 32.6% at n=469), versus the class-0 subset's 7/14 = 50.0% (was 57.1%). More data pulled both numbers down slightly but did not change the qualitative picture at all.

## The real structure, confirmed with 2x the data: a small number of universal "sink" classes, not general dominant-class tracking

This replicates cleanly from the earlier (n=469) pass -- same classes play the same roles:

- **Class 2 remains a perfect universal sink**: 10/10 self-match, AND it pulls
  substantial mass from EVERY other dominant-class group (dominant=1: 4/10,
  dominant=3: 1/10, dominant=4: 1/10, dominant=5: 5/10, dominant=6: 1/10,
  dominant=7: 3/10, dominant=9: 1/10). No other class comes close to being both
  this self-sticky and this much of a draw for everyone else.
- **Class 8 remains a secondary sink** (6/7 = 86% self-match), **class 0 remains
  a moderate, middling sink** (7/14 = 50%, down from 57% but still clearly above
  the non-sink classes).
- **Classes 1, 7, and 9 still have ZERO self-match** (0/10, 0/11, 0/9) -- a
  client dominant in any of these three classes essentially NEVER has its own
  class as its attractor, confirmed again with double the data.
- New at this sample size: class 3 also looks like a mild, inconsistent sink
  (30% self-match, up from smaller signal before) and class 4 looks weak but
  non-zero (20%). Neither approaches class 2's strength.

**This is now a well-replicated finding, not a single-sample artifact: Stage 1/2's optimization has a small set of preferred destination classes (2 >> 8 > 0 > 3/4 > 1/6/7/9) largely independent of what any given client's data is actually dominated by.** Class 0 benefits from being a real target class (what this run's attacker is coded to exploit) AND a moderate generic sink -- but it is demonstrably not the strongest sink in the system, and three classes (1, 7, 9) are essentially never self-attracting at all, confirmed now at twice the sample size.

## Are all tp events explained by class-0 dominance?

| client | dominant partition class | dominant share | tp count (this client) |
|---|---|---|---|
| 45 | 0 | 30% | 9 |
| 47 | 7 | 40% | 1 |
| 84 | 1 | 29% | 5 |

Total tp events in the run so far: 15. From class-0-dominant clients: 9.

**9/15 (60%) tp events are from class-0-dominant clients** -- still NOT all of them (consistent with the n=469 finding of 9/14 = 64%, essentially unchanged as 1 more tp event accumulated, attributed to client 84). The same two non-class-0-dominant exceptions persist: client 47 (dominant class 7, a non-sink class, 1 tp) and client 84 (dominant class 1, also a non-sink class, now 5 tp events, up from 4) continue to be explained by something other than static partition dominance -- most likely calendar-time/maturation drift for 84 (as previously flagged) and noise around a non-0 baseline attractor for 47.

## On the attacker-control framing

Still needs hedging, confirmed with more data rather than resolved. The general "attractor tracks dominant class" claim holds strongly for class 2, moderately for 8 and 0, and essentially not at all for 1, 7, and 9 -- a pattern that has now replicated across a doubling of the sample. An attacker choosing a target class would see very different collateral-damage profiles depending on which class they pick: targeting class 2 would likely produce the strongest, cleanest collateral-exclusion effect of all ten classes; targeting 1, 7, or 9 would likely produce little to no clean dominant-class exclusion pattern. What's supported without hedging, now on firmer footing: for this run's actual target (0), the measured collateral-damage mapping in `exp_partition_profile_NOTE.md` is real and replicates.

## Answer to the user's direct question

**No, it does not hold for clients whose dominant class isn't 0, in general.** It holds well for class 2 (100%) and class 8 (86%), moderately for class 0 (50%), and not at all for classes 1, 7, 9 (0% each). So the mechanism is NOT cleanly "Stage 1/2 recovers a client's dominant class, period" -- that stronger, fully general claim is falsified by the 1/7/9 counterexamples, confirmed at n=919. The correct, narrower description: **Stage 1/2 has a small number of class-dependent "sink" destinations (strongest: 2; moderate: 8, 0; weak/none: 1, 3, 4, 6, 7, 9) that it tends toward regardless of a client's own data, and class 0 happens to be one of the moderate sinks** -- which is why the class-0-specific finding (n=5 vs n=5 in `exp_partition_profile_NOTE.md`) is real, but should not be read as evidence of a fully general dominant-class-tracking mechanism.
