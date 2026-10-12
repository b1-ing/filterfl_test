# Third-term ablation -- ||x_l|| and visual structure, with vs without

Faithful reimplementation of `reference/filterfl/gan.py`'s Stage 1 (identical) and
Stage 2 (with a `use_third_term` toggle the shipped code doesn't expose), using the
SAME reference discriminator weights (d1/d2/malicious-client-36) as
`exp3_reference_check.py`. RNG reset to an identical snapshot before each of the
two Stage 2 variants per run, so the only difference between "with" and "without"
is the loss term itself.

x_l = the raw, single-channel trigger generator output (pre-repeat to 3 channels)
at Stage 2's final (converged) epoch.

## Results (5 runs x 2 conditions)

| run | condition | mean &#124;&#124;x_l&#124;&#124; | pixel min | pixel max | pixel std | pred_class (vs malicious client) | confidence |
|---|---|---|---|---|---|---|---|
| 0 | with | 5.218 | -0.556 | 0.639 | 0.163 | 4 | 0.5068 |
| 0 | without | 4.419 | -0.348 | 0.647 | 0.113 | 4 | 0.4070 |
| 1 | with | 4.716 | -0.635 | 0.552 | 0.145 | 4 | 0.5734 |
| 1 | without | 3.583 | -0.369 | 0.555 | 0.101 | 4 | 0.4985 |
| 2 | with | 4.893 | -0.643 | 0.517 | 0.145 | 4 | 0.5408 |
| 2 | without | 3.403 | -0.465 | 0.523 | 0.106 | 4 | 0.4500 |
| 3 | with | 4.666 | -0.536 | 0.518 | 0.146 | 4 | 0.5243 |
| 3 | without | 4.336 | -0.283 | 0.537 | 0.095 | 4 | 0.4529 |
| 4 | with | 4.685 | -0.570 | 0.523 | 0.143 | 4 | 0.5211 |
| 4 | without | 3.567 | -0.313 | 0.544 | 0.103 | 4 | 0.4252 |

**Mean ||x_l|| with third term:    4.836**
**Mean ||x_l|| without third term: 3.861**
**Difference: +25.2%**

## Verdict

The third term correlates with a LARGER ||x_l|| on average (4.836 vs 3.861, +25.2%). This does NOT support the 'de facto norm penalty' reading -- if anything the term is associated with a larger perturbation, not a smaller one.

**Framing for the writeup:** this is a stronger finding than "the term doesn't
matter for class recovery" might suggest. The published equations already have no
norm bound on x_l (Experiment 3's original finding). The shipped code's
undocumented third term doesn't fill that gap -- it makes it worse, actively
relaxing the search to a *larger* perturbation (+25.2% mean ||x_l||) for no gain
in target-class recovery (still class 4, never the true target 0, with or without
it). So the implementation is further from a constrained trigger search than the
paper already is, not closer. Carry this framing into the writeup, not just the
raw percentage.

Class recovery (pred_class vs. the malicious client, same construction as
`exp3_reference_check_NOTE.md`): see per-run table -- consistent with that
experiment's finding (class 4, not the true target 0) regardless of whether the
third term is present, confirming the term's effect here is on trigger magnitude/
structure, not on which class gets recovered.
