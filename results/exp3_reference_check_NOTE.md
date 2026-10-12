# Third-term check -- does the REFERENCE code recover the target class where ours didn't?

**Why this check matters:** our paper-faithful implementation locked onto class 4
instead of the true target (0) in 10/10 cold detection runs
(exp1_detection_baseline_NOTE.md). The most obvious objection is that our
implementation is missing something the reference code has -- specifically the
undocumented third term on the raw trigger pattern (CODE_NOTES.md discrepancy #1).
This runs the REFERENCE repo's own `gan_defense` directly (not our port) against
the identical d1/d2 construction used in the detection-baseline experiment, so the
only thing that changes is the detector implementation.

## Method

- d1 (G_old) = epoch-60 checkpoint; d2 (G_agg) = that checkpoint
  aggregated with ONLY malicious client 36's update
  (poison_rate=0.5, poison_epochs=6, unscaled) -- identical
  construction to exp1_detection_baseline.py.
- Malicious client: MA=24.62, ASR=85.21 (same backdoor, same 85%-ish
  strength as the earlier check).
- Reference ResNet8 (`reference/filterfl/models/ResNet8.py`) is architecturally
  identical to ours (same BasicBlock, same channel widths, same [1,1,1] block
  counts) but uses different state_dict key names -- a key-remapping function
  converts our checkpoint into the reference's naming.
- **Sanity check before trusting anything downstream:** same weights loaded via
  our keys vs. the remapped reference keys, same input -> max absolute logit
  difference = 0.000000e+00 (passes; confirms the remapping is a true
  identity, not just shape-compatible).
- `reference/filterfl/gan.py`'s `gan_defense()` called directly, unmodified,
  10 times (fresh/independent each time, RNG not reset between calls --
  matching the detection-baseline experiment's methodology). Both
  `tri_imgs[0]` (what `algorithm.py` actually uses, per discrepancy #2) and
  `tri_imgs[-1]` (the converged batch) are checked against the malicious model.

## Results

| run | which batch | pred_class | confidence | target-class hit |
|---|---|---|---|---|
| 0 | first (tri_imgs[0], algorithm.py's actual indexing) | 4 | 0.3558 | no |
| 0 | last (tri_imgs[-1], converged) | 4 | 0.3917 | no |
| 1 | first (tri_imgs[0], algorithm.py's actual indexing) | 4 | 0.4074 | no |
| 1 | last (tri_imgs[-1], converged) | 4 | 0.4607 | no |
| 2 | first (tri_imgs[0], algorithm.py's actual indexing) | 4 | 0.3975 | no |
| 2 | last (tri_imgs[-1], converged) | 4 | 0.4513 | no |
| 3 | first (tri_imgs[0], algorithm.py's actual indexing) | 4 | 0.3419 | no |
| 3 | last (tri_imgs[-1], converged) | 4 | 0.4062 | no |
| 4 | first (tri_imgs[0], algorithm.py's actual indexing) | 4 | 0.3926 | no |
| 4 | last (tri_imgs[-1], converged) | 4 | 0.4510 | no |
| 5 | first (tri_imgs[0], algorithm.py's actual indexing) | 4 | 0.3428 | no |
| 5 | last (tri_imgs[-1], converged) | 4 | 0.4037 | no |
| 6 | first (tri_imgs[0], algorithm.py's actual indexing) | 4 | 0.3560 | no |
| 6 | last (tri_imgs[-1], converged) | 4 | 0.4040 | no |
| 7 | first (tri_imgs[0], algorithm.py's actual indexing) | 4 | 0.3947 | no |
| 7 | last (tri_imgs[-1], converged) | 4 | 0.4519 | no |
| 8 | first (tri_imgs[0], algorithm.py's actual indexing) | 4 | 0.4090 | no |
| 8 | last (tri_imgs[-1], converged) | 4 | 0.4620 | no |
| 9 | first (tri_imgs[0], algorithm.py's actual indexing) | 4 | 0.3416 | no |
| 9 | last (tri_imgs[-1], converged) | 4 | 0.3761 | no |

**tri_imgs[0] pred_class distribution:** {4: 10} -- target-class hits: 0/10
**tri_imgs[-1] pred_class distribution:** {4: 10} -- target-class hits: 0/10

## Verdict

FINDING STANDS: the reference code, including its undocumented third term, ALSO fails to reliably recover the true target class (tri_imgs[0] hits: 0/10, tri_imgs[-1] hits: 0/10). The 20/20-miss result (10 ours + 10 reference) is not an artifact of our implementation choice -- this is a property of the detection mechanism itself, not of which spec (paper vs. code) is used to implement it.
