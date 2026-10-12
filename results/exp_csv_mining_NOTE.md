# Step 1 -- mining the existing cellA_stealthy detection log (zero compute)

All 61 already-completed rounds of `step2_cellA_stealthy` parsed directly
from `results/step2_cellA_stealthy.csv`'s `per_client_confidence` field. No model
loaded, no GPU used -- pure analysis of data already on disk.

## 0. pred_class distribution vs. the cold single-shot 0/20

| | n observations | pred_class distribution | hits target_class 0 |
|---|---|---|---|
| malicious clients (real run) | 61 | {0: 11, 2: 24, 4: 14, 5: 1, 6: 8, 8: 3} | 11/61 = 18.0% |
| benign clients (real run) | 549 | {0: 109, 2: 203, 3: 21, 4: 113, 5: 28, 6: 23, 7: 13, 8: 37, 9: 2} | 109/549 = 19.9% |
| cold single-shot (exp1 + exp3 reference check) | 20 | {4: 20} | 0/20 = 0.0% |

Malicious-client confidence: mean=0.592,
range=[0.273, 0.941] (n=61).
Benign-client confidence: mean=0.684,
range=[0.139, 1.000] (n=549).

## 1. true-positive events -- which clients, how often

Log totals: detection_tp summed = 5, detection_fp summed = 85.
Recovered directly from per-client parsing: 5 tp event(s).

| epoch | client | pred_class | confidence |
|---|---|---|---|
| 23 | 45 | 0 | 0.5357 |
| 30 | 45 | 0 | 0.5117 |
| 32 | 45 | 0 | 0.8946 |
| 51 | 84 | 0 | 0.5071 |
| 51 | 45 | 0 | 0.5859 |

tp events by client: {45: 4, 84: 1}

**REAL DETECTION SIGNAL (partial): at least one client ([45]) was flagged more than once across rounds -- repeated capture of the same client is not explained by independent per-round coincidence alone.**

## 2. per-client pred_class stability (malicious clients sampled >=2 times)

| client | times sampled | pred_class history | dominant class | dominant share | times flagged |
|---|---|---|---|---|---|
| 34 | 6 | [5, 2, 2, 2, 2, 2] | 2 | 83% | 0 |
| 36 | 5 | [4, 4, 4, 0, 2] | 4 | 60% | 0 |
| 45 | 5 | [0, 0, 0, 0, 0] | 0 | 100% | 4 |
| 47 | 9 | [6, 6, 6, 6, 6, 0, 0, 0, 6] | 6 | 67% | 0 |
| 50 | 5 | [6, 6, 2, 2, 8] | 6 | 40% | 0 |
| 54 | 7 | [2, 2, 2, 2, 2, 2, 2] | 2 | 100% | 0 |
| 60 | 8 | [2, 2, 2, 2, 2, 2, 2, 2] | 2 | 100% | 0 |
| 69 | 4 | [2, 4, 8, 8] | 8 | 50% | 0 |
| 84 | 6 | [4, 4, 4, 4, 0, 0] | 4 | 67% | 1 |
| 99 | 6 | [4, 4, 4, 4, 4, 4] | 4 | 100% | 0 |

**Per-client pred_class is STABLE round-to-round for clients sampled multiple times (dominant class share is high) -- consistent with a per-model attractor that doesn't change much even as G_old/G_agg evolve round to round.**

## 3. does the statistic distinguish benign from malicious at all?

Malicious top-3 classes: [(2, 24), (4, 14), (0, 11)]
Benign top-3 classes: [(2, 203), (4, 113), (0, 109)]

**Malicious and benign clients land on target_class at essentially the SAME rate -- the statistic carries little to no information distinguishing backdoored from clean clients in the real run either.**
