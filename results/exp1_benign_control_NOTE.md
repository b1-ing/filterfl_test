# Experiment 1 -- benign-peer control for the strong-backdoor MA floor

Tests whether malicious client 34's MA collapse (43.39 global -> 24.02
local, from exp1_strong_backdoor.py) is caused by 6 local epochs on a small non-IID
shard in general, or is specific to poisoning. Trains a benign client from the same
epoch-60 global weights, same local-epoch count (6) and same lr magnitude (0.005,
matching poison_lr) -- no trigger/label poisoning -- on a partition matched in size
to client 34's.

Original floor (global average, 43.39 - 5 = 38.39) was the wrong baseline:
"trivially detectable" means anomalous relative to local-model peers, not relative
to the global aggregate -- a global model's accuracy is already an average over
many clients' local models, most of which are below it individually.

| | partition size | local epochs | lr | poisoned | MA | ASR |
|---|---|---|---|---|---|---|
| global model (epoch 60) | - | - | - | - | 43.39 | 4.56 |
| benign control (client 7) | 352 | 6 | 0.005 | no | 40.08 | - |
| malicious client 34 | 349 | 6 | 0.005 | yes (rate=0.5) | 24.02 | 73.71 |

**Malicious-vs-benign-peer MA delta: -16.06 points.**

Decision rule result: control MA is NOT within ~5 points of
the malicious client's 24.02 -> collapse is poison-specific; gridding poison_epochs in {2,3,4}.

**Why this number matters even though FilterFL itself doesn't look at it:**
FilterFL's Stage 3 only thresholds on confidence over the generated trigger set T,
so a client's own local MA is invisible to it. But a similarity- or
clustering-based defence (weight-distance, cosine similarity to peer updates, etc.)
would see this delta directly. If this attack is later claimed to transfer beyond
FilterFL, this is the number that determines whether that claim survives contact
with a similarity-based defence -- recorded here rather than discovered in review.
