# s11: CIFAR-10 controlled class-0 skew test — predictions registered BEFORE the run (2026-10-15)

Control: `hpc_s10_cifar_mean_a_s0` (uncapped, same seed 0, same config, 1000 rounds, mean pooling, --dump-probe-responses).
Intervention: `--cap-class-share 0.15` — benign clients' class-0 (target-class) share capped at 15% by swapping
(shard sizes unchanged; see CODE_NOTES.md "Class-share cap intervention"). A constructed condition, not a reproduction
of the paper's setup. Malicious clients are neither donors nor receivers.

Hypothesis under test: benign clients with target-class-dominated shards cause CIFAR's benign false positives and the
benign/malicious overlap. Original s10 numbers (control) and predictions if skew is causal:

| Quantity | s10 control | Prediction for s11 (capped) |
|---|---|---|
| Benign false positives (live mean gate, 1000 rounds) | 301 | falls toward ~0 |
| AUC malicious vs benign, mean pooling | 0.920 | rises toward ~0.96+ |
| AUC malicious vs benign, top-16 | 0.926 | rises toward ~0.969 |
| Benign max score, mean pooling | 0.813 | falls toward ~0.33 |
| Held-out threshold calibration (zero-FPR, mean) | TPR 0.00-0.04 | starts working as on GTSRB (TPR ~0.9+, few FP) |
| Receiving clients' mean pooled score | (baseline) | RISES (two-directional prediction) |
| Capped (donor) clients' mean pooled score | (baseline) | FALLS |
| Blind-spot attack acceptance (per-round, mean gate) | 74% (502/674) | LARGELY UNCHANGED; a large move means the intervention was not clean |

Falsifiers: FPs stay high / AUC barely moves -> skew is not the (main) cause. Attack acceptance moves a lot ->
capping changed something besides benign skew. Receivers do not rise -> effect is not a class-0-exposure mechanism.
