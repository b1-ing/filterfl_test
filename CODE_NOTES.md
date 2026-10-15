# CODE_NOTES — E2E FilterFL Blind-Spot Experiment

## Execution Path (discovered 2026-10-11)

### Federated training loop (`filterfl/run.py`)

```
run.py main()
  ├─ Config.from_yaml()                            # load hyperparams
  ├─ seed_everything(cfg.seed)
  ├─ load_dataset / dirichlet_partition            # data.py
  ├─ adversarial_namelist = random.sample(...)     # fixed at start, ~10% of 100 clients
  │
  └─ for epoch in 1..cfg.epochs:
       ├─ [1] sampled = random.sample(participants, 10)
       │        → malicious_sampled = sampled ∩ adversarial_namelist
       │
       ├─ [2] ori_weight = model.state_dict()      # G_old
       │
       ├─ [3] for cid in sampled:
       │        local_train(model, loader, cfg, device,
       │                    is_malicious=(cid in malicious_sampled),
       │                    attacker=attacker)
       │        → if is_malicious: poison cfg.poison_rate of each batch,
       │            train for cfg.poison_epochs epochs at cfg.poison_lr
       │        → if benign: train for cfg.local_epochs epochs at cfg.lr
       │
       ├─ [4] betas = [1, 1, ...]  (one per sampled client)
       ├─ [5] new_state = aggregate(ori_weight, client_states, betas, eta)
       │        (initial unfiltered aggregate = G_agg)
       │
       ├─ [6] IF defence != "none" AND is_detection_round(epoch):
       │        run_detection(ori_weight, new_state, cfg, ...)  # Stage 1+2 GAN
       │        stage3_classify_clients(client_states, ...)     # Stage 3 probe
       │        for each client flagged (pred==target AND conf > rho):
       │            betas[i] = 0
       │
       ├─ [7] IF any beta==0: re-aggregate with updated betas
       │
       ├─ [8] model.load_state_dict(new_state)
       │
       ├─ [9] ma  = evaluate(model, test_loader)
       │        asr = evaluate(model, asr_loader, poison=True)
       │
       └─ [10] log epoch, ma, asr, detected, tp, fp, per_client_confidence
```

**Key locations:**
1. Local training: `filterfl/client.py:local_train()` (line 6)
2. Malicious modification: `filterfl/client.py:26-29` — poison_batch() called per batch
3. FilterFL evaluation: `filterfl/defense.py:run_detection()` + `stage3_classify_clients()`
4. Rejected clients: `filterfl/run.py:221-227` — betas[i]=0 sets exclusion
5. Aggregation: `filterfl/aggregator.py:aggregate()` (ignores betas==0)
6. Global ASR: `filterfl/run.py:255` — `evaluate(model, asr_loader, poison=True)`

---

## Existing E2E Results (10 rounds, seed=0)

### Condition A — No defence (poison_rate=0.25, poison_epochs=6)
Source: `results/e2e/e2e_s0_nodef25.out`

| Round | MA | ASR | Detected |
|------:|----:|----:|---------:|
| 1 | 15.04 | 18.19 | 0 |
| 2 | 16.13 | 0.50 | 0 |
| 3 | 19.98 | 19.78 | 0 |
| 4 | 21.98 | 0.00 | 0 |
| 5 | 25.36 | 51.09 | 0 |
| 6 | 29.91 | 18.70 | 0 |
| 7 | 29.37 | 0.67 | 0 |
| 8 | 27.14 | 31.49 | 0 |
| 9 | 29.92 | 0.06 | 0 |
| 10 | 29.02 | 55.26 | 0 |

ASR is volatile early (model immature); peaks ~55% by round 10. No defence.

### Condition B — FilterFL + ordinary attack (poison_rate=0.50, poison_epochs=6)
Source: `results/e2e/e2e_s0_ord50.out`

| Round | MA | ASR | Detected | TP |
|------:|----:|----:|---------:|---:|
| 1 | 15.21 | 21.49 | 0 | 0 |
| 2 | 15.80 | 0.32 | 0 | 0 |
| 3 | 20.63 | 24.89 | 0 | 0 |
| 4 | 21.43 | 0.00 | 0 | 0 |
| 5 | 26.09 | 44.78 | 1 | 1 |
| 6 | 26.85 | 37.01 | 0 | 0 |
| 7 | 27.32 | 0.36 | 0 | 0 |
| 8 | 27.13 | 35.54 | 0 | 0 |
| 9 | 27.04 | 0.09 | 0 | 0 |
| 10 | 28.63 | 44.66 | 1 | 1 |

Malicious client detected in rounds 5 and 10. Despite detection, ASR remains moderate
because many rounds pass undetected and only 1 malicious client may be sampled.

### Condition C — FilterFL + blind-spot attack (poison_rate=0.25, poison_epochs=6)
Settings: `--stage3-probe-mode train --stage2-loss reference --seed 0 --epochs 10`

| Round | MA | ASR | Mal_sel | Mal_flagged | Mal_accepted | FP | Stage3_conf_mal |
|------:|----:|----:|--------:|------------:|-------------:|---:|-----------------|
| 1 | 12.9 | 6.53 | 1 | 0 | 1 | 1 | 0.310 |
| 2 | 15.6 | 0.08 | 1 | 0 | 1 | 0 | 0.405 |
| 3 | 20.3 | 16.48 | 1 | 0 | 1 | 1 | 0.405 |
| 4 | 22.3 | 0.00 | 1 | 0 | 1 | 0 | 0.395 |
| 5 | 26.5 | 45.69 | 1 | 0 | 1 | 0 | 0.366 |
| 6 | 30.4 | 22.97 | 2 | 0 | 2 | 0 | 0.368, 0.386 |
| 7 | 29.1 | 0.26 | 1 | 0 | 1 | 0 | 0.688 (pred=2, not target) |
| 8 | 27.6 | 32.34 | 1 | 0 | 1 | 1 | 0.465 |
| 9 | 29.4 | 0.41 | 1 | 0 | 1 | 0 | 0.393 |
| 10 | 31.0 | 34.79 | 2 | 0 | 2 | 1 | 0.469, 0.417 |

**Summary (10 rounds, seed=0):**
- Total malicious appearances: 12
- Total malicious flagged (TP): 0 → **Malicious acceptance rate: 100%**
- Benign FP rate: 4/88 = **4.5%**
- Mean Stage3 confidence (malicious): **0.42** (rho=0.5 threshold; all except round 7 predict class 0)
- Peak global ASR: **45.7%** (round 5)
- Final global ASR: **34.8%** (round 10)
- Final clean accuracy: **31.0%**

---

## Cross-seed robustness — Condition C (blind-spot, 10 rounds each)

| Seed | Mal appearances | Flagged (TP) | Acceptance | FPR | Peak ASR | Final ASR |
|-----:|----------------:|-------------:|-----------:|----:|---------:|----------:|
| 0 | 12 | 0 | **100%** | 4.5% | 45.7% | 34.8% |
| 1 | 10 | 0 | **100%** | 8.9% | 47.4% | 0.1% |
| 2 | 8 | 1 | 87.5% | 6.5% | 30.0% | 0.0% |

**Seed 2 anomaly — round 9, client 13: Stage-3 confidence 0.810 (flagged).**
All other 29 malicious appearances across the 3 seeds had conf < 0.5.
This is consistent with the prompt's note that "RNG can move individual configurations
across the Stage-3 threshold" — the live global model at round 9/seed 2 apparently
pushed one client's signature above rho=0.5. The frozen-checkpoint characterization
(epoch 200, 3 seeds at rate=0.25) never hit this; the live training context introduces
additional variability.

**Seed 1 note — many malicious clients predict class 7 (not target class 0).**
Clients c47, c54, c69 all had probe pred_class=7. The detection condition requires
pred_class == target_class (0), so these pass unconditionally regardless of confidence.
This is "natural evasion" from the probe mispredicting the target.

### Mean Stage-3 conf across seeds (excluding seed 2 flagged round):
- Seed 0: 0.422
- Seed 1: 0.391 (mostly p7 predictions, so flagging never triggered regardless)
- Seed 2: 0.487 (including the 0.81 outlier; without it: 0.424)

---

## Stage3 Probe Mode Note

The blind-spot boundary was characterized using the **reference code** (`reference/filterfl/gan.py`)
which never calls `.eval()` on the probe model — i.e., **train mode**.

Correspondence in our framework: `--stage3-probe-mode train --stage2-loss reference`.

The existing Condition B run used the default `stage3_probe_mode: eval`.
Condition C should use `train` mode to match the blind-spot characterization.

**The boundary (epochs=6, rate=0.20–0.28 → unflagged across 3 seeds) was established
at a frozen checkpoint (epoch 200), using train mode + reference GAN loss.**

---

## Command for Condition C

```
python -m filterfl.run \
  --config configs/cifar_primary.yaml \
  --defense filterfl \
  --epochs 10 \
  --poison-rate 0.25 \
  --poison-epochs 6 \
  --seed 0 \
  --run-name e2e_s0_blind25 \
  --stage3-probe-mode train \
  --stage2-loss reference
```

---

## Key open question

> Does the blind-spot configuration (poison_rate=0.25, poison_epochs=6, scale=1.0)
> **evade FilterFL during live federated training** and produce **global-model backdoor success**?

Expected (if the client-level finding transfers):
- Most malicious rounds accepted (confidence < 0.5)
- Global ASR growing (not necessarily as fast as no-defence due to dilution)

If malicious clients are accepted but global ASR stays low: FedAvg dilution may be the
limiting factor, not FilterFL. The effective malicious contribution would be
~10% × 1/(10 benign + 1 malicious selected) ≈ 9% of the aggregate per round.

---

## Partition reproducibility across platforms (2026-10-15) — RESOLVED, no cross-platform discrepancy

**Question:** a rebuild of the Dirichlet partition gave client 73 = 670 samples / 8.5% class-0, but
`results/exp_partition_profile_NOTE.md` has 500 samples / 67%. Same code, different machine?

**Answer: no. It was a seed mismatch in the comparison, not a platform difference.** The note's run
(`step2_cellA_stealthy`) used **seed 1** (`configs/cifar_primary.yaml` ships `seed: 1`); the HPC runs
(`hpc_s1a_s0`, `hpc_s10_cifar_mean_a_s0`, ...) pass `--seed 0`. Rebuilding with seed 1 on the HPC machine
(`scripts/partition_repro_check.py`) gives client 73 = 500 / 335 class-0 and client 49 = 426 / 168 class-0,
exactly the note; the epoch 1-3 `sampled_clients` sequences also match the Windows CSV exactly. Seed 0 gives
670 / 57 and 194 / 20, and its sampled-client sequence matches the HPC CSVs exactly.

**Why it is deterministic:** `dirichlet_partition` uses only `random.shuffle` and `np.random.dirichlet` on global
state seeded by `seed_everything` before the call; classes are iterated by `range(num_classes)`; no set/dict-order,
hashing, multiprocessing or torch RNG is involved (the torch-RNG transforms in `build_class_index` do not touch
python/numpy RNG). `git diff` shows `data.py` changes are only the GTSRB loader. Environment on HPC:
python 3.11.17, numpy 2.4.6, torch 2.5.1+cu121, torchvision 0.20.1 — the Windows environment's versions were
not recorded here, but identical output for seed 1 shows they do not change the seeded draws.

**Survey of runs on disk** (epoch 1-3 `sampled_clients` + `adversarial_sampled`, grouped by dataset/seed/alpha/N/k/
poison_client_num): all 11 groups consistent — every same-seed pair of runs, Windows-era or HPC, sampled the same
clients (the two apparent differences were one-row `e2e_*_nodef25` CSVs, a length artefact).

**Consequence:** same-seed Windows-vs-HPC comparisons are on identical data distributions. The genuine hazard is
**cross-seed** comparison: Windows-era analyses default to seed 1 (partition-profile, FP-breakdown, attractor notes,
and the frozen-checkpoint `separability_*` sweeps, which load `cifar_primary.yaml` + a seed-1 checkpoint), while
the HPC arms (s1a/s2/s4/s5/s10) are seed 0 (some s1a/s1b/s1c also seeds 1, 2). The partition-profile note's client IDs
(49, 73, 65, 22, 89) are properties of the seed-1 partition and do NOT carry over to seed-0 runs (there the class-0-heavy
benign clients are 9, 75, 18, 4). The class-0-skew *pattern* replicating across seeds is a replication, not a rerun.

### Seed hazard (standing caveat, 2026-10-15)
Windows-era analyses default to **seed 1** (`configs/cifar_primary.yaml` ships `seed: 1`); the HPC arms are mostly
**seed 0** (some s1a/s1b/s1c also seeds 1, 2). Seeds change the partition, the adversarial namelist and the sampling
sequence. **Any claim that pairs a Windows-era result with an HPC result must state both seeds.** I have not audited every
older note for mixed-seed pairings; do this when writing up.

## Class-share cap intervention (2026-10-15) — constructed condition, NOT the paper's setup
`--cap-class-share F` (default off; `cfg.cap_class_share = 0`) in `filterfl/run.py`, implemented by
`data.cap_class_share`. Purpose: test causally whether benign clients with target-class-dominated shards cause CIFAR-10's
benign false positives / benign-malicious score overlap (`scripts/benign_partition_skew.py`).
Method (deterministic, **RNG-free** — touches no python/numpy global state, applied AFTER the seeded adversarial namelist is
drawn, so seeded sampling and the adversaries are identical to the uncapped run; verified in
`scripts/cap_class_share_check.py`):
1. Protected = malicious clients: neither donors nor receivers, shards untouched (their own class-0 shares, e.g. client 82 at 45%, are unchanged).
2. Donors = benign clients with class-0 count > floor(F * shard size). Each gives its excess class-0 samples away.
3. Receivers = other benign clients with headroom floor(F * size) - class-0 count > 0. Total excess is spread over receivers proportionally to headroom (largest remainder); no receiver exceeds F.
4. It is a **swap, not a one-way move**: each receiver hands back as many non-class-0 samples (evenly spaced over its non-class-0 samples sorted by (label, index)) as it received class-0 samples, and those go to the donors. Shard sizes and the overall sample set are unchanged (asserted). This deviates from a plain "move excess to others" so local-training volume per client is unchanged — only composition changes.
Seed-0 CIFAR-10, F=0.15: 1300 class-0 samples moved, 17 donors (cids 4, 9, 18, 43, 44, 45, 51, 53, 54, 55, 64, 66, 68, 72, 75, 77, 93), 73 receivers (mean class-0 share 5.1% -> 8.6%); max benign share 0.517 -> 0.150.
Side effect (unavoidable): donors also receive non-class-0 samples from the receivers' largest-population classes, so their non-target class mix shifts slightly.
Run: `jobs/s11_cifar_cap15.sh` -> `results/hpc_s11_cifar_cap15_a_s0.{csv,_probe.npz,_capreport.json}`; control = `hpc_s10_cifar_mean_a_s0`;
predictions registered in `results/s11_cifar_capskew_PREDICTIONS.md` before the run.

---

## Stage 3 pooling taxonomy, CIFAR-10 benign false positives, and the hypotheses that died (2026-10-15)

Runs (all seed 0, blind-spot attacker rate 0.25 / 6 poison epochs, probe_mode=train, `--dump-probe-responses`):
s8 GTSRB top-16 (`hpc_s8_gtsrb_topk16_a05_s0`); s9 GTSRB mean (`hpc_s9_gtsrb_mean_a05_s0`, re-run of s6 full a05, reproduces it);
s10 CIFAR mean (`hpc_s10_cifar_mean_a_s0`, reproduces `hpc_s1a_s0`); s11 CIFAR mean + class-0 cap 0.15 (`hpc_s11_cifar_cap15_a_s0`);
s12 CIFAR no-attacker control (`hpc_s12_cifar_noattack_s0`, PENDING). Probe log = one consolidated `_probe.npz` per run: per (round, client) the
full 32 x C softmax, ground-truth `is_mal`, rho, and the live rule's pred/conf/flagged. Scripts: `pooling_taxonomy.py`, `pooling_heldout.py`,
`benign_partition_skew.py`, `cap_class_share_check.py`, `cap_outcomes.py`, `cap_residual.py`, `spike_vs_asr.py`.
"Acceptance" = adversarial rounds where no sampled malicious client is flagged / adversarial rounds (per round, not per appearance).

### Findings
1. **Top-16 pooling restores detection on GTSRB.** s8: 7/341 adversarial rounds accepted (2.1%, all in rounds 1-50), 0 FP, ASR ~0, MA ~77 at 500 rounds,
   vs s6 full a05 82.6% acceptance / ASR ~90 (s6 baseline is 70-91%, not 100%; the std attacker, s7, is ~1%). s9 reproduces s6 round-for-round (85% overall, 100% from round ~200).
2. **Acceptance at a fixed rho is ordered by how much weight the pooling gives the upper tail** (min < q25 < median ~ trimmed mean ~ mean < q75 < top-k(k small) < max), on GTSRB and CIFAR-10 alike.
   Trimming/median do no better than the mean (malicious clients respond to only some probe images). High-tail rules pay in benign FPs once benign clients respond too (CIFAR top-8 12%, max 47%).
3. **GTSRB blind spot = threshold calibration.** Mean AUC(target score, mal vs benign) 0.9999; benign max 0.32 < malicious median 0.385. A threshold calibrated on benign records only
   (zero-FPR) generalises held-out (TPR 0.92-0.99, acceptance 2-4%). Caveats: one seed; logged vectors are conditional on the live gate's trajectory; calibration needs a benign reference the defence does not have.
4. **CIFAR-10 is different.** Mean AUC 0.920 (top-16 0.926), 301 benign FPs/1000 rounds, benign max 0.81 > malicious median 0.30; held-out zero-FPR TPR <= 0.11, 1%-FPR TPR <= 0.28.
5. **Cause of the CIFAR FPs: benign target-class (class-0) skew.** Spearman(class-0 share, benign mean score) = 0.84 across 90 clients; the 6 clients with share >= 0.30 hold all 301 FPs
   (3 of them, cids 9/75/18 at ~50%, dominate); GTSRB benign shares are 0-6% and it has 0 FPs. Replicated on a second partition (seed 1, cids 49/73/65/22/89; `exp_partition_profile_NOTE.md`).
6. **Causal test (s11, cap 0.15, swap intervention above):** benign FPs 301 -> 0; AUC mean 0.920 -> 0.971, top-16 0.926 -> 0.975; benign max (mean) 0.813 -> 0.389; donors' mean score 0.256 -> 0.136, receivers 0.075 -> 0.101
   (movement in both directions; weak dose-response only); ASR/MA unchanged (88.0/83.5). Attack acceptance 74.5% -> 67.4% (454 vs 502 of 674): within the seed-to-seed spread of s1a (74.2/76.2/68.8%, spread 7.4 pts, n=3, seeds also change partitions) -> prediction "largely unchanged" counted as held, with that caveat.
   Held-out calibration (mean, benign-quantile thresholds): 1% target TPR 0.66/0.84/0.73, held-out FPR 1.1/0.5/0.6% (early->late, late->early, even->odd); top-16 0.84/0.92/0.84 (FPR 3.7/0.4/0.9%). Acceptance with calibrated threshold 11-26% (mean), 5-12% (top-16) -- viable, but not GTSRB-quality; early->late (benign-score drift) is the hardest split.
7. **Residual after the cap is real and not all skew.** Donors enriched among top benign scorers (8/17 of top-17 vs ~3 by chance) and class-0 share still correlates (Spearman 0.70, compressed 0.03-0.15 range) -- so part of the residual is the same mechanism not fully removed at 15%.
   But the benign MAX (0.389) is a tail statistic set by a few single-round spikes in LOW-share clients (cid 79 at 7.3%: 5 of the top 8 records; cid 2 at 7.2%; cid 39 at 9.5%); benign p99 = 0.208, p99.9 = 0.290.
   The spike records look like backdoor responses (pooled argmax class 0, 8-12 of 32 images > 0.5, comparable to typical malicious records). **Their cause is UNEXPLAINED.**

### Hypotheses tried and abandoned (record kept deliberately)
- **"CIFAR's blind spot is just a threshold problem, like GTSRB."** Tested pre-cap: held-out calibration fails on CIFAR (finding 4). Abandoned; skew was the blocker.
- **Class-8 lead** (class-8 share correlates with the capped-run benign score). Spearman +0.25 over 90 clients, +0.32 over the 45 with class-0 share < 0.10 (control: +0.02 / +0.14), p ~ 0.03 UNCORRECTED over 10 classes tested, n = 45 -> not significant after correction.
  Class 8 is in the pooled top-3 of only 3 of the 8 top spike records (top class in one). Dropped. (Mild thematic link to the 'sink class' finding in `exp_attractor_vs_dominant_scale_NOTE.md`, which was a seed-1 partition; no evidence of a mechanism.)
- **Backdoor-inheritance hypothesis** (proposed by me, endorsed, then tested and retracted): benign clients start the round from a backdoored global model and some retain the backdoor behaviour, producing spikes. Test (`scripts/spike_vs_asr.py`, s11; ASR = the model the client started from = previous round's post-aggregation ASR; spike = benign record > p99 = 0.208, 90 records):

  | prev-round ASR | benign records | spike rate (>0.208) | records > 0.30 |
  |---|---|---|---|
  | 0-10 | 27 | 0.0% | 0 |
  | 10-30 | 231 | **2.2%** (5) | 1 |
  | 30-60 | 2039 | 0.7% | 2 |
  | 60-80 | 2523 | 0.8% | 1 |
  | 80-100 | 4156 | 1.2% | 4 |

  Spikes occur at LOW ASR (highest rate there, tiny n, early-training confounded); the overall correlation of benign score with ASR (+0.47) is stage (+0.48 vs MA, +0.51 vs epoch); within epoch windows it is -0.06/+0.12/+0.06/+0.18.
  The repeat spikers that can be tested (cids 31, 39, 9) spike at ASR 21-27; cid 79 (sets the max) has no records at ASR < 30 -- untestable. Retracted: spikes do not require a backdoored global model.

### Open / pending
- **s12 no-attacker control** (`jobs/s12_cifar_noattack.sh`, `--no-attack`: namelist drawn then discarded -> partition, cap and client sampling identical to s11, nobody trains a backdoor). If benign records still reach s11's spike levels (p99 0.208 / >0.30), the cause is in benign clients + the probe mechanism, independent of any backdoor; if they vanish the backdoor matters through some route other than inheritance. Compare with `scripts/spike_vs_asr.py`.
- **Possible link to Experiment 3** (`exp3_benign_uap_NOTE.md`, `exp_fullbudget_benign_uap_NOTE.md`, seed-1 runs): against a fully benign G_agg, Stage 1/2 probes lock onto attractor classes (2, 8; 4/9 in composite-flip) and benign clients respond to them at 0.9+ confidence -- i.e. the probes behave like universal perturbations. NOTE the attractor classes are NOT the target class 0, so this does not by itself explain class-0 spikes; it is a hypothesis for s12, not a finding.
- Audit older notes for mixed-seed pairings (see seed hazard above) at write-up time.
