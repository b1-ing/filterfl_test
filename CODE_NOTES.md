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
