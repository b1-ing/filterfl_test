# Top-k fix analysis: s4/s5 full series, acceptance trend, and reconciliation

Zero compute. All numbers derived from `results/hpc_s4_topk16_s0.csv` (1000 epochs)
and `results/hpc_s5_topk16_warmup_s0.csv` (700 epochs, warm-started from
`results/clean_ref_e200_ckpt.pt`).

---

## 0. Run configurations (brief)

| | s4 | s5 |
|---|---|---|
| epochs | 1000 | 700 (attack phase only) |
| model start | cold (MA ~13% at e1) | warm (MA ~65% at e1, after 200 clean rounds) |
| `stage3_pool_topk` | 16 | 16 |
| `poison_rate` | 0.25 | 0.25 |
| `poison_epochs` | 6 | 6 |
| seed | 0 | 0 |

---

## 1. Complete epoch-by-epoch series (50-epoch windows)

### s4 — cold start, 1000 epochs

Columns: Adv = adversarial-client appearances that round (some rounds have 2),
Accepted = appearances where all sampled adversaries evade detection (TP=0),
Acc% = Accepted / Adv-rounds (not per-appearance), Det% = TPs / total adv appearances.

| Window | Adv rnds | Accepted | Acc% | Det% | Mean ASR | Mean MA |
|--------|---------|---------|------|------|----------|---------|
| e001-050 | 38 | 8 | 21.1% | 81.3% | 11.9% | 21.2% |
| e051-100 | 34 | 2 |  5.9% | 92.1% | 11.6% | 43.6% |
| e101-150 | 35 | 2 |  5.7% | 91.9% | 11.9% | 55.6% |
| e151-200 | 36 | 1 |  2.8% | 92.4% | 13.0% | 61.3% |
| e201-250 | 33 | 2 |  6.1% | 88.2% | 16.4% | 66.0% |
| e251-300 | 36 | 3 |  8.3% | 91.8% | 25.5% | 70.8% |
| e301-350 | 31 | 0 |  0.0% | 95.6% | 22.4% | 73.8% |
| e351-400 | 36 | 0 |  0.0% | 95.8% | 24.1% | 75.5% |
| e401-450 | 39 | 3 |  7.7% | 87.2% | 26.8% | 76.7% |
| e451-500 | 30 | 3 | 10.0% | 85.4% | 33.1% | 77.6% |
| e501-550 | 36 | 5 | 13.9% | 83.8% | 40.2% | 78.5% |
| e551-600 | 26 | 2 |  7.7% | 85.8% | 41.7% | 79.4% |
| e601-650 | 33 | 1 |  3.0% | 89.2% | 38.3% | 79.4% |
| e651-700 | 29 | 1 |  3.4% | 85.9% | 42.2% | 80.2% |
| e701-750 | 33 | 6 | 18.2% | 74.6% | 47.2% | 81.5% |
| e751-800 | 34 | 4 | 11.8% | 76.4% | 50.0% | 82.0% |
| e801-850 | 34 | 12 | 35.3% | 64.6% | 55.4% | 82.9% |
| e851-900 | 34 | 9 | 26.5% | 64.1% | 60.5% | 83.5% |
| e901-950 | 36 | 8 | 22.2% | 62.3% | 61.9% | 83.4% |
| e951-1000 | 31 | 8 | 25.8% | 64.2% | 64.2% | 83.4% |

### s5 — warm start, 700 attack epochs

| Window | Adv rnds | Accepted | Acc% | Det% | Mean ASR | Mean MA |
|--------|---------|---------|------|------|----------|---------|
| e001-050 | 38 | 1 |  2.6% | 90.4% |  6.0% | 66.0% |
| e051-100 | 34 | 2 |  5.9% | 90.5% | 10.1% | 66.5% |
| e101-150 | 35 | 2 |  5.7% | 92.9% | 11.7% | 69.1% |
| e151-200 | 36 | 1 |  2.8% | 93.3% | 15.6% | 71.7% |
| e201-250 | 33 | 2 |  6.1% | 89.5% | 19.9% | 72.4% |
| e251-300 | 36 | 3 |  8.3% | 90.5% | 29.4% | 74.9% |
| e301-350 | 31 | 0 |  0.0% | 95.8% | 28.9% | 75.2% |
| e351-400 | 36 | 1 |  2.8% | 91.3% | 30.2% | 75.7% |
| e401-450 | 39 | 3 |  7.7% | 83.3% | 33.5% | 76.4% |
| e451-500 | 30 | 4 | 13.3% | 82.4% | 39.8% | 78.4% |
| e501-550 | 36 | 7 | 19.4% | 72.1% | 46.8% | 78.3% |
| e551-600 | 26 | 4 | 15.4% | 77.5% | 50.6% | 80.0% |
| e601-650 | 33 | 1 |  3.0% | 83.3% | 49.7% | 80.5% |
| e651-700 | 29 | 1 |  3.4% | 81.2% | 51.8% | 80.3% |

---

## 2. Acceptance rate trend: scatter with late-run elevation, not a monotone series

100-epoch window acceptance rates for s4:
`13.9%, 4.2%, 7.2%, 0.0%, 8.7%, 11.3%, 3.2%, 14.9%, 30.9%, 23.9%`

**Kendall tau = 0.422.** There is a weak positive trend (32 concordant, 13 discordant
pairs), but the series is decidedly NOT monotone. Specifically:

- e101-400 (4 windows): acceptance is **0–7.2%**, well below the e001-100 opening (13.9%).
  The run gets MORE defended through e401, not less.
- e501-700 (3 windows): back to 3.2–11.3% — scattered, not clearly above e101-400.
- e701-1000 (3 windows): jumps to **14.9–30.9%** — a genuine structural shift.

The correct characterisation is: **acceptance is low and scattered (~2–14%) for 700
of the 1000 epochs; the meaningful change is a step increase starting around e701,
concentrated in the final 300 epochs, not a gradual rise over the full run.** A
claim that acceptance "rises from 9.3% to 36.4%" (a linear series) is not supported:
the middle section is flatter than the endpoints, and the e301-400 window has 0
accepted rounds in 67 adversarial appearances.

---

## 3. Per-appearance detection rate DOES decline — but only in the tail

The "intrinsic degradation" framing is directionally correct but the timing is wrong.

| Epochs (s4) | Det% | Mean ASR |
|-------------|------|----------|
| 1-100  | 81.7% | 12.2% |
| 101-200 | 92.2% | 13.0% |
| 201-400 | 93.0% | 22.2% |
| 401-600 | 85.5% | 35.4% |
| 601-700 | 87.5% | 41.1% |
| 701-800 | 75.5% | 49.0% |
| 801-1000 | 63.8% | 60.2% |

Detection actually **improves** from e1-100 (81.7%) to e101-400 (≈93%), then holds
steady through e700 (85–88%), then falls to **63–75% in e701-1000**. The drop is
real: in the last 300 epochs, about 1 in 3 adversarial appearances slips through
undetected vs about 1 in 10 during e101-700.

**What drives the decline?** The global model has reached near-plateau accuracy
(MA ≈ 82-84% from e700 onward). At this point the model's feature space has
consolidated and the GAN-reconstructed trigger is competing against a more
discriminative reference distribution. The stronger the global model's clean
classification, the harder it is for Stage 3 to distinguish "updates from a client
that was poisoned with our specific reconstructed trigger" from "updates from a
client whose local partition looks unusual for other reasons." This is a structural
property of the algorithm, not a run artifact — but it requires explicitly saying
*when* it kicks in (tail, not throughout) rather than calling it "intrinsic
degradation" without qualification.

The same pattern appears in s5 (Det% drops from ~93% early to ~73% in e501-600)
and at a faster rate, which is consistent with s5 starting from an already-converged
model.

---

## 4. Why s5 has higher mean ASR than s4 over epochs 1-700

| | s4 (e1-700) | s5 (e1-700) |
|---|---|---|
| Mean ASR | **25.9%** | **30.3%** |
| Mean MA (last 100 of window) | 77.9% | 80.4% |
| Per-appearance det rate | 88.2% | 87.0% |
| Per-round acceptance | 7.0% | 6.8% |
| MA at e1 | 12.9% | 64.9% |

s5 detects marginally **more** attackers per appearance (87.0% vs 88.2% is within
noise; early windows of s5 are clearly better: s5 e1-50 acceptance 2.6% vs s4's
21.1%). Despite slightly better detection, s5's mean ASR over the same 700 epochs is
**4.4 pp higher**.

**Correct explanation:** the warm start provides a model that is already useful at
classification when the attack starts. In s4, epochs 1-150 have mean MA 12-55% —
the model is still learning basic CIFAR-10 structure, and backdoor signal that enters
the aggregate has little to compound with (a model that predicts 5/10 classes
randomly cannot carry a backdoor effectively). Mean ASR in s4 e1-200 is 12.5%.

In s5, the model enters the attack phase at MA=64.9%, so the very first accepted
adversarial update injects backdoor signal into a model that already classifies well
— each compounding step multiplies off a higher base. Mean ASR in s5 e1-200 is
9.2%, lower than s4's 12.5% because early detection is better; but by e251-300 s5
is already at 29.4% mean ASR vs s4's 25.5%, and the gap widens from there.

**Backwards claim to correct:** "s5 achieves higher ASR because the model is already
warm so the GAN trigger is better from round 1, which causes detection to improve
less than expected, giving the attacker more opportunities." This is backwards on
two counts. First, GAN trigger quality depends on G_old (the global model) improving,
which helps the *defender* more than the attacker — s5's detection is in fact
better in early rounds (2.6% vs 21.1% acceptance in e1-50). Second, the attacker's
higher later ASR is not caused by more opportunities (acceptance is similar: 6.8%
vs 7.0%); it is caused by each opportunity having a larger compounding effect because
the model is already strong.

The cleaner statement: **top-16 pooling suppresses early acceptance effectively in
both runs, but a warm-started model converts accepted updates into global ASR faster
because backdoor compounding multiplies off a better-trained base. Detection is not
the bottleneck for s5's higher ASR; starting model quality is.**

---

## 5. Reconciliation of the "two accounts" of s4's tail

The **9.3→36.4 series account** appears to be derived from per-appearance detection
rate inverted (1 - Det%) in the last few windows, OR from a non-overlapping
summary that picked early and late endpoints. Neither matches the full 20-window
table above. The **scattered account** is what the complete data shows: acceptance
is not monotone, dips to 0% in e301-400, and the elevation in the final 300 epochs
appears as a step, not a gradual slope. Both observations are partially right:
the *tail* (e701-1000) genuinely has higher acceptance than the *middle*
(e101-700), but presenting this as a linear "9.3→36.4" series misrepresents the
structure by dropping the flat/zero middle.

**What to write instead:**
"Top-16 pooling maintains near-complete detection (81-95% per appearance, ~0-10%
acceptance per round) through epoch 700. In the final 300 epochs — when the global
model has reached near-plateau accuracy — per-appearance detection falls to 63-75%
and acceptance rises to 18-35% per 50-epoch window. This late-run degradation
coincides with ASR rising from ~40% to ~64%, consistent with a structural limit
where detection is harder against a more discriminative reference model. The
degradation is not gradual from round 1; it is a step change beginning around
epoch 700."

---

## 6. Summary for the s4/s5 table

| Metric | s4 (1000 ep) | s5 (700 ep) |
|--------|-------------|------------|
| Overall per-appearance det rate | 81.7% | 87.0% |
| Overall per-round acceptance | 11.9% | 6.8% |
| Mean ASR (full run) | 34.5% | 30.3% |
| Mean ASR (last 100 ep) | 64.7% | 52.3% |
| MA at end (last 10 ep mean) | 82.9% | 80.8% |
| Early acceptance e1-50 | 21.1% | 2.6% |
| Convergence effect confirmed? | n/a | **Yes** — early acceptance nearly eliminated |
| Intrinsic late-run degradation? | Yes, from e701 | Yes, from e501 |
| "Intrinsic from round 1"? | **No** — detection improves through e400 | No |
