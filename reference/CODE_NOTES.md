# FilterFL official code — notes (reference/filterfl)

Source: https://github.com/easywood0204/FilterFL (no LICENSE file — read-only
reference; do not copy into `filterfl/`, my own implementation is independent).

Commit read: `df3d66f` (head at clone time, 2026-10-08).

## Environment — what I actually ran

The repo ships no `requirements.txt`/environment file. The task brief assumed
Python 3.8 / CUDA 11.3 / PyTorch 1.13, but nothing in the code is actually
version-pinned to that — `torch.cuda.amp.autocast`/`GradScaler`, the tensor ops,
and the `torchvision.datasets` calls are all still valid APIs in current PyTorch.
I **ported rather than pinned**:

- Python 3.12 venv (`.venv/` at repo root, via `uv venv --python 3.12`)
- `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`, `pyyaml==6.0.3`
- Verified on this machine's GPU (RTX 4050 Laptop, 6GB) — `torch.cuda.is_available()` is True.

No source changes were needed to run `main.py` as-is on this stack. I ran
3-epoch smoke tests for both `--defense None` and `--defense ours` on CIFAR-10
and both completed without errors (see command log below) — so behavior is
being exercised on the current stack, not just imported.

One pre-existing fragility carried over from upstream, not introduced by the
port: `device.py` hardcodes `torch.device('cuda:0')` unconditionally — the repo
has no CPU fallback path. Fine here since a GPU is present; would crash on a
CPU-only box.

Commands that worked:
```
python main.py --params cifar_params.yaml --defense None
python main.py --params cifar_params.yaml --defense ours
```
(I used a copy, `smoke_params.yaml`, with `epochs: 3` for the smoke test —
original config requests `epochs: 2000`.)

## File-by-file

- **`main.py`** — entry point. Parses `--params` (yaml path) and `--defense`
  (string, default `'None'`, or `'ours'` for FilterFL). Seeds numpy/torch/random
  from `params['seed']`. Builds a `Handle`, loads data, builds model, dumps the
  resolved params back to `saved_models/.../params.yaml`, then calls
  `algorithm.FedAvg(handle)`. Only `algorithm == 'FedAvg'` is wired up even though
  the yaml has an `algorithm` field (no dispatch for anything else).

- **`handle.py`** — `Handle` class: owns dataset loading (CIFAR10/MNIST/CIFAR100/
  GTSRB/TinyImageNet — **correction, 2026-10-09: MNIST is NOT actually complete
  either, see discrepancy #9 below** — GTSRB imports `models.gtsrb.GTSRB` which
  doesn't exist in this repo snapshot, TinyImageNet references an undefined
  `TinyImageNet` class), Dirichlet non-IID partitioning
  (`sample_dirichlet_train_data`), building poisoned test sets
  (`poison_test_dataset_ind` — removes target-label(0) test samples from the
  clean set and keeps a separate target-label pool), and model construction.
  `self.target = 0` is hardcoded — the backdoor target class is always class 0,
  not a free parameter. `random_get_clients` (unused by the `ours`-defense path
  actually taken — `algorithm.get_clients` is the one `FedAvg` calls) supports a
  `single_attack`/`multiple_attack` distinction that isn't reachable from
  `cifar_params.yaml` as shipped (`poison_type` key is absent from the yaml, so
  this whole method is dead code for the configs provided).

- **`algorithm.py`** — the actual training loop (`FedAvg`), client sampling
  (`get_clients` — simple `random.sample`, no attacker-only pass-through logic
  like `random_get_clients` has), and `Aggregation` (plain FedAvg-with-eta: new
  global = old + eta · mean_i(beta_i · (w_i − old))). `Aggregation` takes
  `defence_method` and `params` arguments but never reads them — dead
  parameters, filtering is applied entirely in `FedAvg` by zeroing `beta[i]`
  before a second aggregation call, not inside `Aggregation` itself.
  This is where FilterFL's Stage 3 (model filtering) actually lives — see below.

- **`gan.py`** — `gan_defense(d1, d2, params, the_epoch)`: Stage 1 (knowledge
  extraction) and Stage 2 (trigger filtering) CGAN training. `d1`/`d2` are
  state_dicts loaded directly into fresh `ResNet8` instances to serve as the two
  frozen discriminators — there is no separately-trained discriminator network;
  the classifier itself (at two points in time) is the discriminator, matching
  the paper's framing.

- **`train.py`** — `standard_train`: local SGD training for one client, with
  GPU-side BadNets-style trigger application (`apply_pattern_batch_gpu`) when
  `poison=True`. Uses AMP (`torch.cuda.amp.autocast`/`GradScaler`) when CUDA is
  available — these are the deprecated-but-functional pre-2.x AMP APIs, still
  usable in torch 2.5 with a `FutureWarning`, not touched since behavior is
  unaffected.

- **`test.py`** — `normal_test`: plain accuracy/loss eval; when `poison=True` it
  applies the fixed pixel pattern to every sample in the batch and measures
  accuracy against the all-poisoned labels, i.e. this produces ASR directly
  (fraction still classified as target after patch), not a defended/undefended
  split — single function reused for both MA (`poison=False`) and ASR
  (`poison=True`) depending on which dataloader (`test_data` vs
  `test_un_target_label_data`) is passed from `algorithm.FedAvg`.

- **`device.py`** — `device = torch.device('cuda:0')`, unconditional, no
  CPU/MPS fallback (see Environment note above).

- **`models/ResNet8.py`** — ResNet with `BasicBlock`, `[1,1,1]` block counts,
  `in_planes` starting at 128 (not the usual 64) — this is a much wider ResNet-8
  than the textbook CIFAR ResNet (channels 128→256→512 instead of 64→128→256).
  Used both as the FL classifier and as both GAN discriminators.

## Hyperparameters — actual values used (code, not paper)

From `cifar_params.yaml` (authoritative for FL/attack config):
| param | value |
|---|---|
| epochs | 2000 |
| local_epochs (benign) | 2 |
| poison_epochs | 6 |
| lr (benign) | 0.01, decayed ×0.998/epoch |
| poison_lr | 0.005, decayed ×0.998/epoch |
| poison_rate | 0.5 |
| num_clients sampled/round | 10 |
| total participants | 100 |
| poison_client_num | 10 |
| dirichlet_alpha | 1 (**not 0.5** — see discrepancy below) |
| eta (global step size) | 1 |
| momentum / weight decay | 0.9 / 0.0005 |
| `ours_standard` (ρ schedule) | `[0.50, 0.40, 0.30, 0.20]` |

From `gan.py`, hardcoded (**not exposed in any yaml**):
| param | value | meaning |
|---|---|---|
| γ (`Alpha`) | 0.75 | Stage 1 and Stage 2 loss weight — same constant reused everywhere a γ-like term appears |
| λ | **does not exist** | see discrepancy below |
| n_epochs1 | 10 | Stage 1 CGAN training rounds per class |
| n_epochs2 | 5 | Stage 2 CGAN training rounds |
| generator lr | 0.0005 (Adam, β=(0.5,0.999)) | |
| image_num | 32 | images generated per class per round |
| latent_dim | 100 | |

ρ is **not a single scalar** — it's a 4-step schedule indexed by
`(epoch-1)//500` over the 2000-epoch run (0–499→0.50, 500–999→0.40, …), i.e.
the defense gets strictly more aggressive (easier to trip) as training
progresses. Any fixed-ρ framing from the paper abstract is an approximation.

## Design assumptions and where they are enforced

**G_old is trusted with nothing checking it.** `algorithm.py:75`:
`ori_weight = handle.model.state_dict()` is taken at the top of each round
*before* that round's aggregation, and this is what's passed as `d1` into
`gan_defense`. So "G_old" = whatever global model survived filtering last
round. There's no separate buffer, no staleness window, no "last N accepted
models" — just the immediately-prior accepted global model.

This is not a bug and not a discrepancy from the paper — the paper explicitly
assumes the previous round's output is benign, and the code implements exactly
that assumption, nothing more and nothing less. It's **architectural**: FilterFL
is data-free by design, so it has no independent clean reference to check a
model against. Its only candidate for "ground truth clean model" is the
previous output of the detector itself. Round 1's G_old is the random init
(trivially benign); every subsequent round's G_old is "whatever Stage 3 didn't
catch last time." Nothing in the code checks, bounds, or restores this — there
is no mechanism by which a bad G_old could be detected or rolled back, because
the detector has no other source of truth to check it against.

**This circularity is the central finding so far.** The defence's trust
chain is: round *k*'s detector quality depends entirely on round *k−1*'s
detector having worked. A false negative at round *k−1* doesn't just let one
bad model through — it corrupts the reference the detector uses at round *k*
and every round after, with no self-correcting mechanism. This is the natural
attack surface for generalizing beyond FilterFL to the whole generative-probing
class: any defence in this family that bootstraps its "clean reference" from
its own prior output inherits the same circularity, regardless of what
scheduling or trust-history machinery (SiftFL) is layered on top.

## Implementation discrepancies vs. the Algorithm 1–3 description given in the task brief

1. **No λ, no three-way split in Stage 2 — net effect: weaker.** The task's
   description of Stage 2 gives a loss with three independently-weighted terms
   (γ, λ, 1−γ−λ) applied separately to the "added to other classes" and
   "subtracted from own class" variants of I'. The actual code in
   `gan.py:140-184` computes a **single** `g_loss` that sums both variants
   together using only one constant (0.75/0.25, the same `Alpha` as Stage 1)
   and adds a **third, undocumented term**
   `out3 = softmax(D1(new_gen_trigger))[0]` evaluated on the raw trigger
   pattern itself (not composited onto any base image). There is no λ anywhere
   in the codebase. Losing the independent λ weight removes a degree of
   freedom the paper's formulation has to balance the two I' variants against
   each other — the deployed trigger-generation objective is less flexible
   than described, which should make the resulting trigger set a *less precise*
   approximation of the true backdoor trigger than the paper's version would
   produce. The real decision surface being optimized is
   `0.75·std(D1) terms + 0.25·(class-0-confidence-under-D2 terms) +
   0.25·D1's-own-class-0-confidence-on-the-bare-trigger`, not the cleaner
   3-term split the paper presents.

2. **Stage 3 only uses the trigger set from the first Stage-2 iteration —
   net effect: unclear, flag as dead-code-or-bug.** `gan_defense` returns
   `tri_imgs`, a list of n_epochs2=5 generated trigger batches (one per
   training epoch, each using the best-loss generator seen so far).
   `algorithm.py:95` calls `gan_defense(...)` then indexes `g_imgs[0]` — the
   *first* epoch's batch — when classifying client models in Stage 3. The
   other 4 refinement epochs' trigger batches are computed (at real GPU cost)
   and discarded. Either this is intentional (first pass is already stable and
   further refinement overfits) and the detector is just as strong as
   intended, or it's a leftover indexing bug and the shipped detector is
   running on an under-trained trigger set — i.e. **weaker** than the
   published algorithm, which presumably means "run Stage 2 to convergence."
   I can't resolve which from the code alone. Either way: **the shipped
   detector is not the one the paper describes**, since 4/5 of the compute it
   spends on Stage 2 never reaches Stage 3. This directly bounds how much
   "trigger quality" an attacker actually has to beat in practice.

3. **ρ is a schedule, not a constant — net effect: stronger over time.** See
   hyperparameter table above. The confidence check itself
   (`standard < out_val and out_n == 0`) does match the paper's
   `confidence > ρ` framing structurally, but ρ *decreases* 4 times over a run
   (0.50→0.40→0.30→0.20), which lowers the bar for exclusion as training
   progresses — a client needs less and less confidence on the trigger set to
   get filtered. The deployed detector becomes strictly more aggressive in
   later rounds than a fixed-ρ=0.50 reading of the paper would predict; any
   attack analysis using a single ρ value is only valid for whichever 500-epoch
   block it targets.

4. **Target class is hardcoded to 0, everywhere — net effect: neutral in
   isolation, but welds in an untested assumption.** Not read from params:
   `handle.target = 0` (`handle.py:45`), `out_n == 0` in Stage 3
   (`algorithm.py:120`), and the index-0-is-special treatment of `ori_imgs[0]`
   in Stage 2 (`gan.py:151`,`164`). The paper and task brief present the target
   class as general; the code only ever defends/attacks class 0. Doesn't make
   the detector weaker or stronger per se, but it means the paper's claimed
   generality across target classes has never actually been exercised by this
   code — worth treating target-class choice as a live attack axis rather than
   assuming it's interchangeable.

5. **`dirichlet_alpha` in the shipped yaml is 1, not 0.5 — net effect:
   likely weaker non-IID stress-test than the paper's headline numbers.** The
   task brief's experimental config (and the paper's stated setup) says α=0.5;
   the repo's own `cifar_params.yaml` ships α=1, which is *more* IID (closer to
   uniform per-client class balance) than the paper's config. A more-IID setup
   gives the detector cleaner, more homogeneous benign signal to contrast
   against, which should make detection easier than at the paper's actual
   α=0.5. Per Part B of the task brief: **α=0.5 is now the primary config**,
   α=1 kept as a side run specifically to test whether the repo's own default
   is where the defence looks best (i.e. whether the shipped default was
   quietly chosen to flatter the results).

6. **`Aggregation()` silently ignores `defence_method`/`params` — net effect:
   none, just confusing to read.** Filtering doesn't happen inside the
   aggregation function despite it taking those arguments — it's implemented
   entirely via `beta[i] = 0` zeroing in `FedAvg`, then calling `Aggregation`
   a second time. Dead parameters, harmless, but misleading if read in
   isolation.

7a. **`Aggregation()`'s mean divides by the number of SAMPLED clients, not the
   number of NON-EXCLUDED ones — net effect: filtering shrinks the effective
   global step size rather than renormalizing to the survivors.**
   `algorithm.py:27-52`: `weighted_diffs` is built by iterating
   `range(len(w_list))` (all `lens` sampled clients, `lens`=10 by default) and
   zeroing excluded clients' diffs via `* beta[i]`, but then
   `torch.mean(torch.stack(weighted_diffs), dim=0)` divides by `len(w_list)`
   unconditionally — the zeroed (excluded) entries still count toward the
   denominator. If 3 of 10 sampled clients are excluded this round, the
   average is `(sum of 7 real diffs + 3 zeros) / 10`, not `/ 7`. This means
   Stage 3 filtering does NOT behave like a standard "FedAvg over the
   survivors" — it behaves like "FedAvg over everyone, with excluded clients'
   contribution replaced by zero," which shrinks the effective step size
   (`eta`) proportionally to how many clients got excluded that round, on top
   of whatever the exclusion was meant to accomplish.
   **Our own `filterfl/aggregator.py` does NOT reproduce this** — it filters
   to `active = [i for i, b in enumerate(betas) if b != 0]` first and divides
   by `len(active)`, i.e. a proper "FedAvg over the survivors," renormalized.
   This is a genuine, previously-undocumented behavioral difference between
   our implementation and the reference's actual released code, independent
   of anything about detection accuracy — found while setting up a
   same-configuration reference run to compare ASR trajectories
   (`exp_reference_asr_comparison_NOTE.md`). Net effect on our run vs. a
   reference run at the same config: unclear without measuring, but plausibly
   relevant to any ASR/MA discrepancy between the two, since heavier false
   positive exclusion (already measured as ~0.8-0.9 fp/round in our run) would
   shrink the reference's effective step size far more than ours per round
   affected, all else equal.

   **Quantified from existing logs (919 rounds of `step2_cellA_stealthy`,
   zero-compute): the reference's denominator rule would have applied a mean
   effective step-size multiplier of 0.933 (arithmetic) / 0.928 (geometric)
   per round.** 522/919 rounds had zero exclusions (multiplier 1.0); the
   worst single round excluded 6/10 sampled clients (multiplier 0.4).
   **Correction to the initial framing:** this does NOT compound multiplicatively
   across rounds like a decay applied to the model itself (NOT `0.933^900`) --
   it scales each round's INCREMENTAL update, so the net effect over a run is
   closer to training at ~93% of the configured learning rate throughout, not
   an exponentially vanishing one. That shifts where training sits on its
   convergence curve at a given epoch count, not the asymptote it eventually
   reaches -- a modest effect, and on its own almost certainly too small to
   explain a large (e.g. 50-point) ASR gap. Still worth testing via the
   ablation flag below, but expectations should be calibrated accordingly.

   **Added as an ablation flag, not a default change**: `filterfl/config.py`'s
   `aggregation_denominator` (`"survivors"` default / `"sampled"` to reproduce
   this behavior), threaded through `filterfl/aggregator.py`. Ours
   (`"survivors"`) is arguably the more defensible engineering choice and may
   be closer to what the paper's description intends -- but theirs
   (`"sampled"`) is what actually shipped and produced the paper's published
   numbers. For reproduction purposes, theirs is the behavior to match: the
   open question is whether FilterFL's reported result reproduces, not
   whether a corrected version of their code would do better.

7. **Trigger geometry: the shipped pattern is scattered pixels, not the
   paper's described "white stripe" — net effect: unclear, flag as a live
   variable.** The CCS paper's prose describes the default BadNets trigger as
   "a white stripe in the upper left corner." The actual shipped pattern
   (`cifar_params.yaml`'s `0_poison_pattern`, faithfully reproduced in our
   `filterfl/config.py`) is **two broken 4-pixel dashes per row, at rows 0 and
   3, columns {0-3, 6-9}** — i.e. 4 disconnected 4-pixel blocks with a 2-pixel
   gap at columns 4-5, NOT a contiguous rectangle (confirmed by rendering both
   as images — see the project's `trigger_pattern.png` vs
   `trigger_pattern_stripe.png`). Stage 2 searches for an *additive*
   perturbation that flips classifications; a contiguous high-contrast stripe
   is a structurally different search target from 4 scattered dashes with
   gaps, so this is not a cosmetic discrepancy — it bears directly on whether
   Stage 2's trigger-reconstruction difficulties (see the project's
   attractor-class findings) are an artifact of this specific trigger's
   geometry rather than a general property of the defense. See
   `exp_stripe_trigger_baseline_NOTE.md` in the project's `results/` for a
   controlled test isolating this variable.

8. **CGAN training budget: shipped values (10/5) differ from 12/15 — net
   effect: plausibly weaker Stage 2.** `n_epochs1=10` (reported elsewhere:
   12), `n_epochs2=5` (reported elsewhere: 15 — a 3x difference).
   **Correction to attribution**: the 12/15 figures are from **SiftFL's**
   paper describing FilterFL's settings (a third-party report of FilterFL's
   config), NOT FilterFL's own Table 2 directly — this project does not have
   independent confirmation of FilterFL's own stated CGAN epoch counts from
   its own text. `image_num=32` (shipped) vs. a reported `1` carries the same
   secondhand-attribution caveat and hasn't been reconfirmed against
   FilterFL's own text either; treat that specific number as less certain
   than the 10/5-vs-12/15 epoch-count comparison. The Stage 2 round-count gap
   is the one that matters most regardless of source: it is what narrows a
   generic perturbation into the actual reconstructed trigger, and 5 rounds
   against a reported 15 is a plausible source of an under-trained trigger
   set. Tested directly in the project
   (`exp_fullbudget_detection_baseline_NOTE.md`,
   `exp_fullbudget_benign_uap_NOTE.md`): running at the 12/15/1 budget did NOT
   change the class-4 attractor-lock result against a known strong backdoor
   (still 0/10 class-match), though that specific test was confounded by also
   using a much later checkpoint than the original baseline — see that note
   for the caveat. Not yet cleanly resolved as budget-independent; flagged
   here as the authoritative place this discrepancy is tracked.

9. **The released code cannot run ANY MNIST experiment at all, attack or no
   attack, defended or undefended. The paper reports MNIST results in Tables
   2 and 4 and uses MNIST for its entire attacker-ratio (η) sweep — so this
   is not a minor path gap, it blocks reproducing most of the paper's own
   reported experiments from this code release.**

   Precise accounting of what's present vs. missing, since "MNIST doesn't
   work" alone isn't exact enough:
   - **Config:** no MNIST params yaml is shipped (only `cifar_params.yaml`
     and `smoke_params.yaml`, both CIFAR-10) — a user would have to write
     their own, as this project did.
   - **Data loader: PRESENT AND FUNCTIONAL.** `handle.py:78-87` has a
     complete MNIST branch (`transforms.ToTensor()` +
     `transforms.Normalize((0.1307,),(0.3081,))`, `datasets.MNIST(...)`) —
     confirmed working by actually running it: it downloaded and loaded MNIST
     successfully before the run crashed at the next step.
   - **Model: MISSING ENTIRELY.** `handle.py:313`: `model = CNNMnist()` —
     `CNNMnist` is referenced but never defined or imported anywhere in this
     repo snapshot (`models/` contains only `ResNet8.py` and an empty
     `__init__.py`).
   - **Defense (Stage 1/2): structurally CIFAR/ResNet8-only, independent of
     the model gap** — see the second bullet below.

   Two independent, compounding breaks, confirmed by actually
   attempting the run (not inferred from reading alone):
   - **`handle.py:313`: `model = CNNMnist()` — `CNNMnist` is referenced but
     never defined or imported anywhere in this repo snapshot** (`models/`
     contains only `ResNet8.py` and an empty `__init__.py`). `create_model()`
     is called unconditionally in `main.py` before `algorithm.FedAvg` starts,
     regardless of `--defense` or `is_poison` — so this crashes
     (`NameError: name 'CNNMnist' is not defined`) immediately on ANY MNIST
     run, confirmed by running
     `python main.py --params mnist_test_params.yaml --defense ours` to
     completion of the traceback.
   - **Even if `CNNMnist` existed, `gan.py:87-88` (`discriminator1 =
     ResNet8(num_classes=classnum)`) would still break the DEFENDED path
     specifically**: Stage 1/2's discriminators are unconditionally
     `ResNet8`, whose first conv (`ResNet8.py:44`:
     `nn.Conv2d(3, 128, ...)`) is hardcoded to 3 input channels and would
     reject MNIST's 1-channel images, AND would need to `load_state_dict` a
     `CNNMnist` state dict (the actual per-round client/global model for
     MNIST) into a structurally different `ResNet8` instance — a key
     mismatch independent of the channel-count problem. Stage 1/2 is
     CIFAR/ResNet8-specific by construction, not just by the current
     config's dataset choice.
   - **Our own implementation inherits the second problem, not the first —
     CONFIRMED BY RUNNING IT, not just predicted.** `filterfl/defense.py` has
     a working `CNNMnist` (`filterfl/models.py`) and `build_model` dispatches
     to it correctly for `type: MNIST`, so a plain FedAvg (undefended) MNIST
     run works fine on our side (confirmed: MA=98.78, ASR=99.83 over the last
     10 of 100 epochs at the arXiv v2 Table VII attack settings, matching the
     paper's own no-defense row of MA 98.46/ASR 99.83 almost exactly). But
     `run_detection()` (`defense.py:43-44`) unconditionally builds `ResNet8`
     probes exactly like the reference's `gan.py` does — running
     `defence_method: filterfl` on the same MNIST config crashes on the very
     first detection round with exactly the predicted error:
     `RuntimeError: Error(s) in loading state_dict for ResNet8: Missing
     key(s): "stem_conv.weight", ... Unexpected key(s): "c1.weight",
     "c1.bias", "c2.weight", "c2.bias", "fc1.weight", "fc1.bias",
     "fc2.weight", "fc2.bias"` — a `CNNMnist` state dict being fed into a
     `ResNet8` instance. This is not a bug we introduced; it's a faithful
     inheritance of the reference's own CIFAR-only Stage 1/2 design, which
     was simply never built out to handle a second architecture/channel
     count. (A third, separate issue if this is ever patched: `GeneratorCIFAR`
     hardcodes 32×32 output regardless of dataset, vs. MNIST's native 28×28 —
     would need resizing or a dedicated generator to reconcile.)
   - **Bottom line for reproduction purposes**: neither the unmodified
     reference code nor our implementation can currently produce a DEFENDED
     MNIST number at all (both blocked by the same root architecture
     mismatch); the reference additionally cannot produce ANY MNIST number,
     defended or not (blocked earlier, by the missing class). Only our
     implementation's undefended (no-defense) MNIST path is currently
     runnable.

## Framing for the shipped-`cifar_params.yaml` unmodified run (2026-10-09)

Before launching `python main.py --defense ours` against the repo's own shipped,
UNMODIFIED `cifar_params.yaml` (no `cella_match_params.yaml`-style edits this
time), four known deviations from the paper's own text, consolidated here since
they bound what this specific run can and cannot show:

1. **`dirichlet_alpha: 1`** — the paper's stated default is 0.5 (discrepancy #5).
2. **`poison_rate: 0.5`** — the CCS text's stated default is 0.3 (see "Correction:
   poisoning rate default" below).
3. **Trigger geometry** — `0_poison_pattern` is four disconnected 4-pixel blocks
   (rows 0 and 3, cols 0-3 and 6-9, gap at cols 4-5), not the paper's described
   "white stripe in the upper left corner" (discrepancy #7). **Now confirmed
   directly from the shipped config file itself**, not inferred from this
   project's port of it.
4. **CGAN epochs 10/5 hardcoded in `gan.py`** — reported elsewhere (SiftFL's
   paper, describing FilterFL's settings) as 12/15 (discrepancy #8, with its
   attribution correction).

Everything else in the shipped config matches the paper's stated setup: 2000
epochs, local_epochs=2, lr=0.01, lr_decay_gamma=0.998, momentum=0.9, decay=0.0005,
100 total clients/10 sampled per round, poison_epochs=6, poison_lr=0.005, rho
schedule [0.50, 0.40, 0.30, 0.20].

**Net: this run targets Table 3's α=1.0 row (MA 86.70, ASR 2.27) on the
non-IID axis, but at poison_rate=0.5 rather than the stated default 0.3 — so it
does not reproduce any single published row exactly.** That is a finding about
the released artifact (it ships a config that doesn't match any one published
cell precisely), not a flaw in this test's design.

## Correction: poisoning rate default (2026-10-09)

Earlier notes and the project's experiment framing treated `poison_rate=0.5` as
"the paper's value" (sourced from the arXiv v2 appendix) and `0.1` as a
deliberately "stealthy" variant below it. Having since found FilterFL's actual
CCS paper text, **the paper's own stated default poisoning rate is 0.3, not
0.5.** This means:
- `cellA_stealthy`'s `poison_rate=0.1` is stealthier relative to the paper's
  actual default than previously framed (0.1 vs 0.3, not 0.1 vs 0.5).
- The Experiment-1 "strong-backdoor" config (`poison_rate=0.5`, used in
  `exp1_strong_backdoor_v2_NOTE.md` and everywhere downstream, e.g.
  `exp_fullbudget_detection_baseline_NOTE.md`) is ABOVE the paper's stated
  default, not AT it — it was built to match the arXiv v2 appendix's number,
  which is apparently a different value than the main CCS text's default.
- This does not invalidate any finding to date (all of them are about Stage
  1/2/3 mechanics, not about calibrating "how strong is a realistic attacker"),
  but every note that frames 0.5 as "the paper's default" or 0.1 as "stealthy
  relative to the paper's default" should be read with this correction in
  mind. Not retroactively editing every such note; recorded here as the
  authoritative correction.

Also worth flagging: the CCS paper text states α (dirichlet_alpha) and learning
rate were varied as experimental factors in the paper's own sensitivity
analysis. There may be a sensitivity-analysis section in the paper bearing
directly on this project's findings (e.g. whether detection reliability is
reported to vary with α) that hasn't been located/reviewed yet — worth checking
before treating any single-α finding here as representative of the paper's own
claimed operating range.

## Anything else hardcoded that the paper presents as free

- CGAN training rounds (`n_epochs1=10`, `n_epochs2=5`), generator learning rate
  (`5e-4`), images-per-class (`32`), and latent dim (`100`) are all hardcoded in
  `gan.py` and not configurable via any yaml — if the paper presents "E" (CGAN
  training rounds) as a tunable hyperparameter, the actual released code bakes
  it in as a magic number. Per the paper's Table 2 (discrepancy #8 above), the
  intended values are 12/15/1, not 10/5/32.
- `γ` (`Alpha = 0.75`) is a literal inside `gan_defense`, not read from
  `params`, despite the task brief (and presumably the paper) treating it as a
  configurable defense hyperparameter.

## Smoke-test results (3 epochs, CIFAR-10, seed=1, batch=128)

`--defense None`: runs clean, MA/ASR both near-random after 3 epochs (expected
— 3 epochs is nowhere near convergence, this only validates plumbing).

`--defense ours`: Stage 1–3 ran without error; in this specific 3-epoch,
seed=1 smoke run, 2 sampled-adversary events occurred across 3 rounds and both
were correctly flagged by Stage 3 (`count_all_attack: 2, count_all_defense: 2,
count_correct_defense: 2`). This is far too short a run to mean anything about
detection accuracy — it only confirms the full Stage 1→2→3 pipeline executes
and produces sane-shaped outputs (per-client `(predicted_class, confidence)`
tuples) end to end on the current PyTorch stack.

## Determinism

`filterfl/run.py` seeds python/numpy/torch/torch.cuda from `cfg.seed` and the
checkpoint restores all four RNG streams on resume, but nothing pins
`torch.backends.cudnn.deterministic`/`benchmark`. cuDNN conv/batchnorm kernels
on CUDA are not bit-reproducible run-to-run even with identical RNG state, so
a resumed epoch (e.g. re-running epoch 61 after `--resume` restores the
epoch-60 checkpoint, since checkpoints only save every 5 epochs) can log
slightly different MA/ASR than the same epoch logged pre-interruption.
Deliberately **not pinning** this — doing so changes performance
characteristics (cudnn autotuning) mid-project and would make new runs
incomparable to what's already been collected. Treat any single-epoch
mismatch across a resume boundary as expected noise, not a bug, and dedupe
the stale pre-interruption row rather than trying to reconcile the two.

## Stage 2 loss: paper-faithful vs shipped-code, empirically (2026-10-09)

Discrepancy #1 above recorded, qualitatively, that the shipped `gan.py` Stage 2
loss is structurally different from the paper's 3-term (γ, λ, 1−γ−λ) framing —
single combined constant instead of an independent λ, plus an undocumented
5th term. That was written before any matched-config run existed to show what
the difference actually costs. It now has a number, and the number is large
enough to be the headline finding of this comparison, not a footnote under
discrepancy #1.

**Setup.** `filterfl/run.py --config configs/cifar_alpha1_side.yaml --defense
filterfl`, i.e. this project's own FedAvg + Stage1/2/3 detector, at the exact
config the shipped `reference/filterfl/cifar_params.yaml` run uses: α=1,
poison_rate=0.5, poison_epochs=6, same trigger geometry, same everything.
`defense.py`'s Stage 2 loss at the time of this run implemented the paper's
own 3-term split, normalized to sum to 1:

```
L_paper = γ·(nat_added + nat_subbed)/2 + λ·(1 − d2_added) + (1−γ−λ)·d2_subbed
        = 0.375·nat_added + 0.375·nat_subbed + 0.125·(1 − d2_added) + 0.125·d2_subbed
```
(γ=0.75, λ=0.125 — see `GanDefenseConfig`.)

The shipped `gan.py:140-184` computes, verbatim, a different and *unnormalized*
loss (sums to ~2.25, not 1) with a 5th term that has no counterpart anywhere
in the paper's equations — the raw trigger pattern's own class-0 confidence
under D1, evaluated on the trigger alone, composited onto no base image:

```
L_reference = γ·nat_added + (1−γ)·(1 − d2_added)
            + γ·nat_subbed + (1−γ)·d2_subbed
            + (1−γ)·D1(trigger_alone)[target]
        = 0.75·nat_added + 0.25·(1 − d2_added)
        + 0.75·nat_subbed + 0.25·d2_subbed
        + 0.25·D1(trigger_alone)[target]
```

Both naturalness terms are at **full γ=0.75 each** in `L_reference` vs
**halved to 0.375 each** in `L_paper` (the paper's `/2` average over the two
I' variants). Both bias terms — the part of the loss that actually makes the
generated trigger diagnostic of the backdoor — are at **0.25 each** in
`L_reference` vs **0.125 each** in `L_paper`: double the gradient signal on
the terms that matter, on top of a 5th regularizing term `L_paper` doesn't
have at all.

**Result, identical config/attacker, `L_paper` (this project's Stage 2) over
24 rounds:**

| Rounds | Attacks | Flagged | TP | FP | FN |
|---|---|---|---|---|---|
| 1–10 | 10 | 3 | 2 | 1 | 8 |
| 11–20 | 10 | 2 | 1 | 1 | 9 |
| 21–24 | 7 | 4 | 1 | 3 | 6 |
| **Total** | **27** | **9** | **4** | **5** | **23** |

The shipped reference code, same rounds, same config: **20/20 attacks caught,
0 FP, 0 FN** through round 20 (and 54/58 flagged / 54/54 caught through round
50 — see `results/shipped_cifar_params_run2.out`). Not a close call: 4/27 vs
20/20 at an identical, loud (poison_rate=0.5) attack.

**This localizes to Stage 1/2 (trigger strength), not Stage 3's gate.** The
gate itself (`pred_class == target and confidence > ρ`) is structurally
identical between the two codebases, same ρ=0.50 for epoch ≤ 500. If the gate
were broken you'd expect wrong-class predictions or a threshold-direction
bug. Instead, `alpha1_match.csv`'s per-client `(pred_class, confidence)`
readout shows the real attacker correctly landing on `pred_class == target`
and missing only on confidence, by a small margin, repeatedly:

- Round 21, attacker = client 50: `pred=0, confidence=0.436` — just under
  ρ=0.5. (A benign client, 13, got `pred=0, confidence=0.885` and was flagged
  instead — the FP.)
- Round 23, attacker = client 84: `pred=0, confidence=0.472` — again just
  under ρ=0.5.

That is the signature of a trigger set T that is directionally right but too
weak to push a backdoored model's confidence over threshold — exactly what
`L_paper`'s halved naturalness weight, halved bias weight, and missing 5th
term would predict.

**The finding, stated plainly: the published loss does not detect backdoors
at a loud, default attack config. The shipped code's undocumented loss does,
reliably.** Whatever the paper's text says Stage 2 optimizes, it is not what
makes FilterFL's own released numbers work. An attacker (or a re-implementer)
reading only the paper and building `L_paper` would face a detector that
catches roughly 1 in 7 attacks; the same attacker facing the actual released
server code would face a detector that catches essentially all of them, on
the very first round.

**Implication for attacker design.** Any local defence replica an attacker
builds (to probe whether a given backdoor update would get caught before
submitting it) has to model whichever loss the server *actually runs* to be
useful. If a real deployment runs the released `gan.py`, that means
`L_reference`, not `L_paper` — the paper's text is not a reliable model of
the server's actual behavior for this purpose. No adaptive/defense-aware
attacker exists in this codebase yet, but if/when one is built, its local
replica of Stage 1/2 should default to `gan.stage2_loss: reference`, not
`paper`, for exactly this reason.

**Implemented:** `cfg.gan.stage2_loss` ("paper" | "reference") in
`filterfl/config.py`, with `L_reference` added verbatim to `defense.py`
alongside the existing `L_paper` path. Default stays `"paper"` so nothing
changes silently. `--stage2-loss {paper,reference}` CLI override in
`filterfl/run.py`.

### Validation run result: the loss-weighting hypothesis is WRONG

Reran `alpha1_match` under `--stage2-loss reference` for 24 epochs, same
config/seed, to see whether `L_reference` closes the gap toward the shipped
code's 20/20. **It does not.** Same 24-round window, `run_name=alpha1_match_refloss`:

| Rounds | Attacks | TP | FP | FN |
|---|---|---|---|---|
| 1–10 | 10 | 2 | 1 | 8 |
| 11–20 | 10 | 1 | 1 | 9 |
| 21–24 | 7 | 0 | 3 | 7 |
| **Total** | **27** | **3** | **5** | **24** |

vs. `L_paper`'s 4/27 TP over the identical window. **Worse, not better** — within
noise of each other, certainly not the jump to ~20/27 the hypothesis predicted.
One concrete regression: round 22's real attacker (client 99) went from
`pred=0, confidence=0.520` (correctly caught under `L_paper`) to `pred=0,
confidence=0.490` (missed) under `L_reference` — the "stronger" loss produced
a *weaker* signal on that specific round.

**Why the predicted mechanism doesn't actually apply:** re-deriving the two
losses side by side, `L_reference`'s naturalness and bias coefficients are
each exactly 2× `L_paper`'s (0.75/0.375=2, 0.25/0.125=2) — a uniform rescale
of every term `L_paper` also has. Stage 1/2 trains the generator with Adam,
which normalizes updates by a running estimate of each parameter's gradient
RMS; a constant overall loss scale is close to invariant under that
normalization after the first step or two of warm-up. So the apparent "double
the gradient signal on the terms that matter" framing in the first write-up
of this section was wrong — under Adam, that 2× is nearly a no-op, and the
*only* term that's structurally new in `L_reference` is the 5th
(bare-trigger-under-D1) term, which this result shows is not, on its own,
responsible for the reference code's 20/20.

**Conclusion: Stage 2's loss formula is not the (or not the only) explanation
for the detection gap.** The confidence near-miss evidence (0.436, 0.472,
0.490 — real attackers landing on the right class just under ρ=0.5) is real
and reproducible, but it doesn't discriminate between "our trigger probes are
weak" and "our confidence computation or Stage 3 gate differs in some way
not yet found" — both would produce exactly this near-miss signature. Ruled
out by direct comparison so far: the confidence formula itself (both
implementations compute `mean(softmax(logits), dim=images).max(dim=classes)`,
verified identical by inspection) and `use_first_stage2_batch_only` (True on
both sides already, not a live variable in this comparison). Not yet
ruled out: something upstream in Stage 1 (the per-class image set `I` itself),
or a subtler Stage 3 discrepancy than the gate's threshold/class-match logic.
**Next: a real cross-feed test** — fix one (G_old, G_agg) pair and one set of
client models, generate T under both implementations' full Stage 1+2, and
feed each T into both implementations' Stage 3, to localize the gap to a
specific stage rather than continuing to infer from aggregate TP/FP/FN counts.

## Stage 3 probe BatchNorm mode: the actual source of detection power (2026-10-09)

The cross-feed test above (`scripts/crossfeed_diagnostic.py`) resolves this
decisively, and the answer is not the Stage 2 loss.

**Setup.** Fixed one real (G_old, G_agg, client_states) tuple from one FedAvg
round (10 sampled clients, 1 real attacker, client 47), saved to
`scripts/crossfeed_round.pt`. Ran Stage 1/2 under both implementations on the
*identical* (G_old, G_agg) pair to get `T_ours` and `T_ref` (a
key-remapping table converts between this project's and the reference's
differently-named-but-topologically-identical `ResNet8` state dicts, since
Stage 1/2 trains *using* the live global model as D1/D2). Then fed each T
into each implementation's Stage 3, all four combinations, on the same
client weights:

| client | malicious | ours Stage3 on T_ours | ours Stage3 on T_ref | ref Stage3 on T_ref | ref Stage3 on T_ours |
|---|---|---|---|---|---|
| 47 | **True** | p=0 c=0.395 | p=0 c=0.395 | **\*p=0 c=0.670** | **\*p=0 c=0.673** |
| (9 benign clients) | False | none flagged | none flagged | none flagged | none flagged |

Read this carefully: **reference's Stage 3 flags the real attacker with
confidence ~0.67 regardless of which trigger set it's fed — theirs or
ours.** Our own `T_ours` is just as diagnostic as `T_ref` once run through
*their* Stage 3. Conversely, our Stage 3 gets confidence only ~0.395 on
*either* trigger set. The variable that matters is which Stage 3
implementation does the classifying, not which Stage 1/2 produced the
trigger. **This rules out Stage 1/2 (and therefore the Stage 2 loss
formula, the prior section's hypothesis) as the explanation.**

**Root cause, found by inspecting `algorithm.py`'s inline Stage 3 block
(lines 105-122):** `ResNet8_model = ResNet8(num_classes=10).to(device)` is
constructed once; nothing ever calls `.eval()` on it. A freshly constructed
`nn.Module` is in **train mode** by default. So every one of `ResNet8_model`'s
`BatchNorm2d` layers normalizes using the statistics of *the current forward
batch* (the 32 trigger images) — not the `running_mean`/`running_var` buffers
that were just loaded from the client's own trained weights via
`load_state_dict(parameter)`. Those loaded running-stat buffers are
completely inert for this forward pass; only the loaded conv weights and
BatchNorm affine (γ, β) parameters affect the output, composed with
per-batch-computed normalization.

**Confirmed by ablation**, same script: adding `probe.eval()` to the
reference Stage 3 replica before classifying collapses its output to
*exactly* match ours — `p=0 c=0.395` on client 47, to three decimal places,
on both T_ref and T_ours. Eval-mode reference Stage 3 ≡ our Stage 3,
bit-for-bit in this test. The entire gap is this one missing method call.

**Why it matters so much:** BatchNorm in train mode re-centers and
re-scales every activation to the *current 32-image batch's* own mean/var at
every layer. For a batch that's entirely synthetic CGAN trigger images (not
natural-image-distributed), this produces a very different, apparently much
more class-separable activation geometry than evaluating with the stored
running stats from ordinary training data — stronger relative signal between
"backdoored toward class 0" and "not," at every layer, compounding through
depth. It is almost certainly an accident (a well-known class of PyTorch bug
— forgetting `.eval()` before inference), not a deliberate design choice;
there is no comment or justification for it anywhere in `algorithm.py` or
`gan.py`. But it is the actual mechanism making the shipped detector work as
well as it does.

**Validated at scale.** `filterfl/run.py --stage2-loss paper --stage3-probe-mode
train` (i.e. the *paper's own* Stage 2 loss, formula unchanged, with only the
Stage 3 probe left in train mode to match the reference's accidental
behaviour), same config, 24 epochs:

| Rounds | Attacks | TP | FP | FN |
|---|---|---|---|---|
| 1–10 | 10 | 10 | 0 | 0 |
| 11–20 | 10 | 10 | 0 | 0 |
| 21–24 | 7 | 7 | 0 | 0 |
| **Total** | **27** | **27** | **0** | **0** |

27/27, 0 FP, 0 FN — matching (and in this window, slightly exceeding) the
actual reference run's own 20/20-then-54/58 pattern, using the **published**
Stage 2 loss. One line (`probe.eval()`, or rather its absence) fully accounts
for the detection-power gap this section originally misattributed to the loss
formula.

**Implemented:** `cfg.stage3_probe_mode` (`"eval"` default, `"train"` to
reproduce the reference's behaviour) in `filterfl/config.py`, threaded
through both `stage3_classify_clients` calls in `filterfl/run.py`, with a
`--stage3-probe-mode {eval,train}` CLI override. `stage2_loss` stays at its
default (`"paper"`) — this result shows it never needed to change.

**Corrected framing, replacing the previous section's conclusion:** the paper's
*text*, read as "the loss Stage 2 should optimize," is not what makes
FilterFL's own numbers work — but neither is some other property of the
*loss*. It's that the released code's Stage 3 classifier, due to an
apparent bug, evaluates in training mode. **An attacker's local defence
replica needs to match `stage3_probe_mode`, not `stage2_loss`, to accurately
model what a real deployment running the released code would actually
catch** — default the replica to `"train"` for that reason, superseding the
prior section's (wrong) recommendation to default to `stage2_loss: "reference"`.

This reframes what Stage 3 is actually measuring: not "does this model
respond to the trigger" but "how do this model's BatchNorm layers behave
when renormalized against a batch of `images_per_class` synthetic images" —
a quantity that depends on an undocumented hyperparameter and on model state
(BN running stats) the attacker fully controls. Two cheap ablations follow,
run before building any attack on this, to find out how robust the
mechanism actually is.

### Ablation 1: batch-size (`images_per_class`) dependence

If detection power comes from renormalizing against the trigger batch, it
should depend on that batch's size. Swept `images_per_class` ∈
{32, 16, 8, 4, 2, 1} on the exact same fixed round (`scripts/batchsize_sweep.py`,
reusing `scripts/crossfeed_round.pt`), `stage3_probe_mode="train"`, everything
else unchanged:

| images_per_class | attacker pred | attacker confidence | flagged |
|---|---|---|---|
| 32 | 0 | 0.6731 | YES |
| 16 | 0 | 0.6726 | YES |
| 8 | 0 | 0.6778 | YES |
| 4 | 0 | 0.6723 | YES |
| 2 | 0 | 0.6791 | YES |
| 1 | 0 | 0.6846 | YES |

**Result: flat. No collapse, not even a visible trend, all the way down to a
single image.** Hypothesis refuted. The reason, on reflection, is that
`BatchNorm2d` normalizes per channel over all of `(N, H, W)`, not just `N` —
even at `images_per_class=1`, the early layers still have 32×32 = 1024
spatial positions per channel to compute statistics from, and even the
deepest layer (8×8 spatial after two /2 stages) has 64. There's no N→1
degeneracy to hit at the sizes this architecture ever uses. **This also
means SiftFL's reported `images_per_class=1` is not, by itself, a reason to
expect their reproduction's detection to be weaker** — at least not through
this mechanism.

### Ablation 2: architecture (BatchNorm) dependence

If the mechanism really is BN renormalization, a model with no BatchNorm
layers at all has nothing for train/eval mode to change — detection should
either collapse or become mode-independent. Added `ResNet8NoNorm` to
`filterfl/models.py` (same topology, every `BatchNorm2d` → `nn.Identity()`),
generalized `run_detection`/`stage3_classify_clients` to take a `model_cls`
parameter (default unchanged: `ResNet8`), and rebuilt the *identical* round
(same seed, same participants, same sampled clients, attacker=47 — client
sampling uses the `random` module, model construction uses torch's RNG, so
swapping architectures doesn't perturb which clients get sampled; asserted
this in `scripts/arch_dependence.py` rather than assuming it) with
`ResNet8NoNorm` clients trained end to end instead of `ResNet8`:

| | attacker pred | attacker confidence | flagged |
|---|---|---|---|
| `ResNet8NoNorm`, probe_mode=train | 0 | 0.2381 | no |
| `ResNet8NoNorm`, probe_mode=eval | 0 | 0.2381 | no |

**Two results here.** First, train and eval modes are now bit-identical
(0.2381 = 0.2381) — the expected sanity check: with zero `BatchNorm2d`
layers, there is nothing for `.train()`/`.eval()` to change, confirming the
train/eval distinction really is doing 100% of the work found in the
previous section (not some other side effect of `.eval()`, like dropout —
this architecture has none anyway, but now it's verified structurally, not
just by elimination). Second, and more directly answering the question:
**confidence drops to 0.2381**, below ρ=0.5 and below even this project's
plain `probe_mode="eval"` baseline on the BatchNorm architecture (~0.39-0.47
seen earlier) — **detection does not merely weaken, it fails outright**, on
a model that otherwise differs from the working case only in having no
normalization layers.

Caveat on this second number, stated plainly: `ResNet8NoNorm` is not a
drop-in architecturally-neutral substitution — removing all normalization
from an 8-layer ResNet changes its training dynamics and probably its
overall representational quality too, trained for the same handful of local
epochs as the BN version. This result can't cleanly separate "detection
fails because there's no BN to renormalize against" from "detection fails
because this particular ad hoc BN-free net is just a worse classifier
end to end." Both ablations point the same direction and the train=eval
identity is a clean structural confirmation of *part* of the claim, but a
fully isolated test (e.g. holding classification accuracy roughly constant
while varying only normalization) hasn't been done.

**Taken together: detection power survives arbitrarily small trigger
batches but (with the caveat above) looks like it collapses without
BatchNorm at all.** The mechanism isn't fragile to the one hardcoded
hyperparameter most likely to vary across re-implementations
(`images_per_class`), but it does appear to lean on an architectural
property — BatchNorm specifically — that's never stated as a defense
requirement anywhere in the paper or the task brief. That's still a real,
undocumented deployment constraint: a client running a BatchNorm-free
architecture (GroupNorm, LayerNorm, or no normalization at all — all
legitimate, increasingly common choices, e.g. for small-batch or
cross-client distribution-shift robustness in FL specifically) may be
invisible to Stage 3 regardless of how backdoored its update is, through no
adversarial effort at all.

### Ablation 3: GroupNorm — isolates the mechanism further, and corrects the framing

Ablation 2's `ResNet8NoNorm` result had a real caveat: a BN-free net also
just trains worse from scratch, confounding "no renormalization" with
"worse classifier." `ResNet8GroupNorm` (added to `filterfl/models.py`,
same topology, `BatchNorm2d` → `GroupNorm(32, C)`) separates these: it
normalizes **per-sample** (each image normalized using only its own
activations, no batch dependence at all) and has no train/eval distinction
whatsoever, but trains far better than the zero-normalization net.

Ran all three architectures (`ResNet8` BatchNorm control, `ResNet8GroupNorm`,
`ResNet8NoNorm`) through 15 rounds of matched plain benign-only FedAvg
pretraining (same seed, same client-sampling sequence, so differences are
attributable to the architecture, not round count), then one attack round
with two real attackers this time (clients 36 and 99 — reporting client 36;
did not re-check client 99 individually, noted as an incompleteness, but
the magnitude of the result below doesn't hinge on which attacker):

| Architecture | pretrain clean acc | attacker conf (train mode) | attacker conf (eval mode) | flagged |
|---|---|---|---|---|
| ResNet8 (BatchNorm) | 34.66% | 0.6093 | 0.5745 | **YES** (both modes) |
| ResNet8GroupNorm | 22.60% | 0.6292 | 0.6292 (train=eval, as expected) | **YES** |
| ResNet8NoNorm | 18.51% | 0.2050 | 0.2050 (train=eval) | no |

**This refutes "batch renormalization" as the precise mechanism.**
GroupNorm does not renormalize against the batch at all — it's immune to
batch composition by construction — yet it detects the attacker with
*higher* confidence (0.629) than even BatchNorm's own eval mode (0.575) in
this matched-pretraining setting, and comparably to BatchNorm's train mode
(0.609). It does this **despite being the weaker classifier** (22.6% vs.
34.7% clean accuracy), which also rules out "better classifier -> better
detection" as a confound: GroupNorm detects better while classifying worse.

**Corrected mechanism:** what distinguishes the two detecting
configurations (BN-train, GroupNorm) from the two non-detecting ones
(BN-eval, NoNorm) is not "uses the batch" — it's **whether normalization
statistics are computed from the current input itself, vs. from a fixed
external reference.** BN-train and GroupNorm both normalize using
statistics derived *from the trigger images being classified right now*
(batch-level for BN-train, per-sample for GroupNorm) — the synthetic,
out-of-distribution input gets whitened relative to itself. BN-eval
normalizes against *stored running stats from ordinary training data* — a
distribution mismatch against OOD synthetic triggers that apparently
destroys the signal. NoNorm has no adaptive whitening of any kind and the
raw activation scale apparently isn't separable enough for the classifier
head regardless. "Self-referential/input-adaptive normalization vs.
fixed/stale-reference normalization (or none)" is the real axis, not
"batch vs. not-batch" — the earlier framing in this section ("BatchNorm
renormalizing against the trigger batch's own statistics") was directionally
right about BN-train specifically but over-attributed the mechanism to
*batchness*, which GroupNorm shows isn't the active ingredient.

**This also closes the BN-running-stats attack even more thoroughly than
the section below's refutation already does on its own:** the mechanism
doesn't depend on stored buffers *in either detecting configuration* —
GroupNorm has no running-stat buffers to attack at all, and BN-train's
forward ignores the ones it has. An attack on this detector would have to
act on the conv/affine weights that carry the backdoor and simultaneously
shape how those weights respond under self-referential normalization of
whatever synthetic images Stage 1/2 happens to produce — not a cheap
state-manipulation attack, a much harder one, and out of scope here.

**Caveat, stated plainly:** GroupNorm's 22.60% pretrain accuracy is not
fully "comparable" to BatchNorm's 34.66% — it's a real ~12-point gap, likely
because GroupNorm converges slower in this specific low-round-count
federated setting, not because it's a fundamentally worse architecture.
The result is still informative specifically because it points the
opposite direction from the confound it was meant to rule out (worse
classifier, yet *better* detector) — but "comparable accuracy" was only
partially achieved, and a longer pretraining budget would be needed to
close that gap fully if a more rigorous isolation is wanted later.

## Probe-prediction fidelity: does a client need to predict T at all? (2026-10-09)

The attack direction after Ablation 3: a malicious client runs its own local
replica of Stage 1/2 using models it legitimately receives (`G_old(t)`,
and the client never sees `G_temp(t)`, the PRE-filtering aggregate the
server actually uses as d2), and optimizes its update to stay below ρ on
the *predicted* trigger set, not the server's real one. Tested two
questions before building any optimizer: does prediction work at all, and
is prediction even necessary.

### Existing logs/checkpoints were insufficient — flagged, not substituted

Past runs' checkpoints save only periodic POST-filter global-model
snapshots (`checkpoint_every`); `G_temp(t)` (pre-filter) and per-client raw
models were never persisted for historical rounds. Re-running was required:
`scripts/probe_fidelity.py` does a fresh instrumented FedAvg+detection
sequence that explicitly saves, per round, `G_old(t)`, `G_temp(t)`,
`client_states(t)`, and which clients got excluded.

### Suppression transfer: oracle vs. stale vs. linear, 8 rounds

`T_oracle = Stage1/2(G_old(t), G_temp(t))` (server's actual computation) vs.
`T_stale = Stage1/2(G_old(t-1), G_old(t))` (computable from models the
client legitimately has) vs. `T_linear = Stage1/2(G_old(t), 2·G_old(t) -
G_old(t-1))` (one-step extrapolation), `stage2_loss="paper"`,
`stage3_probe_mode="train"` (the validated 27/27 combination — an eval-mode
replica would model a detector that doesn't exist):

| t | exclusions? | attacker | oracle (pred,conf,rank) | stale (pred,conf,rank) | linear (pred,conf,rank) | MSE(stale) | MSE(linear) |
|---|---|---|---|---|---|---|---|
| 2 | Yes | 47 | p=0 c=0.671 r=1 | p=0 c=0.671 r=1 | p=0 c=0.670 r=1 | 0.0001 | 0.0001 |
| 3 | No | — | (none sampled) | | | 0.0000 | 0.0007 |
| 4 | Yes | 69 | p=0 c=0.647 r=1 | p=0 c=0.648 r=1 | p=0 c=0.649 r=1 | 0.0005 | 0.0006 |
| 5 | Yes | 69 | p=0 c=0.628 r=1 | p=0 c=0.626 r=1 | p=0 c=0.627 r=1 | 0.0012 | 0.0007 |
| 6 | Yes | 45 | p=0 c=0.583 r=2 | p=0 c=0.584 r=2 | p=0 c=0.584 r=2 | 0.0010 | 0.0008 |
| 6 | Yes | 84 | p=0 c=0.683 r=1 | p=0 c=0.682 r=1 | p=0 c=0.681 r=1 | 0.0010 | 0.0008 |
| 7 | Yes | 84 | p=0 c=0.711 r=1 | p=0 c=0.710 r=1 | p=0 c=0.710 r=1 | 0.0015 | 0.0002 |
| 8 | No | — | (none sampled) | | | 0.0008 | 0.0006 |

All four metrics (in priority order) pass cleanly: suppression transfer
(confidence agreement within ~0.002 absolute, including rounds *with*
exclusions where `G_temp(t)` and `G_old(t)` genuinely differ), predicted
class (0 in every row, no disagreement), rank (identical in every row,
including the 2-attacker round 6), pixel MSE (small throughout, consistent
with the above, not weighted per the brief). Caveat: only 6 attacker-rows
from one seeded sequence — thin, though every row agrees with no
counterexample. A 30-round extension is running to get a larger sample
before leaning on this for the optimizer design.

### Is T round-specific at all? No — and that's the more damaging finding

The near-identical stale/linear-vs-oracle agreement above is good for the
attacker but also suspicious, especially combined with the already-noted
stable-attractor-classes observation. Tested directly: feed Stage 1/2
deliberately WRONG `(d1, d2)` pairs and check whether the resulting T still
flags round 7's real attacker (client 84, oracle: p=0, c=0.711, rank=1)
with similar confidence.

| (d1, d2) pair fed to Stage 1/2 | attacker pred | confidence | rank |
|---|---|---|---|
| Oracle (reference) | 0 | 0.711 | 1 |
| A) far pair: `G_old(1)`, `G_old(2)` — early rounds, unrelated to round 7 | 0 | 0.709 | 1 |
| B) unrelated checkpoint: `G_old(1)` vs. a model from a **completely different training run** (`alpha1_match_trainprobe`, epoch 24) | 0 | 0.710 | 1 |
| C) degenerate: `G_old(7)`, `G_old(7)` — **the same model fed as both d1 and d2** | 0 | 0.709 | 1 |

**T is not round-specific.** Even the degenerate case — zero information
about the round's incremental update, by construction, since d1 and d2 are
literally identical — reproduces the oracle result within 0.002. An
unrelated checkpoint from an unrelated training run does too. The `(d1,
d2)` input barely affects what Stage 1/2 converges to, at least for this
attack configuration (α=1, poison_rate=0.5, target class 0).

**This reframes the claim from "an attacker can accurately predict the
hidden state" to "the hidden state is barely hidden to begin with" — the
cleaner and more damaging version.** Stage 1/2 appears to converge toward
an attractor (class 0) nearly independent of its inputs, consistent with
and sharpening the earlier stable-attractor-classes finding. **Consequence
for the optimizer's design:** the attacker does not need per-round
prediction at all. A trigger set T can be computed ONCE (from any
reasonable, or even nonsense, model pair) and reused across rounds — a
simpler and structurally stronger attack than "predict G_temp(t) each
round," since it requires no staleness-tracking and degrades no further
than the already-tiny oracle/stale gap measured above.

### Extended sample: 29 attacker-observations across 30 rounds

The 6-row sample above was thin; re-ran `scripts/probe_fidelity.py` with
`N_ROUNDS=30` (two bugs fixed along the way, noted below since they cost
real wall-clock time and are worth not repeating) to get 29 attacker-rows.
The finding holds, more sharply than the small sample suggested:

- **Predicted class:** `p=0` in every single oracle/stale/linear value — 29
  rows x 3 methods = 87 data points, zero disagreement.
- **Confidence transfer:** typical spread ~0.001-0.003; loosest case (round
  29, client 50) still within 0.007 (oracle 0.713 vs. linear 0.706). No
  case where stale/linear crosses ρ=0.5 in a different direction than
  oracle.
- **Rank agreement:** identical in 28 of 29 rows. One exception (round 24,
  client 54): oracle ranks it 1st (c=0.637), stale/linear both rank it 2nd
  (c=0.631/0.633) — a single-step swap at a tight margin, not a qualitative
  miss.
- **Detection stayed essentially perfect throughout** — every attacker-row
  occurred in an exclusion round, i.e. every sampled attacker across all 30
  rounds got caught by the oracle (and, by the above, would have been
  caught under stale/linear too).

**Two implementation bugs hit and fixed while producing this, both costing
real wall-clock time (over 2 hours lost to the first one before it was
caught) and worth flagging for any future script in this style:**

1. Linear extrapolation `2*G_old(t) - G_old(t-1)` applied naively to every
   floating-point state-dict entry, including BatchNorm `running_var`. A
   linear combination of two variance tensors can go negative, producing
   NaN inside `sqrt(var)` and silently poisoning Stage 1 (every
   `loss.item() < best_loss` comparison against a NaN loss is False, so
   `best_state` stays `None` and `load_state_dict(None)` crashes much later
   with a confusing error, far from the actual cause). Fixed by excluding
   any key containing `"running_"` from the extrapolation.
2. Accumulating all `N_ROUNDS` rounds' full state dicts (`ori_weight` +
   `new_state` + 10 `client_states`, ~20MB each) **on GPU**, in a Python
   list kept alive for the whole run — at `N_ROUNDS=30` that's
   30 x 12 x 20MB ~= 7GB against a 6GB card, and the failure mode isn't a
   clean OOM, it's severe allocator thrashing: the process kept running,
   GPU sat at 100% util, and 131 minutes produced zero output before this
   was caught and killed. The same bug reappeared in the analysis script
   via `torch.load(..., map_location=device)`, which eagerly moves
   everything in the file onto GPU on load regardless of how it was saved.
   Fixed by storing/loading on CPU throughout and letting
   `run_detection`/`stage3_classify_clients` move tensors to GPU transiently
   via `load_state_dict`'s automatic device conversion, plus
   `torch.cuda.empty_cache()` after each round. Both scripts also now use
   `flush=True`/`python -u` since Python fully buffers stdout when
   redirected to a file, which is why the stuck run produced no visible
   output at all until it would have exited.

## Correction and strengthening: the "T isn't round-specific" claim, properly tested (2026-10-10)

The original degenerate-pair result (class 0, then replicated at class 7)
was reported as T landing on the same prediction "to three decimal places"
regardless of what was fed as (d1, d2) — including a literal random-init
model. That framing was **an overstatement, flagged and corrected here, not
quietly adjusted**: every comparison script up to this point reseeded the
RNG to the *same fixed value* immediately before each `run_detection` call,
specifically to make comparisons "fair." That methodology cannot distinguish
"d1/d2 genuinely don't matter" from "every compared run was handed the same
random numbers" — a same-seed comparison is not evidence against a seed
dominance explanation, since it never varies the one thing (randomness)
that would reveal it. This should have been checked before the claim was
written down, not after it was challenged.

**Four checks, run before trusting the finding further:**

1. **Caching.** None in `run_detection` — verified by code inspection (fresh
   `Generator` instances every call, no file I/O, no module-level state)
   and empirically: T's own pixel statistics move measurably across seeds
   (below), which a cached/frozen T could not do.
2. **Does Stage 1/2 shortcut to the known target class?** No — Stage 1
   genuinely loops `for c in range(num_classes)`, building real images for
   all 10 classes regardless of `cfg.target_class`. The target is only used
   afterward to select which class's images become `target_imgs` vs.
   `other_imgs` for Stage 2's ⊕/⊖ construction — identical in spirit to the
   reference's own hardcoded `ori_imgs[0]`. Inherent to the algorithm (both
   implementations), not a shortcut introduced here; already covered by
   discrepancy #4.
3. **Generator seed dominance.** Real, and the right thing to have checked.
   Rerunning the SAME (d1, d2) pair with seeds 123/456/789 instead of one
   fixed seed shows T's raw pixel statistics genuinely move
   (e.g. one pair: std 0.066→0.071 across seeds; not frozen, not cached).
4. **Are d1/d2 actually used?** Confirmed: a random-init model and the
   actual post-30-epoch trained checkpoint have very different parameter
   fingerprints (sums 5387.78 vs. 3237.88), and T's pixel std responds to
   which is used (~0.07 for random-init vs. ~0.09-0.10 for trained) — d1/d2
   are genuinely different tensors, genuinely loaded, genuinely influencing
   T's raw content.

**So d1/d2 measurably affect T's pixel statistics. The open question was
whether that measurable effect propagates into a different Stage 3
classification OUTCOME — and whether a single confidence scalar (prior
framing: 0.708 vs. 0.704, one attacker, one round) was strong enough
evidence either way.** It wasn't, on its own — a ±0.004 noise floor on one
number is thin ground. Re-ran as a full 3×3 grid: three discriminator pairs
(A: random-init degenerate, B: clean-trained-for-10-rounds-never-poisoned
degenerate, C: the TRUE ORACLE, non-degenerate `(G_old(1), G_temp(1))` pair
from the real round) × three GAN-training seeds (123, 456, 789) = 9 runs,
reporting the FULL predicted-class vector, confidence vector, and flagged
set across all 10 sampled clients each time — not just the attacker's
confidence.

**Result: the predicted-class vector is bit-for-bit identical across all 9
runs:**

```
[9, 9, 0, 1, 7, 9, 9, 0, 3, 8]
```

(attacker = client 47, index 4, correctly at class 7 every time). The
flagged set is `{47}` in every single one of the 9 runs — no flips, no
near-misses crossing ρ in either direction, despite the method having
real reachable margins elsewhere (0.467, 0.472 near-misses seen in earlier
sections). Confidence values move within a ~0.002-0.01 band across both
seeds and pairs (e.g. attacker confidence: 0.703-0.708 across all 9 runs;
full per-client confidence vectors available in
`scripts/degenerate_vs_oracle_full.out`), the same order of magnitude as
the seed-noise floor characterized in the single-scalar check above — and
critically, **the true oracle pair (real G_temp(1), not degenerate) lands
in the exact same band as both degenerate pairs.** This is a qualitatively
different and much harder-to-dismiss result than one confidence value
landing close: the detector's entire output — every client's predicted
class, every client's flagged/not decision — is stable across
discriminator identity and across randomness, at the same time, in the one
round tested.

**Corrected, final framing: d1/d2 content measurably affects T's pixel
statistics (fingerprints differ, pixel std differs ~0.07 vs ~0.09-0.10) but
does not move the detection outcome beyond the method's own seed-to-seed
noise floor (~0.002-0.01 on confidence; zero flips in the predicted-class
vector or the flagged set across 9 runs spanning 3 seeds and 3 model
pairs, including the true oracle).** This is weaker than "T is literally
independent of model content" and stronger than "one number looked
similar twice" — it's the claim actually supported by evidence, and it's
the one worth building on.

### Does this survive at α=0.1, where the pipeline is supposed to get fragile?

Everything above (class 0, class 7, the 3×3 grid) ran at α=1.0, the
repo's own shipped default — a relatively favorable, close-to-IID setting.
The paper's own ablation shows detection becoming fragile at low α. Reran
the exact same 3-pair × 3-seed grid (`scripts/degenerate_vs_oracle_full.py
--config configs/cifar_alpha0.1_class7.yaml`, dirichlet_alpha=0.1,
otherwise identical to the class-7 config) to see whether the
d1/d2-independence finding is itself a property of the easy regime, or
survives where the mechanism is under real stress.

**Two separate results, and only one of them degrades:**

1. **Detection quality genuinely gets worse at α=0.1** — a new false
   positive appears that wasn't present at α=1.0. The predicted-class
   vector `[5, 9, 9, 3, 7, 7, 4, 9, 1, 0]` and flagged set `{47, 13}` are,
   again, bit-for-bit identical across all 9 runs — but this time the
   flagged set includes client 13, an innocent client whose locally-trained
   model predicts class 7 with confidence ~0.82-0.83 (well above ρ=0.5),
   alongside the real attacker (client 47, confidence ~0.875, still
   correctly caught — no false negative). Confidence values run much
   hotter across the board at this setting (0.33-1.00, vs. 0.2-0.7 at
   α=1.0), consistent with extreme non-IID local models producing more
   extreme per-client responses under train-mode BatchNorm
   renormalization. This matches the "pipeline becomes fragile at low α"
   pattern the paper's own ablation shows — found independently here, via
   an unrelated test.
2. **The d1/d2-independence finding does NOT degrade — at all.** The
   output (predicted-class vector, confidence band, flagged set) is just
   as perfectly stable across all 3 discriminator pairs and all 3 seeds at
   α=0.1 as it was at α=1.0. Robustness of "T doesn't depend on which
   discriminators you feed it" isn't contingent on being in an easy
   regime; it holds exactly where the mechanism is supposedly under the
   most stress.

**Net: the headline finding is stronger for having been tested here, not
weaker.** It would have been easy for "T isn't round-specific" to turn out
to be an artifact of the easy α=1.0 setting specifically — it isn't. At
the same time, this test surfaced a real, separate fragility (the α=0.1
false positive) that has nothing to do with d1/d2 and everything to do
with detection quality under extreme non-IID partitioning — worth keeping
as its own finding, not folded into or used to qualify the robustness
result.

## Confirming the mechanism findings on the actual reference code, not just the stand-in (2026-10-10)

Everything in the degenerate-pair / seed-dominance / Stage-2 sections above
was run on this project's own reimplementation (`filterfl/`), validated
against the reference's aggregate TP/FP/FN numbers. That validation is
necessary but not sufficient for a mechanism claim: the reimplementation
matched the reference on topology, hyperparameters, and attack setup
*before* the `.eval()` fix too, and still produced 4/27 against the
reference's 20/20 — aggregate agreement at one point in time doesn't rule
out a second, undetected divergence producing a false "doesn't matter"
result. This is especially live for Stage 2, since the reimplementation's
default is the paper's 3-term loss, not the reference's undocumented
5-term one — removing a weaker Stage 2 and seeing no change is weaker
evidence than removing the reference's actual Stage 2 and seeing the same.

So: patched `reference/filterfl/` directly. Additive, flag-gated, defaults
preserve existing behaviour (the original shipped-config runs stay valid):
- `gan.py`: `gan_defense(..., skip_stage2=False, degenerate_pair=False)`.
  `degenerate_pair=True` loads d1 into both discriminators, ignoring d2.
  `skip_stage2=True` returns `[ori_imgs[0]]` (Stage 1's raw per-class image
  for the hardcoded target class 0) immediately after Stage 1, without
  computing Stage 2 at all.
- `algorithm.py`: reads `handle.params.get('skip_stage2'/'degenerate_pair', False)`,
  passes through to `gan_defense`.
- `main.py`: `--skip-stage2` / `--degenerate-pair` CLI flags.
- New `cifar_alpha0.5_params.yaml` / `cifar_alpha0.1_params.yaml` (copies of
  the shipped `cifar_params.yaml` with only `dirichlet_alpha` and `epochs`
  changed, for a 30-round test instead of the full 2000).

### Degenerate pair, on the reference code directly

One real FedAvg round (`degenerate_pair_test.py`, α=0.5, seed=1, this
file's own `handle.py`/`train.py`/`algorithm.py` pipeline, unmodified): real
attacker = client 47. Normal pair (d1=ori_weight, d2=w_glob) vs. degenerate
pair (d1=d2=ori_weight, zero information about the round's actual update):

| | predicted-class vector | flagged |
|---|---|---|
| NORMAL pair | `[3, 3, 9, 1, 0, 7, 0, 9, 3, 9]` | `{47, 73}` |
| DEGENERATE pair | `[3, 3, 9, 1, 0, 7, 0, 9, 3, 9]` | `{47, 73}` |

Identical predicted-class vector, confidence differences ~0.001-0.002 (the
same noise-floor magnitude as the reimplementation's version of this test),
same flagged set — including the same false positive (client 73, at this
α=0.5 setting, consistent with the α-sensitivity found independently).
**This replicates directly on the reference's own, unmodified-except-for-
the-flag code.** The validated-stand-in caveat for this specific finding is
resolved: it isn't an artifact of the reimplementation.

## Correction: Table IV is a GTSRB-only experiment, not cross-dataset (2026-10-10)

Read the actual paper PDF (`filterfl_noncameraready.pdf`) for the first time
in this project, rather than working from the task brief's secondhand
framing. This corrects a real misreading that shaped the entire Stage-2
ablation work below.

**Table IV's caption, verbatim: "EFFECTIVENESS OF THE ABLATION OF OUR
APPROACH AGAINST MULTIPLE ATTACKS ON GTSRB, WITH ONE IID AND THREE
NON-IID SCENARIOS (α = 1, 0.5, 0.1)."** The table's 8 numeric columns are
**4 distribution scenarios (IID, α=1, α=0.5, α=0.1) × 2 metrics (MA%,
ASR%), all on GTSRB** — not 4 datasets at one α, and not CIFAR-10 at two
α's. The 41.61% and 89.69% ASR figures (α=0.5 and α=0.1 "Without Stage 2")
are GTSRB numbers. **The paper's Stage-2 ablation was never run on
CIFAR-10 at all.** Full row, for the record:

| Scenario | Full Approach MA / ASR | Without Stage 2 MA / ASR |
|---|---|---|
| IID | 96.36 / 0.11 | 85.51 / 0.13 |
| α=1 | 95.57 / 0.14 | 71.2 / 1.32 |
| α=0.5 | 95.45 / 0.14 | 94.68 / 41.61 |
| α=0.1 | 96.04 / 0.48 | 93.31 / 89.69 |

This means the GTSRB-vs-CIFAR-10 caveat already attached to the Stage-2
ablation sections below wasn't a hedge about a *possible* dataset-specific
effect — it's the entire basis of the comparison. Every Stage-2-skip number
in this project (reimplementation and reference-code versions alike) is on
a dataset the paper's own ablation never tested.

**Checked whether GTSRB can actually be run on the released code, to close
this properly rather than leave it as a permanent caveat: it cannot.**
`handle.py`'s GTSRB branch (`load_data()`) does `from models.gtsrb import
GTSRB` — `models/gtsrb.py` does not exist anywhere in the repository (only
`models/ResNet8.py` and `models/__init__.py` are present). This is the same
class of bug as discrepancy #9's missing `CNNMnist` for MNIST, now found a
second time for GTSRB, and a third time for `ResNet18_cifar10` (used by
both the CIFAR-100 and TinyImageNet branches in `create_model()`) — also
referenced, never defined. **Of the four datasets in the paper's main
results tables, only CIFAR-10 actually runs on the released code.** MNIST,
CIFAR-100, GTSRB are all broken via a missing class, each a different
missing class, same underlying pattern: the released artifact was
apparently only ever exercised end-to-end on CIFAR-10, and the other three
datasets' code paths were left in a state that cannot execute, structurally
not just as a configuration gap.

**Revised framing for everything below:** the Stage-2 non-replication
result (Stage 2 doesn't matter at α=0.5 on CIFAR-10) is not in tension with
the paper's Table IV (Stage 2 matters a lot on GTSRB at α=0.5/0.1) in the
way "same claim, different result" would be — they are two different
datasets, and the released code cannot produce the GTSRB side of that
comparison to check directly. This doesn't resolve whether Stage 2 would
matter on GTSRB if it could be run (it plausibly would, per the paper's own
numbers) — it resolves that **this project's non-replication finding is
specifically a CIFAR-10 finding**, and should be stated that way rather
than as a general refutation of the paper's ablation claim. The C-dependent
Stage-2 loss-balance observation (referenced in the original caveat) is
still a live, unresolved reason a 10-class vs. 43-class comparison could
genuinely differ, on top of this.

**On the side question that prompted re-reading the paper: does it specify
CGAN network architecture?** No — checked both `filterfl_noncameraready.pdf`
and `siftfl.pdf` directly. FilterFL's text discusses CGANs only
conceptually (what Stage 1/2 are for, the chosen training-round count,
Table XIII's round-count sweep) with no layer-level description anywhere.
SiftFL's Algorithm 1 pseudocode treats the generators as opaque `g1`/`g2`
with `RANDOM()`/`OPTIMIZE()` calls — also no architecture detail. The
DCGAN-style generator (`Linear → reshape → BatchNorm → Upsample → Conv ×2 →
Tanh`) used identically by both `reference/filterfl/gan.py`'s
`Generator_CIFAR` and this project's `GeneratorCIFAR` was always an
implementation choice neither paper specifies — consistent with, and now
confirmed rather than inferred from, the existing hyperparameters table
above (which only ever sourced training-round counts and learning rates,
never architecture).

**One primary-source correction this also surfaces**: the paper's own text
(Section I, "Choice of Training Rounds") states *"we chose 10 as the number
of training rounds for our approach to train the CGANs"* — a first-party
number, not the secondhand SiftFL-reported "12/15" figure discrepancy #8
relies on. The passage doesn't distinguish Stage 1 vs. Stage 2 round counts
explicitly, so it's consistent with the shipped `n_epochs1=10` but doesn't
by itself confirm or deny whether `n_epochs2=5` matches the paper's own
intent — Table XIII (the training-rounds sweep, on CIFAR-100 at α=0.1)
reports results at rounds {2, 5, 10, 15, 20} without separating stages
either. Discrepancy #8 should be read as partially, not fully, resolved by
this: the first-party "10" is now confirmed for *a* CGAN round count, but
which stage(s) it applies to remains inferred, not stated.

## Stage-2 ablation: new CIFAR-10 evidence, not a replication of the paper's Table IV (2026-10-10)

**Corrected framing (this section originally described this as testing
against the paper's Table IV ablation; that was a misreading, corrected
above — Table IV is GTSRB-only, and the released code cannot run GTSRB at
all, so the paper's Stage-2 ablation cannot be checked with this artifact
on any dataset it was actually performed on).** What follows is a CIFAR-10
Stage-2 skip experiment in its own right, not a test of whether Table IV
reproduces. The degenerate-pair findings above show the models fed into
Stage 1/2 don't affect the detection decision; this ablation asks the
related but separate question of whether Stage 2's search/construction
step itself matters, on the one dataset (CIFAR-10) this code can actually
run.

**Implemented:** `--skip-stage2` in `filterfl/run.py` / `skip_stage2` param
in `defense.run_detection` — Stage 2's trigger-filtering/⊕⊖ construction is
not just ignored but not computed at all; Stage 3 is instead fed
`per_class_images[target_class]`, Stage 1's raw per-class image set for the
target class, directly.

**Setup:** CIFAR-10, target_class=7, stage2_loss="paper",
stage3_probe_mode="train", 30 rounds each:

| | Epochs 1-10 | 11-20 | 21-30 | **Total** |
|---|---|---|---|---|
| FULL, α=0.5 | MA=23.18 ASR=0.20 TP=9 FP=3 FN=1 | MA=30.79 ASR=1.31 TP=10 FP=6 FN=0 | MA=34.14 ASR=2.22 TP=15 FP=5 FN=0 | **TP=34/35 FP=14 FN=1** |
| SKIP-STAGE-2, α=0.5 | MA=23.41 ASR=0.32 TP=9 FP=2 FN=1 | MA=31.78 ASR=0.90 TP=10 FP=5 FN=0 | MA=34.20 ASR=2.49 TP=15 FP=4 FN=0 | **TP=34/35 FP=11 FN=1** |
| SKIP-STAGE-2, α=0.1 | MA=15.98 ASR=0.00 TP=10 FP=9 FN=0 | MA=19.00 ASR=0.00 TP=8 FP=2 FN=2 | MA=22.98 ASR=0.11 TP=12 FP=7 FN=3 | **TP=30/35 FP=18 FN=5** |

**On CIFAR-10 at α=0.5: removing Stage 2 changes almost nothing.** TP
and FN are identical between FULL and SKIP-STAGE-2 (34/35 attacks caught,
1 missed); FP is actually slightly *lower* without Stage 2 (11 vs. 14);
ASR stays in the same low band (0.2-2.5%) either way. This is a new
CIFAR-10 finding, not a non-replication of the paper's Table IV — Table IV
never tested CIFAR-10 (it's GTSRB-only), and the released code can't run
GTSRB to check the dataset the paper's ablation actually covers.

**This is consistent with, and adds a layer to, the degenerate-pair
findings — not a resolution of a conflict with Table IV, since Table IV
was never in play on this dataset.** The
degenerate-pair work already showed d1/d2's specific content barely
affects the detection decision, within the method's seed-noise floor. This
ablation now shows Stage 2's entire search/construction, layered *on top
of* that content, also barely affects the decision. Both point the same
direction: in this implementation, Stage 3's train-mode BatchNorm
renormalization (see the earlier probe-mode section) appears to be doing
nearly all of the real work, largely independent of what synthetic image
it's classifying — a carefully-optimized Stage-2 trigger or a raw Stage-1
per-class image give the same outcome, just as a random-init or
clean-trained discriminator pair did.

α=0.1 does show real degradation (TP 30/35, FP 18) even without Stage 2 —
but this matches the α-sensitivity already found independently in the
class-7 degenerate-pair test, not something specific to removing Stage 2
here. No FULL-approach run at α=0.1 was done (not asked for, and the
SKIP-vs-SKIP α=0.5-vs-0.1 comparison already shows the α-sensitivity shows
up with Stage 2 absent, which is the point of that comparison); if a
FULL@0.1 number is wanted later for a cleaner 2x2, it's cheap to add.

**Two things this result cannot establish, stated plainly rather than as
hedged caveats:**

1. **It says nothing about whether Table IV holds, because Table IV's
   ablation cannot be checked at all from this artifact.** Grepped
   `reference/filterfl/*.py` for `skip`/`ablat`/`Table`/`without`: nothing
   — no ablation code of any kind exists in the released repository, and
   the dataset Table IV reports (GTSRB) cannot be executed on this code
   (see the Table IV correction above — missing `models/gtsrb.py`). There
   is no way to check from the artifact whether their ablation even ran
   under the same probe mode (eval vs. train) as their main Table 2/3
   results, let alone whether its conclusion holds.
2. **It is a CIFAR-10-only result.** The C-dependent Stage-2 loss-balance
   finding means a 43-class dataset could genuinely behave differently —
   this isn't ruled out, just untestable with the released code.

### Stage-2 skip, on the reference code directly — the prioritized test

Same three runs as the reimplementation's table (FULL α=0.5, SKIP-STAGE-2
α=0.5, SKIP-STAGE-2 α=0.1), 30 rounds each, but on
`reference/filterfl/main.py --params cifar_alphaX_params.yaml --defense ours
[--skip-stage2]` directly — their actual `algorithm.py`, their actual
undocumented 5-term `gan.py` Stage 2 loss (when not skipped), their
hardcoded target class 0, only the additive `skip_stage2` flag changed:

| | MA (1-10/11-20/21-30) | ASR (1-10/11-20/21-30) | Attacks | TP | FP | FN |
|---|---|---|---|---|---|---|
| FULL, α=0.5 | 23.81 / 32.85 / 37.26 | 1.09 / 3.00 / 3.89 | 35 | **34** | 27 | **1** |
| SKIP-STAGE-2, α=0.5 | 23.91 / 33.83 / 37.24 | 1.20 / 3.73 / 3.67 | 35 | **34** | 28 | **1** |
| SKIP-STAGE-2, α=0.1 | 15.46 / 18.65 / 23.69 | 0.13 / 0.00 / 3.02 | 35 | 26 | 25 | 9 |

**Confirmed directly on their code, not just the reimplementation's
weaker-loss stand-in — but still a CIFAR-10 finding, not a check of Table
IV.** At α=0.5, FULL and SKIP-STAGE-2 are statistically indistinguishable:
identical TP (34/35) and FN (1), FP within noise (27 vs. 28), ASR in the
same low band throughout. This closes the gap the reimplementation's
version of this test left open — that result used the paper's 3-term loss
as Stage 2, so "removing a weaker Stage 2 changes nothing" was a real
possible objection. This run removes *their* actual undocumented 5-term
Stage 2 and gets the same answer: it still changes nothing at α=0.5, now
on the dataset and detector their own release actually ships.

At α=0.1, detection quality (TP, FP) does degrade, consistent with the
α-sensitivity found independently elsewhere — but **ASR never exceeds
3.02%** in any block, a long way from the paper's own reported 89.69% ASR
figure. That figure, though, is GTSRB's — there is no corresponding
CIFAR-10 number in the paper to compare against, so this is new evidence
about CIFAR-10 at low α, not a shortfall against a published CIFAR-10
result.

**The sharper, accurate statement of what this whole ablation section
shows: the paper's Stage-2 ablation (Table IV) cannot be checked, because
the released code cannot run the dataset it was performed on.** GTSRB is
unrunnable (missing `models/gtsrb.py`), and so are MNIST and CIFAR-100
(each missing a different referenced-but-undefined model class) and
TinyImageNet (undefined `TinyImageNet` class) — of the five datasets in
the paper's main tables, only CIFAR-10 executes on this artifact, and
CIFAR-10 is not one of them. The CIFAR-10 Stage-2-skip results above (both
the reimplementation's and the reference code's) stand as new evidence
about this implementation's CIFAR-10 behaviour — Stage 2 appears to
contribute little to detection once Stage 3's train-mode BatchNorm effect
is accounted for — not as a replication attempt, successful or failed, of
the paper's own ablation claim.

**Patches made to `reference/filterfl/` for this (additive, flag-gated,
defaults preserve existing behaviour):** `gan.py`'s `gan_defense(...,
skip_stage2=False, degenerate_pair=False)`, `algorithm.py` reading
`handle.params.get('skip_stage2'/'degenerate_pair', False)` through to it,
`main.py`'s `--skip-stage2`/`--degenerate-pair` CLI flags, and
`cifar_alpha0.5_params.yaml`/`cifar_alpha0.1_params.yaml` (copies of the
shipped `cifar_params.yaml` with only `dirichlet_alpha` and `epochs`
changed). The original shipped-config run path is untouched.

### The BN-running-stats attack: dead on arrival

The natural attack from Ablation 2: BN running stats are part of the
submitted model and an attacker sets them freely without touching the conv
weights carrying the backdoor — submit running stats that make the model
look benign under batch-renormalization while the backdoor stays exactly
where it is. Checked the load-bearing assumption first, since it's a
two-minute check that can kill the idea before building anything:

**Does Stage 3's train-mode forward pass even read `running_mean`/
`running_var`, or only write them?** Only write. `nn.BatchNorm2d.forward()`
in training mode normalizes using the *current batch's own* mean/var,
computed fresh from the input; the running buffers are updated afterward via
the momentum-blended moving average, purely for later `eval()`-mode use —
they are not consulted to produce this forward's output at all. Verified
generically (isolated `BatchNorm2d`, perturb running stats, re-run same
input in train mode: `0.0` max-abs output difference, vs. a large difference
once switched to `eval()`), and verified directly on the real attacker model
from the fixed round:

```
baseline (train mode):                        confidence = 0.683197021484375
after running_mean+=1000, running_var*=0.0001: confidence = 0.683197021484375  (bit-identical)
```

**Dead on arrival, cleanly.** Whatever an attacker writes into
`running_mean`/`running_var` has zero effect on Stage 3's output, because
train-mode BatchNorm never reads those buffers to begin with — only the
conv/fc weights and the BN affine (`weight`/`bias`) parameters matter, and
those are exactly the parameters carrying the backdoor the attacker can't
give up without breaking it. Did not run the full 3-variant table (copy a
benign client's running stats / copy G_old's / copy the attacker's own clean
data's) — all three write to the same buffers this test already proved
inert in train mode, so they'd all trivially reproduce "no effect."

**This is the reproducibility-paper outcome, not the security-paper one, via
this specific route.** The train/eval bug makes the shipped detector work
far better than the paper's description would predict, and makes it depend
on an undocumented architectural property (BatchNorm) — but it does not, by
itself, hand an attacker a free way to defeat Stage 3 by manipulating state
it ignores. A BN-stats attack is closed. (Open question, not pursued here:
whether an attacker could instead shape the conv/affine weights themselves
— the parameters that DO matter under batch-stat renormalization — to
produce benign-looking batch statistics for whatever 32 synthetic images
Stage 1/2 happens to generate, while preserving the backdoor. That's a much
harder, less mechanical attack than "overwrite two buffers," and wasn't
asked for here.)

## Budget sweep: the normal/degenerate attractor gap is budget-independent (2026-10-10)

The seed-dominance / degenerate-pair line of findings showed detector output
is stable across GAN-training seeds at the shipped fixed 10/5 (Stage1/Stage2)
budget. That's consistent with two different explanations that a seed sweep
alone can't separate: (a) the loss landscape has one dominant basin any
input converges to (genuine attractor), or (b) 5 Stage-2 rounds is simply
too few for the loss to differentiate ANY input, including d1/d2 content
itself, and the apparent basin is an artifact of truncated optimisation.

**Setup:** `scripts/budget_sweep.py`, on the reference code directly
(`reference/filterfl/gan.py`'s `gan_defense`, verbatim, with additive
`n_epochs1`/`n_epochs2` override parameters added — default `None` preserves
the shipped 10/5), reusing the exact fixed (G_old, G_agg, 10 client states,
real attacker = client 47) round already saved in `scripts/crossfeed_round.pt`.
At each budget cell, ran Stage 1/2 twice with the same fixed seed (123) —
once for the **normal** pair (d1=G_old, d2=G_agg, the real round-1 aggregate)
and once for the **degenerate** pair (d1=d2=G_old) — so z-sampling sequences
are aligned and any difference in the resulting trigger images is
attributable to d1/d2 content alone, not seed variation. Then classified all
10 clients with Stage 3 in train mode (reference's actual, un-`.eval()`'d
behaviour) under each trigger set.

**Stage 2 sweep** (n_epochs2 ∈ {5, 15, 50, 200} at n_epochs1=10, plus the
SiftFL-reported 12/15 pairing as its own cell):

| n_epochs2 | pixel MSE(normal, degenerate) | attacker conf (normal) | attacker conf (degenerate) | abs gap |
|---|---|---|---|---|
| 5 (shipped) | 0.001131 | 0.6716 | 0.6723 | 0.0007 |
| 15 | 0.001123 | 0.6717 | 0.6710 | 0.0007 |
| 50 | 0.001193 | 0.6712 | 0.6728 | 0.0016 |
| 200 (40x shipped) | 0.001241 | 0.6707 | 0.6722 | 0.0015 |
| 12/15 (SiftFL pairing) | 0.001223 | 0.6732 | 0.6729 | 0.0004 |

**Stage 1 sweep** (n_epochs1 ∈ {5, 10, 20, 40} at n_epochs2=5):

| n_epochs1 | pixel MSE(normal, degenerate) | attacker conf (normal) | attacker conf (degenerate) | abs gap |
|---|---|---|---|---|
| 5 | 0.001191 | 0.6737 | 0.6738 | 0.0002 |
| 10 (shipped) | 0.001341 | 0.6705 | 0.6703 | 0.0001 |
| 20 | 0.000954 | 0.6723 | 0.6735 | 0.0013 |
| 40 | 0.001278 | 0.6709 | 0.6733 | 0.0024 |

**Result: flat, not widening — explanation (a), genuine basin, holds.**
Across a 40x range of Stage-2 budget (5→200) and a 8x range of Stage-1
budget (5→40), pixel MSE between the normal and degenerate trigger output
stays in a narrow 0.0009–0.0013 band with no monotonic trend, the attacker's
confidence under both pairs stays within 0.0001–0.0024 of each other (noise
floor, not a growing gap), and the full 10-client predicted-class vector is
byte-for-byte identical across every single cell in both sweeps — same
`[9, 9, 0, 1, 0, 9, 9, 0, 3, 8]`, same flagged set `{47}`, at every budget
tested. If (b) (truncated-optimisation artifact) were true, the gap should
have widened as the budget rose toward 200 — a full round-count sweep
(Table XIII territory) in that range would have had time to show it. It
did not, at any point in the range. **This closes the open question the
seed sweep alone left unresolved: the attractor is a property of the loss
landscape at this d1/d2 pair, not an artifact of the shipped code's
specific 5-round truncation.** The existing framing throughout this file
(Stage 1/2's trigger generation converges to essentially the same output
regardless of d1/d2 content, and Stage 3's train-mode BatchNorm effect does
nearly all of the real detection work) holds at any tested budget, not just
the shipped one.

**Patch:** `reference/filterfl/gan.py`'s `gan_defense(..., n_epochs1=None,
n_epochs2=None)` — additive, `None` preserves the shipped 10/5 exactly;
threaded through `algorithm.py`'s call site via
`handle.params.get('n_epochs1'/'n_epochs2', None)` for consistency with the
existing `skip_stage2`/`degenerate_pair` flags, though `scripts/budget_sweep.py`
calls `gan_defense` directly and doesn't exercise that path.

## Is the attractor basin the same across training rounds? (2026-10-10)

Also previously flagged as unresolved: whether the degenerate-pair
predicted-class vector is a single fixed basin throughout a run, or
round-dependent (but still seed-stable within whichever round it's checked
at).

**Setup:** `scripts/attractor_rounds.py` — 13 rounds of plain FedAvg (usual
pixel-pattern attack active) on `configs/cifar_alpha1_side.yaml`, saving
G_old at rounds {1, 4, 7, 10, 13}. At each of those 5 rounds, ran the
degenerate-pair detector (d1=d2=G_old(round)) with 3 different GAN-training
seeds (123, 456, 789), classifying that round's actual sampled clients in
Stage 3 (train mode).

**Result: round-dependent, not a single fixed basin — but perfectly
seed-stable within whichever round it's checked at.**

| round | modal predicted class | seed-stable (3/3 seeds identical vector)? |
|---|---|---|
| 1 | 9 | True |
| 4 | 3 | True |
| 7 | 0 | True |
| 10 | 7 | True |
| 13 | 2 | True |

All 5 distinct: {9, 3, 0, 7, 2} — no two of the five checked rounds share a
modal class. Within each round, though, the full 10-client predicted-class
vector is byte-for-byte identical across all 3 GAN-training seeds (same
exact integers, not just the same mode) — the same seed-stability found
throughout this project's degenerate-pair work, now confirmed at 5
independent points across training rather than one. **So the basin moves as
G_old itself evolves over training, but at any fixed point in training it's
a reliable, deterministic landing spot independent of GAN-training
randomness.** This refines, not contradicts, the budget-sweep finding above:
"genuine basin, not a truncation artifact" and "basin is round-dependent"
are compatible — it's a basin whose location is a property of G_old's
current weights, found reliably regardless of budget or seed, and that
location drifts as those weights drift across rounds.

**A second finding, incidental to the question asked but worth recording:**
despite the modal (most-common) class moving every round, the real
attacker's own predicted class was `0` (the true backdoor target) and
confidence exceeded ρ=0.5 at **every single one of the 5 checked rounds**
(flagged in all 5), even in rounds where class 0 was far from the batch's
modal class (e.g. round 1's modal class is 9, round 4's is 3, round 13's is
2 — the attacker still individually lands on 0 and gets flagged in all
three). This is consistent with, not a reversal of, the degenerate-pair
line of findings: d1/d2 content barely affecting the *overall* trigger set
doesn't mean every client lands on the same predicted class as each other —
it means a given client's own weights reliably produce the same predicted
class regardless of which (degenerate) trigger set probes it, and for this
specific real attacker across these 5 rounds that class happened to keep
being the true target.

## Does the attractor track d1, d2, or neither — and does d1/d2 role matter? (2026-10-10)

Follow-up to the round-dependence result, to find out *mechanistically*
whether the basin tracks the specific model fed in (and whether it matters
which one is d1 vs d2), rather than just observing that it moves across
training. Two matched, controlled comparisons on a single fixed consecutive
model pair, so there's no confound from different client samplings or
different attackers across rounds (unlike the round-dependence test, which
necessarily varies those too):

**Setup:** `scripts/d1_d2_role_test.py`, reusing `scripts/crossfeed_round.pt`
directly — its `ori_weight` is G_old(t) and its `new_state` is G_agg(t),
which (per the existing "Design assumptions" section's trust-chain
description) IS the candidate G_old(t+1) if accepted. Verified these are
genuinely different models first (`sum|G_old(t) − G_old(t+1)| = 2465.26`
over all floating-point parameters — not a no-op round). Same 10 client
models (client 47 = real attacker) classified under all four cells, shipped
10/5 budget, fixed seed:

| cell | d1 | d2 | predicted-class vector |
|---|---|---|---|
| A (degenerate @ t) | G_old(t) | G_old(t) | `[9,9,0,1,0,9,9,0,3,8]` |
| B (degenerate @ t+1) | G_old(t+1) | G_old(t+1) | `[9,9,0,1,0,9,9,0,3,8]` |
| C (normal order) | G_old(t) | G_old(t+1) | `[9,9,0,1,0,9,9,0,3,8]` |
| D (swapped order) | G_old(t+1) | G_old(t) | `[9,9,0,1,0,9,9,0,3,8]` |

**All four predicted-class vectors are identical, down to the integer.**
A vs B: pixel MSE(T_A, T_B) = 0.001387 — in the same band as every other
pixel-MSE number in this project's degenerate-pair work, despite A and B
being fed genuinely different models one FedAvg round apart. C vs D: pixel
MSE(T_C, T_D) = 0.002763 — the trigger images themselves do differ visibly
more under a role swap than under either degenerate pair, but the
downstream classification decision is completely unaffected.

**Reading this together with the round-dependence result above, not instead
of it:** that result (modal class different at 5 of 5 widely-spaced rounds:
9, 3, 0, 7, 2) is NOT contradicted by this one (identical output across a
single round's worth of drift) — they operate at different distance scales.
**The basin is locally sticky: robust to the amount of weight change one
FedAvg round produces, but it does eventually move once enough rounds'
worth of drift accumulates.** Neither which model is fed (A vs B) nor which
structural role it plays, naturalness-reference vs target-bias-reference (C
vs D), changes the classification outcome at this one-round distance —
consistent with, and now directly demonstrating a mechanism for, the
existing "Stage 3's train-mode BatchNorm renormalization does nearly all of
the real work, trigger content barely matters" finding: if trigger pixel
content already barely matters for a FIXED model (shown by the degenerate
vs. oracle pixel-MSE numbers throughout this file), it's unsurprising that
swapping which near-identical model generated that content also doesn't
matter. The round-dependence result shows this insensitivity has a limit —
it does not mean the detector is blind to G_old forever, only that it takes
more than one round's change to move it.

**Patch:** none — this reuses the existing `degenerate_pair`-adjacent
pattern (`gan_defense(d1, d2, ...)` called directly with explicit state
dicts) already supported by the additive patches from the budget-sweep
section above; no new flags needed.

### Rerun at the distance that actually showed movement (rounds 1 vs 13) — this relocates the round-dependence finding, it doesn't confirm it (2026-10-10)

The adjacent-round test above (one FedAvg round apart) found no effect from
either model identity or role, but that distance was too short to move the
basin at all — round-dependence was only established at the widely-spaced
1/4/7/10/13 scale (modal classes 9, 3, 0, 7, 2). The natural next step is
the SAME A/B/C/D design at that distance, where the degenerate cases should
actually differ if the earlier result reflects a real property of the
trigger-generating model:

  A) degenerate @ round 1:  d1=d2=G_old(1)
  B) degenerate @ round 13: d1=d2=G_old(13)
  C) normal order:          d1=G_old(1),  d2=G_old(13)
  D) swapped order:         d1=G_old(13), d2=G_old(1)

`scripts/far_role_test.py` regenerates the identical deterministic 13-round
FedAvg sequence (same seed/config as the round-dependence test above, since
that test didn't persist its checkpoints) to get real G_old(1)/G_old(13)
tensors (confirmed genuinely different: `sum|G_old(1) − G_old(13)| =
4636.02`, nearly double the adjacent-round test's delta), and classifies
**round 1's own client set (fixed, including real attacker 47) under all
four conditions** — the one methodological choice that turns out to matter.

**Result: A and B are identical.** `[9,9,0,1,0,9,9,0,3,8]` for both —
the trigger generated from G_old(13) classifies round 1's clients
*exactly* the same as the trigger generated from G_old(1), despite the
large weight delta between those two global models. Consequently C and D
are trivially identical to A and B too (once A==B, "C tracks A" and "C
tracks B" are the same statement) — **the role-swap question this test set
out to answer is undecidable from this result**, because its precondition
(A ≠ B) never held.

**This doesn't contradict the round-dependence finding — it relocates its
cause.** The round-dependence test varied two things together: which
model generated the trigger (G_old at that round) AND which client models
were being classified (that round's own freshly-sampled, freshly-trained
clients, trained starting from an ever-more-converged global model as
training progresses). This test holds the second thing fixed and only
varies the first, and finds the first has no effect even across the exact
distance (1 vs 13) where the original test saw the modal class move. The
straightforward reading: **the round-dependence seen earlier is driven by
the classified client models evolving round to round, not by the
trigger-generating model (d1/d2) changing what it produces.** This is
consistent with, and sharpens, everything else in this file about Stage 3's
train-mode BatchNorm doing nearly all the work on whatever client it's
given — it now appears the trigger set is close to irrelevant not just
pixel-for-pixel at a fixed round (the degenerate-vs-oracle finding) but
across 13 rounds of trigger-generating-model drift as well, and what
actually moves the needle is the client being probed, not the probe.

**Caveat, stated plainly:** this is evidence toward that reading, not a
full proof — it tests one fixed client set (round 1's) against two
trigger-generating models (round 1's and round 13's G_old), not the
reverse (round 13's client set against both trigger models), which would
be the cleaner isolation and is cheap to run as a follow-up if this
distinction needs to be nailed down further. The d1-vs-d2 asymmetry
question from the original request remains genuinely open — answering it
requires finding (or constructing) a pair of models where the degenerate
cases differ, then running the same C/D role-swap on *that* pair; this
particular pair didn't provide the needed precondition.

## MA/ASR ablation, revisited: flagged-rate check and a 30-round-horizon concern (2026-10-10)

Three concerns were raised about the Stage-2 MA/ASR ablation numbers reported
earlier (reference-code runs: `ref_stage2_full_a05.out`, `ref_stage2_skip_a05.out`,
`ref_stage2_skip_a01.out`, all 30 rounds).

**1. What fraction of sampled clients get flagged per round, in the reference
runs?** Parsed directly from the per-round logs (300 sampled-client-slots =
10 clients × 30 rounds, in each file):

| Run | Total flagged | Flagged rate |
|---|---|---|
| FULL (α=0.5) | 61/300 | 0.203 |
| SKIP-STAGE-2 (α=0.5) | 62/300 | 0.207 |
| SKIP-STAGE-2 (α=0.1) | 51/300 | 0.170 |

So ~2/10 per round, not 3-4/10 — the "caught by volume" concern doesn't hold
at face value. But FULL and SKIP-STAGE-2 are statistically flat (0.203 vs
0.207, a 1-client difference over 30 rounds) — Stage 2 is not visibly moving
the flagged rate at α=0.5 either direction.

**2. Does the FP direction support Stage 2's claimed job (reducing false
exclusions under heterogeneity)?** With ~35 real attackers sampled over 30
rounds and TP≈34/35 in both FULL and SKIP-STAGE-2 runs, nearly all flagged
clients beyond the true attackers are FPs: FP=27 (FULL) vs FP=28
(SKIP-STAGE-2) at α=0.5. The direction is the one the paper's story would
predict (FP goes up slightly without Stage 2), but the size (1 client over 30
rounds) is not distinguishable from noise, and it's the opposite direction
from what this project's own reimplementation showed (14→11) on the same
nominal comparison. No reliable FP-reduction effect from Stage 2 is
demonstrated by either codebase at this setting.

**3. Is 30 rounds long enough for the backdoor to establish at all, independent
of Stage 2?** This was flagged as the most likely explanation for ASR never
exceeding ~3-4% in any ablation condition (vs. the paper's 92.80% no-defence
CIFAR-10 ASR at α=0.5), and tested directly: two new reference-code runs, both
α=0.5, poison_rate=0.5 (the exact ablation config), both from the same seed=1
base config (`cifar_alpha0.5_params.yaml`):

- **No-defence control**, 100 rounds (`--defense None`, `cifar_alpha0.5_params_control100.yaml`).
- **SKIP-STAGE-2**, 200 rounds (`--skip-stage2`, `cifar_alpha0.5_params_long200.yaml`) — now complete (all 200 rounds).

Global poison-test accuracy (ASR) by round, both runs, same config apart from
defence:

| Round | SKIP-STAGE-2 (Stage1+3 active) | No-defence control |
|---|---|---|
| 5 | 8.40% | 61.02% |
| 15 | 1.66% | 37.61% |
| 20 | 0.56% | 37.33% |
| 30 | 0.07% | 46.22% |
| 45 | 2.34% | 52.40% |
| 60 | 6.80% | 64.68% |
| 80 | 0.92% | 69.74% |
| 85 | 15.36% | 50.43% |

**The no-defence control answers the horizon question directly: the backdoor
does establish well within 100 rounds when no defence runs at all** — ASR
climbs into the 40-70% range by round ~20-85 (noisy round-to-round, as
expected for FedAvg without any stabilizing defence, but clearly separated
from zero, unlike every ablation condition tested at 30 rounds).

**The SKIP-STAGE-2 run, now complete at all 200 rounds, stays low for the
entire run — it does not separate from zero the way the no-defence control
does, at any point:**

| | SKIP-STAGE-2, 200 rounds |
|---|---|
| Mean ASR, all 200 rounds | 3.29% |
| Mean ASR, last 50 rounds (151-200) | 3.55% |
| Max ASR, any single round | 21.54% (one spike; everything else well below) |
| Rounds with ASR > 10% | 13 / 200 |
| Rounds with ASR > 20% | 1 / 200 |

Over the full 200 rounds — nearly 7x the original 30-round ablation window —
ASR never settles anywhere close to the no-defence control's 40-70% range,
and the mean over the last 50 rounds (3.55%) is barely different from the
mean over the first 30 (consistent with the original ablation's flat
near-zero reading). Cumulative FP also keeps climbing roughly in step with
attack volume throughout (136 FP / 205 attacks at round 200, ratio ~0.66,
essentially unchanged from round 30's ~0.80) — no sign of Stage 3's filtering
capacity degrading over this horizon either.

**Reading this against the original framing ("if the backdoor still doesn't
take hold without Stage 2, the finding is solid; if it does, thirty rounds was
measuring nothing"): the finding is solid.** The no-defence control proves
the backdoor mechanics and dataset/rate settings are fully capable of
establishing within this time budget (40-70% ASR by round 20-85). Yet
SKIP-STAGE-2 holds ASR under ~4% on average across the entire 200-round run,
with no upward trend by the end. This is not a horizon artifact — it's Stage
3 (not Stage 2) doing essentially all of the suppression work, consistent
with the earlier BatchNorm-mode finding. The original 30-round ablation's
flat, near-zero ASR without Stage 2 is confirmed, not explained away, by
running nearly 7x longer.

**External corroboration (SiftFL, a follow-on paper building on FilterFL):**
SiftFL's own independent reimplementation of FilterFL, run at a matching 10%
attacker ratio (non-IID, CIFAR-10), reports ASR = 2.26% — over 1500 rounds,
the same single-digit range found here at 30-200 rounds. SiftFL also
documents a real long-horizon failure mode for FilterFL (confidence overlap
and detector collapse, ASR reaching 97-99%), but only at 40% and 70% attacker
ratios over ~400+ rounds — a different regime (much higher attacker share,
much longer horizon) from this ablation's 10%-attacker, ≤200-round setup, and
not something SiftFL attributes to Stage 2 specifically (their own ablation
varies CGAN scheduling frequency and trust history, not Stage 2 presence/absence).
This supports reading the low ASR here as a property of the attacker-ratio/
horizon regime being measured, not a broken or insufficiently-run experiment.

## Does the degenerate-pair invariance hold at SiftFL's failure-mode attacker ratios? (2026-10-10)

A real tension, flagged rather than smoothed over: SiftFL's documented
collapse mechanism (confidence overlap at 40%/70% attacker ratios, ~400+
rounds) is attributed to the discriminator reference models (G_old/G_temp)
becoming contaminated by backdoor features, degrading the trigger set's
discriminative content. This project's degenerate-pair finding says the
opposite — that the discriminators' content doesn't matter to the detection
decision at all (a (G_old, G_old) pair detects just as well as the real
(G_old, G_agg) pair). These don't sit together comfortably: if discriminator
content is irrelevant to the outcome, contaminating it shouldn't be able to
collapse detection either. One live possibility was that the invariance
result is simply bounded to low attacker ratios (this project's existing
tests were all at ~10%) and breaks down at the ratios where SiftFL's failure
mode actually appears.

**Tested directly** (`scripts/attacker_ratio_degenerate.py`): the same
normal-vs-degenerate-pair comparison, reference code, rerun at
`poison_client_num` = 10 (baseline), 40, and 70, holding the dataset
partition, model init, and the 10 sampled clients fixed across all three
trials (only the population-level attacker pool differs, so each trial
actually realizes 2, 3, and 6 attackers respectively among the fixed 10
sampled clients):

| Trial | Attackers sampled | Pixel MSE (normal vs degenerate) | Preds identical | Flagged set (normal) | Flagged set (degenerate) |
|---|---|---|---|---|---|
| 10% | 2/10 | 0.001030 | **True** | [10, 73, 68] | [10, 73, 68] |
| 40% | 3/10 | 0.000916 | **True** | [72, 73, 40] | [72, 73, 40] |
| 70% | 6/10 | 0.001046 | **True** | [10,72,73,62,61,16,40] | [10,72,73,62,61,16,40] |

**The invariance holds, unchanged, at every attacker ratio tested — the gap
does not widen.** Pixel MSE stays in the same ~0.0009-0.0010 band regardless
of how many real attackers are in the sampled set, predicted-class vectors
are bit-for-bit identical between normal and degenerate pairs in all three
trials, and every individual attacker's confidence differs by ≤0.003 between
the two pairs (e.g. 70%-trial attacker 16: 0.8074 vs 0.8072).

**This rules out "the invariance is a low-attacker-ratio artifact" and
sharpens the tension rather than resolving it.** The result is a single-round
test, so it does not directly test SiftFL's claimed long-horizon (~400+
round) contamination mechanism — but it does establish that at any given
round, regardless of how many real attackers are present, the discriminator
pair's actual content (G_old vs G_agg, real vs degenerate) contributes
nothing measurable to that round's detection decision. If detection in no
single round depends on discriminator content, it is not obvious how
*accumulating* contamination in that content, round over round, could be the
mechanism behind SiftFL's documented collapse. The remaining candidate
readings, as before: (a) SiftFL's collapse is driven by something other than
discriminator contamination despite their framing (most consistent with this
result), (b) discriminator content matters through some channel not
captured by a single generated-trigger-set snapshot (e.g. an accumulating
effect across many rounds that a one-round swap can't reveal), or (c) the
mechanism genuinely differs between the reference implementation (tested
here) and SiftFL's own reimplementation of FilterFL (which is where their
400+-round runs were done) — the two codebases have already shown measurable
divergence elsewhere in this project's testing (the Stage 2 loss and FP-count
discrepancies documented above).

## Does the Stage-2-removable finding hold under the paper's own loss formula? (2026-10-10)

Every Stage-2 ablation result so far (the MA/ASR tables, the flagged-rate
numbers) used the reference code's shipped, undocumented Stage 2 loss
(`L_reference`, documented above). That leaves open whether "Stage 2 is
removable" is a property of FilterFL as a defence, or an artifact of that
specific undocumented loss formula — the paper's own text describes a
different 3-term loss (`L_paper`, γ=0.75, λ=0.125). This project's own
reimplementation already showed `L_paper` performs far worse than
`L_reference` at normal-vs-skip detection (4/27 vs ~20/20 TP, see "Stage 2
loss: paper-faithful vs shipped-code" above) — but that comparison ran on
this project's reimplementation, not the reference pipeline the skip-stage2
ablation itself used. **The honest framing here matters: the shipped code is
what produced FilterFL's actual published results, so it is the ground truth
for what FilterFL is — the paper's loss is the version that doesn't reliably
work, not the other way around.** This is not "fixing" the reference code;
it's testing whether the removability finding generalizes to the one other
loss formulation available, on the reference's own architecture and pipeline
(the paper does not specify an architecture to match, so nothing about
Stage 1/2/3's model structure was touched).

**Implemented:** `stage2_loss` ('reference' | 'paper') added additively to
`reference/filterfl/gan.py`'s `gan_defense`, threaded through
`algorithm.py` and exposed as `--stage2-loss {reference,paper}` in
`main.py`. Default stays `'reference'`, unchanged behaviour. `'paper'`
replaces Stage 2's loss with the 3-term formulation (`0.375*nat_added +
0.375*nat_subbed + 0.125*(1-bias_added) + 0.125*bias_subbed`, no 5th term),
derived in the section above.

**Sanity check that the flag does something:** same fixed round
(`scripts/crossfeed_round.pt`), same GAN-training seed, `'reference'` vs
`'paper'` loss produces genuinely different trigger images — pixel MSE =
0.00087 (the same order of magnitude as the normal-vs-degenerate-pair gap
elsewhere in this project's testing), not the ~0 that would indicate the
flag silently no-oping.

**Result: FULL defence (α=0.5, poison_rate=0.5, 30 rounds), `stage2_loss='paper'`
vs `stage2_loss='reference'`, is bit-for-bit identical, every single round:**

| Run | Attack (cum.) | Defense (cum.) | Correct (cum.) | FP |
|---|---|---|---|---|
| FULL, reference loss | 35 | 61 | 34 | 27 |
| FULL, paper loss | 35 | 61 | 34 | 27 |
| SKIP-STAGE-2 | 35 | 62 | 34 | 28 |

Not just the final tallies — every one of the 30 per-round cumulative
`(attack, defense, correct)` tuples matches exactly between the reference-loss
and paper-loss runs, despite the trigger images themselves differing
pixel-for-pixel each round. **Stage 2's loss formula — like trigger content
in general, per the degenerate-pair finding — makes no detectable difference
to the detection outcome.** This is not a coincidence or a wiring bug (the
sanity check above rules that out); it's the same mechanism showing up a
third way: Stage 3's train-mode BatchNorm exploit (documented above) makes
detection almost entirely insensitive to what Stage 1/2 actually produce,
whether that's degenerate input, a different training budget, or now a
different loss function entirely. **The Stage-2-removable finding is not an
artifact of the shipped code's specific loss — it holds under the paper's
own loss spec too, on the reference pipeline.**

## Probe-backdoor separability: is there a Stage-3 blind spot? (2026-10-11)

Motivation: every finding above shows Stage 3's decision tracks client
*weights*, not trigger content (degenerate-pair invariance, attacker-ratio
invariance, Stage-2-loss invariance). The next question is whether that
weight-sensitivity actually separates benign from malicious clients at all,
or whether there's a region of weight-space that is simultaneously
high-ASR, acceptable clean accuracy, and low enough Stage-3 confidence to
survive filtering — an attack-feasible blind spot, distinct from (and
prior to) any adaptive search for one.

Method (`scripts/separability_scatter.py` / `scripts/separability_scatter_ckpt.py`):
build many single-client weight updates from a fixed G_old — a spread of
benign configs (client identity x local_epochs x lr) and a spread of poisoned
configs (poison_rate x poison_epochs x a post-hoc scale-down factor applied
to the delta, to probe the "scale the malicious update down until it looks
benign" lever specifically). For every resulting client model, record the
full tuple (ASR, CleanAcc, c_stage3, ||delta_w||) — not just ASR, so an
apparent evasion that destroys clean accuracy or an apparent low-confidence
model whose backdoor is already dead can't be mistaken for a real
attack-feasible point. Stage 3's confidence is computed via the actual
reference pipeline (RefResNet8 + train-mode BatchNorm probe), with a single
real (G_old, G_agg) trigger set reused for every probe — per the
degenerate-pair finding, trigger content doesn't measurably affect the
decision, so regenerating it per client would add noise, not information.

**First pass, random-init G_old:** found one overlap candidate
(poison_rate=0.2, epochs=6, scale=1.0: ASR=56.2%, CleanAcc=19.2%,
c_stage3=0.311, not flagged), but at round-1/random-init the benign
clean-accuracy band (9.97-21.41%) is close to chance, so "acceptable clean
accuracy" wasn't a meaningful bar — the global model hasn't learned real
features yet, so nothing is well-separated. Also notable: scaling the
malicious delta down mostly just killed ASR outright rather than sneaking it
under the confidence threshold (smaller ||delta_w|| -> lower ASR, not
smaller ||delta_w|| -> same ASR + lower confidence), suggesting the naive
"scale the update down" attack surface may not work — a genuine evasion, if
one exists, likely needs to change the update's direction/content, not just
its magnitude.

**Second pass, G_old = `results/stage2_full_a05_ckpt.pt` (this project's own
30-round FilterFL run, alpha=0.5, final MA=34.68%/ASR=0% at epoch 30; one
fixed change from the first pass, nothing else):** the overlap reappears,
and more clearly, confirming it is not a random-init artifact.

Benign bands at this checkpoint: confidence 0.292-0.830, clean acc
14.19-31.32 (read as "comparable to this checkpoint's own benign clients at
this point in training," not as absolute model quality -- 30 rounds is not a
converged model, MA=34.68%).

Two unflagged poisoned configs land inside both bands with majority-successful
backdoors, no stealth scaling required:

| config | ASR | CleanAcc | c_stage3 | flagged |
|---|---|---|---|---|
| rate=0.2, epochs=6, scale=1.0 | 73.1% | 23.78 | 0.399 | No |
| rate=0.5, epochs=6, scale=0.5 | 62.6% | 29.53 | 0.436 | No |

The first needs no scaling trick at all — just a moderate poison rate with
the full local-epoch count. This matches the first pass's reading: the
overlap doesn't come from the obvious "scale the update down" lever.

One wrinkle versus the first pass: the scale-down behaviour is not uniformly
"ASR and confidence drop together" at this checkpoint. At rate=0.5,
epochs=6, scale=1.0 -> 0.5: ASR drops from 93.0% to 62.6% while confidence
drops from 0.69 to 0.44, crossing the rho=0.5 threshold while ASR stays
majority-successful -- exactly the decoupling the first pass's "scaling
doesn't work" reading would not have predicted. But at rate=1.0 (max
poisoning), scaling still drags ASR and confidence down together (e.g.
scale=0.25 -> ASR=59.3%, c=0.669, still flagged). So the decoupling is
config-dependent (moderate poison rate + longer local training), not
universal.

### What distinguishes the blind-spot config, and is it an averaging artifact?

Diagnostic (`scripts/separability_diagnostic.py`, 5 hand-picked contrast
configs against ckpt-200) ruled out two naive explanations and surfaced two
candidates: cos(w, G_old) is saturated (0.985-0.9998 for all five configs,
uninformative) and raw ||delta_w|| does not predict flagged status (the
unflagged WINNER has a *larger* delta than a flagged neighbor). The two
candidates were cos(delta_w, mean_benign_delta) (direction alignment with
normal training noise) and sigma_probe (std of the Stage-3 probe's
per-trigger-image target-class probability, i.e. response heterogeneity
across the 32-image batch).

**Epoch sweep** (`scripts/separability_epoch_sweep.py`, poison_rate=0.2,
scale=1.0, local poison epochs 1-10, same ckpt-200 G_old): sigma_probe rises
close to monotonically with local epochs (0.148 at epochs=1 to ~0.25-0.29 at
epochs=6-8) -- a real, reproducible trend. cos_to_benign_centroid is
directionally consistent (negative correlation with c_stage3 across all nine
points) but noisy, not a tight single-variable predictor. ||delta_w||
(control) grows ~monotonically with epochs but tracks neither ASR nor
c_stage3 cleanly -- reconfirms magnitude isn't the driver. Also surfaced an
important correction: re-running epochs=2 with a different RNG trajectory
flipped it from flagged (c=0.575) to unflagged (c=0.490) -- the whole
epochs=1-10 region sits *near* the rho=0.5 threshold and individual points
can land on either side from SGD-order noise alone, not just the
hyperparameter. ASR climbs toward 83-84% by epochs=8/10 while c_stage3 stays
flat in the 0.37-0.49 band throughout -- extended local training buys real
backdoor strength without a corresponding rise in Stage-3's confidence
signal.

**Is the blind spot an averaging artifact?** (`scripts/separability_distribution.py`,
3 groups x ckpt-200 G_old: BENIGN = 5 distinct client identities,
DETECTED-POISONED = rate=1.0/epochs=6/scale=1.0 x5 SGD-order seeds,
UNDETECTED-POISONED = rate=0.2/epochs=6/scale=1.0 x5 SGD-order seeds.
For every repetition, the full 32-value per-image target-class probability
vector was kept, not just its mean, and three aggregation statistics were
compared: the current mean-based flagging rule, majority-vote (>50% of the
32 images individually exceed rho), and max (any single image exceeds rho).)

| group | mean-based flag rate | majority-vote | max-based |
|---|---|---|---|
| BENIGN | 0.00 | 0.00 | 0.00 |
| DETECTED-POISONED (rate=1.0) | 1.00 | 1.00 | 1.00 |
| UNDETECTED-POISONED (rate=0.2, epochs=6) | **0.20** | **0.00** | **1.00** |

**Yes, it is an averaging artifact -- but the mechanism is the opposite of
the majority-vote hypothesis.** The pooled per-image histogram for
UNDETECTED-POISONED (160 samples, 5 seeds) is `[15, 68, 29, 30, 18]` across
bins [0-.2)...[.8-1]: 70% of individual trigger images sit below 0.5, only
30% cross it. So it is not "most images vote backdoor, the mean waters it
down" -- it is a small but consistent high-confidence TAIL (11% of images
land in [.8-1]) embedded in a batch where most images respond weakly. The
mean discards that tail; majority-vote (which needs the backdoor signal to
dominate, not merely appear) is actually *less* sensitive than the mean here
(0/5 vs 1/5); max-based detection (does ANY of the 32 images individually
exceed rho) catches the group 100% of the time, with zero false positives on
BENIGN (0/160 pooled benign images ever exceed 0.5 under either alternative
statistic). DETECTED-POISONED is the clean positive control: fully
saturated, every one of 160 pooled values in [.8-1], sigma_probe=0 in every
rep, all three statistics agree at 1.00.

**Precise claim:** Stage 3's mean-pooling over the 32-image trigger batch
discards tail/outlier information, and the undetected-poisoned config's
backdoor signal lives disproportionately in that tail (a handful of
strongly-triggered images) rather than in the batch's central tendency. A
max- or top-k-style aggregation statistic instead of a mean would close this
specific blind spot on this evidence. Caveat: 5 seeds per poisoned group (160
pooled images each), 5 client identities for benign -- a clean, non-
overlapping separation in this sample, not large enough to rule out rare
benign false positives under max-based flagging at scale. No attack
optimization was attempted here -- this characterizes the three
already-identified groups, it does not search for a stronger evasion.

**Top-k aggregation curve** (`scripts/separability_topk_curve.py`, same three
groups/reps as above): for k in {1,2,3,4,5,8,10,16,32}, A_k = mean of the k
largest of the 32 per-image target-class probabilities (k=1 is max, k=32 is
the current mean). Flag rate at rho=0.5:

| k | 1 | 2 | 3 | 4 | 5 | 8 | 10 | 16 | 32 |
|---|---|---|---|---|---|---|---|---|---|
| BENIGN A_k | 0.100 | 0.085 | 0.072 | 0.063 | 0.057 | 0.043 | 0.038 | 0.028 | 0.016 |
| BENIGN flag rate | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DETECTED-POISONED A_k | 1.000 (all k) | | | | | | | | |
| DETECTED-POISONED flag rate | 1.00 (all k) | | | | | | | | |
| UNDETECTED-POISONED A_k | 0.891 | 0.869 | 0.849 | 0.831 | 0.816 | 0.777 | 0.752 | 0.668 | **0.471** |
| UNDETECTED-POISONED flag rate | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | **1.00** | **0.20** |

**The transition is a cliff, not a slope.** Every k from 1 through 16 (the
mean of even just the top HALF of the 32-image batch) flags
UNDETECTED-POISONED 100% of the time, with A_k staying comfortably above 0.5
(0.89 down to 0.67) the whole way; the flag rate collapses to 20% only at
k=32, the full-batch mean actually implemented. BENIGN never approaches
threshold at any k -- even its max (A_1=0.100) sits 5x below rho, a wide
margin, not a sensitivity/false-positive tradeoff. This sharpens the
averaging-artifact claim: the vulnerability is specific to full-batch
averaging, not fragile to exactly where an aggregation cutoff is drawn --
essentially any statistic weighting the top half of the batch more than the
bottom half would close this blind spot with room to spare, on this
evidence.

**Poison-rate x epochs characterization grid** (`scripts/separability_rate_epoch_grid.py`,
poison_rate in {0.1,0.2,0.3,0.4,0.5,0.7,1.0} x poison_epochs in {2,6,10},
scale=1.0, same malicious client and G_old as above, single seed per cell --
not averaged, so read as "this seed's realization" of each cell, not an
exact deterministic function of the two hyperparameters):

| epochs | rate | ASR | CleanAcc | c_stage3 | A_16 | flag |
|---|---|---|---|---|---|---|
| 2 | 0.1 | 13.6 | 34.4 | 0.280 | 0.389 | No |
| 2 | 0.2-1.0 | 18.5-100 | 32.5-10.0 | 0.57-1.00 | 0.73-1.00 | Yes (all) |
| 6 | 0.1 | 34.8 | 39.6 | 0.342 | 0.520 | No |
| 6 | 0.2 | **62.9** | 42.0 | 0.482 | 0.706 | **No** |
| 6 | 0.3-1.0 | 78.5-100 | 31.6-10.0 | 0.61-1.00 | 0.81-1.00 | Yes (all) |
| 10 | 0.1 | 53.6 | 43.0 | 0.332 | 0.500 | No |
| 10 | 0.2 | 54.2 | 44.8 | 0.491 | 0.711 | **No** |
| 10 | 0.3-1.0 | 86.4-100 | 32.3-10.0 | 0.53-1.00 | 0.77-1.00 | Yes (all) |

**The escape region is consistently rate in {0.1, 0.2} at every epoch count
tested, and rate=0.3 flips to flagged every time**, with a thin margin
(c_stage3 only 0.53-0.67 there, not saturated). No clean-accuracy cost in the
escape region: CleanAcc at the unflagged cells (32-45%) is often *higher*
than at flagged cells, which collapse toward chance as rate climbs (10.0 at
rate=1.0 regardless of epochs). epochs=6/rate=0.2 and epochs=10/rate=0.2 are
the strongest points in the grid -- ASR 54-63%, CleanAcc 42-45% (the best in
the table), unflagged.

**The top-k cliff generalizes across the useful part of the grid, with one
refinement.** A_16 catches every rate=0.2 cell (0.71-0.73, well above 0.5) at
both epochs=6 and epochs=10, confirming the single-point top-k finding is not
specific to the one config it was found at -- the rate=0.2 blind spot is a
full-batch-mean-pooling artifact, not a genuine absence of signal. Only
rate=0.1 sits close to the A_16 threshold too (0.39-0.52) -- a qualitatively
weaker, more-variable-ASR backdoor, not just a scaled-down version of the
rate=0.2 escape.

**Boundary sweep** (`scripts/separability_boundary_sweep.py`, rate in
{0.20,0.22,0.24,0.25,0.26,0.28,0.30} x epochs in {6,10}, 3 seeds/cell):

At epochs=6 the boundary is sharp and fully reproducible -- every rate from
0.20 to 0.28 is unflagged across all 3 seeds (F,F,F at every single rate,
zero exceptions), ASR ranging 55-84%, c_stage3 climbing smoothly 0.39-0.49;
the flip to flagged on all 3 seeds happens between rate=0.28 and rate=0.30.
At epochs=10 the boundary is noisier -- one of the three seeds is already
flagged starting at rate=0.2, two of three flip by rate=0.26 -- so epochs=6
gives a more robust, reproducible escape region than epochs=10, even though
individual epochs=10 points sometimes reach higher ASR.

**Combined scatter, ASR vs c_stage3** (`results/separability_asr_vs_confidence.png`,
78 points pooled across the boundary sweep, the rate x epochs grid, and the
three-group distribution characterization): confirms this is a genuine
region, not a threshold accident at one or two points. There is a visibly
dense horizontal band of unflagged points sitting consistently at confidence
0.39-0.49 across ASR 55-85% -- a real structural cluster just under rho=0.5,
not a handful of lucky outliers. 34/78 pooled points (44%) have ASR>50% AND
c_stage3<rho. Flagged points occupy the same ASR range but at visibly higher
confidence -- the plot shows two roughly separable populations, not one
threshold splitting a single cloud.

No attack optimization attempted at this stage (explicit instruction): top-k
remains a diagnostic of where the defender loses information, not an
objective being optimized against. The natural next step, once/if pursued,
would be an explicit objective like max ASR subject to A_32 < rho -- but only
after this low-confidence region's reproducibility was established, which is
what this sweep and plot were for.

Full data: `results/separability_diagnostic.out`,
`results/separability_epoch_sweep.csv`, `results/separability_distribution.csv`,
`results/separability_topk_curve.csv`, `results/separability_rate_epoch_grid.csv`,
`results/separability_boundary_sweep.csv`, `results/separability_asr_vs_confidence.png`.

Full data: `results/separability_scatter.csv` (round-1),
`results/separability_scatter_ckpt.csv` (round-30). Caveat worth stating
plainly: these are single-round, single-client probes constructed directly
from G_old, not emergent behaviour from an actual adversarial federated run
-- they establish that an unflagged, majority-successful backdoor *can
exist* at this point in training, not that an attacker following the
standard protocol produces one. **Persistence test (same run continued to epoch 100 and 200, snapshots
`results/stage2_full_a05_snap_e00100_ckpt.pt` / `_e00200_ckpt.pt`, final
epoch-200 MA=65.44%/ASR=0.94%):** one candidate persists, one does not.

`rate=0.2, epochs=6, scale=1.0` (the no-scaling-trick config) is essentially
unchanged across the whole horizon:

| round | ASR | CleanAcc | c_stage3 | flagged |
|---|---|---|---|---|
| 30 | 73.1% | 23.78 | 0.399 | No |
| 100 | 67.4% | 32.23 | 0.397 | No |
| 200 | 67.5% | 35.28 | 0.390 | No |

ASR and confidence are flat across 30/100/200 rounds while clean accuracy
rises with the maturing model (reflecting the checkpoint's own improving
benign band, not the attack weakening) — this is not a round-30 artifact;
a majority-successful, unflagged backdoor sits inside the benign confidence
band at every horizon tested.

`rate=0.5, epochs=6, scale=0.5` (the round-30 "decoupling" candidate) does
NOT survive: unflagged at round 30 (c=0.436), ASR collapsed on its own by
round 100 (29.6%, c=0.325, still technically unflagged), and by round 200
confidence crossed the rho=0.5 threshold (c=0.516, flagged) even as ASR
partially recovered (54.8%). So the overlap region is not static — some
near-boundary points close as the model matures, at least one does not.

Full data: `results/separability_scatter_ckpt100.csv`,
`results/separability_scatter_ckpt200.csv`. Same scope caveat as above:
single-round, single-client probes built directly from each snapshot's
G_old, not behaviour emerging from an actual adversarial federated run.

## End-to-end live federated test of the blind-spot attack (2026-10-11)

Everything above establishes a CLIENT-LEVEL result: a single-client weight
snapshot, probed in isolation against a frozen checkpoint, can have high ASR
and low Stage-3 confidence simultaneously. This does not establish that the
blind spot survives an actual multi-round federated run where malicious
clients repeatedly participate, get (not-)filtered, get aggregated, and the
global model evolves round over round. This section tests that directly.

### Discovered execution path (this project's own `filterfl/` pipeline, not reference/filterfl)

- Entry point: `filterfl/run.py` `main()`. Per round: client sampling ->
  local training -> FedAvg aggregation -> (if `cfg.defence_method != 'none'`)
  detection + re-aggregation with excluded clients zeroed -> global eval.
- Client sampling: `random.sample(participants, cfg.num_clients)` each round
  (`run.py` ~line 175); malicious clients are whichever sampled ids are in
  `adversarial_namelist` (fixed per run, drawn once at startup from
  `cfg.poison_client_num` out of `cfg.number_of_total_participants`).
- Local training: `filterfl/client.py:local_train`. Malicious clients use
  `cfg.poison_epochs`/`cfg.poison_lr` and poison `cfg.poison_rate` fraction
  of each batch via `PixelPatternAttacker.poison_batch` (`attacker.py`).
  These are the exact knobs the separability experiments swept.
- FilterFL detection: `filterfl/defense.py:run_detection` (Stage 1/2, CGAN
  trigger construction) then `stage3_classify_clients` (Stage 3, the
  train-mode-BN probe characterized throughout this document). Called from
  `run.py` only when `cfg.defence_method != 'none'`.
- Filtering: in `run.py`, any client with
  `pred_class==target_class and confidence>active_rho` gets `betas[i]=0`;
  excluded clients are counted (`detected_count`, `detection_tp`/`detection_fp`
  against `malicious_sampled`) and re-aggregated out via a second
  `aggregate()` call.
- Aggregation: `filterfl/aggregator.py:aggregate`, weighted FedAvg over
  `betas`.
- Global eval: `filterfl/client.py:evaluate` on the clean test set (MA) and
  on the held-out non-target-class set with the trigger patched in (ASR),
  called once per round on the POST-aggregation global model.
- Per-round logging: `filterfl/metrics_logger.py`, already writes
  `sampled_clients`, `adversarial_sampled`, `detection_tp`, `detection_fp`,
  `detected_count`, and `per_client_confidence` (`cid:pred:conf;...`) for
  every sampled client every round -- every quantity this experiment needs
  (malicious/benign acceptance rates, mean Stage-3 confidence split by
  malicious/benign status) is derivable from the existing CSV by
  post-processing. **No changes to `run.py`/`defense.py`/`client.py`/
  `aggregator.py`/`metrics_logger.py` were needed** -- the smallest clean
  extension is a post-processing/plotting script reading the existing
  per-round CSVs, plus the existing `--poison-rate`/`--poison-epochs`/
  `--seed`/`--defense`/`--epochs` CLI flags already in `run.py` to configure
  the three conditions. `scripts/e2e_analyze.py` (new) does the
  post-processing and plotting; no other new files.

### Experiment design

Base config: `configs/cifar_primary.yaml` (alpha=0.5, CIFAR-10,
poison_client_num=10/100 participants, num_clients=10/round, target_class=0
-- the project's standard config used throughout this document).

- **Condition A (no-defence, blind-spot recipe)**: `--defense none
  --poison-rate 0.25 --poison-epochs 6` -- upper bound for the blind-spot
  attack's achievable ASR with zero filtering, paired directly against
  Condition C to isolate what filtering changes.
- **Condition B (FilterFL, ordinary attack)**: `--defense filterfl
  --poison-rate 0.5 --poison-epochs 6` (config defaults, unchanged) --
  defended baseline, expected to reproduce the low-ASR result already
  established repeatedly in this document.
- **Condition C (FilterFL, blind-spot attack)**: `--defense filterfl
  --poison-rate 0.25 --poison-epochs 6` -- the actual test: does the
  client-level blind spot survive the live loop.

3 seeds (0, 1, 2) per condition, paired (same seed across A/B/C), 50 rounds,
default checkpointing every 5 epochs. No changes to FilterFL itself (Stage
2/3, BatchNorm mode, pooling, rho, architecture) -- the primary experiment
evaluates the implementation exactly as characterized throughout this
document.



- Need a longer run (documented round-count decision) to get real MA/ASR/TP/FP
  numbers — 3 epochs is plumbing-only.
- Running **both** α=0.5 (primary) and α=1 (repo default, side run) per
  discrepancy #5, so the comparison table isn't apples-to-oranges.
- Config corrections (Part B, in progress): α=0.5 default, target class exposed
  as a parameter (behavior unchanged, default stays 0), ρ schedule exposed as
  config and logged every round, gan.py magic numbers (γ, CGAN rounds,
  generator LR, images-per-class) surfaced as config with unchanged defaults.
