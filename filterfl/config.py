"""Config schema for the FilterFL/SiftFL testbed.

Independently written — informed by reference/CODE_NOTES.md, not copied from
reference/filterfl. Where a value has no documented paper default (because the
released code hardcodes it with no stated justification), that is called out
in a comment at the field, not silently guessed.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import yaml


@dataclass
class RhoSchedule:
    """Stage-3 confidence threshold rho, as a step schedule over epoch blocks.

    Reference behaviour (CODE_NOTES.md hyperparameters table): rho is read as
    `ours_standard[(epoch - 1) // 500]` with `ours_standard = [0.50, 0.40, 0.30, 0.20]`.
    That is a fixed block size of 500 with no explicit epoch breakpoints in the
    original code. Here the schedule is explicit (epoch, rho) pairs so a
    randomised or custom schedule can be substituted without touching code.
    """
    breakpoints: list = field(default_factory=lambda: [0, 500, 1000, 1500])
    values: list = field(default_factory=lambda: [0.50, 0.40, 0.30, 0.20])

    def rho_at(self, epoch: int) -> float:
        idx = 0
        for i, bp in enumerate(self.breakpoints):
            if epoch - 1 >= bp:
                idx = i
        return self.values[idx]


@dataclass
class GanDefenseConfig:
    """Stage 1/2 CGAN hyperparameters.

    All defaults below are the literal values hardcoded in reference
    gan.py (not exposed in any yaml there) — see CODE_NOTES.md "Hyperparameters"
    table. Surfacing them here does not change behaviour by default.
    """
    gamma: float = 0.75          # reference: `Alpha = 0.75`, reused for both stage1 and stage2
    # NOTE: the released code has no independent lambda term at all (CODE_NOTES.md
    # discrepancy #1) — there is no ground-truth default to port. This lambda
    # only takes effect in MY implementation's paper-faithful 3-term Stage 2 loss
    # (see defense.py); it does not exist in the reference code's path.
    # Default chosen so gamma + lambda + (1-gamma-lambda) splits the remaining
    # 0.25 evenly; documented as an assumption, not a recovered value.
    lambda_: float = 0.125
    stage1_rounds: int = 10      # reference: n_epochs1
    stage2_rounds: int = 5       # reference: n_epochs2
    generator_lr: float = 5e-4
    adam_b1: float = 0.5
    adam_b2: float = 0.999
    images_per_class: int = 32   # reference: image_num
    latent_dim: int = 100
    # Reference discrepancy #2: algorithm.py only ever uses g_imgs[0], the
    # trigger batch from the FIRST stage-2 training epoch, discarding the
    # other stage2_rounds-1 refinement epochs. Default True reproduces that
    # (possibly unintended) behaviour; set False to use the last (most
    # refined) batch instead, for ablation.
    use_first_stage2_batch_only: bool = True
    # "paper": the independently-written, paper-faithful 3-term (gamma, lambda,
    # 1-gamma-lambda) Stage 2 loss in defense.py, normalized to sum to 1.
    # "reference": gan.py's exact unnormalized 5-term loss verbatim (two
    # naturalness terms at full gamma=0.75 each, two bias terms at only 0.25
    # each, plus an undocumented 5th term pushing the bare trigger pattern
    # itself toward D1-class-0-confidence) — see CODE_NOTES.md "Stage 2 loss:
    # paper-faithful vs shipped-code, empirically (2026-10-09)". Defaults to
    # "paper" so nothing changes silently; an attacker's local defence replica
    # should default to "reference" instead, since a real deployment runs
    # whatever the released server code actually runs, not the paper's text.
    stage2_loss: str = "paper"  # or "reference"


@dataclass
class ScheduleConfig:
    """SiftFL addition #1: run Stage 1-2 (expensive) only every k rounds."""
    enabled: bool = False
    k: int = 10
    randomized: bool = False
    # What happens on a non-detection round. 'last_filter' re-uses the most
    # recently computed trigger set T and client betas from the last
    # detection round (cheap, slightly stale filtering); 'fedavg' disables
    # filtering entirely on non-detection rounds (plain FedAvg). Paper
    # abstract for SiftFL isn't available to us; this default is an explicit
    # assumption, see results/siftfl_assumptions.md.
    non_detection_round_policy: str = "last_filter"  # or "fedavg"


@dataclass
class TrustHistoryConfig:
    """SiftFL addition #2: persistent per-client trust score."""
    enabled: bool = False
    accumulation_rate: float = 0.1   # how much a clean-looking round raises trust
    decay_rate: float = 0.2          # how much a flagged round lowers trust
    exclusion_threshold: float = -1.0  # trust below this -> excluded regardless of this round's T confidence
    initial_trust: float = 0.0


@dataclass
class Config:
    # --- fed ---
    type: str = "CIFAR10"
    algorithm: str = "FedAvg"
    epochs: int = 2000
    eta: float = 1.0
    local_epochs: int = 2
    batch_size: int = 128
    num_clients: int = 10
    number_of_total_participants: int = 100
    # "survivors" (default, our original behaviour): the FedAvg mean divides by the
    # number of NON-EXCLUDED sampled clients, i.e. a renormalized average over whoever
    # survived Stage 3 filtering this round.
    # "sampled" (reference-code behaviour, CODE_NOTES.md discrepancy #7a): divides by
    # the TOTAL number of sampled clients regardless of exclusions -- excluded
    # clients' zeroed diffs still count toward the denominator, so filtering shrinks
    # the effective global step size (eta) proportionally to how much got excluded
    # that round, on top of whatever the exclusion itself does. This is what the
    # reference's actual `Aggregation()` does and is what produced the paper's
    # published numbers -- kept here as an ablation flag, default UNCHANGED, because
    # for reproduction purposes the reference's shipped behaviour is what we're
    # testing against, not a corrected version of it.
    aggregation_denominator: str = "survivors"  # or "sampled"
    lr: float = 0.01
    lr_decay: bool = True
    lr_decay_epoch: int = 1
    lr_decay_gamma: float = 0.998
    momentum: float = 0.9
    decay: float = 0.0005

    # --- non-iid ---
    sampling_dirichlet: bool = True
    dirichlet_alpha: float = 0.5   # Part B correction: paper's headline value, NOT the repo's shipped 1.0

    # --- poison / attacker ---
    is_poison: bool = True
    poison_epochs: int = 6
    poison_lr: float = 0.005
    poison_rate: float = 0.5
    trigger_num: int = 1
    poison_client_num: int = 10
    poison_patterns: dict = field(default_factory=lambda: {
        # index 0: the reference repo's actual shipped pattern (cifar_params.yaml
        # `0_poison_pattern`) -- two broken 4-pixel dashes per row, rows 0 and 3,
        # NOT a contiguous stripe. Faithful to the released code.
        0: [[0, 0], [0, 1], [0, 2], [0, 3], [0, 6], [0, 7], [0, 8], [0, 9],
            [3, 0], [3, 1], [3, 2], [3, 3], [3, 6], [3, 7], [3, 8], [3, 9]],
        # index 1: a genuinely CONTIGUOUS white stripe in the upper-left corner,
        # matching the CCS paper's prose description ("a white stripe in the
        # upper left corner") rather than the code's scattered-dash pattern.
        # Same pixel count (16) as index 0 for a fair comparison: a solid
        # 2-row x 8-col rectangle, rows 0-1, cols 0-7.
        1: [[r, c] for r in (0, 1) for c in range(8)],
    })
    # which poison_patterns key local_train/evaluate actually apply. Defaults to
    # 0 (the reference-faithful pattern) -- existing configs/runs are unaffected.
    active_trigger_index: int = 0
    # Part B correction: target class de-hardcoded. Default unchanged (0).
    target_class: int = 0

    # --- defence ---
    # 'none' | 'filterfl' | 'siftfl' (siftfl = filterfl + schedule/trust flags below)
    defence_method: str = "none"
    rho_schedule: RhoSchedule = field(default_factory=RhoSchedule)
    gan: GanDefenseConfig = field(default_factory=GanDefenseConfig)
    # "eval" (default, textbook-correct): Stage 3's probe model is put in eval
    # mode, so BatchNorm normalizes using each client's own stored running
    # stats. "train" (reference-code behaviour): the probe is left in its
    # default train mode (algorithm.py never calls .eval() on it), so
    # BatchNorm normalizes against the 32-image trigger batch's OWN
    # statistics instead -- see CODE_NOTES.md "Stage 3 probe BatchNorm mode:
    # the actual source of detection power (2026-10-09)". This single flag,
    # not the Stage 2 loss formula, accounts for nearly the entire TP-rate
    # gap between this project's detector and the reference's.
    stage3_probe_mode: str = "eval"  # or "train"
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    trust: TrustHistoryConfig = field(default_factory=TrustHistoryConfig)

    # --- misc ---
    seed: int = 1
    device: str = "cuda"
    results_dir: str = "results"
    run_name: str = "run"

    @staticmethod
    def from_yaml(path: str) -> "Config":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return Config.from_dict(raw)

    @staticmethod
    def from_dict(raw: dict) -> "Config":
        raw = dict(raw)
        rho = raw.pop("rho_schedule", None)
        gan = raw.pop("gan", None)
        sched = raw.pop("schedule", None)
        trust = raw.pop("trust", None)
        cfg = Config(**{k: v for k, v in raw.items() if k in {f.name for f in dataclasses.fields(Config)}})
        if rho:
            cfg.rho_schedule = RhoSchedule(**rho)
        if gan:
            cfg.gan = GanDefenseConfig(**gan)
        if sched:
            cfg.schedule = ScheduleConfig(**sched)
        if trust:
            cfg.trust = TrustHistoryConfig(**trust)
        return cfg

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
