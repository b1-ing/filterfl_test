"""FilterFL/SiftFL detector core — Stage 1 (knowledge extraction), Stage 2
(trigger filtering), Stage 3 (model filtering).

Independently written against the paper-level description in the task brief,
using the real numeric hyperparameters recovered in CODE_NOTES.md. This is
*not* a line-for-line port of reference/filterfl/gan.py: Stage 2's loss comes
in two modes, selected by cfg.gan.stage2_loss ("paper", the default, vs
"reference"). "paper" implements the full 3-term (gamma, lambda,
1-gamma-lambda) loss the paper describes; "reference" reproduces gan.py's
exact unnormalized 5-term loss verbatim, including its undocumented 5th term.
That structural difference is intentional — it's the "my code" column
Step 1 compares against "their code" — but see CODE_NOTES.md "Stage 2 loss:
paper-faithful vs shipped-code, empirically (2026-10-09)" for why this is not
just a fidelity footnote: at identical config, "paper" catches 4/27 attacks
over 24 rounds where "reference" catches ~20/20. The published loss does not
detect backdoors in practice; the undocumented shipped one does.

G_old / G_agg terminology: d1_state == G_old (previous round's accepted global
model, assumed benign — see CODE_NOTES.md "Design assumptions"), d2_state ==
G_agg (this round's raw, unfiltered aggregate).
"""
import logging
from dataclasses import dataclass, field

import numpy as np
import torch

from .models import ResNet8, GeneratorCIFAR, gan_weight_init

logger = logging.getLogger("filterfl")


@dataclass
class DetectionResult:
    trigger_images: torch.Tensor          # (images_per_class, C, 32, 32) - the T set used by Stage 3
    per_class_images: list                # list of (images_per_class, C, 32, 32), one per class - the I set
    stage1_loss_history: list = field(default_factory=list)
    stage2_loss_history: list = field(default_factory=list)


def _sample_z(n, latent_dim, device):
    return torch.randn(n, latent_dim, device=device)


def run_detection(d1_state: dict, d2_state: dict, cfg, num_classes: int, in_channels: int,
                   device, model_cls=ResNet8, skip_stage2: bool = False) -> DetectionResult:
    """model_cls: the D1/D2 discriminator architecture. Defaults to ResNet8
    (unchanged behaviour); CODE_NOTES.md's architecture-dependence ablation
    passes ResNet8NoNorm here to test whether Stage 1/2's naturalness/bias
    signals (not just Stage 3's) depend on having BatchNorm to renormalize.

    skip_stage2: the paper's Table IV ablation design -- skip Stage 2's
    trigger-filtering/⊕⊖ construction entirely (not computed at all, not
    just ignored) and leave `trigger_images` as None; the caller is
    expected to feed Stage 3 `per_class_images[target_class]` directly
    instead. See CODE_NOTES.md "Stage 2 ablation" for why this matters."""
    gan_cfg = cfg.gan
    d1 = model_cls(num_classes=num_classes, in_channels=in_channels).to(device)
    d2 = model_cls(num_classes=num_classes, in_channels=in_channels).to(device)
    d1.load_state_dict(d1_state)
    d2.load_state_dict(d2_state)
    d1.eval()
    d2.eval()
    for p in d1.parameters():
        p.requires_grad_(False)
    for p in d2.parameters():
        p.requires_grad_(False)

    target = cfg.target_class
    gamma = gan_cfg.gamma
    lam = gan_cfg.lambda_

    # ---- Stage 1: knowledge extraction ----
    per_class_images = []
    stage1_losses = []
    for c in range(num_classes):
        gen = GeneratorCIFAR(out_dim=in_channels, latent_dim=gan_cfg.latent_dim).to(device)
        gen.apply(gan_weight_init)
        opt = torch.optim.Adam(gen.parameters(), lr=gan_cfg.generator_lr,
                                betas=(gan_cfg.adam_b1, gan_cfg.adam_b2))
        best_loss, best_state = float("inf"), None
        for _ in range(gan_cfg.stage1_rounds):
            opt.zero_grad()
            z = _sample_z(gan_cfg.images_per_class, gan_cfg.latent_dim, device)
            imgs = gen(z)
            d1_out = torch.softmax(d1(imgs), dim=1)
            naturalness = d1_out.std(dim=1).mean()          # low std -> looks like a confident real class
            d2_out = torch.softmax(d2(imgs), dim=1)[:, c].mean()
            loss = gamma * naturalness + (1 - gamma) * (1 - d2_out)
            loss.backward()
            opt.step()
            if loss.item() < best_loss:
                best_loss, best_state = loss.item(), {k: v.clone() for k, v in gen.state_dict().items()}
        stage1_losses.append(best_loss)
        gen.load_state_dict(best_state)
        with torch.no_grad():
            z = _sample_z(gan_cfg.images_per_class, gan_cfg.latent_dim, device)
            per_class_images.append(gen(z).detach())

    if skip_stage2:
        return DetectionResult(
            trigger_images=None,
            per_class_images=per_class_images,
            stage1_loss_history=stage1_losses,
            stage2_loss_history=[],
        )

    other_imgs = torch.cat([per_class_images[c] for c in range(num_classes) if c != target], dim=0)
    target_imgs = per_class_images[target]

    # ---- Stage 2: trigger filtering ----
    stage2_loss_mode = getattr(gan_cfg, "stage2_loss", "paper")
    trigger_gen = GeneratorCIFAR(out_dim=1, latent_dim=gan_cfg.latent_dim).to(device)
    trigger_gen.apply(gan_weight_init)
    opt = torch.optim.Adam(trigger_gen.parameters(), lr=gan_cfg.generator_lr,
                            betas=(gan_cfg.adam_b1, gan_cfg.adam_b2))
    best_loss, best_state = float("inf"), None
    stage2_losses = []
    trigger_batches = []  # one per epoch, each (images_per_class, C, 32, 32) broadcast to in_channels
    for _ in range(gan_cfg.stage2_rounds):
        opt.zero_grad()
        z = _sample_z(gan_cfg.images_per_class, gan_cfg.latent_dim, device)
        trigger = trigger_gen(z)                                   # (N, 1, 32, 32)
        trigger_full = trigger.repeat(1, in_channels, 1, 1)

        # I' = other-class images + trigger (trying to make them look like `target`)
        added = other_imgs.detach() + trigger_full.repeat(num_classes - 1, 1, 1, 1)[:other_imgs.size(0)]
        naturalness_added = torch.softmax(d1(added), dim=1).std(dim=1).mean()
        d2_added_target_prob = torch.softmax(d2(added), dim=1)[:, target].mean()

        # I' = target-class images - trigger (trying to make them look NOT `target`)
        subbed = target_imgs.detach() - trigger_full
        naturalness_subbed = torch.softmax(d1(subbed), dim=1).std(dim=1).mean()
        d2_subbed_target_prob = torch.softmax(d2(subbed), dim=1)[:, target].mean()

        if stage2_loss_mode == "reference":
            # gan.py:140-184 verbatim: NOT the paper's 3-term (gamma, lambda,
            # 1-gamma-lambda) split. Five unnormalized terms (sum ~2.25, not 1):
            # naturalness at full gamma=0.75 on EACH of added/subbed (not halved),
            # bias terms at only (1-gamma)=0.25 each, plus a 5th term with no
            # counterpart in the paper's equations at all: the bare trigger
            # pattern's own class-0 confidence under D1 (not composited onto any
            # base image). See CODE_NOTES.md "Stage 2 loss: paper-faithful vs
            # shipped-code, empirically (2026-10-09)".
            trigger_alone_target_prob_d1 = torch.softmax(d1(trigger_full), dim=1)[:, target].mean()
            loss = (
                gamma * naturalness_added + (1 - gamma) * (1 - d2_added_target_prob)
                + gamma * naturalness_subbed + (1 - gamma) * d2_subbed_target_prob
                + (1 - gamma) * trigger_alone_target_prob_d1
            )
        elif stage2_loss_mode == "paper":
            loss = (
                gamma * (naturalness_added + naturalness_subbed) / 2
                + lam * (1 - d2_added_target_prob)
                + (1 - gamma - lam) * d2_subbed_target_prob
            )
        else:
            raise ValueError(f"unknown gan.stage2_loss mode: {stage2_loss_mode!r}")
        loss.backward()
        opt.step()
        if loss.item() < best_loss:
            best_loss, best_state = loss.item(), {k: v.clone() for k, v in trigger_gen.state_dict().items()}
        stage2_losses.append(loss.item())

        trigger_gen.load_state_dict(best_state)
        with torch.no_grad():
            z = _sample_z(gan_cfg.images_per_class, gan_cfg.latent_dim, device)
            trigger_batches.append(trigger_gen(z).detach().repeat(1, in_channels, 1, 1))

    trigger_images = trigger_batches[0] if gan_cfg.use_first_stage2_batch_only else trigger_batches[-1]

    return DetectionResult(
        trigger_images=trigger_images,
        per_class_images=per_class_images,
        stage1_loss_history=stage1_losses,
        stage2_loss_history=stage2_losses,
    )


def stage3_classify_clients(client_states: list, ori_weight: dict, betas: list,
                             trigger_images: torch.Tensor, num_classes: int, in_channels: int,
                             target_class: int, device, probe_mode: str = "eval",
                             model_cls=ResNet8) -> list:
    """Returns list of dicts: {pred_class, confidence} per client, in client_states order.

    Mirrors the reference's parameter reconstruction
    (`(w_i - ori) * beta_i + ori`) so a client already zeroed by an earlier
    gate isn't double counted, but still reports its (pred, confidence) for
    logging — the caller decides what to do with betas.

    probe_mode: "eval" (default, textbook-correct) leaves the probe in eval
    mode, so BatchNorm uses each client's own stored running stats.
    "train" reproduces `algorithm.py`'s Stage 3 verbatim: it never calls
    `.eval()` on `ResNet8_model`, so BatchNorm normalizes against the
    32-image trigger batch's OWN statistics instead. See CODE_NOTES.md
    "Stage 3 probe BatchNorm mode: the actual source of detection power
    (2026-10-09)" — this one flag, not the Stage 2 loss formula, explains
    nearly the entire TP-rate gap between this project's detector and the
    reference's.
    """
    probe = model_cls(num_classes=num_classes, in_channels=in_channels).to(device)
    if probe_mode == "eval":
        probe.eval()
    elif probe_mode == "train":
        probe.train()
    else:
        raise ValueError(f"unknown probe_mode: {probe_mode!r}")
    results = []
    with torch.no_grad():
        for i, w_i in enumerate(client_states):
            if betas[i] == 0:
                results.append({"pred_class": None, "confidence": None})
                continue
            params = {
                k: (w_i[k] - ori_weight[k]) * betas[i] + ori_weight[k]
                if k in ori_weight and ori_weight[k].dtype != torch.int64 else w_i[k]
                for k in w_i.keys()
            }
            probe.load_state_dict(params)
            out = torch.softmax(probe(trigger_images), dim=1).mean(dim=0)
            conf, pred = out.max(dim=0)
            results.append({"pred_class": int(pred.item()), "confidence": float(conf.item())})
    return results
