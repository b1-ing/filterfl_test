"""Main orchestration loop — FedAvg + optional FilterFL/SiftFL defense.

Usage:
    python -m filterfl.run --config configs/cifar_filterfl.yaml --defense filterfl --run-name tier1_filterfl
"""
import argparse
import copy
import logging
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import Subset
import yaml

from .config import Config
from .data import (load_dataset, build_class_index, dirichlet_partition,
                    make_client_loader, make_test_loader, split_poison_eval_sets)
from .models import build_model
from .attacker import PixelPatternAttacker
from .client import local_train, evaluate
from .aggregator import aggregate
from .defense import run_detection, stage3_classify_clients
from .trust import TrustState
from .metrics_logger import MetricsLogger
from . import checkpoint

logger = logging.getLogger("filterfl")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def is_detection_round(epoch: int, cfg) -> bool:
    if cfg.defence_method == "none":
        return False
    if cfg.defence_method == "filterfl" or not cfg.schedule.enabled:
        return True
    if cfg.schedule.randomized:
        rng = random.Random((cfg.seed, epoch))
        return rng.random() < (1.0 / cfg.schedule.k)
    return epoch % cfg.schedule.k == 0 or epoch == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--defense", default=None, choices=[None, "none", "filterfl", "siftfl"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--poison-epochs", type=int, default=None, help="malicious clients' local epochs per round")
    ap.add_argument("--poison-rate", type=float, default=None)
    ap.add_argument("--poison-lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--stage2-loss", default=None, choices=[None, "paper", "reference"],
                     help="Stage 2 trigger-generation loss: 'paper' (default, this project's "
                          "faithful reading of the published 3-term loss) or 'reference' (gan.py's "
                          "exact unnormalized 5-term loss verbatim) — see CODE_NOTES.md")
    ap.add_argument("--stage3-probe-mode", default=None, choices=[None, "eval", "train"],
                     help="Stage 3 probe model mode: 'eval' (default, textbook-correct, BatchNorm "
                          "uses each client's stored running stats) or 'train' (reference-code "
                          "behaviour: algorithm.py never calls .eval(), so BatchNorm normalizes "
                          "against the trigger batch's own statistics) — see CODE_NOTES.md")
    ap.add_argument("--skip-stage2", action="store_true",
                     help="Stage-2 ablation (paper's Table IV design): feed Stage 1's raw "
                          "per-class image set I[target_class] directly into Stage 3, skipping "
                          "Stage 2's trigger-filtering/⊕⊖ construction entirely — see CODE_NOTES.md")
    ap.add_argument("--resume", action="store_true",
                     help="resume from results/<run-name>_ckpt.pt if it exists, else start fresh")
    ap.add_argument("--checkpoint-every", type=int, default=5,
                     help="save a checkpoint every N epochs (default 5 — cheap relative to a ~30s epoch)")
    ap.add_argument("--snapshot-every", type=int, default=0,
                     help="in addition to the resumable checkpoint (which overwrites a single file), "
                          "also save a DISTINCT, never-overwritten snapshot every N epochs to "
                          "results/<run-name>_snap_e<epoch>_ckpt.pt — needed to ever replay a specific "
                          "historical round's G_old, since the resumable checkpoint only ever keeps the "
                          "latest epoch. 0 (default) disables this.")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.defense is not None:
        cfg.defence_method = args.defense
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.poison_epochs is not None:
        cfg.poison_epochs = args.poison_epochs
    if args.poison_rate is not None:
        cfg.poison_rate = args.poison_rate
    if args.poison_lr is not None:
        cfg.poison_lr = args.poison_lr
    if args.seed is not None:
        cfg.seed = args.seed
    if args.stage2_loss is not None:
        cfg.gan.stage2_loss = args.stage2_loss
    if args.stage3_probe_mode is not None:
        cfg.stage3_probe_mode = args.stage3_probe_mode
    if cfg.defence_method == "siftfl":
        cfg.schedule.enabled = True
        cfg.trust.enabled = True

    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_set, test_set, num_classes, in_channels = load_dataset(cfg.type)
    class_index = build_class_index(train_set)
    if cfg.sampling_dirichlet:
        per_participant, counts = dirichlet_partition(train_set, cfg.number_of_total_participants,
                                                        cfg.dirichlet_alpha, class_index)
    else:
        raise NotImplementedError("non-dirichlet partitioning not needed for this testbed")

    client_loaders = {p: make_client_loader(train_set, idxs, cfg.batch_size)
                       for p, idxs in per_participant.items()}
    test_loader = make_test_loader(test_set, cfg.batch_size)
    non_target_idx, _ = split_poison_eval_sets(test_set, cfg.target_class)
    asr_loader = make_test_loader(Subset(test_set, non_target_idx), cfg.batch_size)

    participants = list(range(cfg.number_of_total_participants))
    adversarial_namelist = set(random.sample(participants, cfg.poison_client_num))

    model = build_model(cfg.type, num_classes=num_classes).to(device)
    attacker = PixelPatternAttacker(cfg)

    os.makedirs(cfg.results_dir, exist_ok=True)
    out_csv = os.path.join(cfg.results_dir, f"{cfg.run_name}.csv")
    ckpt_path = os.path.join(cfg.results_dir, f"{cfg.run_name}_ckpt.pt")

    trust_state = None
    if cfg.trust.enabled:
        trust_state = TrustState(cfg.trust.accumulation_rate, cfg.trust.decay_rate,
                                  cfg.trust.exclusion_threshold, cfg.trust.initial_trust)

    last_trigger_images = None
    lr, poison_lr = cfg.lr, cfg.poison_lr
    start_epoch = 1

    if args.resume and os.path.exists(ckpt_path):
        ckpt = checkpoint.load(ckpt_path, model, device)
        start_epoch = ckpt["epoch"] + 1
        lr, poison_lr = ckpt["lr"], ckpt["poison_lr"]
        adversarial_namelist = set(ckpt["adversarial_namelist"])  # must match the original run, not re-derived
        if trust_state is not None:
            trust_state.scores = ckpt["trust_scores"]
        last_trigger_images = ckpt["last_trigger_images"]
        if last_trigger_images is not None:
            last_trigger_images = last_trigger_images.to(device)
        logger.info(f"resumed from {ckpt_path} at epoch {start_epoch}")

    mlog = MetricsLogger(out_csv, append=(start_epoch > 1))
    with open(os.path.join(cfg.results_dir, f"{cfg.run_name}_config.yaml"), "w") as f:
        yaml.dump(cfg.to_dict(), f)

    if start_epoch > cfg.epochs:
        logger.info(f"checkpoint already at epoch {start_epoch - 1} >= target {cfg.epochs}, nothing to do")
        mlog.close()
        return

    for epoch in range(start_epoch, cfg.epochs + 1):
        if cfg.lr_decay and epoch % cfg.lr_decay_epoch == 0:
            lr *= cfg.lr_decay_gamma
            poison_lr *= cfg.lr_decay_gamma
            cfg.lr, cfg.poison_lr = lr, poison_lr

        t0 = time.time()
        if cfg.is_poison:
            sampled = random.sample(participants, cfg.num_clients)
        else:
            sampled = random.sample([p for p in participants if p not in adversarial_namelist], cfg.num_clients)
        malicious_sampled = [c for c in sampled if c in adversarial_namelist]

        ori_weight = {k: v.clone() for k, v in model.state_dict().items()}
        client_states = []
        for cid in sampled:
            m = copy.deepcopy(model)
            w = local_train(m, client_loaders[cid], cfg, device,
                             is_malicious=(cid in malicious_sampled), attacker=attacker)
            client_states.append(w)

        # Malicious clients' own local-model MA (clean test set, their trained
        # weights before any aggregation/filtering) — lets us tell "stealthy
        # attacker is just undertrained" apart from "stealthy attacker evades
        # detection while still backdoored", per the Part C reporting requirement.
        malicious_local_ma = {}
        if malicious_sampled:
            probe = build_model(cfg.type, num_classes=num_classes).to(device)
            for i, cid in enumerate(sampled):
                if cid in malicious_sampled:
                    probe.load_state_dict(client_states[i])
                    malicious_local_ma[cid] = evaluate(probe, test_loader, device)

        betas = [1] * len(sampled)
        new_state = aggregate(ori_weight, client_states, betas, cfg.eta,
                               denominator=getattr(cfg, "aggregation_denominator", "survivors"))

        active_rho = None
        detected_this_round = False
        detect_round = is_detection_round(epoch, cfg)
        detection_results = []

        if cfg.defence_method != "none" and detect_round:
            active_rho = cfg.rho_schedule.rho_at(epoch)
            det = run_detection(ori_weight, new_state, cfg, num_classes, in_channels, device,
                                skip_stage2=args.skip_stage2)
            probe_images = det.per_class_images[cfg.target_class] if args.skip_stage2 else det.trigger_images
            last_trigger_images = probe_images
            detection_results = stage3_classify_clients(
                client_states, ori_weight, betas, probe_images,
                num_classes, in_channels, cfg.target_class, device,
                probe_mode=cfg.stage3_probe_mode)
            for i, res in enumerate(detection_results):
                flagged = (res["pred_class"] == cfg.target_class and
                           res["confidence"] is not None and res["confidence"] > active_rho)
                if trust_state is not None:
                    if trust_state.is_excluded_by_trust(sampled[i]):
                        flagged = True
                    trust_state.update(sampled[i], flagged_this_round=flagged)
                if flagged:
                    betas[i] = 0
            detected_this_round = True
        elif cfg.defence_method != "none" and not detect_round:
            # non-detection round under scheduling: reuse policy from config
            active_rho = cfg.rho_schedule.rho_at(epoch)
            if cfg.schedule.non_detection_round_policy == "last_filter" and last_trigger_images is not None:
                detection_results = stage3_classify_clients(
                    client_states, ori_weight, betas, last_trigger_images,
                    num_classes, in_channels, cfg.target_class, device,
                    probe_mode=cfg.stage3_probe_mode)
                for i, res in enumerate(detection_results):
                    flagged = (res["pred_class"] == cfg.target_class and
                               res["confidence"] is not None and res["confidence"] > active_rho)
                    if trust_state is not None:
                        if trust_state.is_excluded_by_trust(sampled[i]):
                            flagged = True
                        trust_state.update(sampled[i], flagged_this_round=flagged)
                    if flagged:
                        betas[i] = 0
            # else: policy == 'fedavg' -> betas stay all 1, plain FedAvg this round

        if any(b == 0 for b in betas):
            new_state = aggregate(ori_weight, client_states, betas, cfg.eta,
                                   denominator=getattr(cfg, "aggregation_denominator", "survivors"))

        model.load_state_dict(new_state)

        ma = evaluate(model, test_loader, device)
        asr = evaluate(model, asr_loader, device, attacker=attacker, poison=True)

        detected_count = sum(1 for b in betas if b == 0)
        detected_ids = {sampled[i] for i, b in enumerate(betas) if b == 0}
        tp = len(detected_ids & set(malicious_sampled))
        fp = len(detected_ids - set(malicious_sampled))

        conf_str = ";".join(
            f"{sampled[i]}:{r['pred_class']}:{r['confidence']}"
            for i, r in enumerate(detection_results)
        )
        trust_str = ""
        if trust_state is not None:
            trust_str = ";".join(f"{cid}:{trust_state.get(cid)}" for cid in sampled)
        malicious_ma_str = ";".join(f"{cid}:{ma_val:.2f}" for cid, ma_val in malicious_local_ma.items())

        mlog.log(epoch=epoch, ma=ma, asr=asr, active_rho=active_rho,
                  is_detection_round=detect_round,
                  sampled_clients=",".join(map(str, sampled)),
                  adversarial_sampled=",".join(map(str, malicious_sampled)),
                  detected_count=detected_count, detection_tp=tp, detection_fp=fp,
                  per_client_confidence=conf_str, per_client_trust=trust_str,
                  malicious_local_ma=malicious_ma_str)

        logger.info(f"epoch {epoch} MA={ma:.2f} ASR={asr:.2f} rho={active_rho} "
                    f"detected={detected_count} tp={tp} fp={fp} ({time.time()-t0:.1f}s)")

        if epoch % args.checkpoint_every == 0 or epoch == cfg.epochs:
            checkpoint.save(ckpt_path, epoch, model, lr, poison_lr,
                             adversarial_namelist, trust_state.scores if trust_state else {},
                             last_trigger_images)

        if args.snapshot_every > 0 and (epoch % args.snapshot_every == 0 or epoch == cfg.epochs):
            snap_path = os.path.join(cfg.results_dir, f"{cfg.run_name}_snap_e{epoch:05d}_ckpt.pt")
            checkpoint.save(snap_path, epoch, model, lr, poison_lr,
                             adversarial_namelist, trust_state.scores if trust_state else {},
                             last_trigger_images)

    mlog.close()


if __name__ == "__main__":
    main()
