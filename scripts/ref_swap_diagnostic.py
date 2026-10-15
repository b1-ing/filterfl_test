"""Reference-swap diagnostic: does swapping in a clean G_old recover detection?

Hypothesis under test: at late rounds (450+), acceptance is high because the
Stage-2 GAN reference (d1_state = G_old) has been contaminated by backdoored
updates, not because something about the client models themselves has changed.

Procedure:
  - Load the contaminated late-round global model (arm A epoch-500 checkpoint).
  - Load a clean early-round reference model (from gen_clean_ref.py).
  - Simulate K detection rounds: for each, sample 1 malicious + (num_clients-1)
    benign clients, run local_train from the contaminated global, aggregate.
  - Run detection twice on the same client set:
      CONT: d1_state = contaminated model (what arm A actually used)
      CLEAN: d1_state = clean reference model (the swap)
  - Compare per-client confidence and TP rate under rho=0.5.

If detection recovers under CLEAN, reference contamination is the cause.
If it stays poor, something about the client models themselves has changed
(arm A's earlier round-dependence results become the relevant prior).

Usage:
    python -m scripts.ref_swap_diagnostic \
        --config configs/cifar_primary.yaml \
        --contaminated-ckpt results/hpc_s1a_s0_ckpt.pt \
        --clean-ckpt results/clean_ref_e50_ckpt.pt \
        --trials 30 \
        --out results/ref_swap_diag.csv
"""
import argparse
import copy
import csv
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filterfl.config import Config
from filterfl.data import (load_dataset, build_class_index, dirichlet_partition,
                            make_client_loader)
from filterfl.models import build_model
from filterfl.client import local_train
from filterfl.aggregator import aggregate
from filterfl.defense import run_detection, stage3_classify_clients
from filterfl.attacker import PixelPatternAttacker
from filterfl import checkpoint


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def load_model_state(ckpt_path, cfg, device):
    model = build_model(cfg.type, num_classes=10).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    return model.state_dict(), ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--contaminated-ckpt", required=True,
                    help="late-round arm A checkpoint (contaminated G_old)")
    ap.add_argument("--clean-ckpt", required=True,
                    help="early-round clean reference checkpoint")
    ap.add_argument("--trials", type=int, default=30,
                    help="number of simulated detection rounds (default 30)")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--rho", type=float, default=0.5,
                    help="detection threshold (default 0.5, matches arm A at epoch 500)")
    ap.add_argument("--diag-seed", type=int, default=7777,
                    help="RNG seed for trial client sampling (default 7777)")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    num_classes, in_channels = 10, 3

    print(f"Device: {device}")
    print(f"Contaminated ckpt: {args.contaminated_ckpt}")
    print(f"Clean ckpt:        {args.clean_ckpt}")
    print(f"Trials: {args.trials}  rho: {args.rho}")

    # Build data partition using arm A's config seed so client_loaders match
    # the original run's distribution.
    seed_everything(cfg.seed)
    train_set, _, _, _ = load_dataset(cfg.type)
    class_index = build_class_index(train_set)
    per_participant, _ = dirichlet_partition(
        train_set, cfg.number_of_total_participants, cfg.dirichlet_alpha, class_index)
    client_loaders = {p: make_client_loader(train_set, idxs, cfg.batch_size)
                      for p, idxs in per_participant.items()}

    # Load contaminated model (arm A epoch 500) and adversarial_namelist.
    # Also loads RNG state from checkpoint — we override it with diag_seed below.
    cont_state, cont_ckpt = load_model_state(args.contaminated_ckpt, cfg, device)
    adversarial_namelist = set(cont_ckpt.get("adversarial_namelist", []))
    if not adversarial_namelist:
        # fallback: shouldn't happen for arm A
        raise RuntimeError("adversarial_namelist missing from checkpoint")
    print(f"Adversarial namelist ({len(adversarial_namelist)} clients): "
          f"{sorted(adversarial_namelist)}")

    # Load clean reference model.
    clean_state, _ = load_model_state(args.clean_ckpt, cfg, device)

    attacker = PixelPatternAttacker(cfg)
    participants = list(range(cfg.number_of_total_participants))
    benign_pool = [p for p in participants if p not in adversarial_namelist]
    mal_list = sorted(adversarial_namelist)

    # Override RNG for reproducible trial sampling (independent of arm A's run).
    seed_everything(args.diag_seed)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fieldnames = ["trial", "mal_client",
                  "cont_conf", "cont_pred", "cont_flagged",
                  "clean_conf", "clean_pred", "clean_flagged"]
    rows = []

    for trial in range(1, args.trials + 1):
        # Sample exactly 1 malicious + (num_clients-1) benign for guaranteed TP opportunity.
        mal_client = random.choice(mal_list)
        benign_sampled = random.sample(benign_pool, cfg.num_clients - 1)
        sampled = [mal_client] + benign_sampled
        random.shuffle(sampled)
        mal_idx = sampled.index(mal_client)

        # Build a fresh contaminated global model for this trial's starting point.
        cont_model = build_model(cfg.type, num_classes=num_classes).to(device)
        cont_model.load_state_dict(cont_state)

        ori_weight = {k: v.clone() for k, v in cont_state.items()}

        # Run local_train from the contaminated global.
        client_states = []
        for cid in sampled:
            m = copy.deepcopy(cont_model)
            w = local_train(m, client_loaders[cid], cfg, device,
                            is_malicious=(cid == mal_client), attacker=attacker)
            client_states.append(w)

        betas = [1] * len(sampled)
        new_state = aggregate(ori_weight, client_states, betas, cfg.eta,
                              denominator=getattr(cfg, "aggregation_denominator", "survivors"))

        # Run detection under CONTAMINATED reference (what arm A actually did).
        det_cont = run_detection(ori_weight, new_state, cfg, num_classes, in_channels, device)
        stage3_cont = stage3_classify_clients(
            client_states, ori_weight, betas, det_cont.trigger_images,
            num_classes, in_channels, cfg.target_class, device,
            probe_mode=cfg.stage3_probe_mode)
        res_cont = stage3_cont[mal_idx]

        # Run detection under CLEAN reference (the swap).
        det_clean = run_detection(clean_state, new_state, cfg, num_classes, in_channels, device)
        stage3_clean = stage3_classify_clients(
            client_states, ori_weight, betas, det_clean.trigger_images,
            num_classes, in_channels, cfg.target_class, device,
            probe_mode=cfg.stage3_probe_mode)
        res_clean = stage3_clean[mal_idx]

        cont_flagged = (res_cont["pred_class"] == cfg.target_class
                        and res_cont["confidence"] is not None
                        and res_cont["confidence"] > args.rho)
        clean_flagged = (res_clean["pred_class"] == cfg.target_class
                         and res_clean["confidence"] is not None
                         and res_clean["confidence"] > args.rho)

        rows.append({
            "trial": trial,
            "mal_client": mal_client,
            "cont_conf": f"{res_cont['confidence']:.4f}" if res_cont["confidence"] else "N/A",
            "cont_pred": res_cont["pred_class"],
            "cont_flagged": int(cont_flagged),
            "clean_conf": f"{res_clean['confidence']:.4f}" if res_clean["confidence"] else "N/A",
            "clean_pred": res_clean["pred_class"],
            "clean_flagged": int(clean_flagged),
        })

        print(f"  trial {trial:3d}: mal={mal_client:3d} "
              f"CONT conf={res_cont['confidence']:.3f} flagged={int(cont_flagged)} | "
              f"CLEAN conf={res_clean['confidence']:.3f} flagged={int(clean_flagged)}")

    # Summary
    n = len(rows)
    cont_tp = sum(r["cont_flagged"] for r in rows)
    clean_tp = sum(r["clean_flagged"] for r in rows)
    cont_mean_conf = np.mean([float(r["cont_conf"]) for r in rows if r["cont_conf"] != "N/A"])
    clean_mean_conf = np.mean([float(r["clean_conf"]) for r in rows if r["clean_conf"] != "N/A"])

    print(f"\n{'='*60}")
    print(f"  Trials: {n}  rho: {args.rho}")
    print(f"  CONTAMINATED ref: TP={cont_tp}/{n} ({cont_tp/n*100:.1f}%)  mean_conf={cont_mean_conf:.4f}")
    print(f"  CLEAN ref:        TP={clean_tp}/{n} ({clean_tp/n*100:.1f}%)  mean_conf={clean_mean_conf:.4f}")
    if clean_tp > cont_tp:
        delta = (clean_tp - cont_tp) / n * 100
        print(f"  -> Detection RECOVERED by {delta:.1f}pp: supports contamination hypothesis")
    elif clean_tp < cont_tp:
        print(f"  -> Detection did NOT recover: client models themselves have changed")
    else:
        print(f"  -> No difference: inconclusive")
    print(f"{'='*60}\n")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
