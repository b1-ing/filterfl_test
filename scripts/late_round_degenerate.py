"""Late-round degenerate-pair invariance test.

Replicates the round-1 d1_d2_role_test / attractor_rounds protocol at the
end of arm A's training (epoch 500) to test whether the d1-invariance finding
holds at late rounds.

Protocol
--------
1. Load arm A's epoch-500 checkpoint as G_old(500).
2. Rebuild the data partition with arm A's seed so client loaders match.
3. Run N_ROUNDS fresh FedAvg rounds (with poisoning) starting from G_old(500),
   saving G_old and the round's client_states at each round.
4. At each round, run the degenerate-pair detector with 3 GAN seeds:
     A) d1=d2=G_old(round)         -- this round's starting model
     B) d1=d2=G_old(round+1)       -- next round's starting model (one step ahead)
   and check whether the pred-class vector and flagged set differ.
5. Print the full per-client pred/conf vectors and the A==B comparison.

This answers:
  - Does one-round-apart d1 invariance hold at round 500? (A vs B per round)
  - Is the basin seed-stable within a late round?
  - Does the basin class change across late rounds (500-505)? (round-dependence)

Usage:
    python -m scripts.late_round_degenerate \
        --ckpt results/hpc_s1a_s0_ckpt.pt \
        --config configs/cifar_primary.yaml \
        --rounds 5 \
        --out results/late_round_degenerate.out
"""
import argparse
import copy
import random
import sys
import os
from collections import Counter
import io

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filterfl.config import Config
from filterfl.data import (load_dataset, build_class_index, dirichlet_partition,
                            make_client_loader)
from filterfl.models import build_model
from filterfl.attacker import PixelPatternAttacker
from filterfl.client import local_train
from filterfl.aggregator import aggregate
from filterfl.defense import run_detection, stage3_classify_clients


GAN_SEEDS = [123, 456, 789]
RHO = 0.5


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def run_degenerate(label, d1_state, client_states, ori_weight, betas, sampled,
                   adversarial_namelist, cfg, num_classes, in_channels, device):
    """Run degenerate detection (d1=d2=d1_state) with GAN_SEEDS seeds.
    Returns list of (seed, preds, confs, flagged) tuples."""
    results = []
    for s in GAN_SEEDS:
        seed_everything(s)
        det = run_detection(d1_state, d1_state, cfg, num_classes, in_channels, device)
        stage3 = stage3_classify_clients(
            client_states, ori_weight, betas, det.trigger_images,
            num_classes, in_channels, cfg.target_class, device,
            probe_mode=cfg.stage3_probe_mode)
        preds = [r["pred_class"] for r in stage3]
        confs = [round(r["confidence"], 3) for r in stage3]
        flagged = [sampled[i] for i, r in enumerate(stage3)
                   if r["pred_class"] == cfg.target_class and r["confidence"] > RHO]
        mal_in_sampled = [c for c in sampled if c in adversarial_namelist]
        tp = len([c for c in flagged if c in adversarial_namelist])
        print(f"    seed={s}: preds={preds}")
        print(f"             confs={confs}")
        print(f"             flagged={flagged}  mal_sampled={mal_in_sampled}  tp={tp}/{len(mal_in_sampled)}")
        results.append((s, tuple(preds), confs, flagged))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="checkpoint to start from")
    ap.add_argument("--config", required=True)
    ap.add_argument("--rounds", type=int, default=5,
                    help="number of fresh rounds to simulate from checkpoint (default 5)")
    ap.add_argument("--diag-seed", type=int, default=9999,
                    help="RNG seed for fresh round simulation (default 9999)")
    ap.add_argument("--out", default=None, help="optional file to tee output to")
    ap.add_argument("--seed", type=int, default=None,
                    help="override config seed for data-partition reconstruction (use arm's training seed)")
    ap.add_argument("--poison-rate", type=float, default=None,
                    help="override config poison_rate (use arm's training poison_rate)")
    ap.add_argument("--stage3-probe-mode", default=None, choices=[None, "eval", "train"],
                    help="override config stage3_probe_mode")
    ap.add_argument("--stage2-loss", default=None, choices=[None, "paper", "reference"],
                    help="override config stage2 loss variant")
    args = ap.parse_args()

    # Tee stdout to --out file if specified.
    _out_fh = None
    if args.out:
        _out_fh = open(args.out, "w")
        _orig_stdout = sys.stdout
        class _Tee:
            def write(self, s):
                _orig_stdout.write(s)
                _out_fh.write(s)
            def flush(self):
                _orig_stdout.flush()
                _out_fh.flush()
        sys.stdout = _Tee()

    cfg = Config.from_yaml(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.poison_rate is not None:
        cfg.poison_rate = args.poison_rate
    if args.stage3_probe_mode is not None:
        cfg.stage3_probe_mode = args.stage3_probe_mode
    if args.stage2_loss is not None:
        cfg.gan.stage2_loss = args.stage2_loss

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    num_classes, in_channels = 10, 3

    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Rounds from checkpoint: {args.rounds}  diag_seed: {args.diag_seed}")
    print(f"Config overrides: seed={cfg.seed} poison_rate={cfg.poison_rate} "
          f"stage3_probe_mode={cfg.stage3_probe_mode} stage2_loss={cfg.gan.stage2_loss}")

    # Rebuild data partition with the training run's seed.
    seed_everything(cfg.seed)
    train_set, _, _, _ = load_dataset(cfg.type)
    class_index = build_class_index(train_set)
    per_participant, _ = dirichlet_partition(
        train_set, cfg.number_of_total_participants, cfg.dirichlet_alpha, class_index)
    client_loaders = {p: make_client_loader(train_set, idxs, cfg.batch_size)
                      for p, idxs in per_participant.items()}

    # Load checkpoint; use its adversarial_namelist so we match arm A's attackers.
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"Checkpoint epoch: {ckpt['epoch']}")
    adversarial_namelist = set(ckpt.get("adversarial_namelist", []))
    if not adversarial_namelist:
        raise RuntimeError("adversarial_namelist missing from checkpoint")
    print(f"Adversarial namelist ({len(adversarial_namelist)}): {sorted(adversarial_namelist)}")

    model = build_model(cfg.type, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    attacker = PixelPatternAttacker(cfg)
    participants = list(range(cfg.number_of_total_participants))

    # Fresh RNG for client sampling in these diagnostic rounds.
    seed_everything(args.diag_seed)

    # We need G_old per round. Collect (round_id, ori_weight, client_states, betas, sampled).
    round_records = []

    for rnd_offset in range(args.rounds + 1):  # +1 so round N+1 exists for the B comparison
        ori_weight = {k: v.clone() for k, v in model.state_dict().items()}
        sampled = random.sample(participants, cfg.num_clients)
        mal_in_sampled = [c for c in sampled if c in adversarial_namelist]

        client_states = []
        for cid in sampled:
            m = copy.deepcopy(model)
            w = local_train(m, client_loaders[cid], cfg, device,
                            is_malicious=(cid in adversarial_namelist),
                            attacker=attacker)
            client_states.append(w)

        betas = [1] * len(sampled)
        new_state = aggregate(ori_weight, client_states, betas, cfg.eta,
                              denominator=getattr(cfg, "aggregation_denominator", "survivors"))
        model.load_state_dict(new_state)

        rnd_label = ckpt["epoch"] + rnd_offset
        print(f"round {rnd_label}: sampled={sampled}  mal_in_sampled={mal_in_sampled}")
        round_records.append((rnd_label, ori_weight, client_states, betas, sampled))

    print(f"\n{'='*70}")
    print("DEGENERATE-PAIR INVARIANCE TEST (late rounds)")
    print("Each round: A=d1=d2=G_old(round), B=d1=d2=G_old(round+1)")
    print("Same client_states classified under both triggers.")
    print(f"{'='*70}\n")

    for i in range(args.rounds):
        rnd_A, ori_A, client_states, betas, sampled = round_records[i]
        rnd_B, ori_B, _, _, _ = round_records[i + 1]

        weight_delta = sum(
            (ori_A[k].float() - ori_B[k].float()).abs().sum().item()
            for k in ori_A if ori_A[k].dtype.is_floating_point)
        print(f"\n{'='*70}")
        print(f"Round {rnd_A} vs {rnd_B}  (sum|G_old({rnd_A})-G_old({rnd_B})| = {weight_delta:.2f})")
        print(f"  sampled={sampled}")
        print(f"  mal_in_sampled={[c for c in sampled if c in adversarial_namelist]}")
        print(f"{'='*70}")

        print(f"\n  --- A) d1=d2=G_old({rnd_A}) ---")
        res_A = run_degenerate(f"A(round={rnd_A})", ori_A, client_states, ori_A, betas, sampled,
                               adversarial_namelist, cfg, num_classes, in_channels, device)

        print(f"\n  --- B) d1=d2=G_old({rnd_B}) ---")
        res_B = run_degenerate(f"B(round={rnd_B})", ori_B, client_states, ori_A, betas, sampled,
                               adversarial_namelist, cfg, num_classes, in_channels, device)

        print(f"\n  >>>  A vs B comparison:")
        all_same = True
        for (sA, predsA, _, _), (sB, predsB, _, _) in zip(res_A, res_B):
            same = predsA == predsB
            all_same = all_same and same
            print(f"    seed={sA}: A={list(predsA)}  B={list(predsB)}  identical={same}")

        seed_stable_A = len(set(r[1] for r in res_A)) == 1
        seed_stable_B = len(set(r[1] for r in res_B)) == 1
        print(f"    seed-stable within A: {seed_stable_A}")
        print(f"    seed-stable within B: {seed_stable_B}")
        print(f"    A==B across all seeds: {all_same}")


if __name__ == "__main__":
    main()
