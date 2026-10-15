"""Noise calibration for degenerate-pair invariance test.

Extends late_round_degenerate with an A vs A' comparison:
  A:  d1=d2=G_old(round), GAN seeds = GROUP_1  (same as existing test)
  A': d1=d2=G_old(round), GAN seeds = GROUP_2  (same trigger, independent GAN inits)
  B:  d1=d2=G_old(round+1), GAN seeds = GROUP_1

Interpretation:
  If A vs A' disagrees as often as A vs B → the pipeline is noisy; A≠B is not a
  trigger-swap signal, it is GAN-init variance on boundary clients.
  If A and A' always agree but A vs B often disagrees → invariance has genuinely
  broken at late rounds and B produces distinct classifications.

Usage:
    python -m scripts.noise_calibration \
        --ckpt results/hpc_s1a_dense_s0_snap_e00498_ckpt.pt \
        --config configs/cifar_primary.yaml \
        --seed 0 \
        --poison-rate 0.25 \
        --stage3-probe-mode train \
        --stage2-loss reference \
        --rounds 5 \
        --diag-seed 9999 \
        --out results/hpc_noise_calibration.out
"""
import argparse
import copy
import random
import sys
import os

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

# Two independent GAN seed groups of equal size.
# GROUP_1 is used for A and B (matches the existing test).
# GROUP_2 is used for A' (independent inits, same trigger).
GAN_GROUP_1 = [123, 456, 789]
GAN_GROUP_2 = [1000, 2000, 3000]

RHO = 0.5


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def run_degenerate(label, d1_state, client_states, ori_weight, betas, sampled,
                   adversarial_namelist, cfg, num_classes, in_channels, device,
                   gan_seeds):
    """Run degenerate detection (d1=d2=d1_state) with the given GAN seeds.
    Returns list of (seed, preds_tuple, confs, flagged) tuples."""
    results = []
    for s in gan_seeds:
        seed_everything(s)
        det = run_detection(d1_state, d1_state, cfg, num_classes, in_channels, device)
        stage3 = stage3_classify_clients(
            client_states, ori_weight, betas, det.trigger_images,
            num_classes, in_channels, cfg.target_class, device,
            probe_mode=cfg.stage3_probe_mode)
        preds = tuple(r["pred_class"] for r in stage3)
        confs = [round(r["confidence"], 3) for r in stage3]
        flagged = [sampled[i] for i, r in enumerate(stage3)
                   if r["pred_class"] == cfg.target_class and r["confidence"] > RHO]
        mal_in_sampled = [c for c in sampled if c in adversarial_namelist]
        tp = len([c for c in flagged if c in adversarial_namelist])
        print(f"    seed={s}: preds={list(preds)}")
        print(f"             confs={confs}")
        print(f"             flagged={flagged}  mal_sampled={mal_in_sampled}  tp={tp}/{len(mal_in_sampled)}")
        results.append((s, preds, confs, flagged))
    return results


def compare_groups(label_x, res_x, label_y, res_y, n_clients):
    """Compare pred vectors across two groups seed-by-seed.

    Returns a list of (per_client_disagree_count, per_client_disagreement_mask)
    aggregated across all seed pairs.
    """
    # Each group has len(GAN_GROUP_*) results. We compare all cross-pairs.
    per_client_flips = [0] * n_clients
    n_pairs = 0
    for (_, preds_x, _, _) in res_x:
        for (_, preds_y, _, _) in res_y:
            n_pairs += 1
            for i in range(n_clients):
                if preds_x[i] != preds_y[i]:
                    per_client_flips[i] += 1
    flip_rate = sum(per_client_flips) / (n_pairs * n_clients) if n_pairs else 0.0
    flipped_clients = [i for i, f in enumerate(per_client_flips) if f > 0]
    return per_client_flips, flip_rate, flipped_clients, n_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--diag-seed", type=int, default=9999)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--poison-rate", type=float, default=None)
    ap.add_argument("--stage3-probe-mode", default=None, choices=[None, "eval", "train"])
    ap.add_argument("--stage2-loss", default=None, choices=[None, "paper", "reference"])
    args = ap.parse_args()

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
    print(f"Rounds: {args.rounds}  diag_seed: {args.diag_seed}")
    print(f"Config: seed={cfg.seed} poison_rate={cfg.poison_rate} "
          f"probe_mode={cfg.stage3_probe_mode} stage2_loss={cfg.gan.stage2_loss}")
    print(f"GAN GROUP_1 (A, B): {GAN_GROUP_1}")
    print(f"GAN GROUP_2 (A'):   {GAN_GROUP_2}")
    print()

    seed_everything(cfg.seed)
    train_set, _, _, _ = load_dataset(cfg.type)
    class_index = build_class_index(train_set)
    per_participant, _ = dirichlet_partition(
        train_set, cfg.number_of_total_participants, cfg.dirichlet_alpha, class_index)
    client_loaders = {p: make_client_loader(train_set, idxs, cfg.batch_size)
                      for p, idxs in per_participant.items()}

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

    seed_everything(args.diag_seed)

    round_records = []
    for rnd_offset in range(args.rounds + 1):
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

    SEP = "=" * 70
    print(f"\n{SEP}")
    print("NOISE CALIBRATION TEST")
    print("A:  d1=d2=G_old(round)   GAN seeds=GROUP_1  (same trigger, group 1)")
    print("A': d1=d2=G_old(round)   GAN seeds=GROUP_2  (same trigger, group 2)")
    print("B:  d1=d2=G_old(round+1) GAN seeds=GROUP_1  (trigger swap, group 1)")
    print(f"{SEP}\n")

    # Accumulate summary stats across rounds.
    summary_rows = []

    for i in range(args.rounds):
        rnd_A, ori_A, client_states, betas, sampled = round_records[i]
        rnd_B, ori_B, _, _, _ = round_records[i + 1]
        n_clients = len(sampled)
        mal_in_sampled = [c for c in sampled if c in adversarial_namelist]

        weight_delta = sum(
            (ori_A[k].float() - ori_B[k].float()).abs().sum().item()
            for k in ori_A if ori_A[k].dtype.is_floating_point)

        print(f"\n{SEP}")
        print(f"Round {rnd_A} vs {rnd_B}  (|G_old delta| = {weight_delta:.2f})")
        print(f"  sampled={sampled}")
        print(f"  mal_in_sampled={mal_in_sampled}")
        print(f"{SEP}")

        print(f"\n  --- A ) d1=d2=G_old({rnd_A})  GROUP_1 seeds={GAN_GROUP_1} ---")
        res_A = run_degenerate("A", ori_A, client_states, ori_A, betas, sampled,
                               adversarial_namelist, cfg, num_classes, in_channels, device,
                               GAN_GROUP_1)

        print(f"\n  --- A') d1=d2=G_old({rnd_A})  GROUP_2 seeds={GAN_GROUP_2} ---")
        res_Ap = run_degenerate("A'", ori_A, client_states, ori_A, betas, sampled,
                                adversarial_namelist, cfg, num_classes, in_channels, device,
                                GAN_GROUP_2)

        print(f"\n  --- B ) d1=d2=G_old({rnd_B})  GROUP_1 seeds={GAN_GROUP_1} ---")
        res_B = run_degenerate("B", ori_B, client_states, ori_A, betas, sampled,
                               adversarial_namelist, cfg, num_classes, in_channels, device,
                               GAN_GROUP_1)

        # Within-condition noise: A vs A' (same trigger, independent GAN inits).
        flips_AAp, rate_AAp, flipped_AAp, n_pairs_AAp = compare_groups(
            "A", res_A, "A'", res_Ap, n_clients)

        # Trigger-swap signal: A vs B (different trigger, same GAN seeds).
        flips_AB, rate_AB, flipped_AB, n_pairs_AB = compare_groups(
            "A", res_A, "B", res_B, n_clients)

        print(f"\n  >>> COMPARISON SUMMARY (round {rnd_A}):")
        print(f"    A vs A' (within-condition noise, {n_pairs_AAp} cross-seed pairs):")
        print(f"      per-client flip counts: {flips_AAp}")
        print(f"      flipped client positions: {flipped_AAp}  "
              f"(clients: {[sampled[p] for p in flipped_AAp]})")
        print(f"      overall flip rate: {rate_AAp:.3f}")

        print(f"    A vs B  (trigger-swap signal,  {n_pairs_AB} cross-seed pairs):")
        print(f"      per-client flip counts: {flips_AB}")
        print(f"      flipped client positions: {flipped_AB}  "
              f"(clients: {[sampled[p] for p in flipped_AB]})")
        print(f"      overall flip rate: {rate_AB:.3f}")

        shared_flips = sorted(set(flipped_AAp) & set(flipped_AB))
        print(f"    Shared unstable positions (in both A vs A' and A vs B): {shared_flips}")
        if shared_flips:
            print(f"      → These {len(shared_flips)} client(s) are boundary cases: "
                  f"GAN init noise and trigger swap both flip them.")
        unique_AB = sorted(set(flipped_AB) - set(flipped_AAp))
        if unique_AB:
            print(f"    Positions flipped ONLY in A vs B (genuine trigger signal): {unique_AB}")
        else:
            print(f"    No positions flip exclusively in A vs B → A≠B is explained by GAN noise.")

        summary_rows.append({
            "round": rnd_A,
            "mal_sampled": len(mal_in_sampled),
            "rate_AA'": rate_AAp,
            "rate_AB": rate_AB,
            "flipped_AA'": len(flipped_AAp),
            "flipped_AB": len(flipped_AB),
            "unique_AB": len(unique_AB),
        })

    print(f"\n\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'Round':>6}  {'Mal':>3}  {'flip_rate(A vs A\')':>18}  "
          f"{'flip_rate(A vs B)':>17}  {'unique(A vs B)':>14}")
    for r in summary_rows:
        print(f"  {r['round']:>4}  {r['mal_sampled']:>3}  "
              f"{r[\"rate_AA'\"]:.4f} ({r[\"flipped_AA'\"]} pos)       "
              f"{r['rate_AB']:.4f} ({r['flipped_AB']} pos)    "
              f"{r['unique_AB']} excl-AB positions")

    all_unique_AB = sum(r["unique_AB"] for r in summary_rows)
    print()
    if all_unique_AB == 0:
        print("VERDICT: All A≠B positions are also A≠A' positions.")
        print("  → The pipeline is noisy at late rounds; A≠B reflects GAN-init")
        print("    variance on boundary clients, NOT a trigger-swap signal.")
        print("  → Correct statement: 'at late rounds the detector output is unstable'.")
    else:
        n_genuine = all_unique_AB
        print(f"VERDICT: {n_genuine} client-positions flip exclusively under the trigger")
        print(f"  swap (A vs B) and are stable within A (A vs A' agrees).")
        print(f"  → Invariance has genuinely broken: B produces distinguishable")
        print(f"    classifications beyond what GAN-init noise can explain.")

    if _out_fh:
        _out_fh.close()


if __name__ == "__main__":
    main()
