"""Full-vector strengthening of the degenerate-pair / seed-dominance check
(CODE_NOTES.md). A single confidence scalar with a ~0.004 seed-noise floor
is thin evidence for "d1/d2 don't matter." This reports, for 3 discriminator
pairs x 3 GAN-training seeds (9 runs total), the FULL predicted-class vector
and confidence vector across all 10 sampled clients, plus the flagged/not
decision per client -- so stability can be judged on the detector's whole
output, not one number.

Three pairs:
  A) random-init degenerate: d1=d2=G_old(1), before any training at all
  B) clean-trained degenerate: d1=d2=a model trained for 10 rounds of plain
     FedAvg with NO poisoning EVER introduced (genuinely clean, not reused
     from the real attack+defense run, which did see poison attempts even
     if filtered)
  C) true oracle (non-degenerate): d1=G_old(1), d2=G_temp(1), the REAL raw
     unfiltered aggregate from the round-1 attack setup

Same round-1 attack setup throughout (client 47 = real attacker) so all 9
runs classify the SAME 10 client models; only T (and the seed that produced
it) varies.
"""
import argparse
import copy
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filterfl.config import Config
from filterfl.data import load_dataset, build_class_index, dirichlet_partition, make_client_loader
from filterfl.models import build_model
from filterfl.attacker import PixelPatternAttacker
from filterfl.client import local_train
from filterfl.aggregator import aggregate
from filterfl.defense import run_detection, stage3_classify_clients

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROBE_MODE = "train"
CLEAN_PRETRAIN_ROUNDS = 10
SEEDS = [123, 456, 789]


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cifar_alpha1_side_class7.yaml")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    print(f"config: {args.config}  dirichlet_alpha={cfg.dirichlet_alpha}  target_class={cfg.target_class}\n")
    assert cfg.target_class == 7

    # ---- clean model: 10 rounds of plain FedAvg, NEVER sampling an adversarial client ----
    seed_everything(cfg.seed)
    train_set, test_set, num_classes, in_channels = load_dataset(cfg.type)
    class_index = build_class_index(train_set)
    per_participant, _ = dirichlet_partition(train_set, cfg.number_of_total_participants,
                                              cfg.dirichlet_alpha, class_index)
    client_loaders = {p: make_client_loader(train_set, idxs, cfg.batch_size)
                       for p, idxs in per_participant.items()}
    participants = list(range(cfg.number_of_total_participants))
    adversarial_namelist = set(random.sample(participants, cfg.poison_client_num))
    benign_namelist = [p for p in participants if p not in adversarial_namelist]

    clean_model = build_model(cfg.type, num_classes=num_classes).to(device)
    attacker_for_clean = PixelPatternAttacker(cfg)
    for _ in range(CLEAN_PRETRAIN_ROUNDS):
        sampled_pre = random.sample(benign_namelist, cfg.num_clients)
        ori = {k: v.clone() for k, v in clean_model.state_dict().items()}
        states = [local_train(copy.deepcopy(clean_model), client_loaders[cid], cfg, device,
                               is_malicious=False, attacker=attacker_for_clean)
                  for cid in sampled_pre]
        new = aggregate(ori, states, [1] * len(sampled_pre), cfg.eta, denominator="survivors")
        clean_model.load_state_dict(new)
    clean_state = {k: v.clone() for k, v in clean_model.state_dict().items()}

    # ---- reconstruct the round-1 attack setup (fresh reseed, identical sequence) ----
    seed_everything(cfg.seed)
    train_set2, test_set2, num_classes2, in_channels2 = load_dataset(cfg.type)
    class_index2 = build_class_index(train_set2)
    per_participant2, _ = dirichlet_partition(train_set2, cfg.number_of_total_participants,
                                               cfg.dirichlet_alpha, class_index2)
    client_loaders2 = {p: make_client_loader(train_set2, idxs, cfg.batch_size)
                        for p, idxs in per_participant2.items()}
    participants2 = list(range(cfg.number_of_total_participants))
    adversarial_namelist2 = set(random.sample(participants2, cfg.poison_client_num))
    assert adversarial_namelist2 == adversarial_namelist

    attack_model = build_model(cfg.type, num_classes=num_classes2).to(device)
    attacker2 = PixelPatternAttacker(cfg)
    sampled = random.sample(participants2, cfg.num_clients)
    malicious_sampled = [c for c in sampled if c in adversarial_namelist2]
    assert malicious_sampled == [47], f"expected attacker 47, got {malicious_sampled}"

    ori_weight = {k: v.clone() for k, v in attack_model.state_dict().items()}  # G_old(1)
    client_states = []
    for cid in sampled:
        m = copy.deepcopy(attack_model)
        w = local_train(m, client_loaders2[cid], cfg, device,
                         is_malicious=(cid in malicious_sampled), attacker=attacker2)
        client_states.append(w)
    betas = [1] * len(sampled)
    new_state = aggregate(ori_weight, client_states, betas, cfg.eta, denominator="survivors")  # G_temp(1), true oracle d2

    pairs = {
        "A) random-init degenerate  ": (ori_weight, ori_weight),
        "B) clean-trained degenerate": (clean_state, clean_state),
        "C) TRUE ORACLE (non-degen) ": (ori_weight, new_state),
    }

    attacker_idx = sampled.index(47)
    rho = 0.5
    print(f"sampled clients (column order below): {sampled}")
    print(f"attacker = client 47, at index {attacker_idx}\n")

    all_flagged_sets = []
    for label, (d1, d2) in pairs.items():
        for s in SEEDS:
            seed_everything(s)
            det = run_detection(d1, d2, cfg, num_classes, in_channels, device)
            results = stage3_classify_clients(client_states, ori_weight, betas, det.trigger_images,
                                               num_classes, in_channels, cfg.target_class, device,
                                               probe_mode=PROBE_MODE)
            preds = [r["pred_class"] for r in results]
            confs = [r["confidence"] for r in results]
            flagged = [sampled[i] for i, r in enumerate(results)
                       if r["pred_class"] == cfg.target_class and r["confidence"] > rho]
            all_flagged_sets.append((label, s, tuple(flagged)))
            print(f"{label} seed={s}")
            print(f"  preds : {preds}")
            print(f"  confs : {[round(c, 3) for c in confs]}")
            print(f"  flagged clients: {flagged}\n")

    print("=== flagged-set summary across all 9 runs ===")
    flagged_sets_only = set(fs for _, _, fs in all_flagged_sets)
    for label, s, fs in all_flagged_sets:
        print(f"  {label} seed={s}: {fs}")
    print(f"\n distinct flagged-sets seen: {len(flagged_sets_only)} -> {flagged_sets_only}")


if __name__ == "__main__":
    main()
