"""Is the degenerate-pair attractor the same basin across rounds, or
round-dependent? (CODE_NOTES.md, task 3: unresolved, flagged as cheap to
check.)

Runs plain FedAvg (with the usual pixel-pattern attack active, matching
existing degenerate-pair test conditions) for several rounds, saving G_old
(the pre-aggregation global model) at each round. At 5-6 spread-out rounds,
runs the degenerate-pair detector (d1=d2=G_old(round), zero information
about that round's incremental update) THREE TIMES with different
GAN-training seeds, classifying that round's actual sampled clients in
Stage 3 (train-mode probe, matching the reference's shipped behaviour).

Reports, per round: the per-client predicted-class vector for each of the 3
seeds (so seed-stability WITHIN a round can be judged directly) and the
modal ("attractor") class. Comparing the attractor class across rounds
answers whether it's one fixed basin throughout training or something that
moves as G_old itself evolves.

Run from the repo root (D:\\filterfl_test_claude).
"""
import copy
import os
import random
import sys
from collections import Counter

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
RHO = 0.5
GAN_SEEDS = [123, 456, 789]
TOTAL_ROUNDS = 13
CHECK_ROUNDS = {1, 4, 7, 10, 13}


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def main():
    cfg = Config.from_yaml("configs/cifar_alpha1_side.yaml")
    seed_everything(cfg.seed)

    train_set, test_set, num_classes, in_channels = load_dataset(cfg.type)
    class_index = build_class_index(train_set)
    per_participant, _ = dirichlet_partition(train_set, cfg.number_of_total_participants,
                                              cfg.dirichlet_alpha, class_index)
    client_loaders = {p: make_client_loader(train_set, idxs, cfg.batch_size)
                       for p, idxs in per_participant.items()}

    participants = list(range(cfg.number_of_total_participants))
    adversarial_namelist = set(random.sample(participants, cfg.poison_client_num))

    model = build_model(cfg.type, num_classes=num_classes).to(device)
    attacker = PixelPatternAttacker(cfg)

    round_records = []  # (round, sampled, malicious_sampled, ori_weight)

    for rnd in range(1, TOTAL_ROUNDS + 1):
        sampled = random.sample(participants, cfg.num_clients)
        malicious_sampled = [c for c in sampled if c in adversarial_namelist]
        ori_weight = {k: v.clone() for k, v in model.state_dict().items()}  # G_old(rnd)

        client_states = []
        for cid in sampled:
            m = copy.deepcopy(model)
            w = local_train(m, client_loaders[cid], cfg, device,
                             is_malicious=(cid in malicious_sampled), attacker=attacker)
            client_states.append(w)
        betas = [1] * len(sampled)
        new_state = aggregate(ori_weight, client_states, betas, cfg.eta, denominator="survivors")
        model.load_state_dict(new_state)  # advance global model for the next round

        print(f"round {rnd}: sampled={sampled} malicious={malicious_sampled}")

        if rnd in CHECK_ROUNDS:
            round_records.append((rnd, sampled, malicious_sampled, ori_weight, client_states, betas))

    print("\n########## Degenerate-pair attractor check at each round, 3 GAN seeds each ##########")
    attractor_per_round = {}
    for rnd, sampled, malicious_sampled, ori_weight, client_states, betas in round_records:
        print(f"\n=== round {rnd}  (sampled={sampled}, malicious={malicious_sampled}) ===")
        seed_pred_vectors = []
        for s in GAN_SEEDS:
            seed_everything(s)
            det = run_detection(ori_weight, ori_weight, cfg, num_classes, in_channels, device)  # degenerate pair
            results = stage3_classify_clients(client_states, ori_weight, betas, det.trigger_images,
                                               num_classes, in_channels, cfg.target_class, device,
                                               probe_mode=PROBE_MODE)
            preds = [r["pred_class"] for r in results]
            confs = [round(r["confidence"], 3) for r in results]
            flagged = [sampled[i] for i, r in enumerate(results)
                       if r["pred_class"] == cfg.target_class and r["confidence"] > RHO]
            seed_pred_vectors.append(tuple(preds))
            print(f"  seed={s}: preds={preds}")
            print(f"           confs={confs}")
            print(f"           flagged={flagged}")
        modal_class = Counter(c for vec in seed_pred_vectors for c in vec).most_common(1)[0][0]
        seed_stable = len(set(seed_pred_vectors)) == 1
        attractor_per_round[rnd] = (modal_class, seed_stable, seed_pred_vectors)
        print(f"  --> round {rnd} modal predicted class = {modal_class}, "
              f"seed-stable across all 3 seeds: {seed_stable}")

    print("\n########## SUMMARY ##########")
    print(f"{'round':>6} {'modal_class':>12} {'seed_stable':>12}")
    for rnd, (modal_class, seed_stable, _) in attractor_per_round.items():
        print(f"{rnd:>6} {modal_class:>12} {str(seed_stable):>12}")
    modal_classes = [v[0] for v in attractor_per_round.values()]
    print(f"\ndistinct modal classes across rounds: {sorted(set(modal_classes))}")
    if len(set(modal_classes)) == 1:
        print("SAME basin across all checked rounds.")
    else:
        print("basin CHANGES across rounds -- round-dependent, not a single fixed attractor.")


if __name__ == "__main__":
    main()
