"""Generate a clean early-round reference model via plain FedAvg (no poisoning).

Saves a checkpoint compatible with checkpoint.load(). Intended as the d1_state
input for ref_swap_diagnostic.py — a stand-in for the arm A global model at
an early round, before backdoor contamination establishes.

Usage:
    python -m scripts.gen_clean_ref \
        --config configs/cifar_primary.yaml \
        --rounds 50 \
        --out results/clean_ref_e50_ckpt.pt
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
from filterfl.data import (load_dataset, build_class_index, dirichlet_partition,
                            make_client_loader, make_test_loader)
from filterfl.models import build_model
from filterfl.client import local_train, evaluate
from filterfl.aggregator import aggregate
from filterfl import checkpoint


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rounds", type=int, default=50,
                    help="number of clean FedAvg rounds (default 50)")
    ap.add_argument("--out", required=True, help="output checkpoint path")
    ap.add_argument("--seed", type=int, default=None,
                    help="override config seed (default: use config seed)")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.is_poison = False
    if args.seed is not None:
        cfg.seed = args.seed

    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  Seed: {cfg.seed}  Rounds: {args.rounds}")

    train_set, test_set, num_classes, in_channels = load_dataset(cfg.type)
    class_index = build_class_index(train_set)
    per_participant, _ = dirichlet_partition(
        train_set, cfg.number_of_total_participants, cfg.dirichlet_alpha, class_index)
    client_loaders = {p: make_client_loader(train_set, idxs, cfg.batch_size)
                      for p, idxs in per_participant.items()}
    test_loader = make_test_loader(test_set, cfg.batch_size)

    participants = list(range(cfg.number_of_total_participants))
    model = build_model(cfg.type, num_classes=num_classes).to(device)

    lr = cfg.lr
    for epoch in range(1, args.rounds + 1):
        if cfg.lr_decay and epoch % cfg.lr_decay_epoch == 0:
            lr *= cfg.lr_decay_gamma

        sampled = random.sample(participants, cfg.num_clients)
        ori_weight = {k: v.clone() for k, v in model.state_dict().items()}
        client_states = []
        for cid in sampled:
            m = copy.deepcopy(model)
            w = local_train(m, client_loaders[cid], cfg, device,
                            is_malicious=False, attacker=None)
            client_states.append(w)

        betas = [1] * len(sampled)
        new_state = aggregate(ori_weight, client_states, betas, cfg.eta,
                              denominator=getattr(cfg, "aggregation_denominator", "survivors"))
        model.load_state_dict(new_state)

        if epoch % 10 == 0 or epoch == args.rounds:
            ma = evaluate(model, test_loader, device)
            print(f"  round {epoch:4d}: MA={ma:.2f}%")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    checkpoint.save(args.out, args.rounds, model, lr, cfg.poison_lr,
                    adversarial_namelist=[], trust_scores={}, last_trigger_images=None)
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
