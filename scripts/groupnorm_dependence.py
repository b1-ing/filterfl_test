"""GroupNorm ablation (CODE_NOTES.md): isolates "no batch-dependent
renormalization to exploit" from "ad hoc net is just a worse classifier."

ResBlockNoNorm (previous ablation) has zero normalization and is known to
train worse than a BatchNorm net from the same random init over few steps --
confounding "detection failed" with "this net just isn't as good." GroupNorm
normalizes per-SAMPLE (no batch dependence, no train/eval distinction) and
trains comparably to BatchNorm. If GroupNorm reaches similar clean accuracy
to BatchNorm but Stage 3 confidence still stays low, that cleanly isolates
the mechanism as "needs batch-dependent renormalization specifically," not
just "needs a well-trained net."

Pretrains each architecture with the SAME number of benign-only FedAvg
rounds (comparable accuracy, not round-1-from-scratch) before running the
one attack round + Stage 1/2/3, for ResNet8 (BatchNorm control, re-run with
matched pretraining rather than reusing the earlier from-scratch numbers),
ResNet8GroupNorm, and ResNet8NoNorm (same pretraining budget, for a full
three-way comparison).
"""
import copy
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filterfl.config import Config
from filterfl.data import load_dataset, build_class_index, dirichlet_partition, make_client_loader, make_test_loader
from filterfl.models import ResNet8, ResNet8NoNorm, ResNet8GroupNorm
from filterfl.attacker import PixelPatternAttacker
from filterfl.client import local_train, evaluate
from filterfl.aggregator import aggregate
from filterfl.defense import run_detection, stage3_classify_clients

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAIN_ROUNDS = 15


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def run_for_architecture(model_cls, label):
    cfg = Config.from_yaml("configs/cifar_alpha1_side.yaml")
    seed_everything(cfg.seed)

    train_set, test_set, num_classes, in_channels = load_dataset(cfg.type)
    class_index = build_class_index(train_set)
    per_participant, _ = dirichlet_partition(train_set, cfg.number_of_total_participants,
                                              cfg.dirichlet_alpha, class_index)
    client_loaders = {p: make_client_loader(train_set, idxs, cfg.batch_size)
                       for p, idxs in per_participant.items()}
    test_loader = make_test_loader(test_set, cfg.batch_size)

    participants = list(range(cfg.number_of_total_participants))
    adversarial_namelist = set(random.sample(participants, cfg.poison_client_num))

    model = model_cls(num_classes=num_classes, in_channels=in_channels).to(device)
    attacker = PixelPatternAttacker(cfg)

    # pretrain: PRETRAIN_ROUNDS of plain benign-only FedAvg, no poisoning, no detection
    for _ in range(PRETRAIN_ROUNDS):
        sampled_pre = random.sample([p for p in participants if p not in adversarial_namelist], cfg.num_clients)
        ori = {k: v.clone() for k, v in model.state_dict().items()}
        states = []
        for cid in sampled_pre:
            m = copy.deepcopy(model)
            w = local_train(m, client_loaders[cid], cfg, device, is_malicious=False, attacker=attacker)
            states.append(w)
        new = aggregate(ori, states, [1] * len(sampled_pre), cfg.eta, denominator="survivors")
        model.load_state_dict(new)

    pretrain_acc = evaluate(model, test_loader, device)

    # the one attack round
    sampled = random.sample(participants, cfg.num_clients)
    malicious_sampled = [c for c in sampled if c in adversarial_namelist]
    ori_weight = {k: v.clone() for k, v in model.state_dict().items()}
    client_states = []
    for cid in sampled:
        m = copy.deepcopy(model)
        w = local_train(m, client_loaders[cid], cfg, device,
                         is_malicious=(cid in malicious_sampled), attacker=attacker)
        client_states.append(w)
    betas = [1] * len(sampled)
    new_state = aggregate(ori_weight, client_states, betas, cfg.eta, denominator="survivors")
    probe_for_acc = model_cls(num_classes=num_classes, in_channels=in_channels).to(device)
    probe_for_acc.load_state_dict(new_state)
    post_attack_acc = evaluate(probe_for_acc, test_loader, device)

    print(f"\n=== {label} ===")
    print(f"sampled: {sampled}  malicious: {malicious_sampled}")
    print(f"pretrain clean acc (after {PRETRAIN_ROUNDS} benign rounds): {pretrain_acc:.2f}%")
    print(f"post-attack-round global clean acc: {post_attack_acc:.2f}%")

    if not malicious_sampled:
        print("no attacker sampled this round for this architecture's RNG stream -- skipping Stage1/2/3")
        return

    atk_idx = sampled.index(malicious_sampled[0])
    rho = 0.5
    for probe_mode in ("train", "eval"):
        seed_everything(123)
        det = run_detection(ori_weight, new_state, cfg, num_classes, in_channels, device, model_cls=model_cls)
        results = stage3_classify_clients(client_states, ori_weight, betas, det.trigger_images,
                                           num_classes, in_channels, cfg.target_class, device,
                                           probe_mode=probe_mode, model_cls=model_cls)
        atk = results[atk_idx]
        flagged = atk["pred_class"] == cfg.target_class and atk["confidence"] > rho
        print(f"  probe_mode={probe_mode:>5}: attacker pred={atk['pred_class']} "
              f"conf={atk['confidence']:.4f} flagged={'YES' if flagged else 'no'}")


if __name__ == "__main__":
    run_for_architecture(ResNet8, "ResNet8 (BatchNorm, matched pretraining)")
    run_for_architecture(ResNet8GroupNorm, "ResNet8GroupNorm")
    run_for_architecture(ResNet8NoNorm, "ResNet8NoNorm (matched pretraining)")
