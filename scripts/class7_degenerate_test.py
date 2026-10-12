"""Class-7 generalization of the degenerate-pair test (CODE_NOTES.md).

Everything in the earlier "is T round-specific" test was at target_class=0,
a moderate Stage-2 attractor (per the earlier sink sweep). This repeats the
degenerate-pair check ((G_old(t), G_old(t)) as d1/d2, zero information about
the round's incremental update) at target_class=7, a zero-sink class, to
rule out the earlier "T isn't round-specific" finding being partly circular
(target happening to coincide with where Stage 2 goes by default anyway).

One round from scratch (same pattern as crossfeed_diagnostic.py): report
the real attacker's oracle Stage 3 result AND the degenerate-pair result,
AND the full predicted-class distribution across all 10 sampled clients for
both, to see directly whether T still predicts 0 (Stage 1/2 recovering
something unrelated to the planted backdoor) or correctly predicts 7.
"""
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


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def main():
    cfg = Config.from_yaml("configs/cifar_alpha1_side_class7.yaml")
    assert cfg.target_class == 7
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

    sampled = random.sample(participants, cfg.num_clients)
    malicious_sampled = [c for c in sampled if c in adversarial_namelist]
    print(f"sampled: {sampled}\nmalicious (ground truth): {malicious_sampled}\n")
    if not malicious_sampled:
        print("no attacker in round 1 at this seed -- rerun with a different round/seed")
        return

    ori_weight = {k: v.clone() for k, v in model.state_dict().items()}
    client_states = []
    for cid in sampled:
        m = copy.deepcopy(model)
        w = local_train(m, client_loaders[cid], cfg, device,
                         is_malicious=(cid in malicious_sampled), attacker=attacker)
        client_states.append(w)
    betas = [1] * len(sampled)
    new_state = aggregate(ori_weight, client_states, betas, cfg.eta, denominator="survivors")

    def report(label, d1, d2):
        seed_everything(123)
        det = run_detection(d1, d2, cfg, num_classes, in_channels, device)
        results = stage3_classify_clients(client_states, ori_weight, betas, det.trigger_images,
                                           num_classes, in_channels, cfg.target_class, device,
                                           probe_mode=PROBE_MODE)
        print(f"=== {label} ===")
        for i, cid in enumerate(sampled):
            tag = "ATTACKER" if cid in malicious_sampled else ""
            r = results[i]
            flagged = r["pred_class"] == cfg.target_class and r["confidence"] > 0.5
            print(f"  client {cid:>3} {tag:>8}: pred={r['pred_class']} conf={r['confidence']:.3f} "
                  f"{'FLAGGED' if flagged else ''}")
        preds = [r["pred_class"] for r in results]
        print(f"  predicted-class distribution: {sorted(preds)}\n")

    report("ORACLE  (G_old(1), G_temp(1))", ori_weight, new_state)
    report("DEGENERATE  (G_old(1), G_old(1))", ori_weight, ori_weight)


if __name__ == "__main__":
    main()
