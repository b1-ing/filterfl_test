"""Architecture-dependence ablation (CODE_NOTES.md): if Stage 3's detection
power comes from BatchNorm renormalizing against the trigger batch (train
mode), a model with NO BatchNorm layers at all has nothing to renormalize.
Rebuilds the identical federated round (same seed, same participants, same
sampled clients/attacker as crossfeed_diagnostic.py's fixed round -- model
architecture doesn't affect the `random` module's state, only torch's, so
client sampling is unaffected) using ResNet8NoNorm instead of ResNet8, then
runs the full Stage 1/2/3 pipeline on it.
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
from filterfl.models import ResNet8NoNorm
from filterfl.attacker import PixelPatternAttacker
from filterfl.client import local_train
from filterfl.aggregator import aggregate
from filterfl.defense import run_detection, stage3_classify_clients

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    model = ResNet8NoNorm(num_classes=num_classes, in_channels=in_channels).to(device)
    attacker = PixelPatternAttacker(cfg)

    sampled = random.sample(participants, cfg.num_clients)
    malicious_sampled = [c for c in sampled if c in adversarial_namelist]
    print(f"sampled: {sampled}")
    print(f"malicious (ground truth): {malicious_sampled}")
    assert malicious_sampled == [47], (
        "expected the same round as crossfeed_diagnostic.py (attacker=47) -- "
        f"got {malicious_sampled}; model architecture should not perturb "
        "the `random` module's state used for client sampling, investigate if this fires"
    )

    ori_weight = {k: v.clone() for k, v in model.state_dict().items()}
    client_states = []
    for cid in sampled:
        m = copy.deepcopy(model)
        w = local_train(m, client_loaders[cid], cfg, device,
                         is_malicious=(cid in malicious_sampled), attacker=attacker)
        client_states.append(w)

    betas = [1] * len(sampled)
    new_state = aggregate(ori_weight, client_states, betas, cfg.eta, denominator="survivors")

    attacker_idx = sampled.index(47)
    rho = 0.5

    def report(label, probe_mode):
        seed_everything(123)
        det = run_detection(ori_weight, new_state, cfg, num_classes, in_channels, device,
                             model_cls=ResNet8NoNorm)
        results = stage3_classify_clients(client_states, ori_weight, betas, det.trigger_images,
                                           num_classes, in_channels, cfg.target_class, device,
                                           probe_mode=probe_mode, model_cls=ResNet8NoNorm)
        atk = results[attacker_idx]
        flagged = atk["pred_class"] == cfg.target_class and atk["confidence"] > rho
        benign = [r["confidence"] for i, r in enumerate(results)
                  if i != attacker_idx and r["pred_class"] == cfg.target_class]
        max_benign = max(benign) if benign else float("nan")
        print(f"{label:>30} | attacker pred={atk['pred_class']} conf={atk['confidence']:.4f} "
              f"flagged={'YES' if flagged else 'no':>3} | max benign(pred=0) conf={max_benign:.4f}")

    print()
    report("ResNet8NoNorm, probe_mode=train", "train")
    report("ResNet8NoNorm, probe_mode=eval", "eval")


if __name__ == "__main__":
    main()
