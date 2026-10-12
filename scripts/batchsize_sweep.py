"""Batch-size dependence sweep.

Stage 3's probe runs BatchNorm in train mode (stage3_probe_mode="train"),
which renormalizes against whichever batch of trigger images it's fed. That
batch's size is `images_per_class`. If detection power depends heavily on
this hardcoded, undocumented value, FilterFL's defense is measuring "how do
this model's BN layers behave when renormalized against N synthetic images,"
not "does this model respond to the trigger" -- and N is an attack surface.

Reuses the EXACT fixed round from crossfeed_diagnostic.py
(scripts/crossfeed_round.pt: one real FedAvg round, 10 sampled clients, one
real attacker, client 47) so only images_per_class varies across the sweep.
"""
import random
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filterfl.config import Config
from filterfl.defense import run_detection, stage3_classify_clients

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES, IN_CHANNELS = 10, 3


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def main():
    saved = torch.load("scripts/crossfeed_round.pt", map_location=device, weights_only=False)
    ori_weight, new_state = saved["ori_weight"], saved["new_state"]
    client_states, sampled, malicious_sampled = saved["client_states"], saved["sampled"], saved["malicious_sampled"]
    betas = [1] * len(sampled)
    attacker_idx = [i for i, cid in enumerate(sampled) if cid in malicious_sampled]
    print(f"sampled: {sampled}\nmalicious: {malicious_sampled} (index {attacker_idx})\n")

    cfg = Config.from_yaml("configs/cifar_alpha1_side.yaml")

    rho = 0.5
    print(f"{'images_per_class':>18} | {'attacker pred':>14} | {'attacker conf':>14} | flagged? | "
          f"max benign conf (pred==0)")
    for ipc in [32, 16, 8, 4, 2, 1]:
        cfg.gan.images_per_class = ipc
        seed_everything(123)  # same GAN-training randomness seed at every sweep point
        det = run_detection(ori_weight, new_state, cfg, NUM_CLASSES, IN_CHANNELS, device)
        results = stage3_classify_clients(client_states, ori_weight, betas, det.trigger_images,
                                           NUM_CLASSES, IN_CHANNELS, cfg.target_class, device,
                                           probe_mode="train")
        atk = results[attacker_idx[0]]
        flagged = atk["pred_class"] == cfg.target_class and atk["confidence"] > rho
        benign_p0 = [r["confidence"] for i, r in enumerate(results)
                     if i not in attacker_idx and r["pred_class"] == cfg.target_class]
        max_benign = max(benign_p0) if benign_p0 else float("nan")
        print(f"{ipc:>18} | {atk['pred_class']:>14} | {atk['confidence']:>14.4f} | "
              f"{'YES' if flagged else 'no':>8} | {max_benign:.4f}")


if __name__ == "__main__":
    main()
