"""Is T round-specific at all? (CODE_NOTES.md)

Stale and linear predictions were near-identical to oracle T in every round
tested -- great for the attacker, but raises the question of whether T is
largely determined by the model's general structure (attractor classes
stable across seeds/checkpoints, per earlier findings) rather than by the
round's specific incremental knowledge. Tests this directly by feeding
Stage 1/2 deliberately WRONG model pairs and checking whether the resulting
T still flags the SAME real attacker model with similar confidence.

Uses round 7's saved data (scripts/probe_fidelity_rounds.pt): real attacker
client 84, oracle result p=0 c=0.711 rank=1 (the reference to compare
against). Three wrong pairs:
  A) far pair:        (G_old(1), G_old(2))        -- from early in a DIFFERENT
                                                       8-round sequence, nowhere
                                                       near round 7
  B) unrelated pair:   (G_old(1) [this sequence],
                        global model from a totally different training run:
                        results/alpha1_match_trainprobe_ckpt.pt, epoch 24)
  C) degenerate pair: (G_old(7), G_old(7))          -- d1 == d2, identical
"""
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filterfl.config import Config
from filterfl.defense import run_detection, stage3_classify_clients
from filterfl.models import ResNet8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROBE_MODE = "train"


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def main():
    cfg = Config.from_yaml("configs/cifar_alpha1_side.yaml")
    num_classes, in_channels = 10, 3
    rounds = torch.load("scripts/probe_fidelity_rounds.pt", map_location=device, weights_only=False)
    r7 = next(r for r in rounds if r["t"] == 7)
    sampled, client_states = r7["sampled"], r7["client_states"]
    atk_cid = 84
    atk_idx = sampled.index(atk_cid)
    oracle_res = r7["oracle_results"][atk_idx]
    print(f"reference (oracle, round 7, client {atk_cid}): "
          f"p={oracle_res['pred_class']} c={oracle_res['confidence']:.3f}\n")

    r1 = next(r for r in rounds if r["t"] == 1)
    r2 = next(r for r in rounds if r["t"] == 2)
    g_old_1, g_old_2, g_old_7 = r1["ori_weight"], r2["ori_weight"], r7["ori_weight"]

    unrelated_ckpt = torch.load("results/alpha1_match_trainprobe_ckpt.pt", map_location=device, weights_only=False)
    unrelated_model = unrelated_ckpt["model_state"]

    betas_all1 = [1] * len(sampled)

    def test(label, d1, d2):
        seed_everything(777)
        det = run_detection(d1, d2, cfg, num_classes, in_channels, device)
        results = stage3_classify_clients(client_states, r7["ori_weight"], betas_all1, det.trigger_images,
                                           num_classes, in_channels, cfg.target_class, device,
                                           probe_mode=PROBE_MODE)
        atk = results[atk_idx]
        confs = [r["confidence"] for r in results]
        rank = sorted(range(len(confs)), key=lambda i: -confs[i]).index(atk_idx) + 1
        print(f"{label:>40}: p={atk['pred_class']} c={atk['confidence']:.3f} rank={rank}")

    print("Feeding round 7's REAL client models through T generated from WRONG (d1,d2) pairs:\n")
    test("A) far pair (G_old(1), G_old(2))", g_old_1, g_old_2)
    test("B) unrelated ckpt (G_old(1), other-run epoch24)", g_old_1, unrelated_model)
    test("C) degenerate (G_old(7), G_old(7))", g_old_7, g_old_7)


if __name__ == "__main__":
    main()
