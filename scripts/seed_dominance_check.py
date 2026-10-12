"""Sanity check before trusting ANY degenerate-pair result (CODE_NOTES.md).

Addresses the critique directly: every prior comparison script reseeded to
the SAME fixed seed immediately before each run_detection call, specifically
"for a fair comparison." That eliminates the one thing (randomness) that
could reveal whether d1/d2 content matters at all. If identical seeds
produce identical-looking T regardless of d1/d2, that's exactly what you'd
ALSO see if the generator's output is seed-dominated and d1/d2 are being
ignored -- the fixed-seed methodology cannot distinguish "d1/d2 genuinely
don't matter" from "we handed every run the same random numbers."

Two checks, using two CLEARLY different, independently-verified model
states (not a clean-vs-random-init rebuild -- a random-init model from the
round-1 setup, and the ACTUAL post-30-epoch trained+defended checkpoint
from the real class7_test run, loaded straight off disk):

1. Hash/sum check: confirm the two d1/d2 candidates are actually numerically
   different before using them (point 4 -- are d1/d2 even being passed
   through, or is some reused-object bug making them the same tensors).
2. For EACH of the two model-pairs (degenerate: d1=d2=random-init;
   degenerate: d1=d2=real-trained), run Stage 1/2 THREE TIMES with three
   DIFFERENT seeds (123, 456, 789) for the GAN training randomness itself
   (data/model construction already fixed beforehand, unaffected). Compare
   trigger-image stats and Stage 3 confidence on the SAME real attacker
   client model across all 6 runs.

If confidence/T barely move across DIFFERENT seeds for the SAME d1/d2
pair -> genuine seed-independent convergence (strengthens the original
reading). If confidence/T vary a lot across seeds -> the earlier same-seed
comparisons were measuring "same seed" not "same d1/d2," and the whole
degenerate-pair line of findings needs to be rerun with proper seed
variation before drawing any conclusion.
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


def state_fingerprint(state_dict):
    total = 0.0
    for v in state_dict.values():
        if v.dtype.is_floating_point:
            total += v.double().sum().item()
    return total


def main():
    cfg = Config.from_yaml("configs/cifar_alpha1_side_class7.yaml")
    assert cfg.target_class == 7

    # ---- rebuild the round-1 attack setup (random-init G_old(1) + real attacker client) ----
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
    assert malicious_sampled == [47], f"expected attacker 47, got {malicious_sampled}"

    random_init_state = {k: v.clone() for k, v in model.state_dict().items()}  # G_old(1)
    client_states = []
    for cid in sampled:
        m = copy.deepcopy(model)
        w = local_train(m, client_loaders[cid], cfg, device,
                         is_malicious=(cid in malicious_sampled), attacker=attacker)
        client_states.append(w)
    betas = [1] * len(sampled)
    attacker_idx = sampled.index(47)

    # ---- load the ACTUAL post-30-epoch trained checkpoint from the real class7_test run ----
    ckpt = torch.load("results/class7_test_ckpt.pt", map_location=device, weights_only=False)
    trained_state = ckpt["model_state"]

    # ---- point 4: confirm these are actually numerically different ----
    fp_random = state_fingerprint(random_init_state)
    fp_trained = state_fingerprint(trained_state)
    print(f"fingerprint(random-init G_old(1))      = {fp_random:.6f}")
    print(f"fingerprint(trained class7_test ckpt)   = {fp_trained:.6f}")
    print(f"these are {'DIFFERENT' if abs(fp_random - fp_trained) > 1e-3 else 'SUSPICIOUSLY IDENTICAL'}\n")

    pairs = {
        "degenerate(random-init, random-init)": (random_init_state, random_init_state),
        "degenerate(trained-ckpt, trained-ckpt)": (trained_state, trained_state),
    }
    seeds = [123, 456, 789]

    for label, (d1, d2) in pairs.items():
        print(f"=== {label} ===")
        for s in seeds:
            seed_everything(s)
            det = run_detection(d1, d2, cfg, num_classes, in_channels, device)
            results = stage3_classify_clients(client_states, random_init_state, betas, det.trigger_images,
                                               num_classes, in_channels, cfg.target_class, device,
                                               probe_mode=PROBE_MODE)
            atk = results[attacker_idx]
            t_mean, t_std = det.trigger_images.mean().item(), det.trigger_images.std().item()
            print(f"  seed={s}: T mean={t_mean:.4f} std={t_std:.4f} | "
                  f"attacker pred={atk['pred_class']} conf={atk['confidence']:.4f}")
        print()


if __name__ == "__main__":
    main()
