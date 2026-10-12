"""Does the degenerate pair's success depend on d1/d2 carrying the backdoor
at all, or is it independent of model content entirely? (CODE_NOTES.md)

The original degenerate test used (G_old(1), G_old(1)) -- the model BEFORE
round 1 training, i.e. random init, never having seen any data at all, let
alone poisoned data. It still produced a T that correctly flagged the real
attacker at class 7. Two candidate explanations:
  (a) Stage 1/2's generation process doesn't need d1/d2 to carry the
      backdoor -- any model (even random noise) produces a T that works,
      because the real "inversion" happens against whatever model Stage 3
      later classifies (the backdoored attacker), not against d1/d2 at all
      ("single-model trigger inversion": Stage 3's probe model itself does
      the work, T's specific content barely matters).
  (b) something very odd -- T is landing on class 7 independent of ANY
      model's content, a pure architectural/loss-landscape artifact.

Test: train a model via SEVERAL ROUNDS of plain FedAvg with NO poisoning
EVER introduced (a genuinely clean, reasonably-trained classifier, not
random noise), use it as the degenerate (d1, d2) pair, generate T from it,
then feed that T to the SAME backdoored attacker model from the round-1
class-7 test (scripts/class7_degenerate_test.py's round, reconstructed
identically here). If T from the clean model still flags class 7 on the
attacker -> consistent with (a), strengthens "single-model inversion."
If it does NOT -> (b), something odder, worth a closer look.
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
CLEAN_PRETRAIN_ROUNDS = 10


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def main():
    cfg = Config.from_yaml("configs/cifar_alpha1_side_class7.yaml")
    assert cfg.target_class == 7

    # ---- Part 1: train a genuinely clean model, no poisoning ever ----
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
    attacker = PixelPatternAttacker(cfg)
    for r in range(CLEAN_PRETRAIN_ROUNDS):
        sampled_pre = random.sample(benign_namelist, cfg.num_clients)  # NEVER an adversarial client
        ori = {k: v.clone() for k, v in clean_model.state_dict().items()}
        states = [local_train(copy.deepcopy(clean_model), client_loaders[cid], cfg, device,
                               is_malicious=False, attacker=attacker)
                  for cid in sampled_pre]
        new = aggregate(ori, states, [1] * len(sampled_pre), cfg.eta, denominator="survivors")
        clean_model.load_state_dict(new)
    clean_state = {k: v.clone() for k, v in clean_model.state_dict().items()}
    print(f"clean model trained for {CLEAN_PRETRAIN_ROUNDS} rounds, NEVER sampled an adversarial client\n")

    # ---- Part 2: reconstruct the EXACT round-1 class-7 attack setup (fresh reseed) ----
    seed_everything(cfg.seed)
    # must redo load_dataset/partition/loaders too: dirichlet_partition consumes
    # np.random state, and we need the identical partition/sampling sequence
    train_set2, test_set2, num_classes2, in_channels2 = load_dataset(cfg.type)
    class_index2 = build_class_index(train_set2)
    per_participant2, _ = dirichlet_partition(train_set2, cfg.number_of_total_participants,
                                               cfg.dirichlet_alpha, class_index2)
    client_loaders2 = {p: make_client_loader(train_set2, idxs, cfg.batch_size)
                        for p, idxs in per_participant2.items()}
    participants2 = list(range(cfg.number_of_total_participants))
    adversarial_namelist2 = set(random.sample(participants2, cfg.poison_client_num))
    assert adversarial_namelist2 == adversarial_namelist, "reseed did not reproduce the same adversarial namelist"

    attack_model = build_model(cfg.type, num_classes=num_classes2).to(device)
    attacker2 = PixelPatternAttacker(cfg)
    sampled = random.sample(participants2, cfg.num_clients)
    malicious_sampled = [c for c in sampled if c in adversarial_namelist2]
    print(f"round-1 attack setup: sampled={sampled} malicious={malicious_sampled}")
    assert malicious_sampled == [47], f"expected the same round-1 attacker (47), got {malicious_sampled}"

    ori_weight = {k: v.clone() for k, v in attack_model.state_dict().items()}
    client_states = []
    for cid in sampled:
        m = copy.deepcopy(attack_model)
        w = local_train(m, client_loaders2[cid], cfg, device,
                         is_malicious=(cid in malicious_sampled), attacker=attacker2)
        client_states.append(w)
    betas = [1] * len(sampled)
    attacker_idx = sampled.index(47)

    # ---- Part 3: degenerate pair from the CLEAN model -> T_clean ----
    seed_everything(123)
    det_clean = run_detection(clean_state, clean_state, cfg, num_classes, in_channels, device)

    # ---- also redo the original (G_old(1), G_old(1)) degenerate pair here, as a same-script control ----
    seed_everything(123)
    det_original = run_detection(ori_weight, ori_weight, cfg, num_classes, in_channels, device)

    def report(label, det):
        results = stage3_classify_clients(client_states, ori_weight, betas, det.trigger_images,
                                           num_classes, in_channels, cfg.target_class, device,
                                           probe_mode=PROBE_MODE)
        atk = results[attacker_idx]
        flagged = atk["pred_class"] == cfg.target_class and atk["confidence"] > 0.5
        preds = sorted(r["pred_class"] for r in results)
        print(f"{label}: attacker pred={atk['pred_class']} conf={atk['confidence']:.3f} "
              f"{'FLAGGED' if flagged else 'not flagged'} | all preds: {preds}")

    print()
    report("T from ORIGINAL degenerate pair (G_old(1), G_old(1), random init)      ", det_original)
    report(f"T from CLEAN-MODEL degenerate pair ({CLEAN_PRETRAIN_ROUNDS}-round trained, never poisoned)", det_clean)


if __name__ == "__main__":
    main()
