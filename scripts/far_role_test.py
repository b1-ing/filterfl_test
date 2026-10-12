"""Discriminating test for which discriminator (d1 or d2) drives the
degenerate-pair attractor's basin, at the SAME distance (rounds 1 vs 13)
where the round-dependence result actually showed movement.

scripts/d1_d2_role_test.py already ran this A/B/C/D design one FedAvg round
apart and found no effect from either model identity or role -- but that
distance was too short to move the basin at all (round-dependence was only
established at the widely-spaced 1/4/7/10/13 scale). This reruns the exact
same design at the 1-vs-13 distance, where the degenerate cases are known
to land on different modal classes (round 1 -> 9, round 13 -> 2, per
CODE_NOTES.md), so the mixed-pair conditions (C, D) can actually
discriminate which input the output tracks:

  A) degenerate @ round 1:  d1 = d2 = G_old(1)
  B) degenerate @ round 13: d1 = d2 = G_old(13)
  C) normal order:          d1 = G_old(1),  d2 = G_old(13)
  D) swapped order:         d1 = G_old(13), d2 = G_old(1)

The paper's design treats d1 ("what the model already knew") and d2 ("what
it just learned") as functionally asymmetric. If C tracks A (i.e. behaves
like pure G_old(1)) and D tracks B (pure G_old(13)), the output is being
driven by WHICHEVER model is d1 -- naturalness/d1 dominates. If C tracks B
and D tracks A, d2 (the target-bias term) dominates instead. If C and D
both land somewhere between A and B (or C==D, matching neither single-model
case cleanly), that's evidence of a genuine blend or of some other dynamic
entirely. If the design's stated asymmetry is actually functioning, C and D
should clearly differ from each other and each should resemble one of
A/B specifically.

Regenerates the SAME deterministic 13-round FedAvg sequence as
scripts/attractor_rounds.py (identical seeding, identical config) to get
real G_old(1)/G_old(13) weight tensors -- that script only printed summary
stats, it didn't persist the checkpoints. Classifies round 1's own sampled
client set (includes the real attacker, client 47) under all four
conditions, so there's one fixed probe target throughout, exactly as
scripts/d1_d2_role_test.py did at the adjacent distance.

Run from the repo root (D:\\filterfl_test_claude).
"""
import copy
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference", "filterfl"))

from filterfl.config import Config
from filterfl.data import load_dataset, build_class_index, dirichlet_partition, make_client_loader
from filterfl.models import build_model
from filterfl.attacker import PixelPatternAttacker
from filterfl.client import local_train
from filterfl.aggregator import aggregate

import gan as ref_gan
from models.ResNet8 import ResNet8 as RefResNet8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_CLASS = 0
RHO = 0.5
SEED = 123
TOTAL_ROUNDS = 13


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def _to_ref_keys(state_dict):
    key_map = {
        "stem_conv.weight": "conv1.weight",
        "stem_bn.weight": "bn1.weight", "stem_bn.bias": "bn1.bias",
        "stem_bn.running_mean": "bn1.running_mean", "stem_bn.running_var": "bn1.running_var",
        "stem_bn.num_batches_tracked": "bn1.num_batches_tracked",
    }
    for ours_stage, ref_layer in [("stage1", "layer1"), ("stage2", "layer2"), ("stage3", "layer3")]:
        key_map[f"{ours_stage}.c1.weight"] = f"{ref_layer}.0.conv1.weight"
        key_map[f"{ours_stage}.n1.weight"] = f"{ref_layer}.0.bn1.weight"
        key_map[f"{ours_stage}.n1.bias"] = f"{ref_layer}.0.bn1.bias"
        key_map[f"{ours_stage}.n1.running_mean"] = f"{ref_layer}.0.bn1.running_mean"
        key_map[f"{ours_stage}.n1.running_var"] = f"{ref_layer}.0.bn1.running_var"
        key_map[f"{ours_stage}.n1.num_batches_tracked"] = f"{ref_layer}.0.bn1.num_batches_tracked"
        key_map[f"{ours_stage}.c2.weight"] = f"{ref_layer}.0.conv2.weight"
        key_map[f"{ours_stage}.n2.weight"] = f"{ref_layer}.0.bn2.weight"
        key_map[f"{ours_stage}.n2.bias"] = f"{ref_layer}.0.bn2.bias"
        key_map[f"{ours_stage}.n2.running_mean"] = f"{ref_layer}.0.bn2.running_mean"
        key_map[f"{ours_stage}.n2.running_var"] = f"{ref_layer}.0.bn2.running_var"
        key_map[f"{ours_stage}.n2.num_batches_tracked"] = f"{ref_layer}.0.bn2.num_batches_tracked"
        key_map[f"{ours_stage}.skip.0.weight"] = f"{ref_layer}.0.shortcut.0.weight"
        key_map[f"{ours_stage}.skip.1.weight"] = f"{ref_layer}.0.shortcut.1.weight"
        key_map[f"{ours_stage}.skip.1.bias"] = f"{ref_layer}.0.shortcut.1.bias"
        key_map[f"{ours_stage}.skip.1.running_mean"] = f"{ref_layer}.0.shortcut.1.running_mean"
        key_map[f"{ours_stage}.skip.1.running_var"] = f"{ref_layer}.0.shortcut.1.running_var"
        key_map[f"{ours_stage}.skip.1.num_batches_tracked"] = f"{ref_layer}.0.shortcut.1.num_batches_tracked"
    key_map["fc.weight"] = "linear1.weight"
    key_map["fc.bias"] = "linear1.bias"
    return {key_map[k]: v for k, v in state_dict.items() if k in key_map}


def ref_stage3_train_mode(client_states, ori_weight, betas, trigger_images, sampled):
    probe = RefResNet8(num_classes=10).to(device)
    out_list = []
    with torch.no_grad():
        for i in range(len(sampled)):
            parameter = {}
            for k in client_states[i].keys():
                if k in ori_weight and ori_weight[k].dtype != torch.int64:
                    parameter[k] = (client_states[i][k] - ori_weight[k]) * betas[i] + ori_weight[k]
                else:
                    parameter[k] = client_states[i][k]
            probe.load_state_dict(_to_ref_keys(parameter))
            out = torch.softmax(probe(trigger_images), dim=1)
            out = torch.mean(out, dim=0)
            out_val, out_n = torch.max(out, dim=0)
            out_list.append({"pred_class": int(out_n.item()), "confidence": float(out_val.item())})
    return out_list


def run_cell(label, d1, d2, probe_ori_weight, client_states, betas, sampled, attacker_idx):
    seed_everything(SEED)
    params = {"type": "CIFAR10"}
    tri = ref_gan.gan_defense(d1, d2, params, the_epoch=1)  # shipped 10/5 budget
    T = tri[0]

    results = ref_stage3_train_mode(client_states, probe_ori_weight, betas, T, sampled)
    preds = [r["pred_class"] for r in results]
    confs = [round(r["confidence"], 4) for r in results]
    flagged = [sampled[i] for i, r in enumerate(results)
               if r["pred_class"] == TARGET_CLASS and r["confidence"] > RHO]

    print(f"\n=== {label} ===")
    print(f"  preds={preds}")
    print(f"  confs={confs}")
    print(f"  flagged={flagged}")
    print(f"  attacker(47) pred={results[attacker_idx]['pred_class']} "
          f"conf={results[attacker_idx]['confidence']:.4f}")
    return tuple(preds), T


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

    round1_record = None
    G_old_1 = None
    G_old_13 = None

    for rnd in range(1, TOTAL_ROUNDS + 1):
        sampled = random.sample(participants, cfg.num_clients)
        malicious_sampled = [c for c in sampled if c in adversarial_namelist]
        ori_weight = {k: v.clone() for k, v in model.state_dict().items()}  # G_old(rnd)

        if rnd == 1:
            G_old_1 = ori_weight
            client_states = []
            for cid in sampled:
                m = copy.deepcopy(model)
                w = local_train(m, client_loaders[cid], cfg, device,
                                 is_malicious=(cid in malicious_sampled), attacker=attacker)
                client_states.append(w)
            round1_record = (sampled, malicious_sampled, ori_weight, client_states)
            betas = [1] * len(sampled)
            new_state = aggregate(ori_weight, client_states, betas, cfg.eta, denominator="survivors")
        else:
            client_states_r = []
            for cid in sampled:
                m = copy.deepcopy(model)
                w = local_train(m, client_loaders[cid], cfg, device,
                                 is_malicious=(cid in malicious_sampled), attacker=attacker)
                client_states_r.append(w)
            betas_r = [1] * len(sampled)
            new_state = aggregate(ori_weight, client_states_r, betas_r, cfg.eta, denominator="survivors")

        model.load_state_dict(new_state)

        if rnd == TOTAL_ROUNDS:
            G_old_13 = ori_weight

        print(f"round {rnd}: sampled={sampled} malicious={malicious_sampled}")

    sampled1, malicious1, ori_weight1, client_states1 = round1_record
    betas1 = [1] * len(sampled1)
    attacker_idx = sampled1.index(47)
    assert malicious1 == [47], f"expected round 1 attacker=[47], got {malicious1}"

    weight_delta = sum((G_old_1[k].float() - G_old_13[k].float()).abs().sum().item()
                        for k in G_old_1 if G_old_1[k].dtype.is_floating_point)
    print(f"\nsanity check -- sum|G_old(1) - G_old(13)| = {weight_delta:.2f}")

    G1 = _to_ref_keys(G_old_1)
    G13 = _to_ref_keys(G_old_13)

    print(f"probing with round 1's client set: {sampled1}, attacker={malicious1} (index {attacker_idx})")

    preds_A, T_A = run_cell("A) degenerate @ round 1:   d1=d2=G_old(1)",
                             G1, G1, ori_weight1, client_states1, betas1, sampled1, attacker_idx)
    preds_B, T_B = run_cell("B) degenerate @ round 13:  d1=d2=G_old(13)",
                             G13, G13, ori_weight1, client_states1, betas1, sampled1, attacker_idx)
    preds_C, T_C = run_cell("C) normal order:           d1=G_old(1),  d2=G_old(13)",
                             G1, G13, ori_weight1, client_states1, betas1, sampled1, attacker_idx)
    preds_D, T_D = run_cell("D) swapped order:          d1=G_old(13), d2=G_old(1)",
                             G13, G1, ori_weight1, client_states1, betas1, sampled1, attacker_idx)

    print("\n########## A vs B: do the two degenerate cases actually land on different basins at this distance? ##########")
    print(f"  A preds: {preds_A}")
    print(f"  B preds: {preds_B}")
    print(f"  identical: {preds_A == preds_B}")

    print("\n########## C vs D: which model does the mixed pair track? ##########")
    print(f"  C preds: {preds_C}")
    print(f"  D preds: {preds_D}")
    print(f"  C == A (tracks d1=G_old(1))?   {preds_C == preds_A}")
    print(f"  C == B (tracks d2=G_old(13))?  {preds_C == preds_B}")
    print(f"  D == A (tracks d2=G_old(1))?   {preds_D == preds_A}")
    print(f"  D == B (tracks d1=G_old(13))?  {preds_D == preds_B}")
    print(f"  C == D (role swap has no effect)? {preds_C == preds_D}")


if __name__ == "__main__":
    main()
