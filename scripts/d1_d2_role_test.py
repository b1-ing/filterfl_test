"""Does the degenerate-pair attractor track d1, d2, or neither -- and does
d1/d2 ROLE (not just content) matter? (CODE_NOTES.md follow-up to the
round-dependence finding.)

The round-dependence result showed the degenerate-pair basin moves across
training rounds. This isolates the mechanism further with two cheap,
controlled comparisons on a SINGLE fixed pair of consecutive models (so
there's no confound from different client samplings or different attackers
across rounds):

  A) degenerate at t:   d1 = d2 = G_old(t)
  B) degenerate at t+1: d1 = d2 = G_old(t+1) = G_agg(t)
  C) normal order:      d1 = G_old(t),   d2 = G_old(t+1)   (the real pipeline's pairing)
  D) swapped order:     d1 = G_old(t+1), d2 = G_old(t)

G_old(t) and G_old(t+1) come straight from scripts/crossfeed_round.pt's
`ori_weight` (G_old) and `new_state` (G_agg -- which, if accepted, IS next
round's G_old; this is exactly the trust-chain relationship CODE_NOTES.md's
"Design assumptions" section already documents). The SAME 10 client models
(client_states) from that fixed round are classified in Stage 3 under all
four T's, so only d1/d2 varies, nothing else.

A vs B: if the predicted-class vector differs between A and B, the
degenerate output tracks WHICH model is fed in (not just "any model, same
seed"), i.e. the basin is a property of the specific weights, confirming
the round-dependence finding mechanistically rather than just observing it
drift. If A and B were identical despite genuinely different models, that
would suggest output is seed-dominated regardless of model content --
already made unlikely by the round-dependence result, but this isolates it
further on a matched pair.

C vs D: if swapping which model plays d1 (naturalness/"does this look
real") vs d2 (target-class bias) changes nothing, that's a second nail --
it would mean the detector's output doesn't depend on the specific
structural ROLE either input plays, just on the SET of weights involved (or
on neither, if the result is flat everywhere). If C and D differ
substantially, role matters and the two discriminators are doing
functionally distinct things, as the loss formula's asymmetric treatment of
d1 vs d2 would predict.

Run from the repo root (D:\\filterfl_test_claude).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference", "filterfl"))

import gan as ref_gan
from models.ResNet8 import ResNet8 as RefResNet8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_CLASS = 0
RHO = 0.5
SEED = 123


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
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
    return tuple(preds), confs, T


def main():
    data = torch.load("scripts/crossfeed_round.pt", map_location=device, weights_only=False)
    ori_weight, new_state = data["ori_weight"], data["new_state"]  # G_old(t), G_old(t+1) candidate
    client_states, sampled, malicious_sampled = data["client_states"], data["sampled"], data["malicious_sampled"]
    betas = [1] * len(sampled)
    attacker_idx = sampled.index(47)
    assert malicious_sampled == [47]

    Gt = _to_ref_keys(ori_weight)      # G_old(t)
    Gt1 = _to_ref_keys(new_state)      # G_old(t+1) candidate (= G_agg(t))

    print(f"sampled: {sampled}")
    print(f"malicious (ground truth): {malicious_sampled}  (attacker index {attacker_idx})")

    weight_delta = sum((ori_weight[k].float() - new_state[k].float()).abs().sum().item()
                        for k in ori_weight if ori_weight[k].dtype.is_floating_point)
    print(f"sanity check -- sum|G_old(t) - G_old(t+1)| = {weight_delta:.2f} (confirms these are genuinely different models)")

    preds_A, confs_A, T_A = run_cell("A) degenerate at t:    d1=d2=G_old(t)",
                                      Gt, Gt, ori_weight, client_states, betas, sampled, attacker_idx)
    preds_B, confs_B, T_B = run_cell("B) degenerate at t+1:  d1=d2=G_old(t+1)",
                                      Gt1, Gt1, ori_weight, client_states, betas, sampled, attacker_idx)
    preds_C, confs_C, T_C = run_cell("C) normal order:       d1=G_old(t),   d2=G_old(t+1)",
                                      Gt, Gt1, ori_weight, client_states, betas, sampled, attacker_idx)
    preds_D, confs_D, T_D = run_cell("D) swapped order:      d1=G_old(t+1), d2=G_old(t)",
                                      Gt1, Gt, ori_weight, client_states, betas, sampled, attacker_idx)

    print("\n########## A vs B: does the degenerate output track WHICH model is fed in? ##########")
    print(f"  A preds: {preds_A}")
    print(f"  B preds: {preds_B}")
    print(f"  identical vector: {preds_A == preds_B}")
    print(f"  pixel MSE(T_A, T_B) = {torch.mean((T_A - T_B) ** 2).item():.6f}")

    print("\n########## C vs D: does swapping d1/d2 ROLE change anything? ##########")
    print(f"  C preds: {preds_C}")
    print(f"  D preds: {preds_D}")
    print(f"  identical vector: {preds_C == preds_D}")
    print(f"  pixel MSE(T_C, T_D) = {torch.mean((T_C - T_D) ** 2).item():.6f}")

    print("\n########## cross-reference: does C (real pairing) differ from A/B (degenerate)? ##########")
    print(f"  pixel MSE(T_C, T_A) = {torch.mean((T_C - T_A) ** 2).item():.6f}")
    print(f"  pixel MSE(T_C, T_B) = {torch.mean((T_C - T_B) ** 2).item():.6f}")


if __name__ == "__main__":
    main()
