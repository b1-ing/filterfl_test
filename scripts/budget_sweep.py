"""Stage 2 (and Stage 1) CGAN training-budget sweep, on the reference code
directly (reference/filterfl/gan.py's gan_defense, verbatim except for the
additive n_epochs1/n_epochs2 override), reusing the exact fixed
(G_old, G_agg, client_states) round already saved in scripts/crossfeed_round.pt
(client 47 = real attacker, target_class=0).

Purpose (CODE_NOTES.md): the seed sweep showed detector output is stable
across GAN-training seeds at the SHIPPED fixed 10/5 budget. That's
consistent with two different explanations that the seed sweep alone cannot
tell apart:
  (a) the loss landscape has one dominant basin any input converges to
      (genuine attractor -> current framing holds at any budget), or
  (b) 5 Stage-2 rounds is too few for the loss to differentiate ANY input,
      including d1/d2 content itself, and the apparent basin is an
      artifact of truncated optimisation that would resolve at a bigger
      budget.

This sweeps n_epochs2 in {5, 15, 50, 200} at n_epochs1=10 (the shipped
Stage-1 budget), PLUS the full SiftFL-reported pairing n_epochs1=12/
n_epochs2=15 as its own cell (the pair most directly comparable to a
published number), and separately sweeps n_epochs1 in {5, 10, 20, 40} at
the shipped n_epochs2=5, for BOTH:
  - the "normal" (non-degenerate) pair: d1=G_old, d2=G_agg (the real,
    unfiltered round-1 aggregate) -- this is what the shipped pipeline
    actually feeds Stage 1/2 every round.
  - the "degenerate" pair: d1=d2=G_old -- zero information about the
    round's incremental update, per the existing degenerate-pair line of
    findings.

At each budget/pair, reports: pixel MSE between the normal-pair and
degenerate-pair trigger output (same budget, same seed, so same z
consumption sequence -> directly comparable image batches), the full
predicted-class vector and confidence vector for all 10 sampled clients
under each pair's T, the flagged set under each (rho=0.5, Stage 3 in TRAIN
mode -- matching algorithm.py's actual, un-.eval()'d behaviour, the thing
that does almost all of the detection work per CODE_NOTES.md), and the real
attacker's (client 47) confidence under each.

If the normal/degenerate gap (both in pixel-MSE and in attacker confidence)
stays ~flat as budget rises -> genuine dominant-basin, current framing
holds at any budget. If it widens substantially -> the shipped 5-round
budget is simply undertrained for Stage 2 to differentiate inputs at all,
and the basin claim needs restating as "at the shipped budget."

Run from the repo root (D:\\filterfl_test_claude).
"""
import os
import random
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

# (label, n_epochs1, n_epochs2)
CELLS = [
    ("n_epochs2=5  (shipped)",      10, 5),
    ("n_epochs2=15",                10, 15),
    ("n_epochs2=50",                10, 50),
    ("n_epochs2=200",                10, 200),
    ("SiftFL-reported pair 12/15",  12, 15),
]
STAGE1_CELLS = [
    ("n_epochs1=5",                  5, 5),
    ("n_epochs1=10 (shipped)",      10, 5),
    ("n_epochs1=20",                20, 5),
    ("n_epochs1=40",                40, 5),
]


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def _to_ref_keys(state_dict):
    # crossfeed_round.pt was saved from filterfl/ (our reimplementation);
    # reuse the same key map crossfeed_diagnostic.py uses to load it into
    # the reference's ResNet8.
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
    """algorithm.py's inline Stage 3 block, verbatim: constructs ResNet8 and
    never calls .eval() -- BatchNorm normalizes against the trigger batch's
    own statistics. This is the actual shipped behaviour, not an ablation."""
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


def run_cell(label, n1, n2, ori_weight_ref, new_state_ref, client_states, betas, sampled, attacker_idx):
    params = {"type": "CIFAR10"}

    seed_everything(SEED)
    tri_normal = ref_gan.gan_defense(ori_weight_ref, new_state_ref, params, the_epoch=1,
                                      n_epochs1=n1, n_epochs2=n2)
    T_normal = tri_normal[0]

    seed_everything(SEED)
    tri_degen = ref_gan.gan_defense(ori_weight_ref, new_state_ref, params, the_epoch=1,
                                     n_epochs1=n1, n_epochs2=n2, degenerate_pair=True)
    T_degen = tri_degen[0]

    pixel_mse = torch.mean((T_normal - T_degen) ** 2).item()

    results_normal = ref_stage3_train_mode(client_states, ori_weight_ref, betas, T_normal, sampled)
    results_degen = ref_stage3_train_mode(client_states, ori_weight_ref, betas, T_degen, sampled)

    def summarize(results):
        preds = [r["pred_class"] for r in results]
        confs = [round(r["confidence"], 4) for r in results]
        flagged = [sampled[i] for i, r in enumerate(results)
                   if r["pred_class"] == TARGET_CLASS and r["confidence"] > RHO]
        return preds, confs, flagged

    preds_n, confs_n, flagged_n = summarize(results_normal)
    preds_d, confs_d, flagged_d = summarize(results_degen)

    print(f"\n=== {label}  (n_epochs1={n1}, n_epochs2={n2}) ===")
    print(f"  pixel MSE(T_normal, T_degenerate) = {pixel_mse:.6f}")
    print(f"  NORMAL     preds={preds_n}")
    print(f"             confs={confs_n}")
    print(f"             flagged={flagged_n}")
    print(f"             attacker(47) conf = {results_normal[attacker_idx]['confidence']:.4f} "
          f"pred={results_normal[attacker_idx]['pred_class']}")
    print(f"  DEGENERATE preds={preds_d}")
    print(f"             confs={confs_d}")
    print(f"             flagged={flagged_d}")
    print(f"             attacker(47) conf = {results_degen[attacker_idx]['confidence']:.4f} "
          f"pred={results_degen[attacker_idx]['pred_class']}")
    attacker_conf_gap = abs(results_normal[attacker_idx]['confidence'] - results_degen[attacker_idx]['confidence'])
    print(f"  attacker confidence gap (normal - degenerate), abs = {attacker_conf_gap:.4f}")

    return {
        "label": label, "n1": n1, "n2": n2, "pixel_mse": pixel_mse,
        "attacker_conf_normal": results_normal[attacker_idx]['confidence'],
        "attacker_conf_degen": results_degen[attacker_idx]['confidence'],
        "attacker_conf_gap": attacker_conf_gap,
        "flagged_normal": flagged_n, "flagged_degen": flagged_d,
    }


def main():
    data = torch.load("scripts/crossfeed_round.pt", map_location=device, weights_only=False)
    ori_weight, new_state = data["ori_weight"], data["new_state"]
    client_states, sampled, malicious_sampled = data["client_states"], data["sampled"], data["malicious_sampled"]
    betas = [1] * len(sampled)
    attacker_idx = sampled.index(47)
    assert malicious_sampled == [47]

    # key-remap G_old/G_agg once -- both pass through gan_defense's load_state_dict
    ori_weight_ref = _to_ref_keys(ori_weight)
    new_state_ref = _to_ref_keys(new_state)

    print(f"sampled: {sampled}")
    print(f"malicious (ground truth): {malicious_sampled}  (attacker index {attacker_idx})")

    print("\n########## Stage 2 (n_epochs2) sweep, n_epochs1=10 fixed ##########")
    stage2_results = []
    for label, n1, n2 in CELLS:
        stage2_results.append(run_cell(label, n1, n2, ori_weight_ref, new_state_ref,
                                        client_states, betas, sampled, attacker_idx))

    print("\n########## Stage 1 (n_epochs1) sweep, n_epochs2=5 fixed ##########")
    stage1_results = []
    for label, n1, n2 in STAGE1_CELLS:
        stage1_results.append(run_cell(label, n1, n2, ori_weight_ref, new_state_ref,
                                        client_states, betas, sampled, attacker_idx))

    print("\n########## SUMMARY: pixel MSE and attacker-confidence gap vs budget ##########")
    print(f"{'cell':>32} {'pixel_mse':>12} {'conf_normal':>12} {'conf_degen':>12} {'abs_gap':>10}")
    for r in stage2_results + stage1_results:
        print(f"{r['label']:>32} {r['pixel_mse']:>12.6f} {r['attacker_conf_normal']:>12.4f} "
              f"{r['attacker_conf_degen']:>12.4f} {r['attacker_conf_gap']:>10.4f}")


if __name__ == "__main__":
    main()
