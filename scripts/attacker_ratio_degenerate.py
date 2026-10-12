"""Degenerate-pair invariance test at higher attacker ratios (40%, 70%).

Motivation (CODE_NOTES.md follow-up): SiftFL's own data shows the original
FilterFL's detection collapsing (ASR 97-99%) at 40% and 70% attacker ratios
over long horizons (~400+ rounds), which they attribute mechanistically to
the discriminator reference models (G_old/G_temp) becoming contaminated by
backdoor features over time -- i.e. the trigger-generating inputs' CONTENT
drifting and losing discriminative power. That sits in tension with this
project's degenerate-pair finding (detector output is near-invariant to
d1/d2 content, at a 10%-attacker baseline): if the reference models'
content doesn't matter to the detection decision, contaminating that content
shouldn't be able to collapse detection either.

This is the cheap, direct test of whether the invariance finding is general
or bounded to low attacker ratios: rerun the normal-vs-degenerate-pair
comparison at poison_client_num=40 and 70 (same round-1-style single-round
setup as scripts/crossfeed_round.pt, same config family, alpha=0.5,
target_class=0), and check whether the normal/degenerate gap (pixel MSE,
predicted-class vector, flagged set, attacker confidence) widens relative to
the 10%-attacker baseline.

To keep the comparison clean, the SAME 10 sampled clients are used across
all three attacker-ratio trials (a separate, fixed-seed RNG stream decides
`sampled`, independent of poison_client_num) -- only which of those 10
happen to be malicious varies, drawn from population-level pools of the
given size. Dataset partition and model initialization are also held fixed
(cfg.seed) across trials. Only Stage 1/2's generator-training seed (SEED,
matching budget_sweep.py's convention) differs between the normal and
degenerate pair within a trial, which is intentional -- it is what makes a
pixel MSE of exactly 0 impossible even if the loss landscape has literally
one basin.

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
SEED = 123  # GAN-training seed, matching budget_sweep.py/crossfeed_round.pt convention

_KEY_MAP = {
    "stem_conv.weight": "conv1.weight",
    "stem_bn.weight": "bn1.weight", "stem_bn.bias": "bn1.bias",
    "stem_bn.running_mean": "bn1.running_mean", "stem_bn.running_var": "bn1.running_var",
    "stem_bn.num_batches_tracked": "bn1.num_batches_tracked",
}
for _ours_stage, _ref_layer in [("stage1", "layer1"), ("stage2", "layer2"), ("stage3", "layer3")]:
    _KEY_MAP[f"{_ours_stage}.c1.weight"] = f"{_ref_layer}.0.conv1.weight"
    _KEY_MAP[f"{_ours_stage}.n1.weight"] = f"{_ref_layer}.0.bn1.weight"
    _KEY_MAP[f"{_ours_stage}.n1.bias"] = f"{_ref_layer}.0.bn1.bias"
    _KEY_MAP[f"{_ours_stage}.n1.running_mean"] = f"{_ref_layer}.0.bn1.running_mean"
    _KEY_MAP[f"{_ours_stage}.n1.running_var"] = f"{_ref_layer}.0.bn1.running_var"
    _KEY_MAP[f"{_ours_stage}.n1.num_batches_tracked"] = f"{_ref_layer}.0.bn1.num_batches_tracked"
    _KEY_MAP[f"{_ours_stage}.c2.weight"] = f"{_ref_layer}.0.conv2.weight"
    _KEY_MAP[f"{_ours_stage}.n2.weight"] = f"{_ref_layer}.0.bn2.weight"
    _KEY_MAP[f"{_ours_stage}.n2.bias"] = f"{_ref_layer}.0.bn2.bias"
    _KEY_MAP[f"{_ours_stage}.n2.running_mean"] = f"{_ref_layer}.0.bn2.running_mean"
    _KEY_MAP[f"{_ours_stage}.n2.running_var"] = f"{_ref_layer}.0.bn2.running_var"
    _KEY_MAP[f"{_ours_stage}.n2.num_batches_tracked"] = f"{_ref_layer}.0.bn2.num_batches_tracked"
    _KEY_MAP[f"{_ours_stage}.skip.0.weight"] = f"{_ref_layer}.0.shortcut.0.weight"
    _KEY_MAP[f"{_ours_stage}.skip.1.weight"] = f"{_ref_layer}.0.shortcut.1.weight"
    _KEY_MAP[f"{_ours_stage}.skip.1.bias"] = f"{_ref_layer}.0.shortcut.1.bias"
    _KEY_MAP[f"{_ours_stage}.skip.1.running_mean"] = f"{_ref_layer}.0.shortcut.1.running_mean"
    _KEY_MAP[f"{_ours_stage}.skip.1.running_var"] = f"{_ref_layer}.0.shortcut.1.running_var"
    _KEY_MAP[f"{_ours_stage}.skip.1.num_batches_tracked"] = f"{_ref_layer}.0.shortcut.1.num_batches_tracked"
_KEY_MAP["fc.weight"] = "linear1.weight"
_KEY_MAP["fc.bias"] = "linear1.bias"


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def _to_ref_keys(state_dict):
    return {_KEY_MAP[k]: v for k, v in state_dict.items() if k in _KEY_MAP}


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


def build_round(cfg, poison_client_num, sample_seed=999):
    seed_everything(cfg.seed)  # dataset partition + model init, identical across trials
    train_set, test_set, num_classes, in_channels = load_dataset(cfg.type)
    class_index = build_class_index(train_set)
    per_participant, _ = dirichlet_partition(train_set, cfg.number_of_total_participants,
                                              cfg.dirichlet_alpha, class_index)
    client_loaders = {p: make_client_loader(train_set, idxs, cfg.batch_size)
                       for p, idxs in per_participant.items()}
    model = build_model(cfg.type, num_classes=num_classes).to(device)
    attacker = PixelPatternAttacker(cfg)

    participants = list(range(cfg.number_of_total_participants))
    sample_rng = random.Random(sample_seed)
    sampled = sample_rng.sample(participants, cfg.num_clients)  # SAME across all trials

    adv_rng = random.Random(cfg.seed * 1000 + poison_client_num)  # trial-specific, deterministic
    adversarial_namelist = set(adv_rng.sample(participants, poison_client_num))
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
    return ori_weight, new_state, client_states, sampled, malicious_sampled, betas


def run_trial(label, poison_client_num, cfg):
    ori_weight, new_state, client_states, sampled, malicious_sampled, betas = build_round(cfg, poison_client_num)
    ori_weight_ref = _to_ref_keys(ori_weight)
    new_state_ref = _to_ref_keys(new_state)
    params = {"type": "CIFAR10"}

    seed_everything(SEED)
    T_normal = ref_gan.gan_defense(ori_weight_ref, new_state_ref, params, the_epoch=1)[0]
    seed_everything(SEED)
    T_degen = ref_gan.gan_defense(ori_weight_ref, new_state_ref, params, the_epoch=1, degenerate_pair=True)[0]

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
    identical_preds = preds_n == preds_d

    print(f"\n=== {label} (poison_client_num={poison_client_num}) ===")
    print(f"  sampled={sampled}")
    print(f"  malicious among sampled={malicious_sampled}  ({len(malicious_sampled)}/10)")
    print(f"  pixel MSE(normal, degenerate) = {pixel_mse:.6f}")
    print(f"  NORMAL     preds={preds_n}")
    print(f"             confs={confs_n}")
    print(f"             flagged={flagged_n}")
    print(f"  DEGENERATE preds={preds_d}")
    print(f"             confs={confs_d}")
    print(f"             flagged={flagged_d}")
    print(f"  predicted-class vectors identical: {identical_preds}")
    for cid in malicious_sampled:
        i = sampled.index(cid)
        print(f"  attacker {cid}: normal conf={confs_n[i]:.4f} pred={preds_n[i]} | "
              f"degenerate conf={confs_d[i]:.4f} pred={preds_d[i]}")
    return {
        "label": label, "poison_client_num": poison_client_num, "pixel_mse": pixel_mse,
        "identical_preds": identical_preds, "flagged_normal": flagged_n, "flagged_degen": flagged_d,
        "n_attackers_sampled": len(malicious_sampled),
    }


def main():
    cfg = Config.from_yaml("configs/cifar_primary.yaml")  # alpha=0.5, target_class=0
    results = []
    for label, pcn in [("baseline", 10), ("40pct", 40), ("70pct", 70)]:
        results.append(run_trial(label, pcn, cfg))

    print("\n########## SUMMARY ##########")
    print(f"{'label':>10} {'pcn':>5} {'n_atk_sampled':>14} {'pixel_mse':>12} {'preds_identical':>16} "
          f"{'flagged_n':>20} {'flagged_d':>20}")
    for r in results:
        print(f"{r['label']:>10} {r['poison_client_num']:>5} {r['n_attackers_sampled']:>14} "
              f"{r['pixel_mse']:>12.6f} {str(r['identical_preds']):>16} "
              f"{str(r['flagged_normal']):>20} {str(r['flagged_degen']):>20}")


if __name__ == "__main__":
    main()
