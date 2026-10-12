"""Diagnose WHY rate=0.2/epochs=6/scale=1.0 sits in the benign Stage-3 region
while preserving the backdoor (CODE_NOTES.md follow-up, 2026-10-11).

The scatter persistence test found one durable blind-spot config and several
flagged near-neighbors differing in only one hyperparameter. This script
looks at what distinguishes them along more axes than the scatter's summary
tuple:
    ||delta_w|| (total and per-layer-group)
    cosine(w_i, G_old)                         -- does the client's full
                                                   weight vector stay aligned
                                                   with the global model?
    cosine(delta_w_i, mean_benign_delta)       -- does the UPDATE direction
                                                   resemble normal training
                                                   noise, rather than just
                                                   being small in magnitude?
    Stage-3 confidence (mean, as before) AND the per-trigger-image
    distribution behind that mean (std/min/max across the 32 probe images,
    fraction of the 32 individually classified as target_class) -- the
    scatter only ever looked at the batch mean.
    ASR, CleanAcc (as before)

G_old = results/stage2_full_a05_snap_e00200_ckpt.pt (the most mature
checkpoint, where the winning config's margin from flagged neighbors is
largest). Contrast set, chosen directly from the ckpt200 scatter table to
isolate one hyperparameter at a time:
    WINNER        rate=0.2 epochs=6 scale=1.0   ASR=67.5% c=0.390 unflagged
    epochs twin   rate=0.2 epochs=2 scale=1.0   ASR=27.3% c=0.592 FLAGGED   (only epochs differs)
    rate twin A   rate=0.5 epochs=6 scale=1.0   ASR=91.4% c=0.725 FLAGGED   (only rate differs)
    rate twin B   rate=1.0 epochs=6 scale=1.0   ASR=100%  c=1.000 FLAGGED   (only rate differs)
    scale twin    rate=0.2 epochs=6 scale=0.5   ASR=33.3% c=0.342 unflagged (only scale differs)
plus a benign reference set for the delta-direction centroid and layer-norm
baseline.

Run from the repo root (D:\\filterfl_test_claude).
"""
import copy
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference", "filterfl"))

from filterfl.config import Config
from filterfl.data import (load_dataset, build_class_index, dirichlet_partition,
                            make_client_loader, make_test_loader, split_poison_eval_sets)
from filterfl.models import build_model
from filterfl.attacker import PixelPatternAttacker
from filterfl.client import local_train, evaluate
from filterfl.aggregator import aggregate

import gan as ref_gan
from models.ResNet8 import ResNet8 as RefResNet8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RHO = 0.5
TARGET_CLASS = 0
GAN_SEED = 123

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

LAYER_GROUPS = ["stem_conv", "stem_bn", "stage1", "stage2", "stage3", "fc"]


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def _to_ref_keys(state_dict):
    return {_KEY_MAP[k]: v for k, v in state_dict.items() if k in _KEY_MAP}


def group_of(key):
    for g in LAYER_GROUPS:
        if key == g or key.startswith(g + "."):
            return g
    return "other"


def flat_delta(w, ori_weight):
    parts = []
    for k, v in w.items():
        if v.dtype == torch.int64:
            continue
        parts.append((v.float() - ori_weight[k].float()).flatten())
    return torch.cat(parts)


def flat_vec(w):
    parts = []
    for k, v in w.items():
        if v.dtype == torch.int64:
            continue
        parts.append(v.float().flatten())
    return torch.cat(parts)


def layerwise_norms(w, ori_weight):
    sums = {g: 0.0 for g in LAYER_GROUPS}
    total = 0.0
    for k, v in w.items():
        if v.dtype == torch.int64:
            continue
        sq = torch.sum((v.float() - ori_weight[k].float()) ** 2).item()
        sums[group_of(k)] = sums.get(group_of(k), 0.0) + sq
        total += sq
    total_norm = total ** 0.5
    fractions = {g: (sums[g] ** 0.5) / total_norm if total_norm > 0 else 0.0 for g in LAYER_GROUPS}
    return total_norm, fractions


def stage3_probe_detail(w, trigger_images):
    """Per-trigger-image breakdown, not just the batch mean used for flagging."""
    probe = RefResNet8(num_classes=10).to(device)
    w_ref = _to_ref_keys(w)
    with torch.no_grad():
        probe.load_state_dict(w_ref)
        out = torch.softmax(probe(trigger_images), dim=1)  # (32, 10)
        per_image_target_prob = out[:, TARGET_CLASS]
        per_image_pred = out.argmax(dim=1)
        mean_out = torch.mean(out, dim=0)
        mean_val, mean_class = torch.max(mean_out, dim=0)
    return {
        "c_stage3_mean": float(mean_val.item()),
        "pred_class_mean": int(mean_class.item()),
        "target_prob_std": float(per_image_target_prob.std().item()),
        "target_prob_min": float(per_image_target_prob.min().item()),
        "target_prob_max": float(per_image_target_prob.max().item()),
        "frac_images_pred_target": float((per_image_pred == TARGET_CLASS).float().mean().item()),
    }


def scale_delta(w, ori_weight, scale):
    if scale == 1.0:
        return w
    out = {}
    for k, v in w.items():
        out[k] = v if v.dtype == torch.int64 else ori_weight[k] + (v - ori_weight[k]) * scale
    return out


def main():
    cfg = Config.from_yaml("configs/cifar_primary.yaml")
    seed_everything(cfg.seed)

    train_set, test_set, num_classes, in_channels = load_dataset(cfg.type)
    class_index = build_class_index(train_set)
    per_participant, _ = dirichlet_partition(train_set, cfg.number_of_total_participants,
                                              cfg.dirichlet_alpha, class_index)
    client_loaders = {p: make_client_loader(train_set, idxs, cfg.batch_size)
                       for p, idxs in per_participant.items()}
    test_loader = make_test_loader(test_set, cfg.batch_size)
    non_target_idx, _ = split_poison_eval_sets(test_set, cfg.target_class)
    asr_loader = make_test_loader(Subset(test_set, non_target_idx), cfg.batch_size)

    model = build_model(cfg.type, num_classes=num_classes).to(device)
    ckpt_path = "results/stage2_full_a05_snap_e00200_ckpt.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    print(f"G_old loaded from {ckpt_path} (epoch {ckpt['epoch']})")
    ori_weight = {k: v.clone() for k, v in model.state_dict().items()}
    ori_weight_ref = _to_ref_keys(ori_weight)
    attacker = PixelPatternAttacker(cfg)

    participants = list(range(cfg.number_of_total_participants))
    rng = random.Random(cfg.seed)
    sampled_for_round = rng.sample(participants, cfg.num_clients)
    benign_pool = [p for p in participants if p not in sampled_for_round][:20]
    malicious_cid = sampled_for_round[0]
    benign_client_ids = sampled_for_round[1:]

    client_states_for_round = []
    for cid in benign_client_ids:
        m = copy.deepcopy(model)
        w = local_train(m, client_loaders[cid], cfg, device, is_malicious=False, attacker=attacker)
        client_states_for_round.append(w)
    betas = [1] * len(client_states_for_round)
    new_state = aggregate(ori_weight, client_states_for_round, betas, cfg.eta, denominator="survivors")
    new_state_ref = _to_ref_keys(new_state)

    seed_everything(GAN_SEED)
    params = {"type": "CIFAR10"}
    T = ref_gan.gan_defense(ori_weight_ref, new_state_ref, params, the_epoch=1)[0]
    print("trigger images generated from one real round, reused for every probe below\n")

    # --- benign reference set: delta-direction centroid + layer-norm baseline ---
    print("=== benign reference set ===")
    benign_results = []
    for cid in benign_pool[:6]:
        m = copy.deepcopy(model)
        w = local_train(m, client_loaders[cid], cfg, device, is_malicious=False, attacker=attacker)
        d = flat_delta(w, ori_weight)
        total_norm, fractions = layerwise_norms(w, ori_weight)
        benign_results.append({"cid": cid, "w": w, "delta": d, "total_norm": total_norm, "fractions": fractions})
        print(f"  benign cid={cid}: ||dw||={total_norm:.3f} layer_fractions={ {k: round(v,3) for k,v in fractions.items()} }")
    mean_benign_delta = torch.mean(torch.stack([r["delta"] for r in benign_results]), dim=0)
    mean_benign_delta = mean_benign_delta / mean_benign_delta.norm()

    def diagnose(label, w):
        d = flat_delta(w, ori_weight)
        total_norm, fractions = layerwise_norms(w, ori_weight)
        cos_to_gold = torch.nn.functional.cosine_similarity(flat_vec(w).unsqueeze(0),
                                                             flat_vec(ori_weight).unsqueeze(0)).item()
        cos_to_benign_centroid = torch.nn.functional.cosine_similarity(
            d.unsqueeze(0), mean_benign_delta.unsqueeze(0)).item()
        probe_model = build_model(cfg.type, num_classes=num_classes).to(device)
        probe_model.load_state_dict(w)
        clean_acc = evaluate(probe_model, test_loader, device)
        asr = evaluate(probe_model, asr_loader, device, attacker=attacker, poison=True)
        detail = stage3_probe_detail(w, T)
        flagged = (detail["pred_class_mean"] == TARGET_CLASS and detail["c_stage3_mean"] > RHO)
        print(f"\n--- {label} ---")
        print(f"  ASR={asr:.2f} CleanAcc={clean_acc:.2f} flagged={flagged}")
        print(f"  ||dw||={total_norm:.3f}  layer_fractions={ {k: round(v,3) for k,v in fractions.items()} }")
        print(f"  cos(w, G_old)={cos_to_gold:.6f}  cos(delta_w, mean_benign_delta)={cos_to_benign_centroid:.4f}")
        print(f"  c_stage3 (mean)={detail['c_stage3_mean']:.4f} pred_class(mean)={detail['pred_class_mean']}")
        print(f"  per-image target-class prob: std={detail['target_prob_std']:.4f} "
              f"min={detail['target_prob_min']:.4f} max={detail['target_prob_max']:.4f} "
              f"frac_of_32_predicting_target={detail['frac_images_pred_target']:.3f}")
        return {"label": label, "asr": asr, "clean_acc": clean_acc, "flagged": flagged,
                "total_norm": total_norm, "fractions": fractions,
                "cos_to_gold": cos_to_gold, "cos_to_benign_centroid": cos_to_benign_centroid, **detail}

    contrast_configs = [
        ("WINNER  rate=0.2 epochs=6 scale=1.0", 0.2, 6, 1.0),
        ("epochs-twin  rate=0.2 epochs=2 scale=1.0", 0.2, 2, 1.0),
        ("rate-twin-A  rate=0.5 epochs=6 scale=1.0", 0.5, 6, 1.0),
        ("rate-twin-B  rate=1.0 epochs=6 scale=1.0", 1.0, 6, 1.0),
        ("scale-twin  rate=0.2 epochs=6 scale=0.5", 0.2, 6, 0.5),
    ]
    results = []
    for label, poison_rate, poison_epochs, scale in contrast_configs:
        cfg_p = copy.deepcopy(cfg)
        cfg_p.poison_rate = poison_rate
        cfg_p.poison_epochs = poison_epochs
        m = copy.deepcopy(model)
        w = local_train(m, client_loaders[malicious_cid], cfg_p, device, is_malicious=True, attacker=attacker)
        w_scaled = scale_delta(w, ori_weight, scale)
        results.append(diagnose(label, w_scaled))

    print("\n########## SUMMARY ##########")
    for r in results:
        print(f"{r['label']:<45} ASR={r['asr']:6.2f} flagged={str(r['flagged']):5} "
              f"||dw||={r['total_norm']:7.3f} cos_gold={r['cos_to_gold']:.5f} "
              f"cos_benign_dir={r['cos_to_benign_centroid']:6.3f} "
              f"c_mean={r['c_stage3_mean']:.3f} frac32_target={r['frac_images_pred_target']:.3f} "
              f"target_prob_std={r['target_prob_std']:.3f}")


if __name__ == "__main__":
    main()
