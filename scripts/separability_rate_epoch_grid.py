"""Poison-rate x epochs characterization grid (CODE_NOTES.md follow-up,
2026-10-11). Not an attack-optimization step -- fills in the grid around the
single blind-spot point (rate=0.2, epochs=6) found so far, to see how ASR,
Stage-3 mean confidence, and the top-k escape property move as poison rate
and local-epoch count vary together. One rep per cell (single seed, no
seed-averaging -- the epoch sweep already showed +/-0.05-0.1 confidence noise
is possible at a fixed hyperparameter, so single-cell values here should be
read as "this seed's realization," not an exact deterministic function of
(rate, epochs); the qualitative pattern across the grid is what matters).

Grid: poison_rate in {0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0} x poison_epochs in
{2, 6, 10}, scale=1.0 fixed, same malicious client id, same G_old
(results/stage2_full_a05_snap_e00200_ckpt.pt, round 200) and same trigger
set T as every other separability_* script. For every cell: ASR, CleanAcc,
c_stage3 (mean, i.e. A_32), A_1 (max) and A_16 (the point just before the
cliff found in the top-k curve script) to see whether the k=16-vs-32 cliff
shape holds across the whole grid or was specific to rate=0.2/epochs=6, and
flagged (mean-based, rho=0.5).

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
    probs = per_image_target_prob.cpu().tolist()
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0001]
    hist = [sum(1 for p in probs if bins[i] <= p < bins[i + 1]) for i in range(5)]
    return {
        "c_stage3_mean": float(mean_val.item()),
        "pred_class_mean": int(mean_class.item()),
        "target_prob_std": float(per_image_target_prob.std().item()),
        "target_prob_min": float(per_image_target_prob.min().item()),
        "target_prob_max": float(per_image_target_prob.max().item()),
        "frac_images_pred_target": float((per_image_pred == TARGET_CLASS).float().mean().item()),
        "per_image_target_prob": probs,
        "histogram_[0-.2)[.2-.4)[.4-.6)[.6-.8)[.8-1]": hist,
        "majority_vote_flagged": (sum(1 for p in probs if p > RHO) / len(probs)) > 0.5,
        "max_flagged": max(probs) > RHO,
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

    def topk_means(probs, ks):
        sorted_probs = sorted(probs, reverse=True)
        return {k: sum(sorted_probs[:k]) / k for k in ks}

    RATE_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    EPOCHS_GRID = [2, 6, 10]
    K_REPORT = [1, 16, 32]

    print("\n########## POISON-RATE x EPOCHS GRID ##########\n")
    csv_rows = []
    grid_results = {}
    for poison_epochs in EPOCHS_GRID:
        for poison_rate in RATE_GRID:
            cfg_p = copy.deepcopy(cfg)
            cfg_p.poison_rate = poison_rate
            cfg_p.poison_epochs = poison_epochs
            m = copy.deepcopy(model)
            w = local_train(m, client_loaders[malicious_cid], cfg_p, device, is_malicious=True, attacker=attacker)
            r = diagnose(f"rate={poison_rate} epochs={poison_epochs}", w)
            ak = topk_means(r["per_image_target_prob"], K_REPORT)
            r["A_1"], r["A_16"], r["A_32"] = ak[1], ak[16], ak[32]
            grid_results[(poison_rate, poison_epochs)] = r
            csv_rows.append({"poison_rate": poison_rate, "poison_epochs": poison_epochs,
                              "asr": r["asr"], "clean_acc": r["clean_acc"],
                              "c_stage3_mean": r["c_stage3_mean"], "A_1": r["A_1"], "A_16": r["A_16"],
                              "flagged": r["flagged"]})

    for poison_epochs in EPOCHS_GRID:
        print(f"--- epochs={poison_epochs} ---")
        print(f"{'rate':>6} {'ASR':>7} {'CleanAcc':>9} {'c_stage3':>9} {'A_1(max)':>9} {'A_16':>7} {'flag':>6}")
        for poison_rate in RATE_GRID:
            r = grid_results[(poison_rate, poison_epochs)]
            print(f"{poison_rate:>6} {r['asr']:>7.2f} {r['clean_acc']:>9.2f} {r['c_stage3_mean']:>9.4f} "
                  f"{r['A_1']:>9.4f} {r['A_16']:>7.4f} {str(r['flagged']):>6}")
        print()

    os.makedirs("results", exist_ok=True)
    import csv
    out_path = "results/separability_rate_epoch_grid.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["poison_rate", "poison_epochs", "asr", "clean_acc",
                                                "c_stage3_mean", "A_1", "A_16", "flagged"])
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    print(f"wrote {len(csv_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
