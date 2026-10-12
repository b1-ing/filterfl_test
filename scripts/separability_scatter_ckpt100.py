"""Probe-backdoor separability scatter, TRAINED-CHECKPOINT variant (CODE_NOTES.md
follow-up, 2026-10-11).

Identical to separability_scatter.py except for exactly one change: G_old is
loaded from results/stage2_full_a05_snap_e00100_ckpt.pt (this project's own 30-round
FilterFL run, alpha=0.5, final MA=34.68%/ASR=0% at epoch 30 -- see
results/stage2_full_a05.csv) instead of a freshly-initialized random model.
The first pass (separability_scatter.py) found one overlap candidate, but at
round-1/random-init the benign clean-accuracy band (9.97-21.41%) is close to
chance, so "acceptable clean accuracy" wasn't a meaningful bar yet. This rerun
asks the same question against an already-trained model, where clean accuracy
actually means something.

Per follow-up feedback: don't pre-filter "attack-feasible" candidates by an
arbitrary clean-accuracy floor -- report the full (ASR, CleanAcc, c_stage3,
||delta_w||) tuple for every point and let it be read off directly.


Question: does the Stage-3 confidence region that high-ASR poisoned clients
occupy overlap with the confidence region benign clients naturally occupy?
Everything established so far (degenerate-pair invariance, attacker-ratio
invariance, Stage-2-loss invariance) shows Stage 3's decision is insensitive
to trigger CONTENT and tracks client WEIGHTS instead. This script asks the
next question directly: within the weight-sensitivity axis, is there a
region that is simultaneously high-ASR, acceptable clean accuracy, and
low enough Stage-3 confidence to survive filtering -- i.e. is there an
attack-feasible blind spot, before any adaptive search for one is attempted.

For every client model (benign and poisoned, many configs each) we record:
    (ASR, CleanAcc, c_stage3, pred_class, ||delta_w||)
not just ASR, so an apparent "evasion" that actually destroys clean accuracy
(hence uninteresting) or an apparent "low confidence" backdoor that is
already dead (ASR ~ 0) can't be mistaken for a real attack-feasible point.

Trigger images are generated ONCE from a real (G_old, G_agg) round and reused
for every probe -- per the degenerate-pair finding, trigger content does not
measurably affect the Stage-3 decision, so re-generating it per client would
only add noise, not information.

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
GAN_SEED = 123  # matches budget_sweep.py / attacker_ratio_degenerate.py convention

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


def stage3_confidence(w, ori_weight_ref, trigger_images):
    """Single-client Stage-3 train-mode probe (reference algorithm.py's inline block,
    verbatim -- no .eval(), BatchNorm normalizes against the 32-image trigger batch)."""
    probe = RefResNet8(num_classes=10).to(device)
    w_ref = _to_ref_keys(w)
    with torch.no_grad():
        probe.load_state_dict(w_ref)
        out = torch.softmax(probe(trigger_images), dim=1)
        out = torch.mean(out, dim=0)
        out_val, out_n = torch.max(out, dim=0)
    return float(out_val.item()), int(out_n.item())


def delta_norm(w, ori_weight):
    sq = 0.0
    for k, v in w.items():
        if v.dtype == torch.int64:
            continue
        sq += torch.sum((v.float() - ori_weight[k].float()) ** 2).item()
    return sq ** 0.5


def scale_delta(w, ori_weight, scale):
    if scale == 1.0:
        return w
    out = {}
    for k, v in w.items():
        if v.dtype == torch.int64:
            out[k] = v
        else:
            out[k] = ori_weight[k] + (v - ori_weight[k]) * scale
    return out


def main():
    cfg = Config.from_yaml("configs/cifar_primary.yaml")  # alpha=0.5, target_class=0
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
    ckpt_path = "results/stage2_full_a05_snap_e00100_ckpt.pt"
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
    benign_client_ids = sampled_for_round[1:]  # used to build a real round for trigger generation

    # --- one real round, purely to get a realistic (G_old, G_agg) pair to feed the CGAN ---
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
    print(f"trigger images generated from one real round, reused for every probe below")

    rows = []

    def record(label, kind, w, extra):
        probe_model = build_model(cfg.type, num_classes=num_classes).to(device)
        probe_model.load_state_dict(w)
        clean_acc = evaluate(probe_model, test_loader, device)
        asr = evaluate(probe_model, asr_loader, device, attacker=attacker, poison=True)
        conf, pred = stage3_confidence(w, ori_weight_ref, T)
        dnorm = delta_norm(w, ori_weight)
        flagged = (pred == TARGET_CLASS and conf > RHO)
        row = {"label": label, "kind": kind, "asr": asr, "clean_acc": clean_acc,
               "c_stage3": conf, "pred_class": pred, "delta_norm": dnorm, "flagged": flagged, **extra}
        rows.append(row)
        print(f"  {label:<40} ASR={asr:6.2f} CleanAcc={clean_acc:6.2f} "
              f"c_stage3={conf:.4f} pred={pred} ||dw||={dnorm:8.3f} flagged={flagged}")
        return row

    print("\n=== BENIGN population (natural variance: client identity x local_epochs x lr) ===")
    benign_configs = []
    for cid in benign_pool[:6]:
        for local_epochs, lr in [(2, 0.01), (4, 0.01), (2, 0.02)]:
            benign_configs.append((cid, local_epochs, lr))
    for cid, local_epochs, lr in benign_configs:
        cfg_b = copy.deepcopy(cfg)
        cfg_b.local_epochs = local_epochs
        cfg_b.lr = lr
        m = copy.deepcopy(model)
        w = local_train(m, client_loaders[cid], cfg_b, device, is_malicious=False, attacker=attacker)
        record(f"benign cid={cid} epochs={local_epochs} lr={lr}", "benign", w,
               {"cid": cid, "local_epochs": local_epochs, "lr": lr, "scale": 1.0})

    print("\n=== POISONED population (poison_rate x poison_epochs x post-hoc scale-down) ===")
    poison_configs = []
    for poison_rate in [0.2, 0.5, 1.0]:
        for poison_epochs in [2, 6]:
            for scale in [1.0, 0.5, 0.25, 0.1]:
                poison_configs.append((poison_rate, poison_epochs, scale))
    for poison_rate, poison_epochs, scale in poison_configs:
        cfg_p = copy.deepcopy(cfg)
        cfg_p.poison_rate = poison_rate
        cfg_p.poison_epochs = poison_epochs
        m = copy.deepcopy(model)
        w = local_train(m, client_loaders[malicious_cid], cfg_p, device, is_malicious=True, attacker=attacker)
        w_scaled = scale_delta(w, ori_weight, scale)
        record(f"poisoned rate={poison_rate} epochs={poison_epochs} scale={scale}", "poisoned", w_scaled,
               {"cid": malicious_cid, "poison_rate": poison_rate, "poison_epochs": poison_epochs, "scale": scale})

    os.makedirs("results", exist_ok=True)
    import csv
    out_path = "results/separability_scatter_ckpt100.csv"
    fieldnames = ["label", "kind", "asr", "clean_acc", "c_stage3", "pred_class", "delta_norm", "flagged",
                  "cid", "local_epochs", "lr", "poison_rate", "poison_epochs", "scale"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"\nwrote {len(rows)} rows to {out_path}")

    benign_rows = [r for r in rows if r["kind"] == "benign"]
    poisoned_rows = [r for r in rows if r["kind"] == "poisoned"]
    benign_conf_band = (min(r["c_stage3"] for r in benign_rows), max(r["c_stage3"] for r in benign_rows))
    benign_acc_band = (min(r["clean_acc"] for r in benign_rows), max(r["clean_acc"] for r in benign_rows))
    print(f"\nbenign confidence band: {benign_conf_band}")
    print(f"benign clean-acc band: {benign_acc_band}")

    print("\n=== all poisoned points, sorted by ASR descending (full tuple, no pre-filtering) ===")
    for r in sorted(poisoned_rows, key=lambda r: -r["asr"]):
        in_benign_conf_band = benign_conf_band[0] <= r["c_stage3"] <= benign_conf_band[1]
        in_benign_acc_band = benign_acc_band[0] <= r["clean_acc"] <= benign_acc_band[1]
        print(f"  {r['label']:<40} ASR={r['asr']:6.2f} CleanAcc={r['clean_acc']:6.2f} "
              f"c_stage3={r['c_stage3']:.4f} ||dw||={r['delta_norm']:8.3f} flagged={r['flagged']} "
              f"in_benign_conf_band={in_benign_conf_band} in_benign_acc_band={in_benign_acc_band}")


if __name__ == "__main__":
    main()
