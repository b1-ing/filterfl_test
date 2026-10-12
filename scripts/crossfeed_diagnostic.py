"""Cross-feed diagnostic: fix one (G_old, G_agg) pair and one set of client
models, run BOTH implementations' Stage 1/2 to get their own trigger set T,
then feed EACH T into BOTH implementations' Stage 3. Localizes a detection gap
to Stage 1/2 (trigger generation) vs Stage 3 (classify/gate), rather than
inferring it from aggregate TP/FP/FN counts across many stochastic rounds.

Not part of the shipped pipeline — a one-off diagnostic script, run from the
repo root (D:\\filterfl_test_claude).
"""
import copy
import random
import sys

import numpy as np
import torch

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, for `filterfl` package
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reference", "filterfl"))

from filterfl.config import Config
from filterfl.data import load_dataset, build_class_index, dirichlet_partition, make_client_loader
from filterfl.models import build_model
from filterfl.attacker import PixelPatternAttacker
from filterfl.client import local_train
from filterfl.aggregator import aggregate
from filterfl.defense import run_detection, stage3_classify_clients

import gan as ref_gan              # reference/filterfl/gan.py
from models.ResNet8 import ResNet8 as RefResNet8  # reference/filterfl/models/ResNet8.py

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Our ResNet8 (filterfl/models.py) and the reference's (reference/filterfl/models/ResNet8.py)
# are the same topology (128->256->512, BasicBlock x[1,1,1]) but independently
# written with different layer names. Map our keys -> theirs so the SAME
# trained weights can be loaded into either class for a fair cross-feed test.
_KEY_MAP = {
    "stem_conv.weight": "conv1.weight",
    "stem_bn.weight": "bn1.weight", "stem_bn.bias": "bn1.bias",
    "stem_bn.running_mean": "bn1.running_mean", "stem_bn.running_var": "bn1.running_var",
    "stem_bn.num_batches_tracked": "bn1.num_batches_tracked",
}
for ours_stage, ref_layer in [("stage1", "layer1"), ("stage2", "layer2"), ("stage3", "layer3")]:
    _KEY_MAP[f"{ours_stage}.c1.weight"] = f"{ref_layer}.0.conv1.weight"
    _KEY_MAP[f"{ours_stage}.n1.weight"] = f"{ref_layer}.0.bn1.weight"
    _KEY_MAP[f"{ours_stage}.n1.bias"] = f"{ref_layer}.0.bn1.bias"
    _KEY_MAP[f"{ours_stage}.n1.running_mean"] = f"{ref_layer}.0.bn1.running_mean"
    _KEY_MAP[f"{ours_stage}.n1.running_var"] = f"{ref_layer}.0.bn1.running_var"
    _KEY_MAP[f"{ours_stage}.n1.num_batches_tracked"] = f"{ref_layer}.0.bn1.num_batches_tracked"
    _KEY_MAP[f"{ours_stage}.c2.weight"] = f"{ref_layer}.0.conv2.weight"
    _KEY_MAP[f"{ours_stage}.n2.weight"] = f"{ref_layer}.0.bn2.weight"
    _KEY_MAP[f"{ours_stage}.n2.bias"] = f"{ref_layer}.0.bn2.bias"
    _KEY_MAP[f"{ours_stage}.n2.running_mean"] = f"{ref_layer}.0.bn2.running_mean"
    _KEY_MAP[f"{ours_stage}.n2.running_var"] = f"{ref_layer}.0.bn2.running_var"
    _KEY_MAP[f"{ours_stage}.n2.num_batches_tracked"] = f"{ref_layer}.0.bn2.num_batches_tracked"
    _KEY_MAP[f"{ours_stage}.skip.0.weight"] = f"{ref_layer}.0.shortcut.0.weight"
    _KEY_MAP[f"{ours_stage}.skip.1.weight"] = f"{ref_layer}.0.shortcut.1.weight"
    _KEY_MAP[f"{ours_stage}.skip.1.bias"] = f"{ref_layer}.0.shortcut.1.bias"
    _KEY_MAP[f"{ours_stage}.skip.1.running_mean"] = f"{ref_layer}.0.shortcut.1.running_mean"
    _KEY_MAP[f"{ours_stage}.skip.1.running_var"] = f"{ref_layer}.0.shortcut.1.running_var"
    _KEY_MAP[f"{ours_stage}.skip.1.num_batches_tracked"] = f"{ref_layer}.0.shortcut.1.num_batches_tracked"
_KEY_MAP["fc.weight"] = "linear1.weight"
_KEY_MAP["fc.bias"] = "linear1.bias"


def to_ref_keys(state_dict):
    return {_KEY_MAP[k]: v for k, v in state_dict.items() if k in _KEY_MAP}


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


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

    # ---- one FedAvg round: this is the fixed (G_old, G_agg, client_states) tuple ----
    sampled = random.sample(participants, cfg.num_clients)
    malicious_sampled = [c for c in sampled if c in adversarial_namelist]
    print(f"sampled: {sampled}")
    print(f"malicious (ground truth): {malicious_sampled}")

    ori_weight = {k: v.clone() for k, v in model.state_dict().items()}  # G_old
    client_states = []
    for cid in sampled:
        m = copy.deepcopy(model)
        w = local_train(m, client_loaders[cid], cfg, device,
                         is_malicious=(cid in malicious_sampled), attacker=attacker)
        client_states.append(w)

    betas = [1] * len(sampled)
    new_state = aggregate(ori_weight, client_states, betas, cfg.eta, denominator="survivors")  # G_agg

    torch.save({"ori_weight": ori_weight, "new_state": new_state, "client_states": client_states,
                "sampled": sampled, "malicious_sampled": malicious_sampled},
               "scripts/crossfeed_round.pt")
    print("saved fixed round to scripts/crossfeed_round.pt")

    # ---- Stage 1/2 under OUR implementation ----
    seed_everything(123)  # fix GAN-training randomness separately from the round above
    det = run_detection(ori_weight, new_state, cfg, num_classes, in_channels, device)
    T_ours = det.trigger_images  # (N, 3, 32, 32), already channel-expanded

    # ---- Stage 1/2 under REFERENCE implementation (gan.py, verbatim) ----
    seed_everything(123)
    params = {"type": "CIFAR10"}
    tri_imgs_ref = ref_gan.gan_defense(to_ref_keys(ori_weight), to_ref_keys(new_state), params, the_epoch=1)
    T_ref = tri_imgs_ref[0]  # first batch, matching use_first_stage2_batch_only=True on both sides

    print(f"T_ours shape={tuple(T_ours.shape)} mean={T_ours.mean().item():.4f} std={T_ours.std().item():.4f}")
    print(f"T_ref  shape={tuple(T_ref.shape)} mean={T_ref.mean().item():.4f} std={T_ref.std().item():.4f}")

    # ---- Stage 3 under OUR implementation, fed T_ours and T_ref ----
    results_ours_on_ours = stage3_classify_clients(client_states, ori_weight, betas, T_ours,
                                                      num_classes, in_channels, cfg.target_class, device)
    results_ours_on_ref = stage3_classify_clients(client_states, ori_weight, betas, T_ref,
                                                     num_classes, in_channels, cfg.target_class, device)

    # ---- Stage 3 under REFERENCE implementation (algorithm.py's inline logic, verbatim), fed T_ref and T_ours ----
    def ref_stage3(trigger_images, force_eval=False):
        probe = RefResNet8(num_classes=10).to(device)
        if force_eval:
            probe.eval()  # ablation: reference's actual code never calls this
        out_list = []
        with torch.no_grad():
            for i in range(len(sampled)):
                parameter = {}
                for k in client_states[i].keys():
                    if k in ori_weight and ori_weight[k].dtype != torch.int64:
                        parameter[k] = (client_states[i][k] - ori_weight[k]) * betas[i] + ori_weight[k]
                    else:
                        parameter[k] = client_states[i][k]
                probe.load_state_dict(to_ref_keys(parameter))
                out = torch.softmax(probe(trigger_images), dim=1)
                out = torch.mean(out, dim=0)
                out_val, out_n = torch.max(out, dim=0)
                out_list.append({"pred_class": int(out_n.item()), "confidence": float(out_val.item())})
        return out_list

    results_ref_on_ref = ref_stage3(T_ref)
    results_ref_on_ours = ref_stage3(T_ours)
    results_ref_on_ref_eval = ref_stage3(T_ref, force_eval=True)   # ablation
    results_ref_on_ours_eval = ref_stage3(T_ours, force_eval=True)  # ablation

    rho = 0.5

    def fmt(r):
        flagged = (r["pred_class"] == cfg.target_class and r["confidence"] > rho)
        mark = "*" if flagged else " "
        return f"{mark}p={r['pred_class']} c={r['confidence']:.3f}"

    print(f"\n{'client':>6} {'malicious':>9} | {'ours_on_ours':>20} | {'ours_on_ref':>20} | "
          f"{'ref_on_ref':>20} | {'ref_on_ours':>20} | {'ref_on_ref(eval)':>20} | {'ref_on_ours(eval)':>20}")
    for i, cid in enumerate(sampled):
        is_mal = cid in malicious_sampled
        print(f"{cid:>6} {str(is_mal):>9} | {fmt(results_ours_on_ours[i]):>20} | "
              f"{fmt(results_ours_on_ref[i]):>20} | {fmt(results_ref_on_ref[i]):>20} | "
              f"{fmt(results_ref_on_ours[i]):>20} | {fmt(results_ref_on_ref_eval[i]):>20} | "
              f"{fmt(results_ref_on_ours_eval[i]):>20}")


if __name__ == "__main__":
    main()
