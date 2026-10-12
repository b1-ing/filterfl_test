"""Analysis phase of probe_fidelity.py, split out so the expensive 8-round
FedAvg sequence (scripts/probe_fidelity_rounds.pt) doesn't need re-running.

Fix vs. the first attempt: linear extrapolation 2*G_old(t) - G_old(t-1) must
NOT be applied to BatchNorm running_mean/running_var buffers -- a linear
combination of two running_var tensors can go negative, which produces NaN
inside BatchNorm's sqrt(var) and silently poisons Stage 1 (best_state stays
None because every loss.item() comparison against a NaN loss is False).
Only conv/fc weights and BN affine (weight, bias) are extrapolated; running
stats are kept at G_old(t)'s own values, matching "the client's best guess
is the current running stats, extrapolate only what's weight-like."
"""
import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from filterfl.config import Config
from filterfl.defense import run_detection, stage3_classify_clients

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROBE_MODE = "train"


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def classify(client_states, ori_weight, betas, trigger_images, cfg, num_classes, in_channels):
    return stage3_classify_clients(client_states, ori_weight, betas, trigger_images,
                                    num_classes, in_channels, cfg.target_class, device,
                                    probe_mode=PROBE_MODE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", default="scripts/probe_fidelity_rounds.pt")
    args = ap.parse_args()

    cfg = Config.from_yaml("configs/cifar_alpha1_side.yaml")
    num_classes, in_channels = 10, 3
    # map_location="cpu", not device: loading all N rounds' state dicts
    # straight onto a 6GB GPU accumulates ~20MB x 12 tensors x N rounds
    # simultaneously and thrashes the allocator well before completion (hit
    # this twice already -- see CODE_NOTES.md). Stays on CPU; run_detection /
    # stage3_classify_clients copy into GPU-resident models transiently via
    # load_state_dict, which handles the device conversion automatically.
    rounds = torch.load(args.rounds, map_location="cpu", weights_only=False)
    print(f"loaded {len(rounds)} rounds from {args.rounds}\n", flush=True)

    print(f"{'t':>3} {'excl?':>6} {'atk':>4} | {'oracle(p,c,rank)':>18} | {'stale(p,c,rank)':>18} | "
          f"{'linear(p,c,rank)':>18} | {'mse(stale)':>11} {'mse(linear)':>12}", flush=True)
    for i in range(1, len(rounds)):
        prev, cur = rounds[i - 1], rounds[i]
        t = cur["t"]
        had_exclusions = len(cur["excluded"]) > 0
        G_old_prev = prev["ori_weight"]
        G_old_cur = cur["ori_weight"]

        seed_everything(1000 + t)
        det_stale = run_detection(G_old_prev, G_old_cur, cfg, num_classes, in_channels, device)

        g_linear = {}
        for k, v in G_old_cur.items():
            if v.dtype.is_floating_point and "running_" not in k:
                g_linear[k] = 2 * v - G_old_prev[k]
            else:
                g_linear[k] = v.clone()
        seed_everything(1000 + t)
        det_linear = run_detection(G_old_cur, g_linear, cfg, num_classes, in_channels, device)

        sampled, malicious = cur["sampled"], cur["malicious_sampled"]
        betas_all1 = [1] * len(sampled)
        results_stale = classify(cur["client_states"], G_old_cur, betas_all1, det_stale.trigger_images,
                                  cfg, num_classes, in_channels)
        results_linear = classify(cur["client_states"], G_old_cur, betas_all1, det_linear.trigger_images,
                                   cfg, num_classes, in_channels)
        results_oracle = cur["oracle_results"]

        def rank_of(results, idx):
            confs = [r["confidence"] for r in results]
            order = sorted(range(len(confs)), key=lambda j: -confs[j])
            return order.index(idx) + 1

        mse_stale = (det_stale.trigger_images.cpu() - cur["oracle_trigger_images"]).pow(2).mean().item()
        mse_linear = (det_linear.trigger_images.cpu() - cur["oracle_trigger_images"]).pow(2).mean().item()
        del det_stale, det_linear
        torch.cuda.empty_cache()

        if not malicious:
            print(f"{t:>3} {str(had_exclusions):>6} {'none':>4} | (no attacker sampled this round) "
                  f"mse(stale)={mse_stale:.6f} mse(linear)={mse_linear:.6f}", flush=True)
            continue

        for cid in malicious:
            idx = sampled.index(cid)
            o, s, l = results_oracle[idx], results_stale[idx], results_linear[idx]
            r_o, r_s, r_l = rank_of(results_oracle, idx), rank_of(results_stale, idx), rank_of(results_linear, idx)
            print(f"{t:>3} {str(had_exclusions):>6} {cid:>4} | "
                  f"p={o['pred_class']} c={o['confidence']:.3f} r={r_o:>2}   | "
                  f"p={s['pred_class']} c={s['confidence']:.3f} r={r_s:>2}   | "
                  f"p={l['pred_class']} c={l['confidence']:.3f} r={r_l:>2}   | "
                  f"{mse_stale:>11.6f} {mse_linear:>12.6f}", flush=True)


if __name__ == "__main__":
    main()
    print("ANALYSIS DONE", flush=True)
