"""Probe-prediction fidelity test (CODE_NOTES.md).

Gates the "predict the server's probes and evade" attack direction. The
server runs Stage 1/2 with d1=G_old(t) (model the client legitimately has),
d2=G_temp(t) (the PRE-filtering aggregate, which the client never sees). The
client only ever legitimately sees G_old(t+1), the POST-filtering aggregate
-- identical to G_temp(t) only in rounds with zero exclusions.

Our existing checkpoints/CSV logs don't carry G_temp or per-client raw
models for past rounds -- only periodic post-filter global-model snapshots.
This script re-runs a fresh instrumented sequence that saves everything
needed, rather than silently substituting something else.

For each round t >= 2, computes three trigger sets:
  T_oracle = Stage1/2(G_old(t),   G_temp(t))              -- what the server actually ran
  T_stale  = Stage1/2(G_old(t-1), G_old(t))                -- what the client can compute, using
                                                               only models it has legitimately received
  T_linear = Stage1/2(G_old(t),   2*G_old(t) - G_old(t-1)) -- one-step extrapolation

then measures, per round, on the real attacker's real client model from round t:
  1. Stage 3 confidence under each T (suppression transfer -- the metric that matters)
  2. predicted class under each T
  3. confidence RANK of the attacker among all 10 sampled clients, under each T
  4. pixel MSE between T_stale/T_linear and T_oracle (reported, not weighted)

stage2_loss stays "paper" (default), stage3_probe_mode="train" -- the
validated 27/27 combination, and the only mode this test is meaningful
under (an eval-mode replica would model a detector that doesn't exist).
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
from filterfl.models import build_model, ResNet8
from filterfl.attacker import PixelPatternAttacker
from filterfl.client import local_train
from filterfl.aggregator import aggregate
from filterfl.defense import run_detection, stage3_classify_clients

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_ROUNDS = 30
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
    cfg = Config.from_yaml("configs/cifar_alpha1_side.yaml")
    assert cfg.gan.stage2_loss == "paper"
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

    rounds = []  # one dict per round: ori_weight (G_old(t)), new_state (G_temp(t)), client_states, sampled, malicious, excluded, oracle_results
    for t in range(1, N_ROUNDS + 1):
        ori_weight = {k: v.clone() for k, v in model.state_dict().items()}  # G_old(t)
        sampled = random.sample(participants, cfg.num_clients)
        malicious_sampled = [c for c in sampled if c in adversarial_namelist]
        client_states = []
        for cid in sampled:
            m = copy.deepcopy(model)
            w = local_train(m, client_loaders[cid], cfg, device,
                             is_malicious=(cid in malicious_sampled), attacker=attacker)
            client_states.append(w)
        betas_all1 = [1] * len(sampled)
        new_state = aggregate(ori_weight, client_states, betas_all1, cfg.eta, denominator="survivors")  # G_temp(t)

        seed_everything(1000 + t)
        det_oracle = run_detection(ori_weight, new_state, cfg, num_classes, in_channels, device)
        oracle_results = classify(client_states, ori_weight, betas_all1, det_oracle.trigger_images,
                                   cfg, num_classes, in_channels)
        betas = [1] * len(sampled)
        for i, res in enumerate(oracle_results):
            if res["pred_class"] == cfg.target_class and res["confidence"] > 0.5:
                betas[i] = 0
        excluded = [sampled[i] for i, b in enumerate(betas) if b == 0]

        if any(b == 0 for b in betas):
            post_filter_state = aggregate(ori_weight, client_states, betas, cfg.eta, denominator="survivors")
        else:
            post_filter_state = new_state

        # Stored on CPU: accumulating N_ROUNDS full state dicts (ori_weight +
        # new_state + 10 client_states, ~20MB each) on GPU for the whole run
        # exceeds this GPU's 6GB at N_ROUNDS=30 and causes severe allocator
        # thrashing well before any OOM error surfaces. The GPU originals are
        # still used above for this round's own live computation; only the
        # stored copy moves to CPU.
        rounds.append({
            "t": t, "ori_weight": {k: v.cpu() for k, v in ori_weight.items()},
            "new_state": {k: v.cpu() for k, v in new_state.items()},
            "client_states": [{k: v.cpu() for k, v in w.items()} for w in client_states],
            "sampled": sampled,
            "malicious_sampled": malicious_sampled, "excluded": excluded,
            "oracle_trigger_images": det_oracle.trigger_images.cpu(),
            "oracle_results": oracle_results,
        })
        print(f"round {t}: sampled={sampled} malicious={malicious_sampled} excluded={excluded}", flush=True)

        model.load_state_dict(post_filter_state)
        del client_states, ori_weight, new_state, post_filter_state, det_oracle
        torch.cuda.empty_cache()

    out_path = f"scripts/probe_fidelity_rounds_n{N_ROUNDS}.pt"
    torch.save(rounds, out_path)
    print(f"saved {out_path}\n")
    print(f"run scripts/probe_fidelity_analysis.py --rounds {out_path} for the oracle/stale/linear comparison")


if __name__ == "__main__":
    main()
