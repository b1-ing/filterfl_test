"""Degenerate-pair test, run directly against the reference implementation
(CODE_NOTES.md), not the reimplementation stand-in. Uses this file's own
handle.py/algorithm.py/gan.py/train.py modules, unmodified except for the
additive degenerate_pair=True hook added to gan.py's gan_defense().

One real FedAvg round (seed=1, cifar_alpha0.5_params.yaml's own data
pipeline), then two gan_defense calls on the SAME round's (ori_weight,
w_glob, client models): normal (d1=ori_weight, d2=w_glob) vs degenerate
(d1=d2=ori_weight). Reports predicted-class vector, confidence vector, and
flagged set for all 10 sampled clients, both cases.
"""
import copy
import random

import numpy as np
import torch
import yaml

import algorithm
import train
from gan import gan_defense
from models.ResNet8 import ResNet8
from handle import Handle
from device import device


def seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    random.seed(s)


def main():
    with open('cifar_alpha0.5_params.yaml', 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    params['defence_method'] = 'ours'
    seed(params['seed'])

    handle = Handle(current_time='degenerate_pair_test', params=params, name='CIFAR10_degentest')
    handle.load_data()
    handle.create_model()
    handle.model.to(device)

    agent_name_keys, adversarial_name_keys = algorithm.get_clients(1, handle)
    lens = len(agent_name_keys)
    ori_weight = {k: v.clone() for k, v in handle.model.state_dict().items()}
    w_locals = []
    for client in agent_name_keys:
        is_poison = client in adversarial_name_keys
        model_copy = copy.deepcopy(handle.model)
        w = train.standard_train(1, handle.clients_data_num[client], client, handle.params,
                                  model_copy.to(device), handle.train_data[client], handle, poison=is_poison)
        w_locals.append(w)

    beta = [1] * lens
    w_glob = algorithm.Aggregation({k: v.clone() for k, v in ori_weight.items()}, w_locals, lens, beta,
                                    handle.params['eta'], handle.params['defence_method'], handle.params)

    print(f"sampled: {agent_name_keys}")
    print(f"adversarial (ground truth): {adversarial_name_keys}\n")

    def fingerprint(state_dict):
        return sum(v.double().sum().item() for v in state_dict.values() if v.dtype.is_floating_point)

    fp_ori = fingerprint(ori_weight)
    fp_glob = fingerprint(w_glob)
    print(f"fingerprint(ori_weight / G_old) = {fp_ori:.6f}")
    print(f"fingerprint(w_glob / G_agg)     = {fp_glob:.6f}")
    print(f"G_old == G_agg (normal pair)?     {'YES -- identical' if fp_ori == fp_glob else 'NO -- different, as expected (real training happened)'}")
    print(f"degenerate pair loads d1=d2=ori_weight explicitly (gan.py's "
          f"`discriminator2.load_state_dict(d1 if degenerate_pair else d2)`), "
          f"so by construction d1==d2 exactly in that condition regardless of what G_agg is.\n")

    def report(label, g_imgs):
        probe = ResNet8(num_classes=10).to(device)
        preds, confs, flagged = [], [], []
        standard = handle.params['ours_standard'][0]
        with torch.no_grad():
            for i, client in enumerate(agent_name_keys):
                parameter = {}
                for k in w_locals[i].keys():
                    if k in ori_weight and ori_weight[k].dtype != torch.int64:
                        parameter[k] = (w_locals[i][k] - ori_weight[k]) * beta[i] + ori_weight[k]
                    else:
                        parameter[k] = w_locals[i][k]
                probe.load_state_dict(parameter)
                out = torch.softmax(probe(g_imgs[0]), dim=1)
                out = torch.mean(out, dim=0)
                out_val, out_n = torch.max(out, dim=0)
                preds.append(int(out_n.item()))
                confs.append(round(float(out_val.item()), 3))
                if standard < out_val and out_n == 0:
                    flagged.append(client)
        print(f"=== {label} ===")
        print(f"  preds   : {preds}")
        print(f"  confs   : {confs}")
        print(f"  flagged : {flagged}\n")

    seed(123)
    g_imgs_normal = gan_defense(ori_weight, w_glob, handle.params, the_epoch=1)
    report("NORMAL pair (d1=ori_weight, d2=w_glob) -- Stage 2 (trigger) output", g_imgs_normal)

    seed(123)
    g_imgs_degenerate = gan_defense(ori_weight, w_glob, handle.params, the_epoch=1, degenerate_pair=True)
    report("DEGENERATE pair (d1=d2=ori_weight) -- Stage 2 (trigger) output", g_imgs_degenerate)

    mse_T = (g_imgs_normal[0] - g_imgs_degenerate[0]).pow(2).mean().item()
    print(f"MSE(T_normal, T_degenerate) = {mse_T:.6f}\n")

    # ---- CGAN 1 (Stage 1) output directly, via skip_stage2=True, for both pairs ----
    seed(123)
    cgan1_normal = gan_defense(ori_weight, w_glob, handle.params, the_epoch=1, skip_stage2=True)[0]
    seed(123)
    cgan1_degenerate = gan_defense(ori_weight, w_glob, handle.params, the_epoch=1,
                                    skip_stage2=True, degenerate_pair=True)[0]
    print(f"=== CGAN 1 (Stage 1) raw output for class 0, NORMAL pair ===")
    print(f"  mean={cgan1_normal.mean().item():.4f} std={cgan1_normal.std().item():.4f}")
    print(f"=== CGAN 1 (Stage 1) raw output for class 0, DEGENERATE pair ===")
    print(f"  mean={cgan1_degenerate.mean().item():.4f} std={cgan1_degenerate.std().item():.4f}")
    mse_I = (cgan1_normal - cgan1_degenerate).pow(2).mean().item()
    print(f"  pixel MSE (normal vs degenerate): {mse_I:.6f}\n")
    print(f"MSE(T_normal, T_degenerate) / MSE(I_normal, I_degenerate) = {mse_T / mse_I:.4f}\n")

    report("NORMAL pair -- CGAN1 output fed DIRECTLY to Stage 3 (skip_stage2)", [cgan1_normal])
    report("DEGENERATE pair -- CGAN1 output fed DIRECTLY to Stage 3 (skip_stage2)", [cgan1_degenerate])


if __name__ == '__main__':
    main()
