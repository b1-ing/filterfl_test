"""Checkpointing so a killed run can resume instead of restarting from scratch.

Saves everything that affects the NEXT epoch's result: model weights, RNG
states (python/numpy/torch/cuda -- client sampling, dirichlet partitioning,
and GPU-side poison-index selection all depend on these), the current
lr/poison_lr (post-decay), and the running attack/defense counters that
FedAvg logs cumulatively every epoch. Also stores adversarial_namelist as a
sanity check: a resumed run reconstructs it via load_data() under the same
seed, and if it doesn't match the checkpoint exactly, something about the
data pipeline changed between the original run and the resume -- better to
fail loudly than silently score a different poisoning setup.

Does NOT save optimizer state per-client -- each client's local SGD
optimizer is freshly constructed every round from the global model (see
train.standard_train), so there's nothing to resume there.
"""
import os
import random

import numpy as np
import torch


def save(path, epoch, model, lr, poison_lr, adversarial_namelist,
         count_all_attack, count_all_defense, count_correct_defense):
    tmp_path = path + ".tmp"
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "lr": lr,
        "poison_lr": poison_lr,
        "adversarial_namelist": list(adversarial_namelist),
        "count_all_attack": count_all_attack,
        "count_all_defense": count_all_defense,
        "count_correct_defense": count_correct_defense,
        "py_random_state": random.getstate(),
        "np_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, tmp_path)
    os.replace(tmp_path, path)  # atomic on same filesystem -- never leaves a half-written checkpoint


def load(path, model):
    # map_location="cpu": torch.set_rng_state requires a CPU ByteTensor, and
    # model.load_state_dict happily copies CPU tensors into a model that's
    # already been moved to its training device.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    random.setstate(ckpt["py_random_state"])
    np.random.set_state(ckpt["np_random_state"])
    torch.set_rng_state(ckpt["torch_rng_state"].to(torch.uint8))
    if ckpt["torch_cuda_rng_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["torch_cuda_rng_state"])
    return ckpt
