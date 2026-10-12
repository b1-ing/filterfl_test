"""Checkpointing so a killed run can resume instead of restarting from scratch.

Saves everything that affects the NEXT epoch's result: model weights, RNG
states (python/numpy/torch/cuda — client sampling and dirichlet partitioning
depend on these), the current lr/poison_lr (post-decay), trust scores, the
adversarial namelist (must stay fixed across resume), and the last computed
trigger set (needed by the scheduling 'last_filter' policy). Does NOT save
optimizer state per-client — each client's local SGD optimizer is freshly
constructed every round from the global model, so there's nothing to resume
there.
"""
import os
import random

import numpy as np
import torch


def save(path: str, epoch: int, model, lr: float, poison_lr: float,
         adversarial_namelist, trust_scores: dict, last_trigger_images):
    tmp_path = path + ".tmp"
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "lr": lr,
        "poison_lr": poison_lr,
        "adversarial_namelist": list(adversarial_namelist),
        "trust_scores": trust_scores,
        "last_trigger_images": last_trigger_images,
        "py_random_state": random.getstate(),
        "np_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, tmp_path)
    os.replace(tmp_path, path)  # atomic on same filesystem — never leaves a half-written checkpoint


def load(path: str, model, device):
    # map_location="cpu" regardless of `device`: torch.set_rng_state requires a
    # CPU ByteTensor, and model.load_state_dict happily copies CPU tensors into
    # a model that's already on `device`.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    random.setstate(ckpt["py_random_state"])
    np.random.set_state(ckpt["np_random_state"])
    torch.set_rng_state(ckpt["torch_rng_state"].to(torch.uint8))
    if ckpt["torch_cuda_rng_state"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["torch_cuda_rng_state"])
    return ckpt
