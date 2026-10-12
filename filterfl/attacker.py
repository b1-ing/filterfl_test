"""Attacker module — BadNets-style fixed-pixel trigger. Drop-in by design.

Kept as a standalone class so later evasion attacks can subclass/replace this
without touching client.py or run.py. Everything that defines "how aggressive
is this attacker" (local epochs, poison rate, lr) is read from config at call
time, which is exactly what Part C's sweep varies.
"""
from functools import lru_cache

import torch


class PixelPatternAttacker:
    def __init__(self, cfg):
        self.cfg = cfg
        self._mask_cache = {}

    def _pattern_mask(self, image_shape, device, trigger_index: int = 0):
        key = (trigger_index, image_shape, str(device))
        if key in self._mask_cache:
            return self._mask_cache[key]
        mask = torch.zeros(image_shape, device=device, dtype=torch.bool)
        positions = self.cfg.poison_patterns.get(trigger_index, [])
        for (r, c) in positions:
            mask[:, r, c] = True
        self._mask_cache[key] = mask
        return mask

    def poison_batch(self, images: torch.Tensor, labels: torch.Tensor, poison_rate: float,
                      trigger_index: int = 0):
        """Patch `poison_rate` fraction of `images` in-place-equivalent and relabel to target.

        Returns (images_after, labels_after). Non-destructive to the input tensors.
        """
        batch_size = images.size(0)
        n_poison = int(batch_size * poison_rate)
        if n_poison == 0:
            return images, labels
        device = images.device
        perm = torch.randperm(batch_size, device=device)
        poison_idx = perm[:n_poison]
        poison_mask = torch.zeros(batch_size, device=device, dtype=torch.bool)
        poison_mask[poison_idx] = True

        pattern = self._pattern_mask(images.shape[1:], device, trigger_index).unsqueeze(0)
        apply_mask = poison_mask.view(-1, 1, 1, 1) & pattern
        images_after = torch.where(apply_mask, torch.ones_like(images), images)

        target = torch.full((batch_size,), self.cfg.target_class, device=device, dtype=labels.dtype)
        labels_after = torch.where(poison_mask, target, labels)
        return images_after, labels_after

    def patch_all(self, images: torch.Tensor, trigger_index: int = 0):
        """Patch every image in the batch (used for ASR eval, not training)."""
        device = images.device
        pattern = self._pattern_mask(images.shape[1:], device, trigger_index).unsqueeze(0)
        return torch.where(pattern, torch.ones_like(images), images)
