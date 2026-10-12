"""Local client training — one FedAvg round's worth of SGD for one client."""
import torch
import torch.nn as nn


def local_train(model: nn.Module, loader, cfg, device, is_malicious: bool, attacker=None):
    """Trains `model` in place for the configured number of local epochs.

    Benign clients use cfg.lr / cfg.local_epochs. Malicious clients use
    cfg.poison_lr / cfg.poison_epochs and poison cfg.poison_rate of each batch
    via `attacker`. This split (benign vs attacker hyperparameters) is exactly
    what Part C's sweep overrides per cell.
    """
    model.train()
    model.to(device)

    local_epochs = cfg.poison_epochs if is_malicious else cfg.local_epochs
    lr = cfg.poison_lr if is_malicious else cfg.lr

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=cfg.decay)

    for _ in range(local_epochs):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if is_malicious and cfg.poison_rate > 0 and attacker is not None:
                images, labels = attacker.poison_batch(images, labels, cfg.poison_rate,
                                                         trigger_index=getattr(cfg, "active_trigger_index", 0))
            optimizer.zero_grad(set_to_none=True)
            out = model(images)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()

    return {k: v.detach().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, attacker=None, poison: bool = False):
    """Plain accuracy. If poison=True, patches every image via `attacker` and reports ASR
    (fraction classified as attacker.cfg.target_class)."""
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if poison:
            images = attacker.patch_all(images, trigger_index=getattr(attacker.cfg, "active_trigger_index", 0))
            labels = torch.full_like(labels, attacker.cfg.target_class)
        out = model(images)
        pred = out.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total if total else float("nan")
