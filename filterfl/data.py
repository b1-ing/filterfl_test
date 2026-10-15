"""Dataset loading, Dirichlet non-IID partitioning, poisoned-test split.

Independently written. Behaviourally equivalent to what CODE_NOTES.md documents
for handle.py (Dirichlet sampling, held-out target-label pool for ASR eval),
but restructured as free functions instead of being baked into a God object.
"""
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms


def load_dataset(data_type: str, data_path: str = "./data"):
    if data_type == "CIFAR10":
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train = datasets.CIFAR10(data_path, train=True, download=True, transform=train_tf)
        test = datasets.CIFAR10(data_path, train=False, download=True, transform=test_tf)
        return train, test, 10, 3
    elif data_type == "MNIST":
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train = datasets.MNIST(data_path, train=True, download=True, transform=tf)
        test = datasets.MNIST(data_path, train=False, download=True, transform=tf)
        return train, test, 10, 1
    elif data_type == "GTSRB":
        # Resize to 32x32 so ResNet-8 / GeneratorCIFAR dimensions match CIFAR-10 exactly.
        # Mean/std from the GTSRB training split (widely-used values for this resize).
        gtsrb_mean = [0.3403, 0.3121, 0.3214]
        gtsrb_std  = [0.2724, 0.2608, 0.2669]
        train_tf = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=gtsrb_mean, std=gtsrb_std),
        ])
        test_tf = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=gtsrb_mean, std=gtsrb_std),
        ])
        train = datasets.GTSRB(data_path, split="train", download=True, transform=train_tf)
        test  = datasets.GTSRB(data_path, split="test",  download=True, transform=test_tf)
        return train, test, 43, 3
    else:
        raise ValueError(f"unsupported dataset type: {data_type}")


def build_class_index(dataset) -> dict:
    classes = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        classes[label].append(idx)
    return classes


def dirichlet_partition(dataset, num_participants: int, alpha: float, class_index: dict = None):
    """Partition `dataset` indices across `num_participants` via a per-class Dirichlet draw.

    Returns (per_participant_indices: dict[int, list[int]], counts: list[int]).
    """
    classes = {k: list(v) for k, v in (class_index or build_class_index(dataset)).items()}
    num_classes = len(classes)
    per_participant = defaultdict(list)
    for c in range(num_classes):
        pool = classes[c]
        random.shuffle(pool)
        class_size = len(pool)
        proportions = class_size * np.random.dirichlet(np.array([alpha] * num_participants))
        cursor = 0
        for p in range(num_participants):
            take = int(round(proportions[p]))
            take = min(take, len(pool) - cursor)
            per_participant[p].extend(pool[cursor:cursor + take])
            cursor += take
    counts = [len(per_participant[p]) for p in range(num_participants)]
    return per_participant, counts


def cap_class_share(per_participant: dict, class_index: dict, target_class: int, cap: float, protected: set):
    """Cap each non-protected client's target-class share at `cap` by SWAPPING samples (shard sizes unchanged).

    Deterministic and RNG-free (touches neither python nor numpy global state, so seeded client sampling and the
    adversarial namelist are unaffected).  Donors = non-protected clients whose target-class count exceeds
    floor(cap*size); each gives its excess target-class samples away and receives the same number of non-target
    samples.  Receivers = other non-protected clients with headroom floor(cap*size)-count0 > 0; the donors' total
    excess is spread over receivers proportionally to headroom (largest remainder), each receiver giving back as many
    non-target samples (evenly spaced over its non-target samples sorted by (label, index)).  Protected (malicious)
    clients are neither donors nor receivers.  Returns (new_per_participant, report).
    """
    lab = {i: l for l, idxs in class_index.items() for i in idxs}
    T = target_class
    shards = {p: sorted(v) for p, v in per_participant.items()}
    size = {p: len(v) for p, v in shards.items()}
    c0 = {p: [i for i in v if lab[i] == T] for p, v in shards.items()}
    nz = {p: sorted((i for i in v if lab[i] != T), key=lambda i: (lab[i], i)) for p, v in shards.items()}
    limit = {p: int(cap * size[p]) for p in shards}
    donors = [p for p in sorted(shards) if p not in protected and len(c0[p]) > limit[p]]
    excess = {p: len(c0[p]) - limit[p] for p in donors}
    recv = [p for p in sorted(shards) if p not in protected and p not in excess and limit[p] - len(c0[p]) > 0]
    head = {p: limit[p] - len(c0[p]) for p in recv}
    E, H = sum(excess.values()), sum(head.values())
    if E > H:
        raise ValueError(f"cannot cap at {cap}: excess {E} exceeds total receiver headroom {H}")
    alloc = {p: (E * head[p]) // H for p in recv}
    rem = E - sum(alloc.values())
    for p in sorted(recv, key=lambda p: (-((E * head[p]) % H), p))[:rem]:
        alloc[p] += 1
    out_c0 = [i for d in donors for i in c0[d][-excess[d]:]]            # target-class samples leaving donors
    out_nz = []                                                          # non-target samples leaving receivers
    for r in recv:
        a = alloc[r]
        if a:
            n = len(nz[r]); out_nz += [nz[r][(k * n) // a] for k in range(a)]
    new = {p: set(v) for p, v in shards.items()}
    for d in donors:
        for i in c0[d][-excess[d]:]: new[d].discard(i)
    cur = 0
    for r in recv:
        a = alloc[r]
        take = out_c0[cur:cur + a]; cur += a
        new[r].update(take)
    for r in recv:
        a = alloc[r]
        if a:
            n = len(nz[r])
            for k in range(a): new[r].discard(nz[r][(k * n) // a])
    cur = 0
    for d in donors:
        new[d].update(out_nz[cur:cur + excess[d]]); cur += excess[d]
    new = {p: sorted(v) for p, v in new.items()}
    assert all(len(new[p]) == size[p] for p in new), "shard sizes changed"
    assert sorted(i for v in new.values() for i in v) == sorted(i for v in shards.values() for i in v), "sample set changed"
    share_after = {p: sum(1 for i in new[p] if lab[i] == T) / size[p] for p in new}
    report = {"cap": cap, "target_class": T, "total_moved_target_samples": E,
              "donors": {p: {"size": size[p], "c0_before": len(c0[p]), "share_before": len(c0[p]) / size[p],
                             "c0_after": round(share_after[p] * size[p]), "share_after": share_after[p], "gave_c0": excess[p]} for p in donors},
              "receivers": {p: {"size": size[p], "c0_before": len(c0[p]), "share_before": len(c0[p]) / size[p],
                                "c0_after": round(share_after[p] * size[p]), "share_after": share_after[p], "received_c0": alloc[p]}
                            for p in recv if alloc[p]},
              "protected_untouched": sorted(p for p in protected if sorted(per_participant[p]) == new[p])}
    return new, report


def make_client_loader(dataset, indices, batch_size: int):
    return DataLoader(dataset, batch_size=batch_size,
                       sampler=SubsetRandomSampler(indices), pin_memory=True, num_workers=0)


def make_test_loader(dataset, batch_size: int):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def split_poison_eval_sets(test_dataset, target_class: int):
    """Returns (non_target_loader_indices, target_loader_indices).

    non_target: every test sample whose true label != target_class (used, with the
    trigger patched in, to compute ASR: did the model predict target_class after the
    trigger was applied to an image that did NOT originally belong to that class).
    target: the held-out pool of samples whose true label == target_class (not used
    for ASR, kept separate so target-class samples don't trivially inflate ASR).
    """
    by_class = defaultdict(list)
    for idx, (_, label) in enumerate(test_dataset):
        by_class[label].append(idx)
    target_indices = by_class[target_class]
    target_set = set(target_indices)
    non_target_indices = [i for i in range(len(test_dataset)) if i not in target_set]
    return non_target_indices, target_indices
