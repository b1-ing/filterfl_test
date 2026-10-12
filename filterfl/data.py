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
