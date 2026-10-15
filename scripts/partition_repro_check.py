"""Is the Dirichlet partition reproducible across platforms? Rebuild with run.py's exact call order for a seed and compare
(a) the partition's class histograms for named clients and (b) the RNG-dependent client-sampling sequence against the
CSV of an actual run (epoch-1..3 sampled_clients + adversarial_sampled). Zero GPU."""
import sys, csv, random
import numpy as np, torch
from collections import Counter
from filterfl.data import load_dataset, build_class_index, dirichlet_partition

def rebuild(seed, alpha=0.5, n=100, n_clients=10, n_adv=10):
    np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
    train, _, _, _ = load_dataset("CIFAR10")
    ci = build_class_index(train)
    pp, _ = dirichlet_partition(train, n, alpha, ci)
    lab = {i: l for l, idxs in ci.items() for i in idxs}
    adv = set(random.sample(list(range(n)), n_adv))
    samp = [sorted(random.sample(list(range(n)), n_clients)) for _ in range(3)]  # epochs 1-3 (is_poison True path)
    hist = {p: Counter(lab[i] for i in idxs) for p, idxs in pp.items()}
    return hist, adv, samp

def csv_rounds(path, k=3):
    r = list(csv.DictReader(open(path)))[:k]
    return [sorted(int(c) for c in x["sampled_clients"].split(",")) for x in r], \
           set(int(c) for x in r for c in x["adversarial_sampled"].split(",") if c)

for seed, path in [(1, "results/step2_cellA_stealthy.csv"), (0, "results/hpc_s10_cifar_mean_a_s0.csv"), (0, "results/hpc_s1a_s0.csv")]:
    hist, adv, samp = rebuild(seed)
    csv_samp, csv_adv = csv_rounds(path)
    print(f"seed {seed} vs {path}")
    print("  client 73: shard %d class-0 %d (%.1f%%) | client 49: shard %d class-0 %d (%.1f%%)" % (
        sum(hist[73].values()), hist[73][0], 100*hist[73][0]/sum(hist[73].values()),
        sum(hist[49].values()), hist[49][0], 100*hist[49][0]/sum(hist[49].values())))
    print("  epoch1-3 sampled match CSV:", samp == csv_samp, "| CSV adversarial seen in epochs 1-3 subset of rebuilt namelist:", csv_adv <= adv)
    print("  rebuilt epoch1:", samp[0], "\n  csv     epoch1:", csv_samp[0])
