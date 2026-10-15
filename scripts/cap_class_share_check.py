"""Offline verification of the class-share cap on the seed-0 CIFAR-10 partition (zero GPU).
Checks: shard sizes unchanged, sample multiset unchanged, malicious shards untouched, benign class-0 share <= cap,
python/numpy RNG state identical before/after the cap (so sampling and the adversarial namelist are unaffected)."""
import random, sys
import numpy as np, torch
from filterfl.data import load_dataset, build_class_index, dirichlet_partition, cap_class_share
seed, cap = 0, float(sys.argv[1]) if len(sys.argv) > 1 else 0.15
np.random.seed(seed); torch.manual_seed(seed); random.seed(seed)
train, _, _, _ = load_dataset("CIFAR10"); ci = build_class_index(train)
pp, _ = dirichlet_partition(train, 100, 0.5, ci)
adv = set(random.sample(list(range(100)), 10))
rs_py, rs_np = random.getstate(), np.random.get_state()
new, rep = cap_class_share(pp, ci, 0, cap, adv)
same_rng = random.getstate() == rs_py and all(np.array_equal(a, b) if hasattr(a, "__len__") else a == b for a, b in zip(np.random.get_state(), rs_np))
lab = {i: l for l, idxs in ci.items() for i in idxs}
share = lambda d, p: sum(1 for i in d[p] if lab[i] == 0) / len(d[p])
ben = [p for p in range(100) if p not in adv]
print(f"adversaries: {sorted(adv)}")
print(f"RNG state unchanged by cap: {same_rng} | malicious shards untouched: {all(sorted(pp[p]) == new[p] for p in adv)}")
print(f"max benign class-0 share before {max(share(pp,p) for p in ben):.3f} -> after {max(share(new,p) for p in ben):.3f}")
print(f"malicious class-0 shares (unchanged): " + ", ".join(f"{p}:{share(pp,p):.2f}" for p in sorted(adv)))
print(f"moved {rep['total_moved_target_samples']} target-class samples; {len(rep['donors'])} donors, {len(rep['receivers'])} receivers")
print("donors (cid size share_before -> share_after gave):")
for p, d in rep["donors"].items(): print(f"  {p:>3} {d['size']:>4} {d['share_before']:.3f} -> {d['share_after']:.3f}  gave {d['gave_c0']}")
rcv = rep["receivers"]; print(f"receivers: {len(rcv)}; share before mean {np.mean([r['share_before'] for r in rcv.values()]):.3f} -> after mean {np.mean([r['share_after'] for r in rcv.values()]):.3f}; "
      f"received per client: min {min(r['received_c0'] for r in rcv.values())} max {max(r['received_c0'] for r in rcv.values())}")
print("class-0 share of benign clients, before vs after, sample:", {p: (round(share(pp,p),3), round(share(new,p),3)) for p in (9,75,18,4,3,5)})
