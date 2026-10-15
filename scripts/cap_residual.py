"""Residual-overlap check on the capped run (s11): is the leftover benign/malicious overlap the same class-0-skew mechanism?
Rebuilds the seed-0 partition + the cap (deterministic) to get each client's POST-cap class-0 share, joins to s11's probe log.
Zero GPU. Also prints the control (s10, pre-cap shares) correlation for comparison."""
import random, json
import numpy as np, torch
from filterfl.data import load_dataset, build_class_index, dirichlet_partition, cap_class_share
T = 0
np.random.seed(0); torch.manual_seed(0); random.seed(0)
train, _, _, _ = load_dataset("CIFAR10"); ci = build_class_index(train)
pp, _ = dirichlet_partition(train, 100, 0.5, ci)
adv = set(random.sample(list(range(100)), 10))
new, rep = cap_class_share(pp, ci, T, 0.15, adv)
lab = {i: l for l, idxs in ci.items() for i in idxs}
sh = lambda d: {p: sum(1 for i in d[p] if lab[i] == T) / len(d[p]) for p in d}
before, after = sh(pp), sh(new)
donors = set(int(c) for c in rep["donors"]); recv = set(int(c) for c in rep["receivers"])
def rank(a): return np.argsort(np.argsort(a)).astype(float)
def sp(a, b): return float(np.corrcoef(rank(np.asarray(a)), rank(np.asarray(b)))[0, 1])
def load(run):
    z = np.load(f"results/{run}_probe.npz"); P = z["probs"]
    return z["cid"], z["is_mal"].astype(bool), z["epoch"], P.mean(1)[:, T]
c1, m1, e1, s1 = load("hpc_s11_cifar_cap15_a_s0"); c0_, m0, e0, s0 = load("hpc_s10_cifar_mean_a_s0")
ben = sorted(set(c1[~m1].tolist()))
ms = {c: s1[~m1 & (c1 == c)].mean() for c in ben}; ms0 = {c: s0[~m0 & (c0_ == c)].mean() for c in ben}
print("=== 1. benign clients ranked by mean pooled score (s11, capped) ===")
order = sorted(ben, key=lambda c: -ms[c])
print(f"{'rank':>4} {'cid':>4} {'class-0 share (post-cap)':>25} {'(pre-cap)':>10} {'mean score':>11} {'group':>9}")
for k, c in enumerate(order[:20], 1):
    print(f"{k:>4} {c:>4} {after[c]:>25.3f} {before[c]:>10.3f} {ms[c]:>11.3f} {'donor' if c in donors else 'receiver' if c in recv else 'other':>9}")
top17 = set(order[:17])
print(f"\ndonors among top-17 scorers: {len(top17 & donors)} / 17 ; among top-10: {len(set(order[:10]) & donors)} / 10")
for nm, g in (("donors", donors), ("receivers", recv)):
    print(f"{nm:<10} n={len(g):>2}: mean share post-cap {np.mean([after[c] for c in g]):.3f} (range {min(after[c] for c in g):.3f}-{max(after[c] for c in g):.3f}); mean score {np.mean([ms[c] for c in g]):.3f}")
print(f"\n=== 2. class-0 share vs mean pooled score, Spearman across the {len(ben)} benign clients ===")
print(f"control s10 (pre-cap shares):   {sp([before[c] for c in ben], [ms0[c] for c in ben]):.3f}")
print(f"capped  s11 (post-cap shares):  {sp([after[c] for c in ben], [ms[c] for c in ben]):.3f}")
print(f"capped, donors only (n={len(donors)}, all ~0.15): {sp([after[c] for c in sorted(donors)], [ms[c] for c in sorted(donors)]):.3f}  (shares nearly constant; low information)")
rc = sorted(recv); print(f"capped, receivers only (n={len(rc)}):      {sp([after[c] for c in rc], [ms[c] for c in rc]):.3f}")
print(f"capped, per benign record:      {sp([after[c] for c in c1[~m1]], s1[~m1]):.3f}")
bins = [(0, .03), (.03, .06), (.06, .10), (.10, .14), (.14, .16)]
print(f"\n{'post-cap share':<15}{'clients':>8}{'mean score':>11}{'max client score':>17}   (s10 pre-cap score for the same clients)")
for lo, hi in bins:
    cs = [c for c in ben if lo <= after[c] < hi]
    if cs: print(f"{lo:.2f}-{hi:.2f}      {len(cs):>8}{np.mean([ms[c] for c in cs]):>11.3f}{max(ms[c] for c in cs):>17.3f}   ({np.mean([ms0[c] for c in cs]):.3f})")
print("\n=== 3. what sets the benign max (mean pooling) in s11 ===")
idx = np.where(~m1)[0]; top = idx[np.argsort(-s1[idx])[:20]]
print("top-20 benign records: (cid, epoch, score, post-cap share, group)")
for i in top[:12]: print(f"   cid {c1[i]:>3}  epoch {e1[i]:>4}  score {s1[i]:.3f}  share {after[c1[i]]:.3f}  {'donor' if c1[i] in donors else 'receiver'}")
from collections import Counter
cnt = Counter(c1[top].tolist()); print("distinct clients in top-20 records:", len(cnt), dict(cnt))
print("benign max overall: %.3f (cid %d, epoch %d); malicious median %.3f; 2nd..5th benign-record scores: %s" % (
    s1[idx].max(), c1[top[0]], e1[top[0]], np.median(s1[m1]), np.round(s1[top[1:5]], 3)))
b99 = np.quantile(s1[idx], .99); print(f"benign p99 {b99:.3f}, p99.9 {np.quantile(s1[idx], .999):.3f}; malicious p25/p50/p75: {np.round(np.quantile(s1[m1], [.25,.5,.75]),3)}")
print("fraction of malicious records scoring below benign p99: %.3f" % (s1[m1] < b99).mean())
