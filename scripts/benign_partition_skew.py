"""Why do benign clients score high on CIFAR-10? Test the Dirichlet-skew hypothesis.

Rebuild the run's Dirichlet partition (same seed/alpha/call order as filterfl.run), take each benign client's
share of target-class (0) samples in its shard, and relate it to that client's Stage-3 behaviour in the logged
probe responses of hpc_s10_cifar_mean_a_s0 (benign records only; malicious clients excluded by ground truth).
Zero GPU. Usage: python scripts/benign_partition_skew.py [run]
"""
import sys, random
import numpy as np
import torch
from filterfl.config import Config
from filterfl.data import load_dataset, build_class_index, dirichlet_partition

run = sys.argv[1] if len(sys.argv) > 1 else "hpc_s10_cifar_mean_a_s0"
cfg = Config.from_yaml(f"results/{run}_config.yaml")
np.random.seed(cfg.seed); torch.manual_seed(cfg.seed); random.seed(cfg.seed)
train, test, nc, ch = load_dataset(cfg.type)
ci = build_class_index(train)
pp, _ = dirichlet_partition(train, cfg.number_of_total_participants, cfg.dirichlet_alpha, ci)
lab = {i: l for l, idxs in ci.items() for i in idxs}
T = cfg.target_class
share, size = {}, {}
for p, idxs in pp.items():
    ls = np.array([lab[i] for i in idxs]); size[p] = len(ls); share[p] = float((ls == T).mean())
# NOTE: results/exp_partition_profile_NOTE.md (client 73: 500 samples/335 class-0; client 49: 426/168) is the SEED-1 partition
# (step2_cellA_stealthy, cifar_primary.yaml seed: 1); the HPC s10/s9 runs use seed 0, so those numbers differ BY SEED, not platform.
# scripts/partition_repro_check.py shows seed 1 rebuilds to exactly 500/335 and 426/168 on this machine.
print("partition (seed %d): client 73 shard %d class-0 share %.3f | client 49 shard %d share %.3f"
      % (cfg.seed, size[73], share[73], size[49], share[49]))

z = np.load(f"results/{run}_probe.npz")
cid, mal, P, ep, rho, flagged = z["cid"], z["is_mal"].astype(bool), z["probs"], z["epoch"], z["rho"], z["flagged"].astype(bool)
ben = ~mal
mean_pool = P.mean(1)                       # (R, C)
i16 = np.argsort(-P[:, :, T], axis=1)[:, :16]
top16 = np.take_along_axis(P, i16[:, :, None], axis=1).mean(1)
s_mean, s_top16 = mean_pool[:, T], top16[:, T]

def rank(a): return np.argsort(np.argsort(a)).astype(float)
def spearman(a, b): return float(np.corrcoef(rank(a), rank(b))[0, 1])

clients = sorted(set(cid[ben].tolist()))
rows = []
for c in clients:
    m = ben & (cid == c)
    rows.append((c, share[c], size[c], int(m.sum()), s_mean[m].mean(), s_top16[m].mean(), flagged[m].mean(), int(flagged[m].sum())))
rows = np.array(rows)
print(f"\n{len(clients)} benign clients with >=1 record; records: {int(ben.sum())}; benign flagged by live gate: {int(flagged[ben].sum())}")
sh, sm, st, fr = rows[:, 1], rows[:, 4], rows[:, 5], rows[:, 6]
print("Spearman across clients: class-0 share vs mean score %.3f | vs top16 score %.3f | vs live flag rate %.3f"
      % (spearman(sh, sm), spearman(sh, st), spearman(sh, fr)))
print("Per-record Spearman (benign records): class-0 share vs mean score %.3f"
      % spearman(np.array([share[c] for c in cid[ben]]), s_mean[ben]))

bins = [(0, .05), (.05, .15), (.15, .30), (.30, .50), (.50, 1.01)]
print(f"\n{'class-0 share':<14}{'clients':>8}{'records':>9}{'mean score':>11}{'p95':>7}{'top16 score':>12}{'flag rate':>10}{'flags':>7}")
tot_flags = int(flagged[ben].sum())
for lo, hi in bins:
    cs = [c for c in clients if lo <= share[c] < hi]
    m = ben & np.isin(cid, cs)
    if not m.any(): continue
    print(f"{f'{lo:.2f}-{min(hi,1):.2f}':<14}{len(cs):>8}{int(m.sum()):>9}{s_mean[m].mean():>11.3f}{np.quantile(s_mean[m],.95):>7.3f}"
          f"{s_top16[m].mean():>12.3f}{flagged[m].mean():>10.3f}{int(flagged[m].sum()):>7}")

order = np.argsort(-fr)
print("\nTop benign clients by live flag rate (client, class-0 share, shard, records, mean score, flag rate):")
for k in order[:10]:
    print("  cid %3d  share %.3f  shard %4d  n=%3d  mean-score %.3f  flag-rate %.3f" % (rows[k,0], rows[k,1], rows[k,2], rows[k,3], rows[k,4], rows[k,6]))
hi = [c for c in clients if share[c] >= 0.30]
mh = ben & np.isin(cid, hi)
print(f"\nclients with class-0 share >= 0.30: {len(hi)} of {len(clients)}; they hold {mh.sum()/ben.sum():.1%} of benign records "
      f"and {flagged[mh].sum()/max(tot_flags,1):.1%} of benign false positives ({int(flagged[mh].sum())}/{tot_flags})")
# does removing them close the overlap? benign max / AUC with and without high-share clients
def auc(pos, neg):
    s = np.concatenate([pos, neg]); r = rank(s) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
for nm, s in (("mean", s_mean), ("top16", s_top16)):
    keep = ben & ~np.isin(cid, hi)
    print(f"  AUC malicious vs benign [{nm}]: all benign %.4f | benign excluding class-0-heavy clients %.4f | benign max %.3f -> %.3f"
          % (auc(s[mal], s[ben]), auc(s[mal], s[keep]), s[ben].max(), s[keep].max()))
