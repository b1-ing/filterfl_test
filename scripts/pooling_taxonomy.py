"""Offline pooling taxonomy from logged Stage 3 probe responses (hpc_s8 run, --dump-probe-responses).

For every logged (round, client): 32 probe images x 43 classes of softmax, ground-truth is_mal, live rule outcome.
Each pooling rule maps the (32, C) matrix to a length-C vector; the real gate then flags the client iff
argmax_c == target_class and max_c > rho (rho as logged for that round).

Two families:
  target-ranked: rank images by target-class prob, average the FULL softmax vectors of the top-k (k=1..32).
                 This is exactly the live top-k rule; k=32 is the original mean; k=16 is the deployed fix.
  columnwise:    per-class statistic over the 32 images (mean, median, trimmed means, max, quantiles, min).
Separation: AUC of the pooled target-class score (mal vs benign) and benign FPR / malicious TPR under the real gate.
Acceptance (usual definition): adversarial rounds where NO sampled malicious client is flagged / adversarial rounds.
Caveat: vectors come from the trajectory produced by the live top-16 gate (malicious updates were being filtered).
"""
import csv, sys
import numpy as np

run = sys.argv[1] if len(sys.argv) > 1 else "hpc_s8_gtsrb_topk16_a05_s0"
TARGET = 0
z = np.load(f"results/{run}_probe.npz")
ep, cid, mal, rho, live, P = z["epoch"], z["cid"], z["is_mal"].astype(bool), z["rho"], z["flagged"].astype(bool), z["probs"]
N, K, C = P.shape
print(f"{N} records, {len(np.unique(ep))} rounds, mal={mal.sum()} benign={(~mal).sum()}, probes={K}, classes={C}")

def ranked_topk(k):
    idx = np.argsort(-P[:, :, TARGET], axis=1)[:, :k]
    return np.take_along_axis(P, idx[:, :, None], axis=1).mean(axis=1)

def trim_mean(a, t, axis):
    n = a.shape[axis]; c = int(t * n)
    return np.sort(a, axis=axis).take(range(c, n - c), axis=axis).mean(axis=axis)

rules = {}
for k in [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 28, 32]:
    rules[f"topk{k}" + (" (max-ish)" if k == 1 else " (mean=orig)" if k == 32 else " (deployed)" if k == 16 else "")] = ranked_topk(k)
rules["col-mean"] = P.mean(1)
rules["col-median"] = np.median(P, axis=1)
for t in [0.1, 0.25, 0.4]:
    rules[f"col-trim{int(t*100)}%"] = trim_mean(P, t, axis=1)
for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
    rules[f"col-q{int(q*100)}"] = np.quantile(P, q, axis=1)
rules["col-max"] = P.max(1)
rules["col-min"] = P.min(1)

def auc(pos, neg):
    s = np.concatenate([pos, neg]); r = s.argsort().argsort() + 1.0
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r); r = (sums / cnt)[inv]
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def gate(vec):
    return (vec.argmax(1) == TARGET) & (vec.max(1) > rho)

adv_rounds = [e for e in np.unique(ep) if (mal & (ep == e)).any()]
windows = {"all": (0, 10**9), "e1-100": (0, 100), "e101-200": (100, 200), "e201-500": (200, 10**9)}
def accept(flag, lo, hi):
    rs = [e for e in adv_rounds if lo < e <= hi]
    n = acc = 0
    for e in rs:
        m = (ep == e) & mal
        n += 1; acc += (not flag[m].any())
    return acc, n

# sanity: reproduce the live rule
live_rule = "col-mean" if "mean" in run else "topk16 (deployed)"
print(f"live rule reproduced by {live_rule}:", bool((gate(rules[live_rule]) == live).all()),
      "| mismatches:", int((gate(rules[live_rule]) != live).sum()))
rows = []
hdr = f"{'rule':<18}{'AUC(tgt)':>9}{'mal med':>8}{'ben max':>8}{'TPR':>7}{'FPR':>7}{'acc all':>12}{'e1-100':>11}{'e101-200':>11}{'e201+':>11}"
print("\n" + hdr)
for name, v in rules.items():
    score = v[:, TARGET]
    f = gate(v)
    a = auc(score[mal], score[~mal])
    tpr, fpr = f[mal].mean(), f[~mal].mean()
    out = []
    for w, (lo, hi) in windows.items():
        ac, n = accept(f, lo, hi); out.append((ac, n))
    print(f"{name:<18}{a:>9.4f}{np.median(score[mal]):>8.3f}{score[~mal].max():>8.3f}{tpr:>7.3f}{fpr:>7.4f}"
          + "".join(f"{f'{ac}/{n} {100*ac/max(n,1):.0f}%':>12}" for ac, n in out))
    rows.append({"rule": name, "auc_target_score": a, "mal_median_score": np.median(score[mal]),
                 "benign_max_score": score[~mal].max(), "tpr": tpr, "fpr": fpr,
                 **{f"acc_{w}": ac / max(n, 1) for w, (ac, n) in zip(windows, out)},
                 **{f"adv_rounds_{w}": n for w, (ac, n) in zip(windows, out)}})
with open(f"results/pooling_taxonomy_{run}.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
print(f"\nrho values seen: {np.unique(rho)}  (benign records {int((~mal).sum())}, malicious {int(mal.sum())})")
