"""Held-out threshold check for the pooling taxonomy.

Question: is the in-sample result "mean pooling separates malicious from benign, it just needs a lower rho"
real? For each pooling rule, calibrate a threshold on a CALIBRATION set of rounds (benign records only: the
threshold is the max benign score, or the (1-FPR) quantile) and evaluate the real gate
(argmax==target AND max > thr) on rounds the threshold never saw.
Splits: early->late (first half calibrates, second half tests), late->early, and interleaved (odd/even rounds).
Reports TPR, benign FPR, and per-round acceptance (adversarial rounds where no sampled malicious client is flagged).
Usage: python scripts/pooling_heldout.py <run-name> [split_epoch]
"""
import sys, csv
import numpy as np

run = sys.argv[1]
TARGET = 0
TARGET_FPRS = (0.0, 0.001, 0.01)  # zero-FPR (benign max), 0.1%, 1% benign-FPR quantiles on the calibration set
z = np.load(f"results/{run}_probe.npz")
ep, mal, P = z["epoch"], z["is_mal"].astype(bool), z["probs"]
last = int(ep.max()); split = int(sys.argv[2]) if len(sys.argv) > 2 else last // 2

def ranked(k):
    i = np.argsort(-P[:, :, TARGET], axis=1)[:, :k]
    return np.take_along_axis(P, i[:, :, None], axis=1).mean(1)

rules = {f"top{k}": ranked(k) for k in [1, 4, 8, 12, 16, 20, 24, 32]}
rules["mean"] = rules.pop("top32")
rules["median"] = np.median(P, axis=1)
rules["trim25%"] = np.sort(P, axis=1)[:, 8:24].mean(1)

def acceptance(flag, mask):
    rs = [e for e in np.unique(ep[mask]) if (mal & (ep == e)).any()]
    acc = sum(1 for e in rs if not flag[mal & (ep == e)].any())
    return acc, len(rs)

splits = {f"early(<= {split})->late": (ep <= split, ep > split),
          f"late->early": (ep > split, ep <= split),
          "even->odd rounds": (ep % 2 == 0, ep % 2 == 1)}
rows = []
for sname, (cal, test) in splits.items():
    print(f"\n=== calibrate on {sname.split('->')[0]}, test on {sname.split('->')[1]}  "
          f"(cal mal={int((cal&mal).sum())} ben={int((cal&~mal).sum())}; test mal={int((test&mal).sum())} ben={int((test&~mal).sum())}) ===")
    print(f"{'rule':<8}{'target FPR':>11}{'thr':>7}{'test TPR':>10}{'test FPR':>10}{'test acc':>14}{'  [live rho=0.5: TPR / acc]'}")
    for name, v in rules.items():
        score = v[:, TARGET]; isT = v.argmax(1) == TARGET
        live = isT & (v.max(1) > 0.5)
        la, ln = acceptance(live, test)
        for tf in TARGET_FPRS:
            b = score[cal & ~mal]
            thr = b.max() if tf == 0 else np.quantile(b, 1 - tf)
            f = isT & (v.max(1) > thr)
            tpr, fpr = f[test & mal].mean(), f[test & ~mal].mean()
            a, n = acceptance(f, test)
            extra = f"  [{live[test & mal].mean():.3f} / {la}/{ln} {100*la/max(ln,1):.0f}%]" if tf == 0 else ""
            print(f"{name:<8}{tf:>11.2f}{thr:>7.3f}{tpr:>10.3f}{fpr:>10.4f}{f'{a}/{n} {100*a/max(n,1):.0f}%':>14}{extra}")
            rows.append({"split": sname, "rule": name, "target_fpr": tf, "thr": thr, "test_tpr": tpr,
                         "test_fpr": fpr, "test_acc": a / max(n, 1), "test_adv_rounds": n,
                         "live_rho_acc": la / max(ln, 1)})
with open(f"results/pooling_heldout_{run}.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
