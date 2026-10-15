"""Compare the capped run (s11) against the uncapped control (s10): FPs, AUCs, benign max vs malicious median,
donor/receiver score movement, attack acceptance. Usage: python scripts/cap_outcomes.py [control] [capped]"""
import csv, json, sys, yaml
import numpy as np
ctl = sys.argv[1] if len(sys.argv) > 1 else "hpc_s10_cifar_mean_a_s0"
cap = sys.argv[2] if len(sys.argv) > 2 else "hpc_s11_cifar_cap15_a_s0"
T = 0
def load(run):
    z = np.load(f"results/{run}_probe.npz"); P = z["probs"]
    i = np.argsort(-P[:, :, T], axis=1)[:, :16]
    return {"ep": z["epoch"], "cid": z["cid"], "mal": z["is_mal"].astype(bool), "flag": z["flagged"].astype(bool),
            "mean": P.mean(1)[:, T], "top16": np.take_along_axis(P, i[:, :, None], axis=1).mean(1)[:, T]}
def rank(a): return np.argsort(np.argsort(a)).astype(float)
def auc(pos, neg):
    s = np.concatenate([pos, neg]); r = rank(s) + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
def acceptance(path):
    r = list(csv.DictReader(open(path))); na = lambda x: len([c for c in x["adversarial_sampled"].split(",") if c])
    a = [x for x in r if na(x)]; acc = sum(1 for x in a if int(x["detection_tp"]) == 0)
    return acc, len(a), sum(int(x["detection_fp"]) for x in r), len(r), float(np.mean([float(x["asr"]) for x in r[-100:]])), float(np.mean([float(x["ma"]) for x in r[-100:]]))
d = {k: load(k) for k in (ctl, cap)}
rep = json.load(open(f"results/{cap}_capreport.json"))
don = [int(c) for c in rep["donors"]]; rcv = [int(c) for c in rep["receivers"]]
print("config diff (control vs capped):")
a, b = (yaml.safe_load(open(f"results/{k}_config.yaml")) for k in (ctl, cap))
for k in sorted(set(a) | set(b)):
    if a.get(k) != b.get(k): print(f"   {k}: {a.get(k)!r} -> {b.get(k)!r}")
print(f"\n{'':<34}{'control':>10}{'capped':>10}")
for k in (ctl, cap):
    pass
ac = {k: acceptance(f"results/{k}.csv") for k in (ctl, cap)}
row = lambda name, f: print(f"{name:<34}" + "".join(f"{f(k):>10}" for k in (ctl, cap)))
row("rounds", lambda k: ac[k][3])
row("benign false positives", lambda k: ac[k][2])
row("attack acceptance (adv rounds)", lambda k: f"{ac[k][0]}/{ac[k][1]}")
row("attack acceptance %", lambda k: f"{100*ac[k][0]/ac[k][1]:.1f}")
row("ASR last100 / MA last100", lambda k: f"{ac[k][4]:.1f}/{ac[k][5]:.1f}")
for nm in ("mean", "top16"):
    row(f"AUC mal vs benign [{nm}]", lambda k: f"{auc(d[k][nm][d[k]['mal']], d[k][nm][~d[k]['mal']]):.4f}")
for nm in ("mean", "top16"):
    row(f"benign max [{nm}]", lambda k: f"{d[k][nm][~d[k]['mal']].max():.3f}")
    row(f"malicious median [{nm}]", lambda k: f"{np.median(d[k][nm][d[k]['mal']]):.3f}")
row("benign p99 [mean]", lambda k: f"{np.quantile(d[k]['mean'][~d[k]['mal']], .99):.3f}")
print("\nmean pooled score of benign clients (all rounds), control -> capped:")
for name, ids in (("donors (capped)", don), ("receivers", rcv)):
    vals = []
    for k in (ctl, cap):
        m = ~d[k]["mal"] & np.isin(d[k]["cid"], ids); vals.append(d[k]["mean"][m].mean())
    print(f"  {name:<18} n={len(ids):>3}: {vals[0]:.3f} -> {vals[1]:.3f}  (delta {vals[1]-vals[0]:+.3f})")
rest = [c for c in set(d[ctl]["cid"][~d[ctl]["mal"]].tolist()) if c not in don and c not in rcv]
m0 = ~d[ctl]["mal"] & np.isin(d[ctl]["cid"], rest); m1 = ~d[cap]["mal"] & np.isin(d[cap]["cid"], rest)
print(f"  {'other benign':<18} n={len(rest):>3}: {d[ctl]['mean'][m0].mean():.3f} -> {d[cap]['mean'][m1].mean():.3f}  (no class-0 change; drift check)")
print("  receivers by amount received (control -> capped mean score):")
rc = {int(c): v["received_c0"] for c, v in rep["receivers"].items()}
qs = np.quantile(list(rc.values()), [0, .5, 1])
for lo, hi, tag in ((qs[0], qs[1], "lower half"), (qs[1], qs[2] + 1, "upper half")):
    ids = [c for c, v in rc.items() if lo <= v < hi]
    v0 = d[ctl]["mean"][~d[ctl]["mal"] & np.isin(d[ctl]["cid"], ids)].mean(); v1 = d[cap]["mean"][~d[cap]["mal"] & np.isin(d[cap]["cid"], ids)].mean()
    print(f"    {tag} (received {int(lo)}-{int(hi-1)}, n={len(ids)}): {v0:.3f} -> {v1:.3f}")
