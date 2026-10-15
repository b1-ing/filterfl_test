"""Do benign 'spike' records (benign clients scoring like backdoored models) occur only when the global model is already
backdoored?  Joins the probe log to the per-round global ASR of the model the clients STARTED from (previous round's
post-aggregation ASR; round 1 = 0).  Zero GPU.  Usage: python scripts/spike_vs_asr.py [run]"""
import csv, sys
import numpy as np
run = sys.argv[1] if len(sys.argv) > 1 else "hpc_s11_cifar_cap15_a_s0"
z = np.load(f"results/{run}_probe.npz"); ep, cid, mal, P = z["epoch"], z["cid"], z["is_mal"].astype(bool), z["probs"]
s = P.mean(1)[:, 0]
rows = {int(x["epoch"]): x for x in csv.DictReader(open(f"results/{run}.csv"))}
asr_prev = np.array([float(rows[int(e) - 1]["asr"]) if int(e) > 1 else 0.0 for e in ep])
ma_prev = np.array([float(rows[int(e) - 1]["ma"]) if int(e) > 1 else 10.0 for e in ep])
ben = ~mal
p99, p999 = np.quantile(s[ben], .99), np.quantile(s[ben], .999)
print(f"benign records {int(ben.sum())}; spike thresholds: p99={p99:.3f} (n={(s[ben]>p99).sum()}), p99.9={p999:.3f} (n={(s[ben]>p999).sum()}), >0.30 (n={(s[ben]>0.30).sum()})")
def rank(a): return np.argsort(np.argsort(a)).astype(float)
def sp(a, b): return float(np.corrcoef(rank(a), rank(b))[0, 1])
print("\n== benign score vs ASR of the model the client started from ==")
print(f"Spearman(benign score, prev-round global ASR), all benign records: {sp(s[ben], asr_prev[ben]):+.3f}   (vs prev-round MA: {sp(s[ben], ma_prev[ben]):+.3f}; vs epoch: {sp(s[ben], ep[ben]):+.3f})")
print(f"\n{'prev-round ASR':<16}{'records':>8}{'mean score':>11}{'p95':>7}{'>p99 rate':>10}{'>0.30 rate':>11}{'#>0.30':>8}")
for lo, hi in ((0, 10), (10, 30), (30, 60), (60, 80), (80, 101)):
    m = ben & (asr_prev >= lo) & (asr_prev < hi)
    if m.sum(): print(f"{lo:>3}-{hi:<12}{int(m.sum()):>8}{s[m].mean():>11.3f}{np.quantile(s[m],.95):>7.3f}{(s[m]>p99).mean():>10.4f}{(s[m]>0.30).mean():>11.4f}{int((s[m]>0.30).sum()):>8}")
print("\n== control for training stage: same analysis inside epoch windows (ASR varies within early windows) ==")
print(f"{'epochs':<10}{'ASR range':>12}{'records':>8}{'Spearman(score,ASR)':>20}{'Spearman(score,MA)':>20}{'#>p99':>7}{'#>p99 @ASR<30':>14}{'recs @ASR<30':>13}")
for lo, hi in ((1, 100), (101, 300), (301, 600), (601, 1000)):
    m = ben & (ep >= lo) & (ep <= hi); lowm = m & (asr_prev < 30)
    print(f"{lo}-{hi:<6}{asr_prev[m].min():>6.0f}-{asr_prev[m].max():<5.0f}{int(m.sum()):>8}{sp(s[m], asr_prev[m]):>+20.3f}{sp(s[m], ma_prev[m]):>+20.3f}{int((s[m]>p99).sum()):>7}{int((s[lowm]>p99).sum()):>14}{int(lowm.sum()):>13}")
print("\n== the repeat spikers: their spike records vs their other records ==")
print(f"{'cid':>4}{'records':>8}{'spikes>p99':>11}{'ASR_prev at spikes (min/med/max)':>34}{'records @ASR<30':>17}{'spikes @ASR<30':>15}")
for c in (79, 2, 39, 31, 9):
    m = ben & (cid == c); sp_m = m & (s > p99); lowm = m & (asr_prev < 30)
    a = asr_prev[sp_m]
    print(f"{c:>4}{int(m.sum()):>8}{int(sp_m.sum()):>11}{('%.0f/%.0f/%.0f' % (a.min(), np.median(a), a.max())) if len(a) else '-':>34}{int(lowm.sum()):>17}{int((lowm & (s > p99)).sum()):>15}")
# spike-rate for 'all benign' at low ASR rounds vs high ASR, matched on the same clients (within-client comparison)
print("\n== within-client: for each of the 90 benign clients, mean score at ASR_prev<30 vs >=60 (clients with >=5 records in both) ==")
d = []
for c in sorted(set(cid[ben].tolist())):
    lo = ben & (cid == c) & (asr_prev < 30); hi = ben & (cid == c) & (asr_prev >= 60)
    if lo.sum() >= 5 and hi.sum() >= 5: d.append((s[lo].mean(), s[hi].mean()))
d = np.array(d); print(f"clients: {len(d)}; mean score low-ASR {d[:,0].mean():.3f} vs high-ASR {d[:,1].mean():.3f}; higher in high-ASR rounds for {int((d[:,1]>d[:,0]).sum())}/{len(d)} clients")
