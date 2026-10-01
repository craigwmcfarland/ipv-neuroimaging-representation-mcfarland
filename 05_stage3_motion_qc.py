"""
---
title: "Stage 3: motion quality control"
output: tables/supp_table3_thresholds.csv; tables/supp_table4_excluded_sample.csv; tables/supp_table5_sensitivity.csv
---
"""
import os
import numpy as np, pandas as pd
from scipy.stats import fisher_exact, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TABLES = os.path.join(HERE, "tables")

def load(name):
    return pd.read_csv(os.path.join(DATA, name))

def num(s):
    return pd.to_numeric(s, errors="coerce")

def header(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)

def save_table(name, df):
    os.makedirs(TABLES, exist_ok=True)
    df.to_csv(os.path.join(TABLES, f"{name}.csv"), index=False)
    print(f"  -> table written to tables/{name}.csv")

def datasets():
    d = dict(scr=load("screening_eligibility.csv"), coh=load("cohort_measures.csv"),
             nsi=load("nsi_items.csv"), red=load("employment_race.csv"), mot=load("fmri_motion.csv"))
    d.update(FMRI=set(d["mot"].participant_id), COHORT=set(d["coh"].participant_id))
    return d

def haldane_or(a, b, c, e):
    A, B, C, E = a+.5, b+.5, c+.5, e+.5
    orr = (A*E)/(B*C); se = np.sqrt(1/A + 1/B + 1/C + 1/E)
    return orr, float(np.exp(np.log(orr)-1.96*se)), float(np.exp(np.log(orr)+1.96*se))

def prop(name, a, b, c, e, quiet=False):
    _, p = fisher_exact([[a, b], [c, e]])
    orr, lo, hi = haldane_or(a, b, c, e)
    d = dict(label=name, kind="binary", a=a, b=b, c=c, d=e, n1=a+b, n0=c+e,
             pct1=100*a/(a+b), pct0=100*c/(c+e), OR=orr, lo=lo, hi=hi, p=p)
    if not quiet:
        print(f"  {name:<42}{d['pct1']:6.1f}% ({a}/{a+b}) vs {d['pct0']:6.1f}% ({c}/{c+e})"
              f"   OR = {orr:.2f} [{lo:.2f}, {hi:.2f}]   Fisher P = {p:.3f}")
    return d


mot = datasets()["mot"]
header(f"05  STAGE 3 - MOTION QUALITY CONTROL (rsfMRI, n = {len(mot)})")
R = {}
fd, inj = mot.mean_fd_mm, mot.ipv_bi_count

rho, prho = spearmanr(fd, inj)
rlog = float(np.corrcoef(fd, np.log1p(inj))[0, 1])
R["fd_injury"] = dict(spearman_rho=rho, spearman_p=prho, pearson_log=rlog, n=len(mot))
print(f"  Mean FD vs BI count: Spearman rho = {rho:+.3f}, P = {prho:.3f}; "
      f"Pearson r on log(1 + count) = {rlog:+.3f}")

print("\n  Supplementary Tables 3 and 4")
R["thresholds"], t3, t4 = {}, [], []
hi, lo = mot[inj >= 10], mot[inj < 10]
for thr in [0.15, 0.20, 0.30]:
    a, c = int((hi.mean_fd_mm >= thr).sum()), int((lo.mean_fd_mm >= thr).sum())
    p = fisher_exact([[a, len(hi) - a], [c, len(lo) - c]])[1]
    exd, ret = mot[fd >= thr], mot[fd < thr]
    R["thresholds"][f"{thr:.2f}"] = dict(a=a, n_hi=len(hi), c=c, n_lo=len(lo), p=p,
        pct_hi=100*a/len(hi), pct_lo=100*c/len(lo), n_excluded=len(exd),
        pct_excluded=100*len(exd)/len(mot), med_inj_excluded=float(exd.ipv_bi_count.median()),
        med_inj_retained=float(ret.ipv_bi_count.median()))
    t3.append([f"Mean FD ≥ {thr:.2f} mm", f"{a}/{len(hi)} ({100*a/len(hi):.0f}%)",
               f"{c}/{len(lo)} ({100*c/len(lo):.0f}%)", f"{p:.3f}".lstrip("0")])
    t4.append([f"Mean FD ≥ {thr:.2f} mm", len(exd), f"{100*len(exd)/len(mot):.1f}%",
               f"{exd.ipv_bi_count.median():.1f}", f"{ret.ipv_bi_count.median():.1f}"])
    print(f"    FD >= {thr:.2f}: >=10 BIs {a}/{len(hi)} vs <10 {c}/{len(lo)}, Fisher P = {p:.3f}; "
          f"{len(exd)} excluded ({100*len(exd)/len(mot):.1f}%), median BIs excluded "
          f"{exd.ipv_bi_count.median():.1f} vs retained {ret.ipv_bi_count.median():.1f}")
save_table("supp_table3_thresholds", pd.DataFrame(t3, columns=["Threshold", "Ten or more injuries excluded",
                                                               "Fewer than ten excluded", "Fisher P"]))
save_table("supp_table4_excluded_sample", pd.DataFrame(t4, columns=["Threshold", "n excluded", "% of sample",
                                                                    "Median injuries, excluded",
                                                                    "Median injuries, retained"]))

print("\n  Supplementary Table 5")
t5 = []
for cut, key, lbl in [(10, "cut10", "Ten or more vs fewer injuries"),
                      (5, "cut5", "Five or more vs fewer injuries"),
                      (1, "cut1", "Any injury vs none")]:
    h, l = mot[inj >= cut], mot[inj < cut]
    a, c = int((h.mean_fd_mm >= 0.20).sum()), int((l.mean_fd_mm >= 0.20).sum())
    d = prop(lbl, a, len(h) - a, c, len(l) - c); R[key] = d
    t5.append([lbl, f"{a}/{len(h)} ({100*a/len(h):.0f}%)", f"{c}/{len(l)} ({100*c/len(l):.0f}%)",
               f"{d['OR']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]", f"{d['p']:.3f}".lstrip("0")])

nex = int((fd >= 0.20).sum())
R["motion"] = dict(n=len(mot), n_excluded=nex, pct_excluded=100*nex/len(mot),
                   n_retained=len(mot) - nex, pct_retained=100*(len(mot) - nex)/len(mot),
                   excluded_injuries=sorted(mot.loc[fd >= 0.20, "ipv_bi_count"].astype(int).tolist()),
                   n_high=int((inj >= 10).sum()), pct_high=100*(inj >= 10).mean(),
                   n_high_excluded=int(((inj >= 10) & (fd >= 0.20)).sum()),
                   max_injuries=int(inj.max()))
print(f"\n  At 0.20 mm: {nex}/{len(mot)} excluded ({100*nex/len(mot):.1f}%), retained {len(mot)-nex} "
      f"({R['motion']['pct_retained']:.1f}%); BI counts of the excluded: {R['motion']['excluded_injuries']}")

print("  Leave-one-out (each excluded woman removed in turn):")
loo = []
for k, (i, row) in enumerate(mot[fd >= 0.20].iterrows(), 1):
    m = mot.drop(i); h, l = m[m.ipv_bi_count >= 10], m[m.ipv_bi_count < 10]
    a, c = int((h.mean_fd_mm >= 0.20).sum()), int((l.mean_fd_mm >= 0.20).sum())
    p = fisher_exact([[a, len(h) - a], [c, len(l) - c]])[1]
    orr, olo, ohi = haldane_or(a, len(h) - a, c, len(l) - c)
    loo.append(dict(injuries=int(row.ipv_bi_count), p=p, a=a, n1=len(h), c=c, n0=len(l),
                    pct1=100*a/len(h), pct0=100*c/len(l), OR=orr, lo=olo, hi=ohi))
    t5.append([f"Leave-one-out {k}: excluded participant with {int(row.ipv_bi_count)} injuries removed",
               f"{a}/{len(h)} ({100*a/len(h):.0f}%)", f"{c}/{len(l)} ({100*c/len(l):.0f}%)",
               f"{orr:.2f} [{olo:.2f}, {ohi:.2f}]", f"{p:.3f}".lstrip("0")])
    print(f"    remove woman with {int(row.ipv_bi_count):>3} BIs -> {a}/{len(h)} vs {c}/{len(l)}, "
          f"OR = {orr:.2f} [{olo:.2f}, {ohi:.2f}], P = {p:.3f}")
R["leave_one_out"] = loo
R["loo_max_p"] = max(x["p"] for x in loo)
save_table("supp_table5_sensitivity", pd.DataFrame(t5, columns=["Analysis", "Exposed group excluded",
                                                                "Comparison excluded", "Odds ratio [95% CI]",
                                                                "Fisher P"]))

g = np.round(np.arange(0.080, 0.4201, 0.0005), 4)
rh = np.array([100*(hi.mean_fd_mm.values >= t).mean() for t in g])
rl = np.array([100*(lo.mean_fd_mm.values >= t).mean() for t in g])
gap = rh - rl; k = int(gap.argmax()); wide = g[gap >= 35]
R["motion"]["sweep"] = dict(t_max=float(g[k]), gap_max=float(gap[k]), hi_at_max=float(rh[k]),
                            lo_at_max=float(rl[k]), t_gap_positive=float(g[np.where(gap > 0)[0][0]]),
                            t_wide_lo=float(wide[0]), t_wide_hi=float(wide[-1]),
                            gap_at={f"{t:.2f}": float(gap[np.argmin(abs(g - t))]) for t in (0.15, 0.20, 0.30)})
print(f"\n  Threshold sweep (Figure 4B): gap between groups largest at {g[k]:.3f} mm "
      f"({rh[k]:.0f}% vs {rl[k]:.0f}%); gap of 35 points or more from {wide[0]:.3f} to {wide[-1]:.3f} mm")

