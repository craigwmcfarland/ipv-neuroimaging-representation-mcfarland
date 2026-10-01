"""
---
title: "Stage 2: participation in neuroimaging"
output: tables/supp_table1_participation.csv
---
"""
import os
import numpy as np, pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu

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

SEED, NBOOT = 0, 5000

def hedges_g(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); na, nb = len(a), len(b)
    if na < 2 or nb < 2: return np.nan
    sp = np.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2))
    if sp == 0: return np.nan
    return (a.mean()-b.mean())/sp * (1 - 3/(4*(na+nb)-9))

def boot_ci(a, b, n=NBOOT, seed=SEED):
    a, b = np.asarray(a, float), np.asarray(b, float); rng = np.random.default_rng(seed)
    return np.nanpercentile([hedges_g(rng.choice(a, len(a), True), rng.choice(b, len(b), True))
                             for _ in range(n)], [2.5, 97.5])

def cliffs_delta(a, b):
    u, _ = mannwhitneyu(a, b, alternative="two-sided")
    return 2*u/(len(a)*len(b)) - 1

def cont(name, a, b, skewed=False, quiet=False):
    a = pd.Series(a).dropna().values.astype(float); b = pd.Series(b).dropna().values.astype(float)
    g = hedges_g(a, b); lo, hi = boot_ci(a, b)
    u, p = mannwhitneyu(a, b, alternative="two-sided")
    d = dict(label=name, kind="continuous", m1=a.mean(), sd1=a.std(ddof=1), n1=len(a),
             m0=b.mean(), sd0=b.std(ddof=1), n0=len(b), g=g, lo=lo, hi=hi, U=float(u), p=p)
    if skewed:
        d.update(md1=float(np.median(a)), q1_1=float(np.percentile(a, 25)), q3_1=float(np.percentile(a, 75)),
                 md0=float(np.median(b)), q1_0=float(np.percentile(b, 25)), q3_0=float(np.percentile(b, 75)),
                 delta=cliffs_delta(a, b))
    if not quiet:
        print(f"  {name:<42}{d['m1']:8.2f} ({d['sd1']:6.2f}) vs {d['m0']:8.2f} ({d['sd0']:6.2f})"
              f"   g = {g:+.2f} [{lo:+.2f}, {hi:+.2f}]   U = {u:7.1f}   P = {p:.3f}   n {len(a)}/{len(b)}")
        if skewed:
            print(f"  {'':<42}median [IQR] {d['md1']:.0f} [{d['q1_1']:.0f}-{d['q3_1']:.0f}] vs "
                  f"{d['md0']:.0f} [{d['q1_0']:.0f}-{d['q3_0']:.0f}]   Cliff's delta = {d['delta']:+.2f}")
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

NSI_SUB = {"vestibular": [1, 2, 3], "somatosensory": [4, 5, 6, 7, 8, 9, 10],
           "cognitive": [11, 12, 13, 14], "affective": [15, 16, 17, 18, 19, 20, 21, 22]}

def participation(D, ids, tag):
    coh, nsi, red, scr, COHORT = D["coh"], D["nsi"], D["red"], D["scr"], D["COHORT"]
    R = {}; P = lambda k: f"{tag}.{k}"
    c = coh.copy(); c["im"] = c.participant_id.isin(ids)
    R[P("n_imaged")], R[P("n_not")] = int(c.im.sum()), int((~c.im).sum())
    print(f"  imaged {int(c.im.sum())} versus not imaged {int((~c.im).sum())}")
    two = lambda df, v: (num(df.loc[df.im, v]).dropna(), num(df.loc[~df.im, v]).dropna())

    print("\n  Socioeconomic")
    R[P("education")] = cont("Education (years)", *two(c, "education_years"))
    r = red[red.participant_id.isin(COHORT)].copy(); r["im"] = r.participant_id.isin(ids)
    e = num(r.employ_curr)
    R[P("employment")] = prop("Currently employed",
        int((r.im & (e == 1)).sum()), int((r.im & (e == 0)).sum()),
        int((~r.im & (e == 1)).sum()), int((~r.im & (e == 0)).sum()))

    print("\n  Psychiatric symptoms")
    for k, lbl, v in [("phq9", "Depression, PHQ-9", "depression_phq9"),
                      ("gad7", "Anxiety, GAD-7", "anxiety_gad7"),
                      ("pcl5", "Post-traumatic stress, PCL-5", "ptsd_pcl5")]:
        R[P(k)] = cont(lbl, *two(c, v))

    print("\n  Neurobehavioral symptoms (NSI; total needs >= 18 of 22 items, subdomains all but one)")
    b = nsi[nsi.participant_id.isin(COHORT)].copy(); b["im"] = b.participant_id.isin(ids)
    it = [f"nsi{i}" for i in range(1, 23)]
    b["nsi_total"] = np.where(b[it].notna().sum(axis=1) >= 18, b[it].sum(axis=1), np.nan)
    for nm, idx in NSI_SUB.items():
        cols = [f"nsi{i}" for i in idx]
        b["nsi_" + nm] = np.where(b[cols].notna().sum(axis=1) >= len(idx) - 1, b[cols].mean(axis=1), np.nan)
    R[P("nsi_total")] = cont("Neurobehavioral symptoms, NSI total", *two(b, "nsi_total"))
    for nm in NSI_SUB:
        R[P("nsi_" + nm)] = cont(f"NSI {nm}", *two(b, "nsi_" + nm))

    print("\n  Injury and abuse history")
    s = scr[scr.participant_id.isin(COHORT)].copy(); s["im"] = s.participant_id.isin(ids)
    pp = num(s.pastpresent)
    s["ong"] = np.where(pp.isin([1, 2]), (pp == 2).astype(float), np.nan)
    R[P("ongoing_coding")] = dict(n_screened=len(s), past=int((pp == 1).sum()), current=int((pp == 2).sum()),
                                  no_abusive_relationship=int((pp == 3).sum()), blank=int(pp.isna().sum()))
    print(f"  ongoing-abuse item among {len(s)} enrolled women with a screening record: "
          f"past {int((pp==1).sum())}, current {int((pp==2).sum())}, "
          f"'no abusive relationship' {int((pp==3).sum())}, blank {int(pp.isna().sum())}")
    a, b_ = int((s.im & (s.ong == 1)).sum()), int((s.im & (s.ong == 0)).sum())
    c_, d_ = int((~s.im & (s.ong == 1)).sum()), int((~s.im & (s.ong == 0)).sum())
    R[P("ongoing")] = prop("Ongoing abuse (share of each imaging group)", a, b_, c_, d_)
    R[P("ongoing_by_status")] = prop("Imaged (share of ongoing vs past abuse)", a, c_, b_, d_)
    R[P("bi_raw")] = cont("Partner-inflicted BI count, raw", *two(c, "ipv_bi_count"), skewed=True)
    R[P("bi_cap")] = cont("Partner-inflicted BI count, winsorized at 25", *two(c, "ipv_bi_count_capped25"), skewed=True)
    R[P("strang")] = cont("Strangulation-related BI count", *two(c, "strangulation_bi_count"), skewed=True)
    R[P("ctq")]    = cont("Childhood maltreatment, CTQ", *two(c, "childhood_maltreatment_ctq"))
    R[P("cas")]    = cont("Abuse severity, CASR-SF (past 12 months)", *two(c, "abuse_severity_cas"))
    R[P("years_since")] = cont("Years since last IPV brain injury", *two(c, "years_since_last_ipv_bi"))

    print("\n  Demographic")
    rc = r[[f"race___{i}" for i in range(7)]].apply(num)
    r["nonwhite"] = np.where(rc.fillna(0).sum(axis=1) == 0, np.nan, (num(r["race___0"]) == 0).astype(float))
    x, y = r.loc[r.im, "nonwhite"].dropna(), r.loc[~r.im, "nonwhite"].dropna()
    R[P("minority")] = prop("Non-White race", int(x.sum()), int(len(x) - x.sum()), int(y.sum()), int(len(y) - y.sum()))
    R[P("age")] = cont("Age (years)", *two(c, "age"))

    print("\n  Highest symptom-burden quartile (sum of z-scored PHQ-9, GAD-7, PCL-5; top 25%)")
    z = lambda v: (v - v.mean()) / v.std()
    comb = z(num(c.depression_phq9)) + z(num(c.anxiety_gad7)) + z(num(c.ptsd_pcl5))
    q4 = comb >= comb.quantile(0.75)
    a, b_ = int((q4 & c.im).sum()), int((q4 & ~c.im).sum())
    c_, d_ = int((~q4 & c.im).sum()), int((~q4 & ~c.im).sum())
    R[P("quartile")] = prop("Imaged (share of top quartile vs rest)", a, b_, c_, d_)
    R[P("quartile_by_group")] = prop("Top quartile (share of each imaging group)", a, c_, b_, d_)
    return R

def fmt_p(p):
    return "<.001" if p < .001 else f"{p:.3f}".lstrip("0")

def supp_table1(R, tag):
    P = lambda k: R[f"{tag}.{k}"]
    def ms(d, k=1): return f"{d[f'm{k}']:.1f} ({d[f'sd{k}']:.1f})"
    def md(d, k=1): return f"{d[f'md{k}']:.0f} [{d[f'q1_{k}']:.0f}–{d[f'q3_{k}']:.0f}]"
    def pc(d, k=1):
        num_, n_ = (d["a"], d["n1"]) if k == 1 else (d["c"], d["n0"])
        return f"{100*num_/n_:.1f}% ({num_}/{n_})"
    rows = []
    def add(lbl, d, kind):
        if kind == "g":
            rows.append([lbl, ms(d, 1), ms(d, 0), f"{d['g']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]",
                         "Mann-Whitney U", f"{d['U']:.1f}", fmt_p(d["p"]), f"{d['n1']}/{d['n0']}"])
        elif kind == "delta":
            rows.append([lbl, md(d, 1), md(d, 0), f"δ {d['delta']:.2f}",
                         "Mann-Whitney U", f"{d['U']:.1f}", fmt_p(d["p"]), f"{d['n1']}/{d['n0']}"])
        else:
            rows.append([lbl, pc(d, 1), pc(d, 0), f"OR {d['OR']:.2f} [{d['lo']:.2f}, {d['hi']:.2f}]",
                         "Fisher exact", "", fmt_p(d["p"]), str(d["n1"] + d["n0"])])
    add("Education (years)", P("education"), "g")
    add("Currently employed", P("employment"), "or")
    add("Depression, PHQ-9", P("phq9"), "g")
    add("Anxiety, GAD-7", P("gad7"), "g")
    add("Post-traumatic stress, PCL-5", P("pcl5"), "g")
    add("Neurobehavioral symptoms, NSI total", P("nsi_total"), "g")
    for nm in NSI_SUB: add(f"NSI {nm}", P("nsi_" + nm), "g")
    add("Ongoing abuse", P("ongoing"), "or")
    add("Raw partner-inflicted BI count median [IQR]", P("bi_raw"), "delta")
    add("Winsorized at 25 partner-inflicted BI count mean (SD)", P("bi_cap"), "g")
    add("Strangulation-related BI count", P("strang"), "delta")
    add("Childhood maltreatment, CTQ", P("ctq"), "g")
    add("Abuse severity, CASR-SF (past 12 months)", P("cas"), "g")
    add("Years since last IPV brain injury", P("years_since"), "g")
    add("Non-White race", P("minority"), "or")
    add("Age (years)", P("age"), "g")
    add("Highest symptom-burden quartile", P("quartile_by_group"), "or")
    rows = [[c.replace("-", "\u2212") if i == 3 else c for i, c in enumerate(r)] for r in rows]
    return pd.DataFrame(rows, columns=["Characteristic", "Imaged", "Not imaged", "Effect size [95% CI]",
                                       "Test (two-sided)", "Statistic", "P", "n"])


D = datasets()
header("04  STAGE 2 - PARTICIPATION: rsfMRI (n = 61) versus no imaging data (n = 146)")
R = participation(D, D["FMRI"], "fmri")
c = D["coh"]
R["fmri.years_since_available"] = int(c.years_since_last_ipv_bi.notna().sum())
print(f"\n  Years since last IPV BI recorded for {R['fmri.years_since_available']} of {len(c)} enrolled women")
save_table("supp_table1_participation", supp_table1(R, "fmri"))
