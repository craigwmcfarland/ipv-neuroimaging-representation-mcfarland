"""
---
title: "Partner-inflicted brain injury count distribution"
output: printed summary
---
"""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TABLES = os.path.join(HERE, "tables")

def load(name):
    return pd.read_csv(os.path.join(DATA, name))

def num(s):
    return pd.to_numeric(s, errors="coerce")

def header(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)

def datasets():
    d = dict(scr=load("screening_eligibility.csv"), coh=load("cohort_measures.csv"),
             nsi=load("nsi_items.csv"), red=load("employment_race.csv"), mot=load("fmri_motion.csv"))
    d.update(FMRI=set(d["mot"].participant_id), COHORT=set(d["coh"].participant_id))
    return d


D = datasets(); coh, mot = D["coh"], D["mot"]
header("02  DISTRIBUTION OF CUMULATIVE PARTNER-INFLICTED BI COUNT")

u = num(coh.ipv_bi_count).dropna()
R = dict(n=len(u), median=float(u.median()), q1=float(u.quantile(.25)), q3=float(u.quantile(.75)),
         p90=float(u.quantile(.90)), mean=float(u.mean()), sd=float(u.std(ddof=1)),
         max=float(u.max()), skew=float(u.skew()))
print(f"  Enrolled cohort (n = {len(u)})")
print(f"    median {R['median']:.0f}, IQR {R['q1']:.0f}-{R['q3']:.0f}, 90th percentile {R['p90']:.0f}, "
      f"maximum {R['max']:.0f}")
print(f"    mean {R['mean']:.1f}, SD {R['sd']:.1f}, adjusted Fisher-Pearson G1 = {R['skew']:.1f}")

cap = num(coh.ipv_bi_count_capped25)
assert (cap == np.minimum(num(coh.ipv_bi_count), 25)).all(), "capped column is not min(count, 25)"
R["cohort_over_cap"] = int((u > 25).sum())
R["fmri_over_cap"] = int((mot.ipv_bi_count > 25).sum())
R["fmri_ten_or_more"] = int((mot.ipv_bi_count >= 10).sum())
R["fmri_n"] = len(mot)
print(f"\n  Winsorizing at 25")
print(f"    enrolled women with a count above 25           : {R['cohort_over_cap']} of {len(u)}")
print(f"    rsfMRI women with a count above 25             : {R['fmri_over_cap']} of {len(mot)}")
print(f"    rsfMRI women with ten or more BIs (for contrast): {R['fmri_ten_or_more']} of {len(mot)}")

bands = coh.exposure_band.value_counts()
R["bands"] = {k: int(v) for k, v in bands.items()}
print("\n  Exposure bands in the enrolled cohort: " + ", ".join(f"{k}: {v}" for k, v in bands.items()))

