"""
---
title: "Stage 1: eligibility for neuroimaging"
output: tables/supp_table2_ineligibility_reasons.csv
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

def save_table(name, df):
    os.makedirs(TABLES, exist_ok=True)
    df.to_csv(os.path.join(TABLES, f"{name}.csv"), index=False)
    print(f"  -> table written to tables/{name}.csv")

def datasets():
    d = dict(scr=load("screening_eligibility.csv"), coh=load("cohort_measures.csv"),
             nsi=load("nsi_items.csv"), red=load("employment_race.csv"), mot=load("fmri_motion.csv"))
    d.update(FMRI=set(d["mot"].participant_id), COHORT=set(d["coh"].participant_id))
    return d


D = datasets(); scr, FMRI, COHORT = D["scr"], D["FMRI"], D["COHORT"]
header("03  STAGE 1 - ELIGIBILITY FOR NEUROIMAGING")

beh, mri = num(scr.remote_study), num(scr.inpersoninclusion)
S1 = scr[(beh == 1) & mri.isin([0, 1])].copy()
elig = num(S1.inpersoninclusion) == 1
n, ne = len(S1), int(elig.sum())
stage1 = dict(n=n, eligible=ne, pct_eligible=100*ne/n, excluded=n-ne, pct_excluded=100*(n-ne)/n)
print(f"  Behaviourally eligible women with an imaging decision : {n}")
print(f"    eligible for imaging   : {ne} ({100*ne/n:.1f}%)")
print(f"    ineligible for imaging : {n-ne} ({100*(n-ne)/n:.1f}%)")

CUR  = [("mri_contra", "Magnetic-resonance contraindication", "nomricontraindication"),
        ("geoloc",     "Geographic location",                 "geographiclocation"),
        ("psych",      "Schizophrenia, bipolar disorder or autism", "no_psych"),
        ("modsev",     "Moderate or severe traumatic brain injury", "nomodsev"),
        ("neuro",      "Confounding neurological disorder",   "noneuro"),
        ("sud",        "Substance use history",               "nosud"),
        ("recent",     "Brain injury within three months",    "months"),
        ("pregnant",   "Pregnancy",                           "not_pregnant")]
HIST = [("psychmed",   "Psychotropic-medication use",         "nopsychmed"),
        ("bicount",    "One or two partner-inflicted brain injuries", "reqtbis"),
        ("agecap",     "Age above the ceiling of 55 years",   "reqage")]

ex = S1[~elig].copy()
items = ex[[v for _, _, v in CUR + HIST]].apply(num)

none_before = ~items.eq(0).any(axis=1)
print(f"\n  Ineligible women with no failing screening item recorded: {int(none_before.sum())} "
      f"(participant_id {sorted(ex.loc[none_before, 'participant_id'].tolist())})")

rc = load("screening_reclassification.csv")
assert set(rc.participant_id) == set(ex.loc[none_before, "participant_id"]), \
    "reclassification file does not match the women with no recorded reason"
if (rc.verified == 0).any():
    print(f"  WARNING: {int((rc.verified == 0).sum())} rows in data/screening_reclassification.csv are unverified")
for _, r in rc.iterrows():
    if r.reclassified_as == "sud":
        items.loc[ex.participant_id == r.participant_id, "nosud"] = 0
bad_mri = ex.participant_id.isin(rc.loc[rc.reclassified_as == "prior_bad_mri", "participant_id"])

rows, reasons = [], {}
print(f"\n  Reasons for imaging ineligibility (n = {len(ex)}); a woman can fail more than one.")
print(f"  {'':44}{'n':>5}{'% excluded':>12}{'% assessed':>12}{'assessed':>11}")
for grp, lst in [("Criteria retained throughout", CUR), ("Initial but dropped criteria", HIST)]:
    print(f"   {grp}")
    for key, lbl, v in lst:
        col = items[v]; f_ = int((col == 0).sum()); k = int(col.notna().sum())
        reasons[key] = dict(label=lbl, n=f_, pct_excluded=100*f_/len(ex), pct_assessed=100*f_/k,
                            assessed=k, denom=len(ex))
        rows.append(dict(group=grp, criterion=lbl, n=f_, pct_of_excluded=round(100*f_/len(ex), 1),
                         pct_of_assessed=round(100*f_/k, 1), assessed=f"{k}/{len(ex)}"))
        print(f"     {lbl:<42}{f_:>5}{100*f_/len(ex):>11.1f}%{100*f_/k:>11.1f}%{k:>7}/{len(ex)}")
    if grp.startswith("Criteria retained"):
        nb = int(bad_mri.sum())
        reasons["bad_mri"] = dict(label="Prior bad MRI experience", n=nb)
        rows.append(dict(group=grp, criterion="Prior bad MRI experience", n=nb,
                         pct_of_excluded="", pct_of_assessed="", assessed=""))
        print(f"     {'Prior bad MRI experience':<42}{nb:>5}{'-':>12}{'-':>12}{'-':>11}")
save_table("supp_table2_ineligibility_reasons", pd.DataFrame(rows))

C = items[[v for _, _, v in CUR]]; H = items[[v for _, _, v in HIST]]
fail_cur  = C.eq(0).any(axis=1)
only_hist = ~fail_cur & H.eq(0).any(axis=1)
no_reason = ~fail_cur & ~only_hist & ~bad_mri
stage1.update(fail_current=int(fail_cur.sum()), pct_fail_current=100*fail_cur.mean(),
              only_relaxed=int(only_hist.sum()), pct_only_relaxed=100*only_hist.mean(),
              bad_mri=int(bad_mri.sum()), no_reason=int(no_reason.sum()),
              projected_eligible=ne + int((~fail_cur & ~bad_mri).sum()))
stage1["pct_projected"] = 100*stage1["projected_eligible"]/n
print(f"\n  Failed at least one retained criterion : {stage1['fail_current']} ({stage1['pct_fail_current']:.0f}%)")
print(f"  Failed only dropped criteria           : {stage1['only_relaxed']} ({stage1['pct_only_relaxed']:.0f}%)")
print(f"  Prior bad MRI experience only          : {stage1['bad_mri']}")
print(f"  No reason after reclassification       : {stage1['no_reason']}")
print(f"  Projected eligibility under the retained criteria alone: "
      f"{stage1['projected_eligible']}/{n} = {stage1['pct_projected']:.1f}%")

ids_hist, ids_cur = set(ex.loc[only_hist, "participant_id"]), set(ex.loc[fail_cur, "participant_id"])
stage1.update(only_relaxed_imaged=len(ids_hist & FMRI),
              pct_only_relaxed_imaged=100*len(ids_hist & FMRI)/len(ids_hist),
              fail_current_imaged=len(ids_cur & FMRI),
              fmri_with_screen=len(set(S1.participant_id) & FMRI),
              fmri_coded_ineligible=len(set(ex.participant_id) & FMRI))
print(f"\n  Of the {len(ids_hist)} women who failed only dropped criteria, {len(ids_hist & FMRI)} "
      f"({stage1['pct_only_relaxed_imaged']:.0f}%) later have rsfMRI data.")
print(f"  Of the {len(ids_cur)} women who failed a retained criterion, {len(ids_cur & FMRI)} later have rsfMRI data.")
print(f"  Of the {len(FMRI)} women with rsfMRI data, {stage1['fmri_with_screen']} have a Stage 1 screening "
      f"record and {stage1['fmri_coded_ineligible']} of those were coded ineligible at screening.")

