"""
---
title: "CTQ-SF and CASR-SF scoring"
output: tables/ctq_casr_scored.csv
---
"""
import sys, re
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


CTQ_REVERSE = [2, 5, 7, 13, 19, 26, 28]
CTQ_MD      = [10, 16, 22]
CTQ_CLIN    = [i for i in range(1, 29) if i not in CTQ_MD]
CTQ_SUBSCALES = {"emotional_abuse": [3, 8, 14, 18, 25], "physical_abuse": [9, 11, 12, 15, 17],
                 "sexual_abuse": [20, 21, 23, 24, 27], "emotional_neglect": [5, 7, 13, 19, 28],
                 "physical_neglect": [1, 2, 4, 6, 26]}

def ctq_columns(df):
    cols = {}
    for c in df.columns:
        m = re.fullmatch(r"ctq_?(\d{1,2})", c.lower())
        if m and 1 <= int(m.group(1)) <= 28:
            cols[int(m.group(1))] = c
    missing = [i for i in range(1, 29) if i not in cols]
    if missing:
        sys.exit(f"CTQ items not found in export: {missing}")
    return cols

def score_ctq(df):
    cols = ctq_columns(df)
    X = pd.DataFrame({i: num(df[cols[i]]) for i in range(1, 29)}, index=df.index)
    lo, hi = np.nanmin(X.values), np.nanmax(X.values)
    if lo == 0 and hi <= 4:
        print("  CTQ items are coded 0-4 in this export; shifting to 1-5.")
        X = X + 1
    bad = ~X.isin([1, 2, 3, 4, 5]) & X.notna()
    if bad.any().any():
        sys.exit("CTQ responses outside 1-5 after recoding; check the export.")
    md = (X[CTQ_MD] == 5).sum(axis=1).where(X[CTQ_MD].notna().all(axis=1))
    S = X.copy()
    S[CTQ_REVERSE] = 6 - S[CTQ_REVERSE]
    total = S[CTQ_CLIN].sum(axis=1).where(S[CTQ_CLIN].notna().all(axis=1))
    out = pd.DataFrame({"ctq_total_25": total, "ctq_minimization_denial": md}, index=df.index)
    for k, items in CTQ_SUBSCALES.items():
        out[f"ctq_{k}"] = S[items].sum(axis=1).where(S[items].notna().all(axis=1))
    return out

def casr_columns(df):
    found = {}
    for c in df.columns:
        m = re.fullmatch(r"(cas\w*?)_?(\d{1,2})_([abc])", c.lower())
        if m: found.setdefault(int(m.group(2)), {})[m.group(3)] = c
    items = sorted(i for i in found if "a" in found[i] and "b" in found[i])
    if len(items) != 15:
        sys.exit(f"Expected 15 CASR-SF items with _a and _b fields, found {len(items)}: {items}")
    return {i: found[i] for i in items}

def score_casr(df):
    cols = casr_columns(df)
    B = {}
    for i, c in cols.items():
        gate, freq = num(df[c["a"]]), num(df[c["b"]])
        B[i] = np.where(gate == 0, 0, freq)
    B = pd.DataFrame(B, index=df.index)
    if (B.max().max() > 5) or (B.min().min() < 0):
        sys.exit("CASR-SF past-12-month frequencies outside 0-5; check the export.")
    return pd.DataFrame({"casr_past12": B.sum(axis=1).where(B.notna().all(axis=1))}, index=df.index)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python 01_score_ctq_casr.py <REDCap item-level export.csv>")
    raw = pd.read_csv(sys.argv[1])
    header("01  CTQ-SF and CASR-SF scoring")
    out = pd.concat([raw[["participant_id"]], score_ctq(raw), score_casr(raw)], axis=1)
    for v in out.columns[1:]:
        print(f"  {v:<28} n {out[v].notna().sum():>3}  mean {out[v].mean():6.1f}  range {out[v].min():.0f}-{out[v].max():.0f}")
    save_table("ctq_casr_scored", out)
