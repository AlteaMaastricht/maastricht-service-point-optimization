"""Cleaning, spatial imputation and VIF screening of CBS 500 m square statistics.

Port of ``legacy/Cleaning_FillingNA.py`` as a function + CLI. Same steps:
1. drop squares with missing population;
2. drop squares with more than ``max_missing`` missing values (the original
   comment says "more than 4" but the code used ``<= 6``; 6 is kept as the
   default and the discrepancy is documented);
3. impute remaining gaps with the median of the 5 spatially nearest squares
   (X/Y), falling back to the column *mean* (as in the original code, despite
   its "median" comment);
4. convert age and household-type counts to percentages, drop raw counts and a
   few manually chosen collinear columns;
5. compute variance inflation factors.

VIF: uses statsmodels' ``variance_inflation_factor`` when installed (as the
original did); otherwise an equivalent numpy implementation (regression of
each feature on the others without intercept, uncentred R^2 - the same
definition statsmodels uses).

    python3 analysis/cbs_cleaning.py squares.csv --out-dir cleaned/

Input: a CSV/XLSX with the CBS column names listed in docs/DATA.md.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

AGE_COLS = ["Age0-14", "Age15-24", "Age25-44", "Age45-64", "Age65+"]
HH_COLS = ["Single-person households", "Multi-person households w/o kids",
           "Single parent households", "Two-parent households"]
DROP_BEFORE_VIF = AGE_COLS + HH_COLS + ["Rental %", "Percentage high income households",
                                        "Age65+_perc", "Two-parent households_perc"]


def vif_numpy(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    out = []
    for i in range(X.shape[1]):
        y = X[:, i]
        others = np.delete(X, i, axis=1)
        beta, *_ = np.linalg.lstsq(others, y, rcond=None)
        resid = y - others @ beta
        r2 = 1 - (resid @ resid) / (y @ y)   # uncentred, as statsmodels OLS without constant
        out.append(np.inf if r2 >= 1 else 1 / (1 - r2))
    return np.array(out)


def vif(X: np.ndarray) -> np.ndarray:
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        return np.array([variance_inflation_factor(X, i) for i in range(X.shape[1])])
    except ImportError:
        return vif_numpy(X)


def clean(df: pd.DataFrame, max_missing: int = 6, n_neighbors: int = 5):
    pop_cols = [c for c in df.columns if "population" in c.lower()]
    df = df.dropna(subset=pop_cols or [df.columns[1]]).copy()
    df = df[df.isna().sum(axis=1) <= max_missing].copy()

    spatial = [c for c in ("X", "Y") if c in df.columns]
    if len(spatial) == 2 and len(df) > n_neighbors:
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(df[spatial].values)
        _, ind = nn.kneighbors(df[spatial].values)
        for col in df.columns[df.isna().any()]:
            if col in spatial:
                continue
            for idx in df.index[df[col].isna()]:
                pos = df.index.get_loc(idx)
                vals = df.iloc[ind[pos][1:]][col].dropna()
                df.at[idx, col] = vals.median() if len(vals) else df[col].median()
    df = df.fillna(df.mean(numeric_only=True))
    imputed = df.copy()

    df = df.drop(columns=[c for c in df.columns if c.lower() in ("female", "male")] +
                 [c for c in ("Households", "Houses") if c in df.columns])
    for cols in (AGE_COLS, HH_COLS):
        present = [c for c in cols if c in df.columns]
        if present:
            tot = df[present].sum(axis=1)
            for c in present:
                df[c + "_perc"] = np.where(tot > 0, df[c] / tot * 100, 0)
    df = df.drop(columns=[c for c in DROP_BEFORE_VIF if c in df.columns])

    feats = df.select_dtypes(include=[np.number]).drop(columns=spatial, errors="ignore")
    vif_df = pd.DataFrame({"feature": feats.columns, "VIF": vif(feats.values)})
    return imputed, df, vif_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--sheet", default="CBS Squares")
    ap.add_argument("--out-dir", default="cleaned")
    a = ap.parse_args()
    df = (pd.read_excel(a.input, sheet_name=a.sheet) if a.input.endswith((".xlsx", ".xls"))
          else pd.read_csv(a.input))
    imputed, final, vif_df = clean(df)
    os.makedirs(a.out_dir, exist_ok=True)
    imputed.to_csv(os.path.join(a.out_dir, "cbs_squares_imputed.csv"), index=False)
    final.to_csv(os.path.join(a.out_dir, "cbs_squares_features.csv"), index=False)
    vif_df.to_csv(os.path.join(a.out_dir, "vif.csv"), index=False)
    print(vif_df.to_string(index=False))


if __name__ == "__main__":
    main()
