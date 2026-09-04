"""
Daily realized measures from the cleaned 1-minute price series (btc_1min.csv).

Mapping to the SDE observables over [t-1, t]  (gmm_math_spec, Thm 5.3):

    RV_t   = sum r_i^2                    -> Int sigma^2 ds + Int h^2 mu(ds,dx)
                                            (total quadratic variation)
    TPV_t  = mu23^-3 * n/(n-2)            -> Int (V^c + V^j) ds   (jump-robust,
             * sum |r_{i-2} r_{i-1} r_i|^{2/3}                     tripower)

    The three exponents sum to 2 (the power of variance): tripower VARIATION.
    Using 2/3-per-return (not 4/3, which sums to 4 and estimates quarticity) is
    what makes TPV -> integrated variance rather than integrated quarticity.
    Verified empirically: at 1-min TPV/RV ~= 0.92, matching MedRV/RV, and
    scale-stable across sampling frequencies (a quarticity estimator would
    instead collapse toward zero as the grid refines).
    MedRV_t= c_med * n/(n-2)              -> Int (V^c + V^j) ds   (jump-robust,
             * sum med(|r_{i-1}|,|r_i|,|r_{i+1}|)^2                second estimator)
    JV_t   = max(RV_t - TPV_t, 0)         -> Int h^2 mu(ds,dx)    (price-jump energy)

TPV and MedRV converge to the SAME integrated diffusive variance by two
independent robust roads (Barndorff-Nielsen-Shephard tripower; Andersen-
Dobrev-Schaumburg 2012 median). Their agreement is a specification robustness
check: JV built from either should match.

Frozen estimation design (out-of-sample by construction):
    TRAIN   [2023-01-01 .. 2025-06-30]  -- excludes the 2025-10-10 crash
    HOLDOUT [2025-07-01 .. 2026-06-30]  -- contains it, never seen by GMM

Units: percent^2 per day (returns scaled by 100 before squaring).
"""

import os

import numpy as np
import pandas as pd
from scipy.special import gamma as gamma_fn

ROOT = os.path.dirname(os.path.abspath(__file__))
BARS_PER_DAY = 1440                      # 1-minute bars in a 24h crypto session
TRAIN_END = pd.Timestamp("2025-06-30")   # inclusive
HOLDOUT_START = pd.Timestamp("2025-07-01")

# tripower normaliser mu23 = E|Z|^{2/3}, Z~N(0,1);  MedRV normaliser (ADS 2012)
MU23 = 2.0 ** (1.0 / 3.0) * gamma_fn(5.0 / 6.0) / gamma_fn(0.5)
C_MED = np.pi / (6.0 - 4.0 * np.sqrt(3.0) + np.pi)


def load_returns_matrix(path=os.path.join(ROOT, "btc_1min.csv")):
    """Percent log returns reshaped to (n_days, 1440); only complete days kept."""
    px = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")["price"]
    r = 100.0 * np.log(px).diff().dropna()
    day = r.index.floor("D")
    counts = r.groupby(day).size()
    full_days = counts[counts == BARS_PER_DAY].index          # drop partial days
    keep = day.isin(full_days)
    R = r[keep].to_numpy().reshape(-1, BARS_PER_DAY)
    dates = pd.DatetimeIndex(sorted(full_days), name="date")
    return R, dates


def daily_measures(path=os.path.join(ROOT, "btc_1min.csv")):
    R, dates = load_returns_matrix(path)
    n = BARS_PER_DAY

    rv = (R ** 2).sum(axis=1)

    a23 = np.abs(R) ** (2.0 / 3.0)
    tpv = MU23 ** (-3.0) * (n / (n - 2.0)) * (a23[:, 2:] * a23[:, 1:-1] * a23[:, :-2]).sum(axis=1)

    A = np.abs(R)
    med3 = np.median(np.stack([A[:, :-2], A[:, 1:-1], A[:, 2:]], axis=2), axis=2)
    medrv = C_MED * (n / (n - 2.0)) * (med3 ** 2).sum(axis=1)

    # finite-sample truncation: jump-robust estimator must not exceed total QV
    # (mirror of the JV = max(RV - . , 0) clip). Report how often it bites.
    n_cap_t = int((tpv > rv).sum())
    n_cap_m = int((medrv > rv).sum())
    tpv = np.minimum(tpv, rv)
    medrv = np.minimum(medrv, rv)

    jv = np.maximum(rv - tpv, 0.0)         # primary jump-variance series
    jv_med = np.maximum(rv - medrv, 0.0)   # MedRV-based cross-check

    out = pd.DataFrame(
        {"RV": rv, "TPV": tpv, "MedRV": medrv, "JV": jv, "JV_med": jv_med},
        index=dates,
    )
    out.attrs["n_cap_tpv"], out.attrs["n_cap_medrv"] = n_cap_t, n_cap_m
    return out


def split(meas):
    train = meas.loc[:TRAIN_END]
    holdout = meas.loc[HOLDOUT_START:]
    return train, holdout


def _summary(name, df):
    jv_ac1 = df["JV"].autocorr(1)
    tpv_med_corr = df["TPV"].corr(df["MedRV"])
    print(f"\n[{name}]  {len(df)} days  {df.index[0].date()} .. {df.index[-1].date()}")
    print(df[["RV", "TPV", "MedRV", "JV"]].mean().round(3).to_dict())
    print(f"  corr(TPV, MedRV) = {tpv_med_corr:.4f}   "
          f"(robustness: ~1 => jump/continuous split is estimator-invariant)")
    print(f"  JV lag-1 autocorr = {jv_ac1:.4f}   "
          f"(>0 => jump clustering => motivates affine intensity)")


def main():
    meas = daily_measures()
    print(f"TPV capped at RV on {meas.attrs['n_cap_tpv']} day(s); "
          f"MedRV capped on {meas.attrs['n_cap_medrv']} day(s).")

    meas.to_csv(os.path.join(ROOT, "daily_measures.csv"), float_format="%.6f")
    train, holdout = split(meas)
    train.to_csv(os.path.join(ROOT, "daily_measures_train.csv"), float_format="%.6f")
    holdout.to_csv(os.path.join(ROOT, "daily_measures_holdout.csv"), float_format="%.6f")

    _summary("FULL", meas)
    _summary("TRAIN", train)
    _summary("HOLDOUT", holdout)
    print("\nsaved -> daily_measures.csv, daily_measures_train.csv, daily_measures_holdout.csv")


if __name__ == "__main__":
    main()
