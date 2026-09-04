"""
Do the filtered states (V^c, V^j) predict FUTURE CHANGES in the variance risk
premium?

This is a different question from the ones already answered. Earlier we asked
whether the states explain the VRP LEVEL contemporaneously (they partly do, via
V^c) and whether VRP predicts the variance-selling payoff (it does, but the
content is the implied-vol level). Here we ask:

    Delta VRP_{t -> t+h}  =  a + b_c V^c_t + b_j V^j_t + (control: VRP_t) + e

If the states carry dynamic information, they should forecast where the premium
is HEADED, not just where it is. Because VRP is highly persistent
(AR(1) = 0.951, half-life ~14 d), any change is dominated by mean reversion, so
the honest specification CONTROLS for VRP_t and asks whether the states add
anything beyond it.

All regressors standardised, so coefficients read as "change in VRP (%^2/day)
per 1 standard deviation of the regressor". Overlapping horizons => Newey-West
HAC standard errors with lags = h + 5.
"""

import numpy as np
import pandas as pd

from vrp import newey_west_t

HORIZONS = (1, 5, 10, 20)


def z(s):
    return (s - s.mean()) / s.std()


def main():
    df = pd.read_csv("vrp_series.csv", parse_dates=["date"]).set_index("date")
    train_end = pd.Timestamp("2025-06-30")

    print("=" * 78)
    print("DO THE STATES PREDICT FUTURE CHANGES IN THE VRP?")
    print("=" * 78)
    print(f"VRP persistence: AR(1) = {df.VRP.autocorr(1):.4f}  "
          f"(half-life {np.log(2)/-np.log(df.VRP.autocorr(1)):.1f} d)")
    print("Coefficients are per 1 SD of the regressor, in %^2/day of VRP change.\n")

    rows = []
    for h in HORIZONS:
        d = pd.DataFrame({
            "dv": df.VRP.shift(-h) - df.VRP,
            "vrp": z(df.VRP), "vc": z(df.Vc), "vj": z(df.Vj),
            "js": z(df.jump_share),
        }).dropna()
        lags = h + 5

        # (A) states only
        Xa = np.column_stack([np.ones(len(d)), d.vc, d.vj])
        ba, ta, r2a = newey_west_t(d.dv.to_numpy(), Xa, lags)
        # (B) states + mean-reversion control
        Xb = np.column_stack([np.ones(len(d)), d.vrp, d.vc, d.vj])
        bb, tb, r2b = newey_west_t(d.dv.to_numpy(), Xb, lags)
        # (C) mean reversion alone (the benchmark the states must beat)
        Xc = np.column_stack([np.ones(len(d)), d.vrp])
        bc, tc, r2c = newey_west_t(d.dv.to_numpy(), Xc, lags)

        print(f"--- horizon h = {h} day(s)   n = {len(d)} ---")
        print(f"  (A) states only        Vc {ba[1]:+.3f} (t {ta[1]:+.2f})   "
              f"Vj {ba[2]:+.3f} (t {ta[2]:+.2f})                       R2 {r2a:.3f}")
        print(f"  (C) VRP level only     VRP {bc[1]:+.3f} (t {tc[1]:+.2f})"
              f"                                       R2 {r2c:.3f}")
        print(f"  (B) VRP + states       VRP {bb[1]:+.3f} (t {tb[1]:+.2f})   "
              f"Vc {bb[2]:+.3f} (t {tb[2]:+.2f})   Vj {bb[3]:+.3f} (t {tb[3]:+.2f})   R2 {r2b:.3f}")
        print(f"      incremental R2 from adding the states: {r2b - r2c:+.4f}")
        rows.append(dict(h=h, r2_states=r2a, r2_vrp=r2c, r2_both=r2b,
                         t_vc=tb[2], t_vj=tb[3], t_vrp=tb[1]))
        print()

    # out-of-sample check at the most informative horizon
    print("=" * 78)
    print("OUT-OF-SAMPLE CHECK (h = 10, holdout 2025-07 onward)")
    h = 10
    d = pd.DataFrame({"dv": df.VRP.shift(-h) - df.VRP, "vrp": z(df.VRP),
                      "vc": z(df.Vc), "vj": z(df.Vj)}).dropna()
    tr, ho = d[d.index <= train_end], d[d.index > train_end]
    Xtr = np.column_stack([np.ones(len(tr)), tr.vrp, tr.vc, tr.vj])
    b, _, _ = newey_west_t(tr.dv.to_numpy(), Xtr, h + 5)
    Xho = np.column_stack([np.ones(len(ho)), ho.vrp, ho.vc, ho.vj])
    pred = Xho @ b
    corr = np.corrcoef(pred, ho.dv)[0, 1]
    # benchmark: mean reversion alone
    b2, _, _ = newey_west_t(tr.dv.to_numpy(),
                            np.column_stack([np.ones(len(tr)), tr.vrp]), h + 5)
    pred2 = np.column_stack([np.ones(len(ho)), ho.vrp]) @ b2
    corr2 = np.corrcoef(pred2, ho.dv)[0, 1]
    print(f"  fit on train, predict holdout (n={len(ho)}):")
    print(f"    VRP + states : corr(pred, actual) = {corr:+.3f}")
    print(f"    VRP only     : corr(pred, actual) = {corr2:+.3f}")
    print(f"    => states add {corr - corr2:+.3f} in OOS correlation")

    summary = pd.DataFrame(rows)
    summary.to_csv("vrp_predictability.csv", index=False)
    print("\nsaved vrp_predictability.csv")


if __name__ == "__main__":
    main()
