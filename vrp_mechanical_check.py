"""
Is the "V^c predicts VRP changes" result real, or mechanical?

THE CONCERN. The model's P leg is an explicit linear function of the states:
    P_t = level + (V^c_t - Vbar_c) taut(kc) + (1+psi)(V^j_t - vbar) taut(kj)
so  dVRP = dQ - dP = dQ - taut(kc) dV^c - (1+psi) taut(kj) dV^j.
V^c mean-reverts, so a HIGH V^c today implies dV^c < 0, hence dP < 0, hence
dVRP > 0 -- a POSITIVE coefficient on V^c arises with no economic content
whatsoever. That is precisely the sign found earlier, so it must be decomposed.

FOUR REGRESSIONS, run at each horizon (all regressors standardised, HAC lags):
  1. d VRP_model  on [VRP_model, V^c, V^j]   -- the original result
  2. d VRP_har    on [VRP_har,   V^c, V^j]   -- P leg from a HAR forecast that
                                               does not contain the states
  3. d Q          on [Q,         V^c, V^j]   -- PURE MARKET DATA, no model output
                                               anywhere: the clean test
  4. d P_model    on [V^c, V^j]              -- isolates the mechanical channel
Regression 3 is decisive: Q is the options market's own quote, so nothing the
model computes can contaminate it.
"""

import numpy as np
import pandas as pd

from vrp import newey_west_t

HORIZONS = (5, 10, 20)


def z(s):
    return (s - s.mean()) / s.std()


def run(dep, ctrl, states, lags, label):
    """dep ~ 1 + ctrl(optional) + Vc + Vj ; returns printed row."""
    cols = {"y": dep}
    if ctrl is not None:
        cols["ctrl"] = ctrl
    cols.update(states)
    d = pd.DataFrame(cols).dropna()
    names = (["ctrl"] if ctrl is not None else []) + list(states.keys())
    X = np.column_stack([np.ones(len(d))] + [d[n].to_numpy() for n in names])
    b, t, r2 = newey_west_t(d["y"].to_numpy(), X, lags)
    parts = []
    for i, n in enumerate(names, start=1):
        parts.append(f"{n} {b[i]:+.3f} (t {t[i]:+.2f})")
    print(f"    {label:34s} " + "   ".join(parts) + f"   R2 {r2:.3f}   n {len(d)}")
    return {n: (b[i], t[i]) for i, n in enumerate(names, start=1)} | {"r2": r2}


def main():
    df = pd.read_csv("vrp_series.csv", parse_dates=["date"]).set_index("date")
    st = {"Vc": z(df.Vc), "Vj": z(df.Vj)}

    print("=" * 96)
    print("IS THE V^c EFFECT REAL OR MECHANICAL?")
    print("=" * 96)
    print("VRP(model) = Q - P_model, and P_model is an explicit function of Vc, Vj.")
    print("VRP(HAR)   = Q - P_har,   P_har is a lagged-RV regression (no states).")
    print("Q          = DVOL^2/365,  pure options-market data.\n")

    print("Correlations among the P legs and the states:")
    print(f"   corr(Vc, P_model) = {df.Vc.corr(df.P):+.3f}   "
          f"corr(Vc, P_har) = {df.Vc.corr(df.P_har):+.3f}   "
          f"corr(P_model, P_har) = {df.P.corr(df.P_har):+.3f}")
    print(f"   corr(VRP_model, VRP_har) = {df.VRP.corr(df.VRP_har):+.3f}\n")

    out = {}
    for h in HORIZONS:
        lags = h + 5
        print(f"--- horizon h = {h} days " + "-" * 66)
        out[h] = {}
        out[h]["model"] = run(df.VRP.shift(-h) - df.VRP, z(df.VRP), st, lags,
                              "1. d VRP(model)  | ctrl=VRP")
        out[h]["har"] = run(df.VRP_har.shift(-h) - df.VRP_har, z(df.VRP_har), st, lags,
                            "2. d VRP(HAR)    | ctrl=VRP_har")
        out[h]["Q"] = run(df.Q.shift(-h) - df.Q, z(df.Q), st, lags,
                          "3. d Q (market)  | ctrl=Q")
        out[h]["Pmech"] = run(df.P.shift(-h) - df.P, None, st, lags,
                              "4. d P(model)  [mechanical]")
        print()

    print("=" * 96)
    print("READING THE TABLE")
    print("=" * 96)
    for h in HORIZONS:
        tm = out[h]["model"]["Vc"][1]
        th = out[h]["har"]["Vc"][1]
        tq = out[h]["Q"]["Vc"][1]
        tp = out[h]["Pmech"]["Vc"][1]
        print(f"  h={h:2d}:  Vc t-stat  model {tm:+.2f} | HAR {th:+.2f} | "
              f"market-Q {tq:+.2f} | mechanical dP {tp:+.2f}")
    print()
    print("  If the model-VRP t-stat is large but the market-Q t-stat is ~0, the")
    print("  effect lives entirely in the model's own P leg and is mechanical.")

    # full detail on the level/mean-reversion term for both VRPs
    print("\n" + "=" * 96)
    print("MEAN REVERSION IN BOTH VRP DEFINITIONS (coefficient on own level)")
    print("=" * 96)
    for nm, series in (("VRP model", df.VRP), ("VRP HAR", df.VRP_har), ("Q level", df.Q)):
        rho = series.autocorr(1)
        hl = np.log(2) / -np.log(abs(rho)) if 0 < abs(rho) < 1 else np.nan
        line = f"  {nm:10s} AR(1) {rho:.4f}  half-life {hl:5.1f} d  |"
        for h in HORIZONS:
            d = pd.DataFrame({"y": series.shift(-h) - series, "x": z(series)}).dropna()
            b, t, r2 = newey_west_t(d.y.to_numpy(),
                                    np.column_stack([np.ones(len(d)), d.x]), h + 5)
            line += f"  h={h}: {b[1]:+.3f} (t {t[1]:+.2f}, R2 {r2:.3f})"
        print(line)


if __name__ == "__main__":
    main()
