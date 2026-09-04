"""
Does the jump-variance state V^j predict compressions in the variance risk premium?

This is the question the whole project was built to answer: the model exists to
extract V^j (the self-exciting jump-variance state) from high-frequency data, so
the payoff has to be that V^j tells you something about where the premium is
going that simpler quantities do not.

Design. Every test below controls for the two things that would otherwise
manufacture a result:
  (a) the premium's own mean reversion  -- high VRP falls regardless of V^j;
  (b) the mechanical channel -- the model's P leg is an explicit function of the
      states, so any state "predicts" a model-based VRP by construction. We
      therefore report the effect on the pure market quote Q = DVOL^2/365 and on
      the HAR-based VRP (whose P leg contains no model states) alongside the
      model VRP.

Tests, in order of how hard they are to fool:
  1. Regression robustness across horizons and subsamples.
  2. Monotonicity across V^j quintiles (an effect that is real should be ordered,
     not concentrated in one bucket).
  3. Double sort: VRP level x V^j -- does V^j add anything within a level bucket?
  4. Compression probability: P(large VRP drop | high V^j) vs unconditional.
  5. Out-of-sample: rule formed on train only, evaluated on the holdout.
  6. Economic magnitude, not just significance.
"""

import numpy as np
import pandas as pd

from vrp import newey_west_t

TRAIN_END = pd.Timestamp("2025-06-30")
H = 10                       # primary horizon (days)


def z(s):
    return (s - s.mean()) / s.std()


def main():
    df = pd.read_csv("vrp_series.csv", parse_dates=["date"]).set_index("date")
    df["dVRP"] = df.VRP.shift(-H) - df.VRP
    df["dQ"] = df.Q.shift(-H) - df.Q
    df["dVRP_har"] = df.VRP_har.shift(-H) - df.VRP_har
    d = df.dropna(subset=["dVRP", "dQ", "dVRP_har"]).copy()
    d["zVj"], d["zVc"], d["zVRP"], d["zQ"] = z(d.Vj), z(d.Vc), z(d.VRP), z(d.Q)
    d["zJS"] = z(d.jump_share)

    print("=" * 88)
    print(f"DOES V^j PREDICT VRP COMPRESSION?   horizon = {H} days,  n = {len(d)}")
    print("=" * 88)

    # ---- 1. regression robustness ---------------------------------------
    print("\n[1] REGRESSION: dependent variable vs V^j, controlling for own level")
    print("    (HAC lags = 15; coefficients per 1 SD of V^j, units %^2/day)")
    specs = [("dQ  (pure market)", "dQ", "zQ"),
             ("dVRP (HAR P-leg)", "dVRP_har", "zVRP"),
             ("dVRP (model P-leg)", "dVRP", "zVRP")]
    for label, dep, ctrl in specs:
        X = np.column_stack([np.ones(len(d)), d[ctrl], d.zVj])
        b, t, r2 = newey_west_t(d[dep].to_numpy(), X, 15)
        # incremental R2 from V^j
        Xc = np.column_stack([np.ones(len(d)), d[ctrl]])
        _, _, r2c = newey_west_t(d[dep].to_numpy(), Xc, 15)
        print(f"    {label:22s} Vj {b[2]:+.3f} (t {t[2]:+.2f})   "
              f"R2 {r2:.3f} (level alone {r2c:.3f}, incremental {r2-r2c:+.4f})")

    print("\n    same, using JUMP SHARE (scale-free) instead of raw V^j:")
    for label, dep, ctrl in specs:
        X = np.column_stack([np.ones(len(d)), d[ctrl], d.zJS])
        b, t, r2 = newey_west_t(d[dep].to_numpy(), X, 15)
        print(f"    {label:22s} JS {b[2]:+.3f} (t {t[2]:+.2f})   R2 {r2:.3f}")

    # ---- 2. monotonicity across V^j quintiles ---------------------------
    print("\n[2] MONOTONICITY: mean subsequent change by V^j quintile")
    d["q"] = pd.qcut(d.zVj, 5, labels=[1, 2, 3, 4, 5])
    print(f"    {'quintile':10s} {'n':>5s} {'mean dQ':>10s} {'mean dVRP':>11s} "
          f"{'P(dVRP<0)':>11s} {'mean VRP now':>13s}")
    for qq in [1, 2, 3, 4, 5]:
        s = d[d.q == qq]
        print(f"    Vj Q{qq:<7d} {len(s):5d} {s.dQ.mean():10.3f} {s.dVRP.mean():11.3f} "
              f"{100*(s.dVRP<0).mean():10.1f}% {s.VRP.mean():13.3f}")
    print("    (a genuine effect should be ORDERED across quintiles, not one bucket)")

    # ---- 3. double sort: does V^j add within a VRP level bucket? --------
    print("\n[3] DOUBLE SORT: mean subsequent dVRP by (VRP level tercile x V^j tercile)")
    d["lv"] = pd.qcut(d.VRP, 3, labels=["VRP low", "VRP mid", "VRP high"])
    d["vj3"] = pd.qcut(d.zVj, 3, labels=["Vj low", "Vj mid", "Vj high"])
    tab = d.pivot_table(index="lv", columns="vj3", values="dVRP",
                        aggfunc="mean", observed=True)
    cnt = d.pivot_table(index="lv", columns="vj3", values="dVRP",
                        aggfunc="size", observed=True)
    print(tab.round(3).to_string())
    print("    cell counts:")
    print(cnt.to_string())
    hi = d[(d.lv == "VRP high")]
    a = hi[hi.vj3 == "Vj high"].dVRP
    b_ = hi[hi.vj3 == "Vj low"].dVRP
    print(f"\n    within VRP-high: Vj-high mean {a.mean():+.3f} (n={len(a)}) vs "
          f"Vj-low {b_.mean():+.3f} (n={len(b_)})  -> gap {a.mean()-b_.mean():+.3f}")

    # ---- 4. compression probability -------------------------------------
    print("\n[4] COMPRESSION EVENTS: bottom-quintile dVRP (large premium drop)")
    thr = d.dVRP.quantile(0.20)
    d["comp"] = d.dVRP <= thr
    base = d.comp.mean()
    print(f"    threshold dVRP <= {thr:.3f};  unconditional P(compression) = {base:.3f}")
    for qq in [1, 3, 5]:
        s = d[d.q == qq]
        print(f"    P(compression | V^j quintile {qq}) = {s.comp.mean():.3f}  "
              f"(lift {s.comp.mean()/base:+.2f}x)")

    # ---- 5. OUT-OF-SAMPLE ------------------------------------------------
    print("\n[5] OUT-OF-SAMPLE (rule formed on train, evaluated on holdout)")
    tr, ho = d[d.index <= TRAIN_END], d[d.index > TRAIN_END]
    cut = tr.zVj.quantile(0.667)                       # top-tercile threshold from TRAIN only
    for nm, s in (("train ", tr), ("holdout", ho)):
        hiv = s[s.zVj >= cut]
        lov = s[s.zVj < cut]
        print(f"    {nm}: n={len(s):4d}  Vj-high n={len(hiv):4d} mean dVRP {hiv.dVRP.mean():+.3f} | "
              f"Vj-low mean dVRP {lov.dVRP.mean():+.3f} | gap {hiv.dVRP.mean()-lov.dVRP.mean():+.3f}")
    if len(ho) > 30:
        X = np.column_stack([np.ones(len(ho)), ho.zVRP, ho.zVj])
        b, t, r2 = newey_west_t(ho.dQ.to_numpy(), X, 15)
        print(f"    holdout regression dQ on [VRP, Vj]: Vj {b[2]:+.3f} (t {t[2]:+.2f})")

    # ---- 6. economic magnitude ------------------------------------------
    print("\n[6] ECONOMIC MAGNITUDE")
    sd_dvrp = d.dVRP.std()
    X = np.column_stack([np.ones(len(d)), d.zVRP, d.zVj])
    b, t, _ = newey_west_t(d.dVRP.to_numpy(), X, 15)
    print(f"    1 SD of V^j moves the {H}-day VRP change by {b[2]:+.3f} %^2/day")
    print(f"    against a standard deviation of dVRP of {sd_dvrp:.3f}  "
          f"-> {abs(b[2])/sd_dvrp:.1%} of one SD")
    print(f"    mean |VRP| level is {d.VRP.abs().mean():.3f}, so the effect is "
          f"{abs(b[2])/d.VRP.abs().mean():.1%} of a typical premium")

    d.to_csv("jump_compression_analysis.csv")
    print("\nsaved jump_compression_analysis.csv")


if __name__ == "__main__":
    main()
