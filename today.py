"""
Current-state readout: where does the model say we are right now?

Prints the latest jump state, variance risk premium, and which cell of the
(premium level x jump state) grid we sit in -- the grid whose in-sample pattern
is the project's central result.

HONESTY ABOUT STALENESS. The Q leg (Deribit DVOL) is fetched live and is current
to today. The P leg needs daily realized measures, which are built from the
1-minute bar file; that file ends whenever the tick data ends. This script
reports the age of the realized data explicitly rather than pretending the whole
readout is live. To refresh the P leg you need new tick data, then:
    python clean_prices.py && python realized_measures.py

INTERPRETATION WARNING. The grid's historical pattern is an IN-SAMPLE result that
did not replicate on held-out data. This readout says where we are, not what will
happen.
"""

import json
import sys

import numpy as np
import pandas as pd

import vrp as V

TERCILE = ("low", "mid", "high")


def tercile_of(value, series):
    """Which tercile of `series` does `value` fall in? Returns (label, index)."""
    lo, hi = series.quantile(1 / 3), series.quantile(2 / 3)
    if value < lo:
        return TERCILE[0], 0
    if value < hi:
        return TERCILE[1], 1
    return TERCILE[2], 2


def main(refresh_dvol=True):
    if refresh_dvol:
        try:
            import fetch_dvol
            print("fetching latest DVOL ...")
            fetch_dvol.main()
            print()
        except Exception as exc:                                # noqa: BLE001
            print(f"  (DVOL refresh failed: {exc}; using stored file)\n")

    meas = pd.read_csv("daily_measures.csv", parse_dates=["date"]).set_index("date")
    dvol = pd.read_csv("dvol_btc_daily.csv", parse_dates=["date"]).set_index("date")["close"]
    theta = np.array(json.load(open("estimate_train_results.json"))["theta"])

    states = V.build_states(meas, theta)
    eP = V.p_forecast(states, theta)
    eQ = (dvol ** 2 / 365.0)

    # P leg is only as fresh as the realized measures
    p_date = meas.index[-1]
    q_date = dvol.index[-1]
    stale = (pd.Timestamp.now().normalize() - p_date).days

    # historical grid, built on the same series the finding used
    hist = pd.read_csv("vrp_series.csv", parse_dates=["date"]).set_index("date")

    vj_now, vc_now = float(states["Vj"].iloc[-1]), float(states["Vc"].iloc[-1])
    p_now = float(eP.iloc[-1])
    q_at_p = float(eQ.reindex([p_date]).iloc[0]) if p_date in eQ.index else float("nan")
    q_latest = float(eQ.iloc[-1])
    vrp_now = q_at_p - p_now

    lab_v, _ = tercile_of(vrp_now, hist["VRP"])
    lab_j, _ = tercile_of(vj_now, hist["Vj"])

    w = 66
    print("=" * w)
    print("CURRENT STATE".center(w))
    print("=" * w)
    print(f"  realized-measure data through   {p_date.date()}   "
          f"({stale} days old)" + ("   <-- STALE" if stale > 7 else ""))
    print(f"  DVOL (options) data through     {q_date.date()}   "
          f"latest DVOL {dvol.iloc[-1]:.2f}")
    print("-" * w)
    print(f"  jump state        V^j           {vj_now:8.3f}   ({lab_j} tercile)")
    print(f"  continuous state  V^c           {vc_now:8.3f}")
    print(f"  jump share        V^j/(V^c+V^j) {vj_now/(vc_now+vj_now):8.3f}")
    print("-" * w)
    print(f"  E^Q implied  (DVOL^2/365)       {q_at_p:8.3f}  %^2/day")
    print(f"  E^P model forecast              {p_now:8.3f}")
    print(f"  VARIANCE RISK PREMIUM           {vrp_now:+8.3f}   ({lab_v} tercile)")
    if not np.isnan(q_latest) and q_date != p_date:
        print(f"    (today's implied leg alone:   {q_latest:8.3f})")
    print("-" * w)
    print(f"  GRID CELL:  premium {lab_v.upper()}  x  jump state {lab_j.upper()}")

    # what that cell did historically, in sample
    hh = hist.copy()
    hh["dVRP"] = hh.VRP.shift(-10) - hh.VRP
    hh = hh.dropna(subset=["dVRP"])
    tr = hh[hh.index <= pd.Timestamp("2025-06-30")]
    lv = pd.qcut(tr.VRP, 3, labels=list(TERCILE))
    vj = pd.qcut(tr.Vj, 3, labels=list(TERCILE))
    cell = tr[(lv == lab_v) & (vj == lab_j)]["dVRP"]
    if len(cell):
        print(f"  historically (in sample, n={len(cell)}): mean 10-day change "
              f"{cell.mean():+.3f}")
    print("=" * w)
    print("  NOTE: the grid pattern is an IN-SAMPLE result that did NOT replicate")
    print("  on held-out data. This is a state readout, not a forecast.")
    print("=" * w)


if __name__ == "__main__":
    main(refresh_dvol="--no-fetch" not in sys.argv)
