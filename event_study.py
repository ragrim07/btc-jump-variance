"""
Out-of-sample event study on the HOLDOUT window (2025-07 .. 2026-06).

THE TEST. kappa_j was estimated on the TRAIN window (2023-01 .. 2025-06) from
unconditional moments. Every event below lies in the holdout, which the estimator
never saw. The model makes a sharp conditional prediction:

    a price jump kicks the jump-variance state V^j (it does NOT touch the
    Brownian-driven continuous factor V^c), so the post-event EXCESS variance is
    pure V^j excess and must decay at the effective rate kappa_j.

Comparing the observed post-event decay to the train-estimated kappa_hat_j is
therefore a genuine out-of-sample structural prediction.

THREE DESIGN POINTS THAT MATTER.
 1. Fit from day +1, not day 0. The shock ARRIVES on day 0; day 0's variance is
    the shock itself, not the state it leaves behind. Including it makes the fit
    measure the shock's own size, not the decay.
 2. Pool several events. One event gives ~20 very noisy daily observations. We
    pool all large jump events in the holdout with a common decay rate and
    per-event levels (event fixed effects in logs), which is what supplies power.
 3. Use MedRV, not tripower, for continuous variance. On 2025-10-10 tripower
    overshoots RV (three consecutive large returns inflate every factor of its
    product), is truncated to RV, and the tripower jump variance collapses to
    0.000 -- it reports "no jumps" on the largest crash in the sample. MedRV
    gives 5.700.

P VERSUS Q. kappa_j is a physical-measure object; DVOL is risk-neutral. The
change of measure rescales the self-excitation, so kappa_j^Q != kappa_j^P in
general. The DVOL decay is NOT a second test of kappa_hat_j -- the WEDGE between
the two is the economic result (how jump-risk persistence is priced).
"""

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

H = 15                 # days after the event used in the decay fit (from day +1)
PRE = 30               # pre-event days for the baseline level
MIN_SEP = 20           # minimum separation between events (non-overlapping windows)
N_EVENTS = 5
N_BOOT = 3000

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#6d6b66", "#e1e0d9", "#c3c2b7"
C_P, C_Q, C_PRED = "#2a6fd0", "#b8860b", "#159e6d"


def pick_events(jv, n=N_EVENTS, sep=MIN_SEP):
    """Largest jump-energy days, greedily, keeping a minimum separation."""
    chosen = []
    for d in jv.sort_values(ascending=False).index:
        if all(abs((d - c).days) >= sep for c in chosen):
            chosen.append(d)
        if len(chosen) == n:
            break
    return sorted(chosen)


def event_panel(s, events, pre=PRE, horizon=H):
    """Excess paths for days +1..+horizon, one row per (event, h)."""
    rows = []
    for e in events:
        base = float(s.loc[e - pd.Timedelta(days=pre): e - pd.Timedelta(days=1)].median())
        post = s.loc[e + pd.Timedelta(days=1): e + pd.Timedelta(days=horizon)]
        for h, (dt, v) in enumerate(post.items(), start=1):
            rows.append(dict(event=str(e.date()), h=h, excess=v - base, base=base))
    return pd.DataFrame(rows)


def pooled_decay(panel, n_boot=N_BOOT, seed=0):
    """Common decay rate with per-event levels: log(excess) = a_e - rate*h.

    Only positive excesses enter (logs). Cluster bootstrap over EVENTS -- the
    unit of independent variation is the event, not the day.
    """
    d = panel[panel["excess"] > 0].copy()
    d["y"] = np.log(d["excess"])
    evs = sorted(d["event"].unique())

    def fit(df):
        E = sorted(df["event"].unique())
        if len(df) < len(E) + 2:
            return np.nan
        X = np.zeros((len(df), len(E) + 1))
        for i, e in enumerate(E):
            X[:, i] = (df["event"] == e).astype(float)
        X[:, -1] = -df["h"].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, df["y"].to_numpy(), rcond=None)
        return beta[-1]                      # the common decay rate

    rate = fit(d)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        samp = rng.choice(evs, size=len(evs), replace=True)
        parts = []
        for j, e in enumerate(samp):
            p = d[d["event"] == e].copy()
            p["event"] = f"{e}__{j}"          # keep resampled events distinct
            parts.append(p)
        r = fit(pd.concat(parts, ignore_index=True))
        if np.isfinite(r):
            draws.append(r)
    draws = np.array(draws)
    return rate, np.percentile(draws, [2.5, 97.5]), draws


def main():
    meas = pd.read_csv("daily_measures_holdout.csv", parse_dates=["date"]).set_index("date")
    dvol = pd.read_csv("dvol_btc_daily.csv", parse_dates=["date"]).set_index("date")["close"]
    train = json.load(open("estimate_train_results.json"))
    kj, kc = train["theta"][3], train["theta"][0]

    events = pick_events(meas["JV_med"])
    print(f"train-estimated kappa_j = {kj:.4f}  (half-life {np.log(2)/kj:.2f} d)")
    print(f"train-estimated kappa_c = {kc:.4f}  (half-life {np.log(2)/kc:.2f} d)")
    print(f"\nevents (largest jump-energy days in the holdout, >={MIN_SEP}d apart):")
    for e in events:
        print(f"   {e.date()}   JV_med={meas.loc[e,'JV_med']:6.2f}   RV={meas.loc[e,'RV']:7.2f}")

    out = {"kappa_j_train": float(kj), "kappa_c_train": float(kc),
           "events": [str(e.date()) for e in events], "H": H}

    print("\n--- P-measure: continuous variance (MedRV) excess, days +1..+%d ---" % H)
    panel = event_panel(meas["MedRV"], events)
    rate, ci, _ = pooled_decay(panel)
    covers = ci[0] <= kj <= ci[1]
    print(f"pooled decay rate = {rate:.4f}/day   95% CI [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"half-life = {np.log(2)/rate:.2f} d   (predicted {np.log(2)/kj:.2f} d)")
    print(f"CI contains train kappa_j_hat = {kj:.4f}:  "
          f"{'YES -- prediction confirmed out of sample' if covers else 'NO'}")
    out["P"] = dict(rate=float(rate), ci=[float(ci[0]), float(ci[1])],
                    halflife=float(np.log(2)/rate), covers_kj=bool(covers))

    # single-event view for the headline crash
    p1 = event_panel(meas["MedRV"], [pd.Timestamp("2025-10-10")])
    r1, c1, _ = pooled_decay(p1, n_boot=1500, seed=2)
    print(f"\n  2025-10-10 alone: rate {r1:.4f} CI [{c1[0]:.4f}, {c1[1]:.4f}] "
          f"(half-life {np.log(2)/r1:.2f} d)")
    out["P_crash_only"] = dict(rate=float(r1), ci=[float(c1[0]), float(c1[1])])

    # jump-energy channel as a secondary read
    pj = event_panel(meas["JV_med"], events)
    rj, cj, _ = pooled_decay(pj, seed=3)
    print(f"  jump-energy (JV_med) channel: rate {rj:.4f} CI [{cj[0]:.4f}, {cj[1]:.4f}]")
    out["P_jv"] = dict(rate=float(rj), ci=[float(cj[0]), float(cj[1])])

    print("\n--- Q-measure: DVOL^2 excess ---")
    dv2 = (dvol / 100.0) ** 2
    pq = event_panel(dv2, events)
    rq, cq, _ = pooled_decay(pq, seed=4)
    print(f"pooled Q decay rate = {rq:.4f}/day  95% CI [{cq[0]:.4f}, {cq[1]:.4f}]"
          f"   half-life {np.log(2)/rq:.2f} d")
    out["Q"] = dict(rate=float(rq), ci=[float(cq[0]), float(cq[1])],
                    halflife=float(np.log(2)/rq))
    print(f"\nP vs Q wedge: P {rate:.4f} vs Q {rq:.4f}  -> "
          f"Q is {'SLOWER (more persistent under Q)' if rq < rate else 'FASTER'}"
          f", ratio {rate/rq:.2f}x")
    out["wedge_ratio"] = float(rate / rq)

    print("\nwindow sensitivity (P, MedRV):")
    sens = {}
    for HH in (8, 10, 15, 20, 25):
        pp = event_panel(meas["MedRV"], events, horizon=HH)
        rr, cc, _ = pooled_decay(pp, n_boot=800, seed=5)
        sens[HH] = [float(rr), float(cc[0]), float(cc[1])]
        print(f"   H={HH:2d}d  rate {rr:.4f}  CI [{cc[0]:.4f}, {cc[1]:.4f}]  "
              f"contains kappa_j: {'yes' if cc[0] <= kj <= cc[1] else 'no'}")
    out["sensitivity"] = sens
    json.dump(out, open("event_study_results.json", "w"), indent=1)

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE); ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"): ax.spines[s_].set_color(AXIS)

    ax = axes[0]
    # normalise each event by its own fitted level so they overlay
    d = panel[panel["excess"] > 0]
    for e in sorted(d["event"].unique()):
        sub = d[d["event"] == e]
        lvl = np.exp(np.mean(np.log(sub["excess"]) + rate * sub["h"]))
        ax.plot(sub["h"], sub["excess"] / lvl, "o", ms=4.4, alpha=0.72,
                mec=SURFACE, mew=0.7, color=C_P, zorder=3)
    hs = np.linspace(1, H, 200)
    ax.plot(hs, np.exp(-rate * hs), color=C_P, lw=2.0, zorder=4,
            label=f"observed pooled decay {rate:.3f}/d (half-life {np.log(2)/rate:.1f} d)")
    ax.plot(hs, np.exp(-kj * hs), color=C_PRED, lw=2.2, ls=(0, (5, 3)), zorder=5,
            label=f"PREDICTED by train $\\hat\\kappa_j$ = {kj:.3f} "
                  f"(half-life {np.log(2)/kj:.1f} d)")
    ax.fill_between(hs, np.exp(-ci[1]*hs), np.exp(-ci[0]*hs), color=C_P, alpha=0.13, lw=0,
                    label="95% CI on the observed rate", zorder=1)
    ax.set_yscale("log")
    ax.set_title("P-measure: excess variance after large jump events",
                 color=INK, fontsize=10.5, loc="left")
    ax.set_xlabel("days since event", color=INK, fontsize=9)
    ax.set_ylabel("normalised excess variance (log scale)", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=7.6, labelcolor=INK, loc="lower left")

    ax = axes[1]
    dq = pq[pq["excess"] > 0]
    for e in sorted(dq["event"].unique()):
        sub = dq[dq["event"] == e]
        lvl = np.exp(np.mean(np.log(sub["excess"]) + rq * sub["h"]))
        ax.plot(sub["h"], sub["excess"] / lvl, "o", ms=4.4, alpha=0.72,
                mec=SURFACE, mew=0.7, color=C_Q, zorder=3)
    ax.plot(hs, np.exp(-rq * hs), color=C_Q, lw=2.0, zorder=4,
            label=f"Q decay {rq:.3f}/d (half-life {np.log(2)/rq:.1f} d)")
    ax.plot(hs, np.exp(-rate * hs), color=C_P, lw=1.8, ls=(0, (4, 3)), zorder=5,
            label=f"P decay {rate:.3f}/d, for comparison")
    ax.set_yscale("log")
    ax.set_title("Q-measure: option-implied variance (DVOL$^2$) after the same events",
                 color=INK, fontsize=10.5, loc="left")
    ax.set_xlabel("days since event", color=INK, fontsize=9)
    ax.set_ylabel("normalised excess implied variance (log)", color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=7.6, labelcolor=INK, loc="lower left")

    fig.suptitle("Out-of-sample event study — these events were never in the estimation window",
                 color=INK, fontsize=12, x=0.006, ha="left", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig("fig_event_study.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("\nsaved event_study_results.json + fig_event_study.png")


if __name__ == "__main__":
    main()
