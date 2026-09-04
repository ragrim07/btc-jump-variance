"""
P-measure estimation on the REAL train window (2023-01-02 .. 2025-06-30).

Runs only after the simulation verification passed (estimation_verification_report).
The 2025-07 .. 2026-06 holdout -- which contains the 2025-10-10 crash -- is never
touched here.

Outputs: estimate_train_results.json + diagnostic figures.
"""

import json
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import moments as mo
import gmm
import damping as dp
from moments import N_TPV, N_JV, N_FWD, N_BWD

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("estimate")

N_BOOT, BLOCK = 600, 50
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"
C_EMP, C_MOD, C_OLD = "#2a78d6", "#1baf7a", "#c0504d"


def _ax(figsize=(8.6, 4.8)):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(AXIS)
    return fig, ax


def _finish(fig, ax, title, xl, yl, path):
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xlabel(xl, color=INK, fontsize=9); ax.set_ylabel(yl, color=INK, fontsize=9)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
    fig.tight_layout(); fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    meas = pd.read_csv("daily_measures_train.csv", parse_dates=["date"]).set_index("date")
    tpv, jv = meas["TPV"].to_numpy(), meas["JV"].to_numpy()
    log.info("train window: %d days  %s .. %s", len(tpv),
             meas.index[0].date(), meas.index[-1].date())

    m_emp, rows = gmm.empirical_moments_ext(tpv, jv)
    W = gmm.build_weighting_matrix(rows)
    # estimate from the contribution-row mean: the same object the bootstrap
    # resamples, so point estimate and interval share one estimator
    g_bar = rows.mean(axis=0)
    res = gmm.calibrate(g_bar, W)
    theta = res.x
    der = mo.derived_quantities(theta)
    ok_b, b_val = gmm.check_transience(theta)

    log.info("J = %.6e   b = %.4f (transience %s)", res.fun, b_val, "OK" if ok_b else "VIOLATED")

    bs = gmm.moving_block_bootstrap(rows, theta, W, n_boot=N_BOOT, block=BLOCK, seed=11)
    lo, hi = bs["theta_pct"][0], bs["theta_pct"][2]

    # bootstrap distribution of the two over-identification residuals
    cons = {"CJS_minus_CBCF": [], "S_minus_implied": []}
    for d in bs["theta"]:
        dd = mo.derived_quantities(d)
        cons["CJS_minus_CBCF"].append(dd["consistency_CJS_minus_CBCF"])
        cons["S_minus_implied"].append(dd["consistency_S_minus_implied"])
    cons_pct = {k: np.percentile(v, [2.5, 50, 97.5]).tolist() for k, v in cons.items()}

    print("\n" + "=" * 78)
    print("P-MEASURE GMM ESTIMATES -- TRAIN WINDOW 2023-01-02 .. 2025-06-30")
    print("=" * 78)
    print(f"{'parameter':12s} {'estimate':>12s} {'95% CI lower':>14s} {'95% CI upper':>14s}")
    for i, n in enumerate(mo.PARAM_NAMES):
        print(f"{n:12s} {theta[i]:12.5f} {lo[i]:14.5f} {hi[i]:14.5f}")
    print("-" * 78)
    print("DERIVED (structural):")
    for k in ("b", "eta", "lam0", "lam1", "vbar", "halflife_c", "halflife_j", "halflife_b"):
        v = der[k]
        if k in bs["derived_pct"]:
            p = bs["derived_pct"][k]
            print(f"  {k:12s} {v:10.5f}   CI [{p[0]:.5f}, {p[2]:.5f}]")
        else:
            print(f"  {k:12s} {v:10.5f}")
    print("-" * 78)
    print("OVER-IDENTIFICATION TESTS (0 = model consistent):")
    for k, p in cons_pct.items():
        flag = "PASS (0 in CI)" if p[0] <= 0 <= p[2] else "REJECT"
        print(f"  {k:20s} median {p[1]:+.5f}  CI [{p[0]:+.5f}, {p[2]:+.5f}]  {flag}")
    print("=" * 78)

    m_fit, _ = mo.theoretical_moments_ext(theta)

    # ---------------- figures ----------------
    ks = np.arange(1, N_TPV + 1)
    fig, ax = _ax()
    ax.plot(ks, m_emp[5:5+N_TPV], "o-", color=C_EMP, ms=5, lw=1.7, label="Empirical")
    ax.plot(ks, m_fit[5:5+N_TPV], "o-", color=C_MOD, ms=5, lw=1.7, label="Model (calibrated)")
    ax.set_xticks(ks)
    _finish(fig, ax, "TPV autocovariance: empirical vs calibrated model (train)",
            "lag k (days)", r"Cov(TPV$_t$, TPV$_{t-k}$)", "fig_train_tpv_acov.png")

    kj_ = np.arange(1, N_JV + 1)
    fig, ax = _ax()
    ax.axhline(0, color=C_OLD, lw=1.6, ls=(0, (5, 3)), label="Old model (constant intensity): 0")
    ax.plot(kj_, m_emp[5+N_TPV:5+N_TPV+N_JV], "o-", color=C_EMP, ms=5, lw=1.7, label="Empirical")
    ax.plot(kj_, m_fit[5+N_TPV:5+N_TPV+N_JV], "o-", color=C_MOD, ms=5, lw=1.7, label="Model (calibrated)")
    ax.set_xticks(kj_)
    _finish(fig, ax, "JV autocovariance: jump clustering the old model cannot produce",
            "lag k (days)", r"Cov(JV$_t$, JV$_{t-k}$)", "fig_train_jv_acov.png")

    # two-sided cross
    kk = np.arange(-N_BWD, N_FWD + 1)
    emp_c, mod_c = [], []
    dt, dj = tpv - tpv.mean(), jv - jv.mean()
    T = len(tpv)
    for k in kk:
        if k >= 1:
            emp_c.append(dt[k:] @ dj[:-k] / T)
            mod_c.append(m_fit[5+N_TPV+N_JV+k-1])
        elif k == 0:
            emp_c.append(dt @ dj / T); mod_c.append(m_fit[4])
        else:
            emp_c.append(dt[:k] @ dj[-k:] / T)
            mod_c.append(m_fit[5+N_TPV+N_JV+N_FWD+(-k)-1])
    fig, ax = _ax((9.2, 4.8))
    ax.axvline(0, color=AXIS, lw=0.8, ls=(0, (3, 3)))
    ax.plot(kk, emp_c, "o-", color=C_EMP, ms=5, lw=1.7, label="Empirical")
    ax.plot(kk, mod_c, "o-", color=C_MOD, ms=5, lw=1.7, label="Model (calibrated)")
    ax.set_xticks(kk)
    ax.annotate("variance leads jumps\n(old model: exactly 0)", (-3, ax.get_ylim()[1]*0.75),
                color=MUTED, fontsize=8, ha="center")
    ax.annotate("jumps lead variance", (3, ax.get_ylim()[1]*0.75),
                color=MUTED, fontsize=8, ha="center")
    _finish(fig, ax, "Two-sided TPV-JV cross-covariance: the asymmetry is the co-jump coupling",
            "lag k (days)   Cov(TPV$_t$, JV$_{t-k}$)", "cross-covariance",
            "fig_train_cross.png")

    # eta bootstrap distribution
    fig, ax = _ax((6.4, 4.2))
    ax.hist(bs["derived"]["eta"], bins=40, color=C_EMP, edgecolor=SURFACE, alpha=0.85)
    p = bs["derived_pct"]["eta"]
    for x, s in ((p[0], "-"), (p[2], "-")):
        ax.axvline(x, color=C_OLD, lw=1.4, ls=(0, (4, 3)))
    ax.axvline(der["eta"], color=C_MOD, lw=2.0, label=f"$\\hat\\eta$ = {der['eta']:.3f}")
    _finish(fig, ax, "Bootstrap distribution of the branching ratio $\\eta$",
            r"$\eta$ (fraction of jumps endogenously triggered)", "count",
            "fig_train_eta_hist.png")

    out = {
        "window": [str(meas.index[0].date()), str(meas.index[-1].date())],
        "n_days": int(len(tpv)), "J": float(res.fun),
        "param_names": mo.PARAM_NAMES,
        "theta": theta.tolist(), "ci_lo": lo.tolist(), "ci_hi": hi.tolist(),
        "derived": {k: float(v) for k, v in der.items()},
        "derived_pct": {k: np.asarray(v).tolist() for k, v in bs["derived_pct"].items()},
        "consistency_pct": cons_pct,
        "transience_ok": bool(ok_b), "b": float(b_val),
        "m_emp": m_emp.tolist(), "m_fit": m_fit.tolist(),
        "moment_labels": mo.moment_labels(),
    }
    with open("estimate_train_results.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("saved estimate_train_results.json + 4 figures")


if __name__ == "__main__":
    main()
