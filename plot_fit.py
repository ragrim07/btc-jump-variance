"""
Model-implied vs empirical autocovariance / cross-autocovariance curves,
with block-bootstrap uncertainty bands on the empirical estimates.

Judging fit by eye WITHOUT error bands is misleading: these moments differ enormously
in precision (Var(TPV) carries a ~26% standard error on 911 days, while the JV
autocovariances are comparatively tight). A curve that looks "far off" may be
well inside noise, and one that looks close may be a real miss. Every panel here
therefore shows the empirical point estimate with a +-2 SE band.

Produces fig_fit_panels.png (4-panel overview) and the individual panels.
"""

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gmm
import moments as mo
from moments import N_TPV, N_JV, N_FWD, N_BWD

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#6d6b66", "#e1e0d9", "#c3c2b7"
C_EMP, C_MOD, C_OLD = "#2a78d6", "#1baf7a", "#c0504d"
BAND = "#c8d8ee"

N_BOOT, BLOCK = 600, 50


def bootstrap_moment_se(rows, n_boot=N_BOOT, block=BLOCK, seed=0):
    """Block-bootstrap SE of each of the 40 empirical moments."""
    rng = np.random.default_rng(seed)
    Tr = rows.shape[0]
    nb = int(np.ceil(Tr / block))
    draws = np.empty((n_boot, rows.shape[1]))
    for i in range(n_boot):
        st = rng.integers(0, Tr - block + 1, nb)
        idx = np.concatenate([np.arange(s, s + block) for s in st])[:Tr]
        draws[i] = rows[idx].mean(axis=0)
    return draws.std(axis=0, ddof=1)


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)


def _panel(ax, x, emp, se, mod, title, xlabel, ylabel, zero_label=None, xticks=None):
    ax.fill_between(x, emp - 2 * se, emp + 2 * se, color=BAND, alpha=0.65, lw=0,
                    label="empirical ± 2 SE", zorder=1)
    if zero_label:
        ax.axhline(0.0, color=C_OLD, lw=1.7, ls=(0, (5, 3)), zorder=2, label=zero_label)
    else:
        ax.axhline(0.0, color=AXIS, lw=0.8, zorder=2)
    ax.plot(x, emp, "o-", color=C_EMP, ms=5, lw=1.7, mec=SURFACE, mew=1.0,
            label="empirical", zorder=4)
    ax.plot(x, mod, "s-", color=C_MOD, ms=4.6, lw=1.9, mec=SURFACE, mew=1.0,
            label="model-implied", zorder=5)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8)
    ax.set_xlabel(xlabel, color=INK, fontsize=8.8)
    ax.set_ylabel(ylabel, color=INK, fontsize=8.8)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.legend(frameon=False, fontsize=7.8, labelcolor=INK)


def main():
    J = json.load(open("estimate_train_results.json"))
    theta = np.array(J["theta"])
    m_emp = np.array(J["m_emp"])
    m_fit = np.array(J["m_fit"])

    meas = pd.read_csv("daily_measures_train.csv", parse_dates=["date"]).set_index("date")
    _, rows = gmm.empirical_moments_ext(meas["TPV"].to_numpy(), meas["JV"].to_numpy())
    se = bootstrap_moment_se(rows)

    # slices
    s_tpv = slice(5, 5 + N_TPV)
    s_jv = slice(5 + N_TPV, 5 + N_TPV + N_JV)
    i_fwd0 = 5 + N_TPV + N_JV
    i_bwd0 = i_fwd0 + N_FWD

    k_tpv = np.arange(1, N_TPV + 1)
    k_jv = np.arange(1, N_JV + 1)

    # two-sided cross: x = k in Cov(TPV_t, JV_{t-k}); k>0 jumps lead, k<0 variance leads
    kx = np.arange(-N_BWD, N_FWD + 1)
    ce, cs, cm = [], [], []
    for k in kx:
        if k > 0:
            j = i_fwd0 + k - 1
        elif k == 0:
            j = 4
        else:
            j = i_bwd0 + (-k) - 1
        ce.append(m_emp[j]); cs.append(se[j]); cm.append(m_fit[j])
    ce, cs, cm = np.array(ce), np.array(cs), np.array(cm)

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.ravel():
        _style(ax)

    _panel(axes[0, 0], k_tpv, m_emp[s_tpv], se[s_tpv], m_fit[s_tpv],
           "TPV autocovariance, the two-factor decay",
           "lag k (days)", r"Cov(TPV$_t$, TPV$_{t-k}$)", xticks=k_tpv[::2])

    _panel(axes[0, 1], k_jv, m_emp[s_jv], se[s_jv], m_fit[s_jv],
           "JV autocovariance, jump clustering",
           "lag k (days)", r"Cov(JV$_t$, JV$_{t-k}$)",
           zero_label="old model: exactly 0 ∀k", xticks=k_jv)

    _panel(axes[1, 0], kx, ce, cs, cm,
           "TPV–JV cross-autocovariance (two-sided)",
           r"lag k  ,   Cov(TPV$_t$, JV$_{t-k}$)", "cross-autocovariance",
           xticks=kx)
    axes[1, 0].annotate("variance leads jumps\n(old model: 0)", xy=(-3.4, 0.72),
                        color=MUTED, fontsize=7.6, ha="center")
    axes[1, 0].annotate("jumps lead variance", xy=(3.0, 0.72),
                        color=MUTED, fontsize=7.6, ha="center")

    # standardized residuals across all 40
    ax = axes[1, 1]
    z = (m_emp - m_fit) / se
    lab = J["moment_labels"]
    colors = ([C_EMP] * 5 + ["#7a5ea6"] * N_TPV + ["#e08a1e"] * N_JV
              + ["#3aa0a0"] * N_FWD + ["#c0504d"] * N_BWD)
    ax.axhspan(-2, 2, color=GRID, alpha=0.55, lw=0, label="±2 SE")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.bar(np.arange(40), z, color=colors, width=0.72, edgecolor=SURFACE, linewidth=0.4)
    ax.set_ylim(-3.2, 3.2)
    ax.set_title("Standardized fit residuals, only 2 of 40 exceed ±2",
                 color=INK, fontsize=10.5, loc="left", pad=8)
    ax.set_xlabel("moment (levels · TPV-acov · JV-acov · cross-fwd · cross-bwd)",
                  color=INK, fontsize=8.8)
    ax.set_ylabel(r"(empirical − model)/SE", color=INK, fontsize=8.8)
    ax.set_xticks([2, 5 + N_TPV // 2, 5 + N_TPV + N_JV // 2,
                   i_fwd0 + N_FWD // 2, i_bwd0 + N_BWD // 2])
    ax.set_xticklabels(["levels", "TPV-acov", "JV-acov", "fwd", "bwd"], fontsize=8)
    ax.legend(frameon=False, fontsize=7.8, labelcolor=INK)

    fig.suptitle("Model fit on the train window (2023-01-02 – 2025-06-30, 911 days)",
                 color=INK, fontsize=12.5, x=0.008, ha="left", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig("fig_fit_panels.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)

    # ---- individual panels ------------------------------------------------
    for name, (x, e, s, m, ttl, xl, yl, zl, xt) in {
        "fig_fit_tpv_acov.png": (k_tpv, m_emp[s_tpv], se[s_tpv], m_fit[s_tpv],
                                 "TPV autocovariance: empirical vs model-implied",
                                 "lag k (days)", r"Cov(TPV$_t$, TPV$_{t-k}$)", None, k_tpv),
        "fig_fit_jv_acov.png": (k_jv, m_emp[s_jv], se[s_jv], m_fit[s_jv],
                                "JV autocovariance: jump clustering the old model cannot produce",
                                "lag k (days)", r"Cov(JV$_t$, JV$_{t-k}$)",
                                "old model: exactly 0 ∀k", k_jv),
        "fig_fit_cross.png": (kx, ce, cs, cm,
                              "TPV–JV cross-autocovariance (two-sided)",
                              r"lag k  ,   Cov(TPV$_t$, JV$_{t-k}$)",
                              "cross-autocovariance", None, kx),
    }.items():
        fig, ax = plt.subplots(figsize=(8.8, 4.9), dpi=150)
        fig.patch.set_facecolor(SURFACE); _style(ax)
        _panel(ax, x, e, s, m, ttl, xl, yl, zl, xt)
        fig.tight_layout(); fig.savefig(name, facecolor=SURFACE, bbox_inches="tight")
        plt.close(fig)

    # report
    print("standardized residuals |z|>2:")
    for i in range(40):
        if abs(z[i]) > 2:
            print(f"  {lab[i]:16s} emp {m_emp[i]:9.4f}  mod {m_fit[i]:9.4f}  z {z[i]:+.2f}")
    print(f"\nJV-acov block: empirical sum {m_emp[s_jv].sum():.4f}, "
          f"model {m_fit[s_jv].sum():.4f}  (old model would be exactly 0)")
    print(f"backward-cross block: empirical sum {m_emp[i_bwd0:i_bwd0+N_BWD].sum():.4f}, "
          f"model {m_fit[i_bwd0:i_bwd0+N_BWD].sum():.4f}  (old model: 0)")
    print("saved fig_fit_panels.png + 3 individual panels")


if __name__ == "__main__":
    main()
