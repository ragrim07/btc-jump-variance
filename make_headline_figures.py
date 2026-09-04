"""
The three headline figures for the project write-up.

  FIG 1  THE FINDING        jump clustering vs the old model's structural zero
  FIG 2  THE VERIFICATION   40 analytic moments vs an exact simulator
  FIG 3  THE HONEST LIMIT   in-sample compression effect vs its holdout replication

One message each: I found something real / my tools are proven / I test my own
results. Figure 2 is regenerated from the stored verification run; 1 and 3 are
computed here.
"""

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import gmm
import moments as mo
from moments import N_TPV, N_JV, N_BWD

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#3d3b34", "#6d6b66"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
C_EMP, C_MOD, C_OLD = "#2a6fd0", "#159e6d", "#b8484a"
BAND = "#c8d8ee"
N_BOOT, BLOCK, H = 2000, 50, 10
TRAIN_END = pd.Timestamp("2025-06-30")


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)


def boot_se(rows, n_boot=N_BOOT, block=BLOCK, seed=0):
    rng = np.random.default_rng(seed)
    Tr = rows.shape[0]
    nb = int(np.ceil(Tr / block))
    B = np.empty((n_boot, rows.shape[1]))
    for i in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block)
                              for s in rng.integers(0, Tr - block + 1, nb)])[:Tr]
        B[i] = rows[idx].mean(axis=0)
    return B.std(axis=0, ddof=1), B


# ---------------------------------------------------------------- FIG 1
def fig1_finding():
    meas = pd.read_csv("daily_measures.csv", parse_dates=["date"]).set_index("date")
    m_emp, rows = gmm.empirical_moments_ext(meas["TPV"].to_numpy(), meas["JV"].to_numpy())
    se, B = boot_se(rows)
    tr = json.load(open("estimate_train_results.json"))
    m_fit, _ = mo.theoretical_moments_ext(np.array(tr["theta"]))

    i0 = 5 + N_TPV
    ks = np.arange(1, N_JV + 1)
    emp, mod, s = m_emp[i0:i0 + N_JV], m_fit[i0:i0 + N_JV], se[i0:i0 + N_JV]
    joint = m_emp[i0:i0 + N_JV].sum()
    joint_t = joint / B[:, i0:i0 + N_JV].sum(1).std(ddof=1)

    ib = 5 + N_TPV + N_JV + 5
    kb = np.arange(1, N_BWD + 1)
    empb, modb, sb = m_emp[ib:ib + N_BWD], m_fit[ib:ib + N_BWD], se[ib:ib + N_BWD]
    jb = m_emp[ib:ib + N_BWD].sum()
    jb_t = jb / B[:, ib:ib + N_BWD].sum(1).std(ddof=1)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        _style(ax)

    for ax, (k, e, mo_, sd, ttl, yl, tstat) in zip(axes, [
        (ks, emp, mod, s, "Jump clustering", r"Cov(JV$_t$, JV$_{t-k}$)", joint_t),
        (kb, empb, modb, sb, "Diffusive variance leads future jumps",
         r"Cov(TPV$_t$, JV$_{t+k}$)", jb_t)]):
        ax.fill_between(k, e - 2 * sd, e + 2 * sd, color=BAND, alpha=0.6, lw=0,
                        label="data ± 2 SE", zorder=1)
        ax.axhline(0, color=C_OLD, lw=2.0, ls=(0, (5, 3)), zorder=3,
                   label="standard model: exactly 0 at every lag")
        ax.plot(k, e, "o-", color=C_EMP, lw=1.8, ms=5.5, mec=SURFACE, mew=1.0,
                label="data", zorder=5)
        ax.plot(k, mo_, "s-", color=C_MOD, lw=1.9, ms=4.8, mec=SURFACE, mew=1.0,
                label="self-exciting model", zorder=4)
        ax.set_xticks(k)
        ax.set_title(ttl, color=INK, fontsize=11, loc="left", pad=8)
        ax.set_xlabel("lag k (days)", color=INK2, fontsize=9)
        ax.set_ylabel(yl, color=INK2, fontsize=9)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
        ax.annotate(f"joint test vs zero:  t = {tstat:.2f}",
                    xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
                    fontsize=10, color=INK,
                    bbox=dict(boxstyle="round,pad=0.45", fc="#eef5ee",
                              ec=C_MOD, lw=1.0))

    fig.suptitle("Bitcoin jumps cluster — the standard model says they cannot",
                 color=INK, fontsize=13, x=0.006, ha="left", y=1.005)
    fig.text(0.006, 0.945,
             "Constant-intensity jump-diffusion predicts both curves are identically zero "
             "for every parameter value. Both are significantly positive.",
             color=MUTED, fontsize=9, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    fig.savefig("FIG1_finding.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"FIG1  jump-clustering joint t = {joint_t:.2f}, backward-cross joint t = {jb_t:.2f}")


# ---------------------------------------------------------------- FIG 2
def fig2_verification():
    J = json.load(open("sim_verify_results.json"))
    res = J["results"]
    pal = ["#2a6fd0", "#159e6d", "#e08a1e", "#b8484a", "#7a5ea6", "#3aa0a0"]

    fig, ax = plt.subplots(figsize=(12.4, 4.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.axhspan(-2, 2, color=GRID, alpha=0.55, lw=0, label="±2 Monte-Carlo SE")
    for c in (-3, 3):
        ax.axhline(c, color=C_OLD, lw=1.3, ls=(0, (4, 3)))
    ax.axhline(0, color=AXIS, lw=0.9)
    for k, r in enumerate(res):
        ax.scatter(np.arange(40) + (k - 2.5) * 0.12, r["z"], s=17, color=pal[k],
                   edgecolor=SURFACE, linewidth=0.4, zorder=3,
                   label=f"{r['name']}  (η={r['eta_true']:.2f})")
    mx = max(r["max_absz"] for r in res)
    ax.set_xlim(-1, 40)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("each of the 40 closed-form moments", color=INK2, fontsize=9)
    ax.set_ylabel("(simulated − analytic) / MC standard error", color=INK2, fontsize=9)
    ax.set_title("Every derived formula checked against an exact simulator, "
                 "at six known parameter sets", color=INK, fontsize=12.5, loc="left", pad=10)
    ax.annotate(f"max |z| = {mx:.2f}    0 of 240 outside ±3",
                xy=(0.985, 0.06), xycoords="axes fraction", ha="right", fontsize=10.5,
                color=INK, bbox=dict(boxstyle="round,pad=0.45", fc="#eef5ee",
                                     ec=C_MOD, lw=1.0))
    ax.legend(frameon=False, fontsize=7.8, ncol=4, loc="upper center", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig("FIG2_verification.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"FIG2  max |z| = {mx:.2f}")


# ---------------------------------------------------------------- FIG 3
def fig3_honest_limit():
    df = pd.read_csv("vrp_series.csv", parse_dates=["date"]).set_index("date")
    df["dVRP"] = df.VRP.shift(-H) - df.VRP
    d = df.dropna(subset=["dVRP"]).copy()

    def grid(s):
        lv = pd.qcut(s.VRP, 3, labels=["low", "mid", "high"])
        vj = pd.qcut(s.Vj, 3, labels=["low", "mid", "high"])
        g = s.groupby([lv, vj], observed=True).dVRP.agg(["mean", "count"])
        return g["mean"].unstack(), g["count"].unstack()

    mtr, ctr = grid(d[d.index <= TRAIN_END])
    mho, cho = grid(d[d.index > TRAIN_END])
    vmax = max(np.abs(mtr.to_numpy()).max(), np.abs(mho.to_numpy()).max())
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig = plt.figure(figsize=(12.4, 5.9), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    # explicit geometry: two panels + a slim colorbar, with reserved header space
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.045],
                          left=0.075, right=0.945, top=0.70, bottom=0.11, wspace=0.22)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])

    for ax, (M, C, ttl, sub) in zip(axes, [
        (mtr, ctr, "In sample  (2023-01 → 2025-06)", "n = 889 days"),
        (mho, cho, "Held out  (2025-07 → 2026-06)", "n = 355 days")]):
        ax.set_facecolor(SURFACE)
        im = ax.imshow(M.to_numpy(), cmap="RdYlGn_r", norm=norm, aspect="auto")
        for i in range(3):
            for j in range(3):
                v, n = M.iloc[i, j], int(C.iloc[i, j])
                weak = n < 25
                # dark cells need light text; pick by distance from the midpoint
                strong_fill = abs(v) > 0.62 * vmax
                fg = "#ffffff" if strong_fill else "#111111"
                ax.text(j, i - 0.10, f"{v:+.2f}", ha="center", va="center",
                        fontsize=16, color=fg, fontweight="bold")
                ax.text(j, i + 0.24, f"n={n}" + ("  (thin)" if weak else ""),
                        ha="center", va="center", fontsize=8.5,
                        fontweight="bold" if weak else "normal", color=fg)
        ax.set_xticks(range(3)); ax.set_xticklabels(["low", "mid", "high"], fontsize=9.5)
        ax.set_yticks(range(3)); ax.set_yticklabels(["low", "mid", "high"], fontsize=9.5)
        ax.set_xlabel("jump state  $V^j$", color=INK2, fontsize=10)
        ax.set_ylabel("variance risk premium level", color=INK2, fontsize=10)
        ax.set_title(f"{ttl}\n{sub}", color=INK, fontsize=11.5, loc="left", pad=9)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0, colors=MUTED)

    cb = fig.colorbar(im, cax=cax)
    cb.set_label("mean change in premium over next 10 days", color=INK2, fontsize=9)
    cb.ax.tick_params(colors=MUTED, labelsize=8)
    cb.outline.set_visible(False)

    tr_row, ho_row = mtr.loc["high"], mho.loc["high"]
    fig.text(0.008, 0.965, "A large in-sample effect that held-out data does not reproduce",
             color=INK, fontsize=13.5, ha="left", va="top")
    fig.text(0.008, 0.905,
             "The bottom row holds the premium rich, so any variation across it must come from the jump state.\n"
             f"In sample the row is ordered ({tr_row['low']:+.2f} → {tr_row['mid']:+.2f} → {tr_row['high']:+.2f}) — "
             f"rich premium plus a hot jump state compresses hardest.\n"
             f"Out of sample the ordering breaks ({ho_row['low']:+.2f} → {ho_row['mid']:+.2f} → {ho_row['high']:+.2f}), "
             "and the only cell favouring the effect holds 8 observations.",
             color=MUTED, fontsize=9.2, ha="left", va="top", linespacing=1.5)
    fig.savefig("FIG3_honest_limit.png", facecolor=SURFACE)
    plt.close(fig)
    print(f"FIG3  train high-VRP row {tr_row.to_dict()}")
    print(f"      holdout high-VRP row {ho_row.to_dict()}")


if __name__ == "__main__":
    fig1_finding()
    fig2_verification()
    fig3_honest_limit()
    print("\nsaved FIG1_finding.png, FIG2_verification.png, FIG3_honest_limit.png")
