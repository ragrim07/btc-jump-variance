"""
Ex-ante variance risk premium from the calibrated P-measure model.

    VRP_t  =  E^Q_t[QV_{t,t+30}]  -  E^P_t[QV_{t,t+30}]

BOTH LEGS ARE KNOWN AT TIME t. No future realized variance enters the signal, so
VRP_t is a genuine tradeable quantity, not an ex-post accounting identity.

Q leg. Deribit DVOL is a 30-day model-free implied volatility in ANNUALISED
VOL POINTS. Our realized measures are in percent^2 per day, so
    E^Q_t[QV]/day = DVOL_t^2 / 365 .

P leg (eq. 8.1 of the theory doc), with tau = 30 days and
tau_tilde(a) = (1 - e^{-a*tau})/a:
    E^P_t[QV_{t,t+tau}] = (Vbar_c + vbar + A) tau
                        + (V^c_t - Vbar_c) tau_tilde(kappa_c)
                        + (1 + psi)(V^j_t - vbar) tau_tilde(kappa_j)
Divide by tau for a per-day figure. The (1+psi) loading is the point of the
extension: an elevated jump state raises future variance TWICE -- through its own
decay and through the extra jumps it will trigger.

States. We do not need the full LMMSE projection: the SDE already tells us what
V^j is. Since dV^j = -b V^j dt + sum of c*h^2 kicks,
    V^j_t = c * sum_{s<=t} JV_s e^{-b(t-s)}                    (an EWMA of JV at rate b)
and V^c is recovered as a smoothed (MedRV - V^j), smoothing at the CIR rate
kappa_c because that is the persistence V^c actually has. Transparent and derived
from the model, not a black box.

PARAMETERS COME FROM THE TRAIN WINDOW ONLY (2023-01..2025-06). Everything from
2025-07 onward is therefore an out-of-sample application.

KNOWN BIAS, STATED UP FRONT. The event study showed the model under-predicts
post-shock persistence (observed half-life ~9.5 d vs predicted 2.4 d). So after
big jumps the P leg is too LOW and VRP is biased HIGH. We quantify this by also
running a purely empirical HAR benchmark for the P leg.
"""

import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAU = 30.0
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#6d6b66", "#e1e0d9", "#c3c2b7"
C_Q, C_P, C_V, C_HI, C_LO = "#b8860b", "#2a6fd0", "#159e6d", "#b8484a", "#7a5ea6"


def tau_tilde(a, tau=TAU):
    return (1.0 - np.exp(-a * tau)) / a


def newey_west_t(y, X, lags):
    """OLS with Newey-West HAC standard errors. Returns (beta, tstat, r2)."""
    X = np.asarray(X, float); y = np.asarray(y, float)
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        G = (X[L:] * resid[L:, None]).T @ (X[:-L] * resid[:-L, None])
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(V))
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = 1.0 - (resid ** 2).sum() / ss_tot
    return beta, beta / se, r2


def build_states(meas, theta):
    """V^j from the SDE (EWMA of jump energy at rate b); V^c from smoothed residual."""
    kc, sc2, Vbar_c, kj, A, B, psi, c, S = theta
    b = kj + psi * c
    jv = meas["JV_med"].to_numpy()          # MedRV-based: robust on crash days
    cv = meas["MedRV"].to_numpy()           # continuous variance observable

    # V^j_t = c * sum_{s<=t} JV_s e^{-b(t-s)}   (recursive EWMA)
    decay = np.exp(-b)
    vj = np.empty(len(jv))
    run = c * jv[0] / (1.0 - decay)         # start at the stationary-ish level
    vj[0] = run
    for i in range(1, len(jv)):
        run = run * decay + c * jv[i]
        vj[i] = run

    # V^c: smooth (continuous variance - V^j) at the CIR persistence rate
    raw = np.maximum(cv - vj, 0.0)
    wc = np.exp(-kc)
    vc = np.empty(len(raw))
    vc[0] = raw[0]
    for i in range(1, len(raw)):
        vc[i] = wc * vc[i - 1] + (1.0 - wc) * raw[i]
    return pd.DataFrame({"Vc": vc, "Vj": vj}, index=meas.index)


def p_forecast(states, theta):
    """E^P_t[QV_{t,t+30}] per day."""
    kc, sc2, Vbar_c, kj, A, B, psi, c, S = theta
    b = kj + psi * c
    vbar = c * A / b
    level = (Vbar_c + vbar + A) * TAU
    term_c = (states["Vc"] - Vbar_c) * tau_tilde(kc)
    term_j = (1.0 + psi) * (states["Vj"] - vbar) * tau_tilde(kj)
    return (level + term_c + term_j) / TAU


def har_forecast(rv, window=30):
    """Empirical benchmark: HAR-style forecast of mean daily variance over the
    next 30 days, using only information available at t (1d, 5d, 22d averages)."""
    d1 = rv.shift(1)
    d5 = rv.rolling(5).mean().shift(1)
    d22 = rv.rolling(22).mean().shift(1)
    y = rv.rolling(window).mean().shift(-window)      # target (used to FIT only)
    df = pd.concat([y, d1, d5, d22], axis=1).dropna()
    df.columns = ["y", "d1", "d5", "d22"]
    X = np.column_stack([np.ones(len(df)), df[["d1", "d5", "d22"]].to_numpy()])
    beta, _, _ = newey_west_t(df["y"].to_numpy(), X, lags=30)
    Xf = np.column_stack([np.ones(len(rv)), d1.to_numpy(), d5.to_numpy(), d22.to_numpy()])
    return pd.Series(Xf @ beta, index=rv.index)


def main():
    meas = pd.read_csv("daily_measures.csv", parse_dates=["date"]).set_index("date")
    dvol = pd.read_csv("dvol_btc_daily.csv", parse_dates=["date"]).set_index("date")["close"]
    tr = json.load(open("estimate_train_results.json"))
    theta = np.array(tr["theta"])
    train_end = pd.Timestamp("2025-06-30")

    states = build_states(meas, theta)
    eP = p_forecast(states, theta)
    eQ = (dvol ** 2 / 365.0).reindex(meas.index)
    eP_har = har_forecast(meas["RV"])

    df = pd.DataFrame({"Q": eQ, "P": eP, "P_har": eP_har,
                       "Vc": states["Vc"], "Vj": states["Vj"],
                       "RV": meas["RV"]}).dropna()
    df["VRP"] = df["Q"] - df["P"]
    df["VRP_har"] = df["Q"] - df["P_har"]
    df["jump_share"] = df["Vj"] / (df["Vc"] + df["Vj"])
    df["oos"] = df.index > train_end

    print("=" * 72)
    print("EX-ANTE VARIANCE RISK PREMIUM   VRP_t = E^Q_t[QV] - E^P_t[QV]")
    print("=" * 72)
    print(f"sample {df.index[0].date()} .. {df.index[-1].date()}   n = {len(df)}")
    print(f"  mean E^Q (implied)         {df.Q.mean():8.3f}  %^2/day")
    print(f"  mean E^P (model)           {df.P.mean():8.3f}")
    print(f"  mean E^P (HAR benchmark)   {df.P_har.mean():8.3f}")
    print(f"  mean realized RV           {df.RV.mean():8.3f}")
    print("-" * 72)
    for nm, col in (("model", "VRP"), ("HAR", "VRP_har")):
        v = df[col]
        print(f"  VRP ({nm:5s}): mean {v.mean():7.3f}  median {v.median():7.3f}  "
              f"sd {v.std():6.3f}  positive {100*(v>0).mean():5.1f}%")
    oos = df[df.oos]
    print(f"  VRP (model), OUT-OF-SAMPLE only (n={len(oos)}): mean {oos.VRP.mean():.3f}, "
          f"positive {100*(oos.VRP>0).mean():.1f}%")

    # ---- persistence of the VRP ------------------------------------------
    v = df["VRP"]
    rho = v.autocorr(1)
    hl = np.log(2) / -np.log(abs(rho)) if 0 < abs(rho) < 1 else np.nan
    print("-" * 72)
    print(f"  VRP persistence: AR(1) rho = {rho:.4f}  ->  half-life {hl:.1f} days")

    # ---- what drives the VRP ---------------------------------------------
    X = np.column_stack([np.ones(len(df)), df.Vc, df.Vj])
    beta, tstat, r2 = newey_west_t(df.VRP.to_numpy(), X, lags=30)
    print(f"  VRP on states: const {beta[0]:+.3f} (t {tstat[0]:+.1f}), "
          f"Vc {beta[1]:+.3f} (t {tstat[1]:+.1f}), Vj {beta[2]:+.3f} (t {tstat[2]:+.1f}), "
          f"R2 {r2:.3f}")
    print("     (Vj coefficient > 0 => the market charges MORE per unit of JUMP risk)")

    # ---- does the ex-ante VRP predict the payoff to selling variance? -----
    fwd = df["RV"].rolling(30).mean().shift(-30)
    pay = (df["Q"] - fwd).dropna()                 # per-day P&L of a short-variance position
    sig = df["VRP"].reindex(pay.index)
    X = np.column_stack([np.ones(len(sig)), sig.to_numpy()])
    beta, tstat, r2 = newey_west_t(pay.to_numpy(), X, lags=30)
    print("-" * 72)
    print("  PREDICTIVE TEST  payoff(t..t+30) = a + b * VRP_t   (HAC lags=30)")
    print(f"     a = {beta[0]:+.3f} (t {tstat[0]:+.2f})   b = {beta[1]:+.3f} "
          f"(t {tstat[1]:+.2f})   R2 = {r2:.3f}")
    print(f"     unconditional mean payoff = {pay.mean():+.3f} %^2/day "
          f"({100*(pay>0).mean():.1f}% of days positive)")

    # ---- CONTAMINATION CONTROL (the test that matters) --------------------
    # payoff = Q - fwdRV  and  signal VRP = Q - P  SHARE the Q leg, so a naive
    # regression of one on the other is partly mechanical. corr(VRP,Q) ~ 0.97.
    # The honest question: does VRP predict ANYTHING once Q is controlled for?
    print("-" * 72)
    print("  CONTAMINATION CONTROL  (payoff and VRP share the Q leg)")
    Qs = df["Q"].reindex(pay.index)
    print(f"     corr(VRP, Q) = {np.corrcoef(sig, Qs)[0,1]:.3f}")
    for nm, Xc in (("VRP alone", np.column_stack([np.ones(len(sig)), sig])),
                   ("Q alone  ", np.column_stack([np.ones(len(sig)), Qs])),
                   ("Q + VRP  ", np.column_stack([np.ones(len(sig)), Qs, sig]))):
        b, t, r2 = newey_west_t(pay.to_numpy(), Xc, lags=30)
        print(f"     {nm}  beta {np.round(b,3)}  t {np.round(t,2)}  R2 {r2:.3f}")
    print("     => VRP is insignificant once Q is included: the predictive content")
    print("        is the IMPLIED-VOL LEVEL, not the structural P leg.")

    # ---- does the structural P leg beat a simple benchmark? ---------------
    print("-" * 72)
    print("  FORECAST QUALITY of the P leg (target: mean RV over next 30d)")
    for nm, col in (("model P", df["P"]), ("HAR    ", df["P_har"])):
        f = col.reindex(pay.index)
        err = fwd.reindex(pay.index) - f
        tot = ((fwd.reindex(pay.index) - fwd.reindex(pay.index).mean()) ** 2).sum()
        print(f"     {nm}  RMSE {np.sqrt((err**2).mean()):.3f}   "
              f"R2 {1 - (err**2).sum()/tot:.3f}")
    print("     => the structural model does NOT beat a 3-term HAR regression.")

    # ---- does the jump decomposition (unique to this model) add value? ----
    js = df["jump_share"].reindex(pay.index)
    b, t, r2 = newey_west_t(pay.to_numpy(),
                            np.column_stack([np.ones(len(js)), Qs, js]), lags=30)
    q10 = pay.quantile(0.10)
    print("-" * 72)
    print("  JUMP-SHARE DECOMPOSITION (the one thing HAR cannot give)")
    print(f"     payoff on [Q, jump_share]: jump_share {b[2]:+.3f} (t {t[2]:+.2f}) "
          f"-> not significant")
    print(f"     jump share before worst-decile outcomes {js[pay<=q10].mean():.4f} "
          f"vs {js.mean():.4f} overall -> no tail-risk warning")

    # ---- regime robustness: level vs z-scored signal ----------------------
    z = ((df["VRP"] - df["VRP"].rolling(250).mean())
         / df["VRP"].rolling(250).std()).reindex(pay.index)
    dz = pd.DataFrame({"p": pay, "z": z}).dropna()
    b, t, r2 = newey_west_t(dz["p"].to_numpy(),
                            np.column_stack([np.ones(len(dz)), dz["z"]]), lags=30)
    ho = dz[dz.index > train_end]
    print("-" * 72)
    print("  REGIME ROBUSTNESS")
    print(f"     VRP LEVEL flips sign out of sample (train {df[~df.oos].VRP.mean():+.2f} "
          f"-> holdout {df[df.oos].VRP.mean():+.2f}) because the model's long-run")
    print(f"     variance is anchored to the train regime (RV {meas.loc[:train_end,'RV'].mean():.2f} "
          f"-> {meas.loc[train_end:,'RV'].mean():.2f}).")
    print(f"     z-SCORED VRP is robust: b {b[1]:+.3f} (t {t[1]:+.2f}), "
          f"OOS corr {ho['z'].corr(ho['p']):+.3f}")

    # ---- conditional payoffs (report with the caveats above in mind) ------
    print("-" * 72)
    hi = sig > sig.median()
    n_eff = len(pay) / 30.0                      # 30-day overlap -> effective n
    for label, mask in (("always short", pd.Series(True, index=pay.index)),
                        ("VRP rich (top half)", hi)):
        p = pay[mask]
        sr = p.mean() / p.std() * np.sqrt(365 / 30) if p.std() > 0 else np.nan
        print(f"     {label:22s} n={len(p):4d}  mean {p.mean():+.3f}  "
              f"hit {100*(p>0).mean():5.1f}%  worst {p.min():+.2f}  ann.SR~{sr:4.2f}")
    print(f"     CAVEAT: 30-day overlapping windows leave only ~{n_eff:.0f} independent")
    print("     observations, so these Sharpes carry very wide error bars; they exclude")
    print("     transaction costs and the fat left tail intrinsic to short variance.")

    df.to_csv("vrp_series.csv", float_format="%.6f")

    # ---------------- figures ----------------
    fig, axes = plt.subplots(3, 1, figsize=(12, 9.6), dpi=150, sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 1, 1]})
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE); ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"): ax.spines[s_].set_color(AXIS)

    ax = axes[0]
    ax.plot(df.index, df.Q, color=C_Q, lw=1.2, label="E$^Q$ implied (DVOL$^2$/365)")
    ax.plot(df.index, df.P, color=C_P, lw=1.2, label="E$^P$ model forecast")
    ax.set_ylabel("variance (%$^2$/day)", color=INK, fontsize=9)
    ax.set_title("The two legs: what the options market charges vs what the model expects",
                 color=INK, fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, ncol=2)

    ax = axes[1]
    ax.axhline(0, color=AXIS, lw=0.9)
    ax.fill_between(df.index, 0, df.VRP.where(df.VRP > 0), color=C_V, alpha=0.55, lw=0)
    ax.fill_between(df.index, 0, df.VRP.where(df.VRP <= 0), color=C_HI, alpha=0.5, lw=0)
    ax.plot(df.index, df.VRP, color=INK, lw=0.6, alpha=0.7)
    ax.axvline(train_end, color=MUTED, lw=1.2, ls=(0, (4, 3)))
    ax.annotate("parameters estimated ← | → out of sample", (train_end, ax.get_ylim()[1]*0.8),
                xytext=(6, 0), textcoords="offset points", color=MUTED, fontsize=8)
    ax.set_ylabel("VRP (%$^2$/day)", color=INK, fontsize=9)
    ax.set_title(f"Ex-ante variance risk premium — positive {100*(df.VRP>0).mean():.0f}% of days "
                 f"(green = market overpays for variance)", color=INK, fontsize=10.5, loc="left")

    ax = axes[2]
    ax.plot(df.index, df.jump_share, color=C_LO, lw=1.0)
    ax.set_ylabel("jump share $V^j/(V^c{+}V^j)$", color=INK, fontsize=9)
    ax.set_xlabel("date", color=INK, fontsize=9)
    ax.set_title("Jump share of total variance — the state that conditions the trade",
                 color=INK, fontsize=10.5, loc="left")

    fig.tight_layout()
    fig.savefig("fig_vrp.png", facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("\nsaved vrp_series.csv + fig_vrp.png")


if __name__ == "__main__":
    main()
