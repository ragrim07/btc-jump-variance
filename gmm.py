"""
GMM estimation machinery for the affine-intensity model M'.

Pipeline:  empirical 40-moments  ->  Ledoit-Wolf weighting matrix W
        ->  J(theta) = g'Wg minimised by L-BFGS-B with analytic gradient
        ->  method-of-moments multi-start  ->  moving-block bootstrap CIs.

Nothing here touches real data on import; callers pass daily (TPV, JV) arrays.
"""

import logging

import numpy as np
from scipy.optimize import minimize

import moments as mo
from moments import N_TPV, N_JV, N_FWD, N_BWD, N_MOM
import damping as dp

log = logging.getLogger("gmm")
LN2 = np.log(2.0)


# ---------------------------------------------------------------------------
# 1. EMPIRICAL MOMENTS  (sample analogues, ordering matches theoretical_moments_ext)
# ---------------------------------------------------------------------------

def empirical_moments_ext(tpv, jv, n_tpv=N_TPV, n_jv=N_JV, n_fwd=N_FWD, n_bwd=N_BWD):
    """Return (m_emp, rows).

    m_emp : (M,) sample moments, each a /T average over all valid pairs.
    rows  : (T', M) per-day moment contributions h_t, aligned on the common
            index t in [n_tpv, T-n_bwd) so that every lead/lag is in range.
            cov(rows) estimates the long-run variance used to build W.
    """
    tpv = np.asarray(tpv, float)
    jv = np.asarray(jv, float)
    T = len(tpv)
    dt = tpv - tpv.mean()
    dj = jv - jv.mean()

    m = np.empty(5 + n_tpv + n_jv + n_fwd + n_bwd)
    m[0], m[1] = tpv.mean(), jv.mean()
    m[2], m[3] = dt @ dt / T, dj @ dj / T
    m[4] = dt @ dj / T
    i = 5
    for k in range(1, n_tpv + 1):                     # TPV autocov
        m[i] = dt[k:] @ dt[:-k] / T; i += 1
    for k in range(1, n_jv + 1):                      # JV autocov
        m[i] = dj[k:] @ dj[:-k] / T; i += 1
    for k in range(1, n_fwd + 1):                     # fwd: Cov(TPV_t, JV_{t-k})
        m[i] = dt[k:] @ dj[:-k] / T; i += 1
    for k in range(1, n_bwd + 1):                     # bwd: Cov(TPV_t, JV_{t+k})
        m[i] = dt[:-k] @ dj[k:] / T; i += 1

    # per-day contribution rows on the common window
    lo, hi = n_tpv, T - n_bwd
    t = np.arange(lo, hi)
    R = np.empty((len(t), len(m)))
    R[:, 0], R[:, 1] = tpv[t], jv[t]
    R[:, 2], R[:, 3] = dt[t] ** 2, dj[t] ** 2
    R[:, 4] = dt[t] * dj[t]
    i = 5
    for k in range(1, n_tpv + 1):
        R[:, i] = dt[t] * dt[t - k]; i += 1
    for k in range(1, n_jv + 1):
        R[:, i] = dj[t] * dj[t - k]; i += 1
    for k in range(1, n_fwd + 1):
        R[:, i] = dt[t] * dj[t - k]; i += 1
    for k in range(1, n_bwd + 1):
        R[:, i] = dt[t] * dj[t + k]; i += 1
    return m, R


# ---------------------------------------------------------------------------
# 2. WEIGHTING MATRIX  (Ledoit-Wolf shrinkage + condition-number guardrail)
# ---------------------------------------------------------------------------

def ledoit_wolf_cov(X, target="diagonal"):
    """Shrinkage of cov(X) toward a structured target.

    target='diagonal' (default): Sigma_hat = rho*diag(S) + (1-rho)*S.
    target='identity'          : Sigma_hat = rho*mu*I  + (1-rho)*S  (classic LW).

    WHY THE DIAGONAL TARGET IS THE RIGHT ONE HERE. The classic Ledoit-Wolf
    identity target assumes all coordinates share a scale. Our 40 moment
    conditions do not: on real BTC data Var(TPV) ~ 47 while the JV
    autocovariances are ~ 0.03 -- five orders of magnitude in variance. Shrinking
    toward mu*I discards that scale information, so W treats a 0.03-sized moment
    as if it were as variable as a 47-sized one; the large TPV moments then
    dominate J(theta) and the JV-clustering moments -- exactly the ones that
    identify psi and kappa_j -- receive almost no weight. The diagonal target
    preserves each moment's own variance (so W ~ inverse-variance weighting) while
    still shrinking the poorly-estimated OFF-diagonal correlations that cause the
    ill-conditioning. Empirically this is decisive: identity-target weighting
    returns kappa_j = 3.43 (half-life 0.2 d, JV-acov fit destroyed), diagonal
    target returns kappa_j = 0.25 (half-life 2.8 d) and fits the clustering.
    """
    Xc = X - X.mean(axis=0)
    T, P = Xc.shape
    S = Xc.T @ Xc / T
    if target == "identity":
        Tgt = (np.trace(S) / P) * np.eye(P)
    else:
        Tgt = np.diag(np.diag(S))
    d2 = np.linalg.norm(S - Tgt) ** 2
    # Ledoit-Wolf optimal intensity: E||S - Sigma||^2 estimated by the sum of
    # per-entry sampling variances of S.
    b2 = 0.0
    for i in range(P):
        for j in range(P):
            wij = Xc[:, i] * Xc[:, j]
            b2 += wij.var(ddof=0) / T
    rho = float(np.clip(b2 / d2, 0.0, 1.0)) if d2 > 0 else 1.0
    return rho * Tgt + (1.0 - rho) * S, rho


def build_weighting_matrix(rows, target="diagonal"):
    """W = inverse of the regularised covariance of the moment contributions.

    Guardrail: warn if raw cond > 100; shrink toward the structured target;
    escalate a Tikhonov ridge until cond < 1e6 before inverting.
    """
    cond_raw = np.linalg.cond(np.cov(rows, rowvar=False))
    if cond_raw > 100:
        log.warning("moment-cov cond = %.3e > 100 -> shrinkage (target=%s)",
                    cond_raw, target)
    S, rho = ledoit_wolf_cov(rows, target=target)
    ridge = 1e-8 * np.trace(S) / S.shape[0]
    while np.linalg.cond(S) > 1e6:
        S = S + ridge * np.eye(S.shape[0]); ridge *= 10.0
    log.info("weighting cond: raw %.2e -> reg %.2e (shrink rho=%.3f, target=%s)",
             cond_raw, np.linalg.cond(S), rho, target)
    return np.linalg.inv(S)


# ---------------------------------------------------------------------------
# 3. OBJECTIVE, BOUNDS, SEEDING, CALIBRATION
# ---------------------------------------------------------------------------

def gmm_objective(theta, m_emp, W):
    """J(theta) = g'Wg and its analytic gradient 2 D' W g, g = m(theta) - m_emp."""
    m, D = mo.theoretical_moments_ext(theta)
    g = m - m_emp
    Wg = W @ g
    return g @ Wg, 2.0 * D.T @ Wg


def default_bounds():
    """Economic-identification bounds on theta' = [kc,sc2,Vc,kj,A,B,psi,c,S].

    kappa_c capped so the continuous factor half-life > 5 d (persistent);
    kappa_j only needs kappa_j > 0  (<=> eta < 1, stationarity is FREE);
    all level/variance/gauge params strictly positive; psi >= 0 (lam1 >= 0).
    The transience mandate b >= ln2/2 is a LINEAR constraint on (kj,psi,c),
    checked and reported post-fit (see check_transience).
    """
    eps = 1e-10
    return [(1e-4, LN2 / 5.0),   # kappa_c   half-life > 5 d
            (eps, None),         # sigma_c^2
            (eps, None),         # Vbar_c
            (1e-4, 15.0),        # kappa_j   free to be persistent
            (eps, None),         # A
            (eps, None),         # B
            (0.0, None),         # psi >= 0
            (eps, None),         # c
            (eps, None)]         # S


def check_transience(theta):
    """b = kj + psi c >= ln2/2  (jump-factor half-life < 2 d). Returns (ok, b)."""
    kc, sc2, Vc, kj, A, B, psi, c, S = theta
    b = kj + psi * c
    return b >= LN2 / 2.0, b


def _clip(theta, bounds):
    return np.array([np.clip(v, lo, hi if hi is not None else np.inf)
                     for v, (lo, hi) in zip(theta, bounds)])


def mom_seeds(m_emp, bounds):
    """Method-of-moments seeds from the diagnostics that motivate the model.

    kappa_j : slope of a log-linear fit to the empirical JV-autocovariance.
    C_F,C_B,C_J : lag-1 cross / JV-acov levels divided by g(kappa_j).
    c  = (C_F - C_B)/B ;  psi = C_J/C_F ;  S = C_B/psi .
    CIR block: S f(kj) removed from Var(TPV) leaves sigma_c^2 f(kc); Vbar_c from
    E[TPV] net of vbar.  A small grid over kappa_c yields several starts.
    """
    ETPV, EJV, VarTPV, VarJV, Cov0 = m_emp[:5]
    jv_acov = m_emp[5 + N_TPV: 5 + N_TPV + N_JV]
    fwd = m_emp[5 + N_TPV + N_JV: 5 + N_TPV + N_JV + N_FWD]
    bwd = m_emp[5 + N_TPV + N_JV + N_FWD:]

    # kappa_j from JV-acov log-slope (fallback 0.5)
    ks = np.arange(1, N_JV + 1)
    pos = jv_acov > 0
    if pos.sum() >= 2:
        slope = np.polyfit((ks[pos] - 1.0), np.log(jv_acov[pos]), 1)[0]
        kj = float(np.clip(-slope, 0.05, 5.0))
    else:
        kj = 0.5
    gj = dp.g(kj)

    A = max(EJV, 1e-6)
    B = max(VarJV, 1e-6)
    CF = max(fwd[0] / gj, 1e-8)
    CB = max(bwd[0] / gj, 1e-8)
    CJ = max(jv_acov[0] / gj, 1e-10)
    c = float(np.clip((CF - CB) / B, 1e-5, None))
    psi = float(np.clip(CJ / CF, 1e-4, None))
    S = float(np.clip(CB / psi, 1e-6, None))
    b = kj + psi * c
    vbar = c * A / b

    seeds = []
    for hl_c in (30.0, 12.0, 6.0):                   # CIR half-life grid (days)
        kc = LN2 / hl_c
        sc2 = float(np.clip((VarTPV - S * dp.f(kj)) / dp.f(kc), 1e-6, None))
        Vc = float(np.clip(ETPV - vbar, 1e-6, None))
        seeds.append(_clip([kc, sc2, Vc, kj, A, B, psi, c, S], bounds))
    return seeds


def calibrate(m_emp, W, bounds=None, extra_starts=None):
    """Bounded multi-start L-BFGS-B minimisation of J(theta) with analytic grad."""
    bounds = bounds or default_bounds()
    starts = mom_seeds(m_emp, bounds)
    if extra_starts:
        starts += [_clip(s, bounds) for s in extra_starts]
    best = None
    for p0 in starts:
        res = minimize(gmm_objective, p0, args=(m_emp, W), jac=True,
                       method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 5000, "ftol": 1e-15, "gtol": 1e-12})
        if best is None or res.fun < best.fun:
            best = res
    return best


# ---------------------------------------------------------------------------
# 4. MOVING-BLOCK BOOTSTRAP  (CIs on theta' and derived quantities)
# ---------------------------------------------------------------------------

def moving_block_bootstrap(rows, theta_hat, W, n_boot=300, block=40, seed=0):
    """Moving-block bootstrap on the moment-CONTRIBUTION rows, re-estimating
    warm-started from theta_hat.

    Critical detail: we resample blocks of the per-day contribution rows h_t
    (returned by empirical_moments_ext) and average them to form each bootstrap
    moment vector -- we do NOT re-slice the raw (TPV,JV) series and recompute
    autocovariances. Each h_t already carries its lag products dt_t*dt_{t-k}
    computed on the original contiguous data, so blocks of rows preserve the
    serial dependence; recomputing autocovariances on concatenated raw blocks
    would instead inject spurious zeros at every block join and shrink the
    apparent moment variance (the cause of CI under-coverage).

    Returns dict with 'theta' (n_boot x 9) and derived-quantity arrays, plus
    2.5/50/97.5 percentiles. W is held fixed (standard efficient-GMM practice).
    """
    rng = np.random.default_rng(seed)
    rows = np.asarray(rows, float)
    Tr = rows.shape[0]
    n_blocks = int(np.ceil(Tr / block))
    bounds = default_bounds()
    draws = []
    for _ in range(n_boot):
        starts = rng.integers(0, Tr - block + 1, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:Tr]
        m_b = rows[idx].mean(axis=0)               # bootstrapped moment vector
        res = minimize(gmm_objective, theta_hat, args=(m_b, W), jac=True,
                       method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 3000, "ftol": 1e-14})
        draws.append(res.x)
    draws = np.array(draws)

    keys = ["b", "eta", "lam0", "lam1", "vbar", "halflife_j", "halflife_b"]
    der = {k: np.array([mo.derived_quantities(x)[k] for x in draws]) for k in keys}
    pct = lambda a: np.percentile(a, [2.5, 50, 97.5], axis=0)
    return {"theta": draws, "theta_pct": pct(draws),
            "derived": der, "derived_pct": {k: pct(v) for k, v in der.items()}}
