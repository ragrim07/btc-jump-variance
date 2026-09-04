"""
Exact simulator for the affine-intensity model M' (Chapter 9).

Produces daily (TPV, JV) whose law is EXACTLY the model's -- no discretization
of the jump dynamics, so any gap between the simulated moments and the analytic
formulas of moments.py is a bug, not scheme bias.

Three independent pieces, assembled per day:

  V^c (CIR)   exact noncentral-chi^2 transition on a fine sub-grid; the daily
              integral  Int_{t-1}^t V^c ds  by trapezoid on the exact path.
              (V^c is very persistent, so a modest sub-grid is more than enough.)

  V^j (jumps) Ogata thinning: between jumps V^j decays at the microscopic rate
              b, so lambda(t)=lambda0+lambda1 V^j(t^-) is monotone decreasing and
              its post-jump value is an exact ceiling. The daily integral
              Int V^j ds is accumulated ANALYTICALLY between events
              (Int_a^b v0 e^{-b(s-a)} ds = v0 (1-e^{-b(b-a)})/b), split at day
              boundaries -- exact, no quadrature.

  marks       only (m_h, m_hh) of the price-jump energy w=h^2 are identified, so
              w is drawn from the two-parameter Gamma matched to E[w]=1 (the
              m_h=1 gauge) and E[w^2]=m_hh=B/A. Each jump adds w to JV and c*w to
              V^j (the translation gauge k=c h^2 couples them: the co-jump).

The variance S=Var(V^j) is NOT an input -- it emerges from the dynamics and
equals c^2 B/(2 kappa_j) automatically (the structural identity). Test points
must therefore be chosen ON that manifold (see make_theta).
"""

import numpy as np
import pandas as pd

import moments as mo


# ---------------------------------------------------------------------------
def make_theta(kappa_c, sigma_c2, Vbar_c, kappa_j, A, B, psi, c):
    """Assemble a valid theta' with S pinned by the structural identity
    S = c^2 B/(2 kappa_j).  Returns the 9-vector and a feasibility dict."""
    S = c ** 2 * B / (2.0 * kappa_j)
    theta = np.array([kappa_c, sigma_c2, Vbar_c, kappa_j, A, B, psi, c, S])
    b = kappa_j + psi * c
    feas = {
        "feller_ok": Vbar_c ** 2 >= sigma_c2,             # CIR stays positive
        "energy_var_ok": (B / A) - 1.0 >= 0.0,            # Var(w) = m_hh - 1 >= 0
        "transience_ok": b >= np.log(2) / 2,              # b >= ln2/2
        "eta": psi * c / b,
    }
    return theta, feas


def _gamma_params(m_hh):
    """Gamma(shape a, scale s) with E[w]=1, E[w^2]=m_hh  =>  var=m_hh-1, s=var, a=1/var.
    Returns None for the degenerate deterministic case (m_hh=1 => w==1)."""
    var = m_hh - 1.0
    if var <= 1e-12:
        return None
    return 1.0 / var, var


# ---------------------------------------------------------------------------
def _cir_daily_integral(kc, sc2, Vbar, n_total, substeps, rng):
    """Int_{t-1}^t V^c ds for each day via exact noncentral-chi^2 path + trapezoid."""
    sig2 = 2.0 * kc * sc2 / Vbar                          # sigma_cv^2
    dt = 1.0 / substeps
    ekt = np.exp(-kc * dt)
    cc = sig2 * (1.0 - ekt) / (4.0 * kc)
    df = 4.0 * kc * Vbar / sig2
    N = n_total * substeps

    # exact CIR path on the sub-grid, started from its Gamma stationary law
    shape, scale = df / 2.0, sig2 / (2.0 * kc)
    P = np.empty(N + 1)
    P[0] = rng.gamma(shape, scale)
    for i in range(N):
        P[i + 1] = cc * rng.noncentral_chisquare(df, P[i] * ekt / cc)

    seg = P[:N].reshape(n_total, substeps)               # p_0..p_{S-1} per day
    right = P[substeps::substeps]                         # p_S per day (day close)
    return dt * (seg.sum(axis=1) - 0.5 * seg[:, 0] + 0.5 * right)


def _vj_daily_integral_and_jv(lam0, lam1, b, c, vbar, gp, n_total, rng):
    """Ogata thinning: returns (Int V^j ds per day, JV per day)."""
    Ij = np.zeros(n_total)
    JV = np.zeros(n_total)
    T = float(n_total)

    def accumulate(t0, t1, v0):
        """Add Int_{t0}^{t1} v0 e^{-b(s-t0)} ds to the day buckets it spans."""
        s, vv = t0, v0
        while s < t1:
            day = int(np.floor(s))
            nb = min(day + 1.0, t1)
            dl = nb - s
            if day < n_total:
                Ij[day] += vv * (1.0 - np.exp(-b * dl)) / b
            vv *= np.exp(-b * dl)
            s = nb

    t, v = 0.0, vbar                                     # start near stationary mean
    while True:
        ceiling = lam0 + lam1 * v                        # exact bound (monotone decay)
        t_next = t + rng.exponential(1.0 / ceiling)
        accumulate(t, min(t_next, T), v)                 # path decays regardless of accept
        if t_next >= T:
            break
        v = v * np.exp(-b * (t_next - t))                # decayed value at proposal
        if rng.random() < (lam0 + lam1 * v) / ceiling:   # accept w.p. true/ceiling
            w = 1.0 if gp is None else rng.gamma(gp[0], gp[1])
            day = int(np.floor(t_next))
            if day < n_total:
                JV[day] += w
            v += c * w                                   # state kick (self-excitation)
        t = t_next
    return Ij, JV


# ---------------------------------------------------------------------------
def simulate(theta, n_days, seed=0, substeps=16, n_burn=100):
    """Simulate n_days of daily (TPV, JV) from theta'. Burn-in days are discarded."""
    kc, sc2, Vbar, kj, A, B, psi, c, S = theta
    d = mo.derived_quantities(theta)
    gp = _gamma_params(B / A)
    n_total = n_days + n_burn
    rng = np.random.default_rng(seed)

    Ic = _cir_daily_integral(kc, sc2, Vbar, n_total, substeps, rng)
    Ij, JV = _vj_daily_integral_and_jv(
        d["lam0"], d["lam1"], d["b"], c, d["vbar"], gp, n_total, rng)

    sl = slice(n_burn, n_total)
    return pd.DataFrame({"TPV": (Ic + Ij)[sl], "JV": JV[sl]})
