"""
The 40-moment population engine for the affine-intensity model M' and its
analytic 40x9 Jacobian (Theorem 5.3 of affine_intensity_theory).

Parameter vector (relaxed gauge, m_h = 1):

    theta' = [ kappa_c, sigma_c^2, Vbar_c, kappa_j, A, B, psi, c, S ]  in R^9

    kappa_c   CIR mean-reversion (continuous factor, persistent)
    sigma_c^2 unconditional variance of V^c
    Vbar_c    long-run mean of V^c
    kappa_j   EFFECTIVE (observable) decay of V^j   -- free to be persistent
    A = lam_bar * m_h    = E[JV]
    B = lam_bar * m_hh   = Var(JV) level
    psi = lam_1 * m_h    self-excitation strength
    c                    translation gauge  k(x) = c h^2(x)
    S = Var(V^j)

Derived (reported, not estimated):
    b     = kappa_j + psi c        microscopic per-jump decay (transient)
    eta   = psi c / b              branching ratio (children per jump); eta < 1
    lam_bar = A,  lam0 = A(1-eta),  lam1 = psi,  vbar = c A / b,  rho_j = -b

Composite coefficients that recur in the moments:
    C_F = c B + psi S     forward cross weight (jumps lead)
    C_B = psi S           backward cross weight (variance leads)
    C_J = psi (psi S + c B) = psi C_F   JV-autocovariance / Var(JV) weight
Over-identifying identities the data can reject:
    C_J S = C_B C_F     and     C_F - C_B = c B     and     S = c^2 B / (2 kappa_j)

Moment vector ordering (length 40 with the default menu):
    [0]   E[TPV]
    [1]   E[JV]
    [2]   Var(TPV)
    [3]   Var(JV)
    [4]   Cov(TPV_t, JV_t)                       (contemporaneous cross)
    [5:20]   Cov(TPV_t, TPV_{t-k}),  k = 1..15   (TPV autocovariance)
    [20:30]  Cov(JV_t,  JV_{t-k}),   k = 1..10   (JV autocovariance)
    [30:35]  Cov(TPV_t, JV_{t-k}),   k = 1..5    (forward cross, jumps lead)
    [35:40]  Cov(TPV_t, JV_{t+k}),   k = 1..5    (backward cross, variance leads)
"""

import numpy as np

import damping as dp

PARAM_NAMES = ["kappa_c", "sigma_c2", "Vbar_c", "kappa_j", "A", "B", "psi", "c", "S"]
IDX = {name: i for i, name in enumerate(PARAM_NAMES)}
KC, SC2, VC, KJ, A_, B_, PSI, C_, S_ = range(9)   # positional aliases

# default moment menu (counts per block)
N_TPV, N_JV, N_FWD, N_BWD = 15, 10, 5, 5
N_MOM = 5 + N_TPV + N_JV + N_FWD + N_BWD          # = 40


def moment_labels(n_tpv=N_TPV, n_jv=N_JV, n_fwd=N_FWD, n_bwd=N_BWD):
    lab = ["E[TPV]", "E[JV]", "Var(TPV)", "Var(JV)", "Cov(TPV,JV)_0"]
    lab += [f"TPVacov_{k}" for k in range(1, n_tpv + 1)]
    lab += [f"JVacov_{k}" for k in range(1, n_jv + 1)]
    lab += [f"crossFwd_{k}" for k in range(1, n_fwd + 1)]
    lab += [f"crossBwd_{k}" for k in range(1, n_bwd + 1)]
    return lab


def derived_quantities(theta):
    """Map the estimated composites to the interpretable structural quantities."""
    kc, sc2, Vc, kj, A, B, psi, c, S = theta
    b = kj + psi * c
    eta = psi * c / b
    out = {
        "b": b, "eta": eta, "lam_bar": A, "lam0": A * (1.0 - eta), "lam1": psi,
        "vbar": c * A / b, "rho_j": -b,
        "halflife_c": np.log(2.0) / kc, "halflife_j": np.log(2.0) / kj,
        "halflife_b": np.log(2.0) / b,
        # over-identification diagnostics (should be ~0 if the model holds)
        "CJ": psi * (psi * S + c * B), "CF": c * B + psi * S, "CB": psi * S,
        "S_implied": c ** 2 * B / (2.0 * kj),
    }
    out["consistency_CJS_minus_CBCF"] = out["CJ"] * S - out["CB"] * out["CF"]
    out["consistency_S_minus_implied"] = S - out["S_implied"]
    return out


def theoretical_moments_ext(theta, n_tpv=N_TPV, n_jv=N_JV, n_fwd=N_FWD, n_bwd=N_BWD):
    """Population moment vector m(theta') and analytic Jacobian D = dm/dtheta'.

    Returns (m, D) with m shape (M,) and D shape (M, 9), M = 5+n_tpv+n_jv+n_fwd+n_bwd.
    Every closed form is Theorem 5.3; every Jacobian row is its exact gradient.
    """
    kc, sc2, Vc, kj, A, B, psi, c, S = theta
    b = kj + psi * c

    # damping scalars
    fc, fj = dp.f(kc), dp.f(kj)
    fpc, fpj = dp.f_prime(kc), dp.f_prime(kj)

    # composites
    CF = c * B + psi * S
    CB = psi * S
    CJ = psi * CF                       # = psi(psi S + cB); also the Var(JV) coef

    M = 5 + n_tpv + n_jv + n_fwd + n_bwd
    m = np.empty(M)
    D = np.zeros((M, 9))

    # ---- [0] E[TPV] = Vbar_c + c A / b ,  b = kj + psi c ----------------------
    m[0] = Vc + c * A / b
    D[0, VC] = 1.0
    D[0, A_] = c / b
    D[0, C_] = A / b - c * A * psi / b ** 2          # db/dc = psi
    D[0, KJ] = -c * A / b ** 2                       # db/dkj = 1
    D[0, PSI] = -c ** 2 * A / b ** 2                 # db/dpsi = c

    # ---- [1] E[JV] = A -------------------------------------------------------
    m[1] = A
    D[1, A_] = 1.0

    # ---- [2] Var(TPV) = sc2 f(kc) + S f(kj) ---------------------------------
    m[2] = sc2 * fc + S * fj
    D[2, SC2] = fc
    D[2, KC] = sc2 * fpc
    D[2, S_] = fj
    D[2, KJ] = S * fpj

    # ---- [3] Var(JV) = B + C_J f(kj) ----------------------------------------
    m[3] = B + CJ * fj
    D[3, B_] = 1.0 + psi * c * fj                    # dCJ/dB = psi c
    D[3, S_] = psi ** 2 * fj                         # dCJ/dS = psi^2
    D[3, PSI] = (2.0 * psi * S + c * B) * fj         # dCJ/dpsi = 2psiS + cB
    D[3, C_] = psi * B * fj                          # dCJ/dc = psi B
    D[3, KJ] = CJ * fpj

    # ---- [4] Cov(TPV_t, JV_t) = (0.5 cB + psi S) f(kj) ----------------------
    Q = 0.5 * c * B + psi * S
    m[4] = Q * fj
    D[4, C_] = 0.5 * B * fj
    D[4, B_] = 0.5 * c * fj
    D[4, PSI] = S * fj
    D[4, S_] = psi * fj
    D[4, KJ] = Q * fpj

    row = 5
    # ---- TPV autocovariance: sc2 g(kc) E^c_k + S g(kj) E_k , k=1..n_tpv ------
    for k in range(1, n_tpv + 1):
        Dc, Dj = dp.lag_term(kc, k), dp.lag_term(kj, k)
        Dcp, Djp = dp.lag_term_prime(kc, k), dp.lag_term_prime(kj, k)
        m[row] = sc2 * Dc + S * Dj
        D[row, SC2] = Dc
        D[row, KC] = sc2 * Dcp
        D[row, S_] = Dj
        D[row, KJ] = S * Djp
        row += 1

    # ---- JV autocovariance: C_J g(kj) E_k , k=1..n_jv -----------------------
    for k in range(1, n_jv + 1):
        Dj, Djp = dp.lag_term(kj, k), dp.lag_term_prime(kj, k)
        m[row] = CJ * Dj
        D[row, PSI] = (2.0 * psi * S + c * B) * Dj
        D[row, S_] = psi ** 2 * Dj
        D[row, C_] = psi * B * Dj
        D[row, B_] = psi * c * Dj
        D[row, KJ] = CJ * Djp
        row += 1

    # ---- forward cross (jumps lead): C_F g(kj) E_k , k=1..n_fwd --------------
    for k in range(1, n_fwd + 1):
        Dj, Djp = dp.lag_term(kj, k), dp.lag_term_prime(kj, k)
        m[row] = CF * Dj
        D[row, C_] = B * Dj
        D[row, B_] = c * Dj
        D[row, PSI] = S * Dj
        D[row, S_] = psi * Dj
        D[row, KJ] = CF * Djp
        row += 1

    # ---- backward cross (variance leads): C_B g(kj) E_k , k=1..n_bwd ---------
    for k in range(1, n_bwd + 1):
        Dj, Djp = dp.lag_term(kj, k), dp.lag_term_prime(kj, k)
        m[row] = CB * Dj
        D[row, PSI] = S * Dj
        D[row, S_] = psi * Dj
        D[row, KJ] = CB * Djp
        row += 1

    return m, D
