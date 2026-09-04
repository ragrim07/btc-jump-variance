"""
Layer-1 verification gates for the analytic engine (run BEFORE any simulation).

Three independent checks, each catching a different class of error:

  GATE A  damping limits    -- f,g,ghat match their a->0 and a->inf expansions.
  GATE B  lambda1 = 0 collapse -- at psi=0 the 40-moment engine reproduces an
          INDEPENDENTLY hand-coded Step-A moment set to machine precision, and
          the self-excitation-only blocks (JV-acov, backward cross) vanish.
  GATE C  analytic Jacobian -- D matches central finite differences at random
          interior points to < 1e-6 relative error.

Gates A/B/C verify that the CODE matches the MATH. They cannot verify that the
math itself (Theorem 5.3) is correct -- that is the job of the simulation phase,
where population moments are checked against an exact Monte-Carlo generator.
"""

import numpy as np

import damping as dp
import moments as mo
from moments import N_TPV, N_JV, N_FWD, N_BWD

rng = np.random.default_rng(0)


# --------------------------------------------------------------------------
def gate_a_damping(tol_small=1e-4, tol_large=1e-3):
    ok = True
    a = 1e-4                                   # a -> 0 expansions
    ok &= abs(dp.f(a) - (1 - a / 3)) < tol_small
    ok &= abs(dp.ghat(a) - (1 - a / 2)) < tol_small
    ok &= abs(dp.g(a) - (1 - a)) < tol_small
    a = 60.0                                    # a -> inf asymptotics
    ok &= abs(dp.f(a) - 2 / a) < tol_large
    ok &= abs(dp.ghat(a) - 1 / a) < tol_large
    ok &= abs(dp.g(a) - 1 / a ** 2) < tol_large
    # derivatives vs finite differences
    for fn, dfn in ((dp.f, dp.f_prime), (dp.ghat, dp.ghat_prime), (dp.g, dp.g_prime)):
        for a in (0.05, 0.5, 3.0):
            fd = (fn(a + 1e-6) - fn(a - 1e-6)) / 2e-6
            ok &= abs(dfn(a) - fd) < 1e-6 * max(1, abs(dfn(a)))
    return ok


# --------------------------------------------------------------------------
def _stepA_reference(theta):
    """Independent Step-A moment formulas at psi=0 (NOT reusing moments.py).

    At psi=0: b=kappa_j, eta=0, C_J=C_B=0, C_F=cB. Returns the 20 Step-A-shared
    moments in the same slots as the extended vector's first 20 entries plus the
    forward-cross prediction, for cross-comparison.
    """
    kc, sc2, Vc, kj, A, B, psi, c, S = theta
    assert psi == 0.0
    shared = np.empty(5 + N_TPV)
    shared[0] = Vc + c * A / kj                        # E[TPV] with b=kj
    shared[1] = A                                      # E[JV]
    shared[2] = sc2 * dp.f(kc) + S * dp.f(kj)          # Var(TPV)
    shared[3] = B                                      # Var(JV)
    shared[4] = 0.5 * c * B * dp.f(kj)                 # Cov(TPV,JV)_0
    for k in range(1, N_TPV + 1):
        shared[4 + k] = (sc2 * dp.g(kc) * np.exp(-kc * (k - 1))
                         + S * dp.g(kj) * np.exp(-kj * (k - 1)))
    fwd = np.array([c * B * dp.g(kj) * np.exp(-kj * (k - 1))
                    for k in range(1, N_FWD + 1)])     # C_F = cB at psi=0
    return shared, fwd


def gate_b_collapse(n=200, tol=1e-11):
    ok = True
    for _ in range(n):
        theta = np.array([
            rng.uniform(0.01, 0.13),   # kc
            rng.uniform(0.5, 8.0),     # sc2
            rng.uniform(0.5, 8.0),     # Vc
            rng.uniform(0.1, 3.0),     # kj
            rng.uniform(0.1, 2.0),     # A
            rng.uniform(0.1, 5.0),     # B
            0.0,                       # psi = 0  <-- collapse point
            rng.uniform(0.1, 3.0),     # c
            rng.uniform(0.1, 5.0)])    # S
        m, _ = mo.theoretical_moments_ext(theta)
        shared_ref, fwd_ref = _stepA_reference(theta)
        ok &= np.allclose(m[:5 + N_TPV], shared_ref, atol=tol, rtol=0)
        # self-excitation-only blocks must vanish at psi=0
        jv_acov = m[5 + N_TPV: 5 + N_TPV + N_JV]
        bwd = m[5 + N_TPV + N_JV + N_FWD:]
        ok &= np.allclose(jv_acov, 0.0, atol=tol)
        ok &= np.allclose(bwd, 0.0, atol=tol)
        # forward cross must equal the Step-A prediction cB g(kj) E_k
        fwd = m[5 + N_TPV + N_JV: 5 + N_TPV + N_JV + N_FWD]
        ok &= np.allclose(fwd, fwd_ref, atol=tol, rtol=0)
    return ok


# --------------------------------------------------------------------------
def gate_c_jacobian(n=200, step=1e-6, atol=1e-8, rtol=1e-6):
    """Combined-tolerance Jacobian check.  A single |D| ranges over ~12 orders of
    magnitude (deep-lag terms ~ g(a)e^{-a(k-1)} are legitimately ~1e-11), far
    below the central-difference roundoff floor (~eps*|m|/h ~ 1e-10). Judging
    those by relative error is meaningless, so we pass on
    |D - Dfd| <= atol + rtol*|D| (numpy.allclose semantics) and separately
    report the worst relative error among entries large enough to resolve."""
    all_ok = True
    max_rel_significant = 0.0
    for _ in range(n):
        theta = np.array([
            rng.uniform(0.01, 0.13), rng.uniform(0.5, 8), rng.uniform(0.5, 8),
            rng.uniform(0.1, 3), rng.uniform(0.1, 2), rng.uniform(0.1, 5),
            rng.uniform(0.0, 2), rng.uniform(0.1, 3), rng.uniform(0.1, 5)])
        _, D = mo.theoretical_moments_ext(theta)
        Dfd = np.empty_like(D)
        for j in range(9):
            tp, tm = theta.copy(), theta.copy()
            h = step * max(1.0, abs(theta[j]))
            tp[j] += h; tm[j] -= h
            mp, _ = mo.theoretical_moments_ext(tp)
            mm, _ = mo.theoretical_moments_ext(tm)
            Dfd[:, j] = (mp - mm) / (2 * h)
        all_ok &= np.allclose(D, Dfd, atol=atol, rtol=rtol)
        sig = np.abs(D) > 1e-4                          # entries FD can resolve
        if sig.any():
            max_rel_significant = max(
                max_rel_significant,
                np.max(np.abs(D[sig] - Dfd[sig]) / np.abs(D[sig])))
    return all_ok, max_rel_significant


# --------------------------------------------------------------------------
def main():
    a = gate_a_damping()
    b = gate_b_collapse()
    c_ok, c_rel = gate_c_jacobian()
    print(f"GATE A  damping limits & derivatives      : {'PASS' if a else 'FAIL'}")
    print(f"GATE B  lambda1=0 collapse to Step-A       : {'PASS' if b else 'FAIL'}")
    print(f"GATE C  analytic Jacobian vs finite diff   : "
          f"{'PASS' if c_ok else 'FAIL'}  (max rel err {c_rel:.2e})")
    ok = a and b and c_ok
    print("-" * 52)
    print("ALL GATES PASS -- analytic engine verified." if ok
          else "GATES FAILED -- do not proceed.")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
