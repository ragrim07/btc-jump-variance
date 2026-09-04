"""
Honest over-identification tests for the affine-intensity model.

Two identities are implied by the theory:

  (I)   C_J * S = C_B * C_F
  (II)  S = c^2 B / (2 kappa_j)

An important distinction.
  Inside the structural parametrisation theta' = [.., A, B, psi, c, S], the
  coefficients are DEFINED as
        C_F = cB + psi S,   C_B = psi S,   C_J = psi (psi S + cB) = psi C_F.
  Then C_J S = (psi C_F) S and C_B C_F = (psi S) C_F are the SAME EXPRESSION.
  Identity (I) is therefore an ALGEBRAIC TAUTOLOGY of the parametrisation: it
  holds for arbitrary parameter values, fitted or random, and evaluating it at
  theta_hat tests NOTHING.

  To make (I) a genuine test we must estimate C_J, C_B, C_F and S as FREE
  coefficients -- letting each moment block choose its own level, with no
  cross-block restriction -- and only then ask whether the free estimates
  satisfy the identity. That is what this module does.

  Identity (II) IS a genuine restriction inside the structural fit, because S is
  estimated as a free parameter while c, B, kappa_j are pinned by other moments.

Free (unrestricted) estimation, given decay rates (kappa_c, kappa_j):
  JV-acov_k   = C_J * g(kj) E_k                  -> C_J   by least squares
  fwdCross_k  = C_F * g(kj) E_k                  -> C_F
  bwdCross_k  = C_B * g(kj) E_k                  -> C_B
  TPVacov_k   = sc2 * g(kc) E^c_k + S * g(kj) E_k -> (sc2, S)  jointly
Each is linear in its coefficient(s), so the free fit is an OLS projection of the
empirical moment block onto the model's SHAPE -- the shape is imposed, the LEVEL
is free. Identity (I) then has real content.
"""

import json

import numpy as np
import pandas as pd

import damping as dp
import gmm
import moments as mo
from moments import N_TPV, N_JV, N_FWD, N_BWD

N_BOOT, BLOCK = 800, 50


def free_coefficients(m, kc, kj):
    """Unrestricted level estimates (C_J, C_F, C_B, S, sigma_c^2) from a moment vector."""
    kJ = np.arange(1, N_JV + 1)
    kF = np.arange(1, N_FWD + 1)
    kB = np.arange(1, N_BWD + 1)
    kT = np.arange(1, N_TPV + 1)

    shape = lambda a, ks: dp.g(a) * np.exp(-a * (ks - 1.0))

    def ls(y, x):                       # least squares through the origin
        return float(x @ y / (x @ x))

    CJ = ls(m[5 + N_TPV: 5 + N_TPV + N_JV], shape(kj, kJ))
    i0 = 5 + N_TPV + N_JV
    CF = ls(m[i0: i0 + N_FWD], shape(kj, kF))
    CB = ls(m[i0 + N_FWD: i0 + N_FWD + N_BWD], shape(kj, kB))

    # TPV block: two shapes, joint least squares -> (sigma_c^2, S)
    X = np.column_stack([shape(kc, kT), shape(kj, kT)])
    y = m[5: 5 + N_TPV]
    sc2, S = np.linalg.lstsq(X, y, rcond=None)[0]
    return dict(CJ=CJ, CF=CF, CB=CB, S=float(S), sc2=float(sc2))


def main():
    J = json.load(open("estimate_train_results.json"))
    theta = np.array(J["theta"])
    kc, kj = theta[0], theta[3]
    B_hat, c_hat = theta[5], theta[7]

    meas = pd.read_csv("daily_measures_train.csv", parse_dates=["date"]).set_index("date")
    tpv, jv = meas["TPV"].to_numpy(), meas["JV"].to_numpy()
    m_emp, rows = gmm.empirical_moments_ext(tpv, jv)

    print("=" * 74)
    print("TEST (I)  C_J * S  =  C_B * C_F")
    print("=" * 74)

    # -- (a) evaluated inside the structural parametrisation: a tautology -------
    d = mo.derived_quantities(theta)
    print("(a) at the fitted structural theta_hat:")
    print(f"    C_J={d['CJ']:.6f}  C_B={d['CB']:.6f}  C_F={d['CF']:.6f}  S={theta[8]:.6f}")
    print(f"    C_J*S - C_B*C_F = {d['CJ']*theta[8] - d['CB']*d['CF']:.3e}"
          "   <-- exactly 0 BY CONSTRUCTION, tests nothing")

    # -- (b) free coefficients: a genuine test ---------------------------------
    fr = free_coefficients(m_emp, kc, kj)
    stat = fr["CJ"] * fr["S"] - fr["CB"] * fr["CF"]
    scale = abs(fr["CB"] * fr["CF"]) + abs(fr["CJ"] * fr["S"])
    print("\n(b) with C_J, C_B, C_F, S estimated FREELY (shape imposed, level free):")
    print(f"    C_J={fr['CJ']:.5f}  C_B={fr['CB']:.5f}  C_F={fr['CF']:.5f}  S={fr['S']:.5f}")
    print(f"    statistic C_J*S - C_B*C_F = {stat:+.5f}   (relative {2*stat/scale:+.3f})")

    rng = np.random.default_rng(7)
    Tr = rows.shape[0]
    nb = int(np.ceil(Tr / BLOCK))
    draws, rel = [], []
    for _ in range(N_BOOT):
        st = rng.integers(0, Tr - BLOCK + 1, nb)
        idx = np.concatenate([np.arange(s, s + BLOCK) for s in st])[:Tr]
        mb = rows[idx].mean(axis=0)
        f = free_coefficients(mb, kc, kj)
        s = f["CJ"] * f["S"] - f["CB"] * f["CF"]
        sc = abs(f["CB"] * f["CF"]) + abs(f["CJ"] * f["S"])
        draws.append(s)
        rel.append(2 * s / sc if sc > 0 else np.nan)
    draws = np.array(draws)
    lo, med, hi = np.percentile(draws, [2.5, 50, 97.5])
    rlo, rmed, rhi = np.nanpercentile(rel, [2.5, 50, 97.5])
    verdict = "NOT REJECTED (0 inside CI)" if lo <= 0 <= hi else "REJECTED"
    print(f"    bootstrap 95% CI = [{lo:+.5f}, {hi:+.5f}]   median {med:+.5f}   -> {verdict}")
    print(f"    scale-free version: {rmed:+.3f}  CI [{rlo:+.3f}, {rhi:+.3f}]")

    print()
    print("=" * 74)
    print("TEST (II)  S  =  c^2 B / (2 kappa_j)")
    print("=" * 74)
    S_impl = c_hat ** 2 * B_hat / (2.0 * kj)
    print(f"    free S (structural fit)      = {theta[8]:.5f}")
    print(f"    implied c^2 B/(2 kappa_j)    = {S_impl:.5f}")
    print(f"    residual                     = {theta[8]-S_impl:+.5f}")
    pct = np.array(J["consistency_pct"]["S_minus_implied"])
    print(f"    bootstrap 95% CI = [{pct[0]:+.4f}, {pct[2]:+.4f}]   median {pct[1]:+.4f}")
    print(f"    -> {'NOT REJECTED (0 inside CI)' if pct[0] <= 0 <= pct[2] else 'REJECTED'}")
    width = pct[2] - pct[0]
    print(f"\n    POWER WARNING: the CI is {width:.1f} wide against a point estimate of")
    print(f"    {theta[8]:.2f}. The interval spans {pct[2]/max(pct[0],1e-9) if pct[0]>0 else float('inf'):.0f}x"
          " in magnitude; a test this wide")
    print("    cannot reject much. 'Not rejected' here means LOW POWER, not strong support.")

    out = {
        "test1_tautology_at_theta": float(d["CJ"] * theta[8] - d["CB"] * d["CF"]),
        "test1_free": {"CJ": fr["CJ"], "CB": fr["CB"], "CF": fr["CF"], "S": fr["S"],
                       "stat": float(stat), "ci": [float(lo), float(hi)],
                       "rel_median": float(rmed), "rel_ci": [float(rlo), float(rhi)],
                       "rejected": bool(not (lo <= 0 <= hi))},
        "test2": {"S_free": float(theta[8]), "S_implied": float(S_impl),
                  "resid": float(theta[8] - S_impl),
                  "ci": [float(pct[0]), float(pct[2])],
                  "rejected": bool(not (pct[0] <= 0 <= pct[2]))},
    }
    json.dump(out, open("overid_test_results.json", "w"), indent=1)
    print("\nsaved overid_test_results.json")


if __name__ == "__main__":
    main()
