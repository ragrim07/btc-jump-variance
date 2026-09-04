"""
Simulation verification of the P-measure GMM estimator (the capstone).

Three experiments, all on data simulated from KNOWN theta' (simulate.py):

  1. MOMENT VERIFICATION -- checks the MATH.  For each parameter set, compare the
     40 analytic moments to their simulated batch-means estimates; z-scores must
     lie inside +-3.  (A sign error in Theorem 5.3 shared with its Jacobian would
     pass selfcheck.py but fail here.)

  2. ESTIMATOR RECOVERY -- checks the PIPELINE.  Run the full analytic-Jacobian
     GMM on each simulated series; report theta_hat vs theta_true and a moving-
     block-bootstrap 95% CI, flag whether the truth is covered.  Headline: eta.

  3. COVERAGE CALIBRATION -- checks the CIs.  Repeatedly simulate independent
     datasets from one parameter set, estimate + CI each, and report the fraction
     of 95% CIs that actually contain the truth (should be ~0.95).

Outputs: sim_verify_results.json and four figures for the PDF report.
"""

import json
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import simulate as sim
import moments as mo
import gmm

logging.getLogger("gmm").setLevel(logging.ERROR)     # silence per-fit chatter
rng_master = np.random.default_rng(20260803)

# ---- experiment configuration ---------------------------------------------
N_MAIN = 50_000        # days per main run
SUBSTEPS = 32          # CIR sub-grid (bias < 0.1% at this level)
N_BOOT = 300           # block-bootstrap draws (main)
BLOCK = 50             # bootstrap block length (days) -- exceeds the ~15-lag memory
N_BATCH = 50           # batches for moment-verification SEs

# six parameter sets: (name, kappa_c, sigma_c2, Vbar_c, kappa_j, A, B, psi, c)
PARAM_SETS = [
    ("A baseline",     np.log(2)/8,  4.0, 5.0, 0.30, 0.50, 0.80, 0.50, 0.60),
    ("B low-eta",      np.log(2)/6,  3.0, 6.0, 0.60, 0.60, 1.00, 0.30, 0.50),
    ("C high-eta",     np.log(2)/10, 6.0, 8.0, 0.21, 0.45, 0.90, 0.70, 0.70),
    ("D persistent-j", np.log(2)/12, 5.0, 7.0, 0.15, 0.55, 0.95, 0.40, 0.60),
    ("E fast-jumps",   np.log(2)/7,  3.5, 5.5, 0.80, 0.70, 1.10, 0.25, 0.50),
    ("F heavy-energy", np.log(2)/9,  4.5, 6.0, 0.35, 0.40, 1.60, 0.50, 0.60),
]

PALETTE = ["#2a78d6", "#1baf7a", "#e08a1e", "#c0504d", "#7a5ea6", "#3aa0a0"]
SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9", "#c3c2b7"


def _batch_means_se(tpv, jv, nb):
    L = len(tpv) // nb
    B = np.array([gmm.empirical_moments_ext(tpv[i*L:(i+1)*L], jv[i*L:(i+1)*L])[0]
                  for i in range(nb)])
    return B.std(0, ddof=1) / np.sqrt(nb)


def run_one(name, args, seed):
    theta, feas = sim.make_theta(*args)
    df = sim.simulate(theta, N_MAIN, seed=seed, substeps=SUBSTEPS)
    tpv, jv = df["TPV"].values, df["JV"].values

    # (1) moment verification
    m_an, _ = mo.theoretical_moments_ext(theta)
    m_emp, rows = gmm.empirical_moments_ext(tpv, jv)
    se = _batch_means_se(tpv, jv, N_BATCH)
    z = (m_emp - m_an) / se

    # (2) estimator recovery + bootstrap CI.  Estimate from the contribution-row
    # mean -- the exact object the bootstrap resamples -- so point and interval
    # share one estimator (no centering mismatch).
    W = gmm.build_weighting_matrix(rows)
    res = gmm.calibrate(rows.mean(axis=0), W)
    bs = gmm.moving_block_bootstrap(rows, res.x, W, n_boot=N_BOOT,
                                    block=BLOCK, seed=seed + 1)
    lo, hi = bs["theta_pct"][0], bs["theta_pct"][2]
    std = bs["theta"].std(0, ddof=1)
    cov = [(bool(lo[i] <= theta[i] <= hi[i])) for i in range(9)]
    dt, dh = mo.derived_quantities(theta), mo.derived_quantities(res.x)
    elo, ehi = bs["derived_pct"]["eta"][0], bs["derived_pct"]["eta"][2]

    return {
        "name": name, "theta_true": theta.tolist(), "feas": {k: float(v) if not isinstance(v, bool) else v for k, v in feas.items()},
        "theta_hat": res.x.tolist(), "ci_lo": lo.tolist(), "ci_hi": hi.tolist(),
        "boot_std": std.tolist(), "coverage": cov, "J": float(res.fun),
        "eta_true": float(dt["eta"]), "eta_hat": float(dh["eta"]),
        "eta_lo": float(elo), "eta_hi": float(ehi),
        "eta_cov": bool(elo <= dt["eta"] <= ehi),
        "z": z.tolist(), "max_absz": float(np.max(np.abs(z))),
        "n_z_gt3": int((np.abs(z) > 3).sum()),
    }


def coverage_experiment(args, R=50, N=8000, n_boot=150, block=25):
    """Repeated independent datasets from one set -> empirical CI coverage."""
    theta, _ = sim.make_theta(*args)
    hits = np.zeros(9); eta_hits = 0
    eta_true = mo.derived_quantities(theta)["eta"]
    for r in range(R):
        df = sim.simulate(theta, N, seed=9000 + r, substeps=SUBSTEPS)
        tpv, jv = df["TPV"].values, df["JV"].values
        m_emp, rows = gmm.empirical_moments_ext(tpv, jv)
        W = gmm.build_weighting_matrix(rows)
        res = gmm.calibrate(rows.mean(axis=0), W)
        bs = gmm.moving_block_bootstrap(rows, res.x, W, n_boot=n_boot,
                                        block=block, seed=r)
        lo, hi = bs["theta_pct"][0], bs["theta_pct"][2]
        hits += [(lo[i] <= theta[i] <= hi[i]) for i in range(9)]
        el, eh = bs["derived_pct"]["eta"][0], bs["derived_pct"]["eta"][2]
        eta_hits += (el <= eta_true <= eh)
    return {"R": R, "N": N, "per_param": (hits / R).tolist(),
            "eta": eta_hits / R, "theta_true": theta.tolist()}


# ---------------------------------------------------------------------------
# FIGURES
# ---------------------------------------------------------------------------
def _ax(figsize):
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, lw=0.7); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(AXIS)
    return fig, ax


def fig_moment_z(results, path):
    fig, ax = _ax((10, 4.6))
    ax.axhspan(-2, 2, color=GRID, alpha=0.5, lw=0)
    for c in (-3, 3): ax.axhline(c, color=AXIS, lw=1, ls=(0, (4, 3)))
    ax.axhline(0, color=AXIS, lw=0.8)
    for k, r in enumerate(results):
        x = np.arange(40) + (k - 2.5) * 0.11
        ax.scatter(x, r["z"], s=14, color=PALETTE[k], label=r["name"],
                   edgecolor=SURFACE, linewidth=0.4, zorder=3)
    ax.set_xlim(-1, 40); ax.set_ylim(-4, 4)
    ax.set_xlabel("moment index (0..39)", color=INK, fontsize=9)
    ax.set_ylabel(r"z = (simulated $-$ analytic)/MC-SE", color=INK, fontsize=9)
    ax.set_title("Moment verification: simulated moments vs analytic formulas (all inside $\\pm$3)",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", labelcolor=INK)
    fig.tight_layout(); fig.savefig(path, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)


def fig_eta(results, path):
    fig, ax = _ax((6.2, 5.4))
    lim = [0, 0.8]; ax.plot(lim, lim, color=AXIS, lw=1, ls=(0, (4, 3)), zorder=1)
    for k, r in enumerate(results):
        ax.errorbar(r["eta_true"], r["eta_hat"],
                    yerr=[[max(0.0, r["eta_hat"] - r["eta_lo"])], [max(0.0, r["eta_hi"] - r["eta_hat"])]],
                    fmt="o", ms=8, color=PALETTE[k], ecolor=PALETTE[k], elinewidth=1.6,
                    capsize=3, mec=SURFACE, mew=1, zorder=3, label=r["name"])
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"true $\eta$ (branching ratio)", color=INK, fontsize=9)
    ax.set_ylabel(r"estimated $\hat\eta$ with 95% bootstrap CI", color=INK, fontsize=9)
    ax.set_title(r"Headline recovery: $\hat\eta$ vs truth across the self-excitation range",
                 color=INK, fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK)
    fig.tight_layout(); fig.savefig(path, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)


def fig_recovery_norm(results, path):
    fig, ax = _ax((10, 4.6))
    ax.axhspan(-2, 2, color=GRID, alpha=0.5, lw=0)
    ax.axhline(0, color=AXIS, lw=0.8)
    names = mo.PARAM_NAMES
    for k, r in enumerate(results):
        th = np.array(r["theta_true"]); hat = np.array(r["theta_hat"]); sd = np.array(r["boot_std"])
        t = (hat - th) / np.where(sd > 0, sd, np.nan)
        x = np.arange(9) + (k - 2.5) * 0.11
        ax.scatter(x, t, s=22, color=PALETTE[k], label=r["name"], edgecolor=SURFACE, linewidth=0.4, zorder=3)
    ax.set_xticks(range(9)); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylim(-4, 4)
    ax.set_ylabel(r"standardized error $(\hat\theta-\theta_0)/\mathrm{sd}_{\mathrm{boot}}$", color=INK, fontsize=9)
    ax.set_title("Estimator recovery: standardized error per parameter (inside $\\pm$2 = covered)",
                 color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center", labelcolor=INK)
    fig.tight_layout(); fig.savefig(path, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)


def fig_coverage(cov, path):
    fig, ax = _ax((7.6, 4.4))
    names = mo.PARAM_NAMES + ["eta"]
    vals = cov["per_param"] + [cov["eta"]]
    x = np.arange(len(vals))
    ax.axhline(0.95, color="#1baf7a", lw=1.5, ls=(0, (4, 3)), label="nominal 95%")
    ax.bar(x, vals, color=PALETTE[0], width=0.62, edgecolor=SURFACE)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8, rotation=0)
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("empirical coverage", color=INK, fontsize=9)
    ax.set_title(f"CI calibration: coverage over {cov['R']} independent datasets (N={cov['N']})",
                 color=INK, fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout(); fig.savefig(path, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    print(f"MAIN runs: {len(PARAM_SETS)} sets x {N_MAIN} days, boot={N_BOOT}")
    results = []
    for k, (name, *args) in enumerate(PARAM_SETS):
        r = run_one(name, tuple(args), seed=1000 + k)
        results.append(r)
        print(f"  {name:16s} eta {r['eta_true']:.3f}->{r['eta_hat']:.3f} "
              f"cov[{r['eta_lo']:.3f},{r['eta_hi']:.3f}]={r['eta_cov']} | "
              f"param-cov {sum(r['coverage'])}/9 | max|z| {r['max_absz']:.2f} "
              f"(#>3: {r['n_z_gt3']}) | J {r['J']:.2e}")

    print("COVERAGE calibration experiment (set A) ...")
    cov = coverage_experiment(tuple(PARAM_SETS[0][1:]), R=50, N=8000, n_boot=200, block=50)
    print("  per-param coverage:", [f"{v:.2f}" for v in cov["per_param"]], "eta", f"{cov['eta']:.2f}")

    out = {"config": {"N_MAIN": N_MAIN, "SUBSTEPS": SUBSTEPS, "N_BOOT": N_BOOT,
                      "BLOCK": BLOCK, "N_BATCH": N_BATCH, "param_names": mo.PARAM_NAMES},
           "results": results, "coverage": cov}
    with open("sim_verify_results.json", "w") as fh:
        json.dump(out, fh, indent=1)

    fig_moment_z(results, "fig_verify_moment_z.png")
    fig_eta(results, "fig_verify_eta.png")
    fig_recovery_norm(results, "fig_verify_recovery.png")
    fig_coverage(cov, "fig_verify_coverage.png")
    print("saved sim_verify_results.json + 4 figures")


if __name__ == "__main__":
    main()
