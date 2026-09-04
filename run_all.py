"""
Reproduce the entire project, end to end.

    python run_all.py            # everything except the raw-tick rebuild
    python run_all.py --full     # also rebuild btc_1min.csv from the raw monthly files
    python run_all.py --quick    # skip the slow simulation-verification study

Stages run in dependency order; each prints its own results. Seeds are fixed
throughout, so figures and numbers reproduce exactly.
"""

import argparse
import os
import subprocess
import sys
import time

# (script, description, raw-tick stage?, input files it needs, output it produces)
# The large intermediates (raw ticks, btc_1min.csv) are gitignored, so a fresh
# clone starts from the committed daily measures. Stages whose inputs are absent
# are skipped with an explanation rather than crashing.
STAGES = [
    ("clean_prices.py",        "raw 1s ticks -> gapless 1-minute series",        True,  [], "btc_1min.csv"),
    ("fetch_dvol.py",          "download Deribit DVOL history",                  False, [], "dvol_btc_daily.csv"),
    ("realized_measures.py",   "daily RV/TPV/MedRV/JV + train/holdout split",     False, ["btc_1min.csv"], "daily_measures.csv"),
    ("selfcheck.py",           "GATE: damping, lambda1=0 collapse, Jacobian",     False, [], None),
    ("verify_simulation.py",   "GATE: 40 moments vs exact simulator + recovery",  False, [], "sim_verify_results.json"),
    ("estimate_train.py",      "GMM estimation on the train window",             False, ["daily_measures_train.csv"], "estimate_train_results.json"),
    ("plot_fit.py",            "fit diagnostics with bootstrap bands",           False, ["estimate_train_results.json"], None),
    ("overid_tests.py",        "over-identification tests (honest versions)",     False, ["estimate_train_results.json"], None),
    ("event_study.py",         "out-of-sample event study on held-out crashes",   False, ["daily_measures_holdout.csv"], None),
    ("vrp.py",                 "ex-ante variance risk premium + diagnostics",     False, ["daily_measures.csv", "dvol_btc_daily.csv"], "vrp_series.csv"),
    ("vrp_predictability.py",  "do the states predict VRP changes?",             False, ["vrp_series.csv"], None),
    ("vrp_mechanical_check.py", "mechanical vs real: decomposition",              False, ["vrp_series.csv"], None),
    ("jump_state_compression.py", "jump state -> VRP compression (the question)", False, ["vrp_series.csv"], None),
    ("make_headline_figures.py",  "FIG1 finding, FIG2 verification, FIG3 limit",  False, ["vrp_series.csv", "sim_verify_results.json"], None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="rebuild btc_1min.csv from the raw monthly tick files")
    ap.add_argument("--quick", action="store_true",
                    help="skip the slow simulation-verification study")
    args = ap.parse_args()

    plan, skipped = [], []
    produced = set()
    for script, desc, raw_stage, needs, out in STAGES:
        if raw_stage and not args.full:
            skipped.append((script, "raw-tick stage; pass --full to rebuild from ticks"))
            continue
        if args.quick and script == "verify_simulation.py":
            skipped.append((script, "skipped by --quick"))
            continue
        missing = [f for f in needs if not os.path.exists(f) and f not in produced]
        if missing:
            skipped.append((script, f"input not present: {', '.join(missing)}"))
            continue
        plan.append((script, desc))
        if out:
            produced.add(out)

    if skipped:
        print("skipping:")
        for s, why in skipped:
            print(f"  - {s:28s} {why}")
        print()

    print(f"running {len(plan)} stages\n" + "=" * 70)
    t0 = time.time()
    for i, (script, desc) in enumerate(plan, 1):
        print(f"\n[{i}/{len(plan)}] {script}  --  {desc}")
        print("-" * 70)
        r = subprocess.run([sys.executable, "-u", script])
        if r.returncode != 0:
            print(f"\nFAILED at {script} (exit {r.returncode}). Stopping.")
            return r.returncode
    print("\n" + "=" * 70)
    print(f"all stages completed in {time.time()-t0:.0f}s")
    print("headline figures: FIG1_finding.png, FIG2_verification.png, FIG3_honest_limit.png")
    print("compile the summary:  pdflatex writeup.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
