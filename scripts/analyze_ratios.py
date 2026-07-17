#!/usr/bin/env python3
"""
Ratio analysis: cross-task significance (RoR) + the difficulty ladder.

Reads one matched-random summary_stats.json per task (control or merged shards)
and computes, on the non-weak bar (primary) and solved bar (companion):

  * per-task ratio + MOVER 95% CI (the developmental-prior multiplier)
  * within-task Fisher exact (MN-competent vs random-competent), table-stakes
  * pairwise ratio-of-ratios RoR = RR_harder / RR_easier with a delta-method CI
    and the saturated-logistic interaction term -- the cross-task
    interaction test (RoR > 1, CI excludes 1)
  * if >= 3 tasks: Spearman rho across the difficulty order
  * if >= 4 tasks: OLS slope of ln(RR) on difficulty rank, with SE

CartPole is injected automatically from the pinned B2 counts; pass the others as
  name=path  e.g.  acrobot=experiments/acrobot/matched_random/full_k50/summary_stats.json

Read (CartPole vs Acrobot): RoR CI lower bound > 1 -> the ratio significantly
increases; CI straddling 1 -> persists; < 1 -> no cross-task increase.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath("code"))

from MorphoNAS_DevPriors.ratio_stats import (  # noqa: E402
    fisher_within_task,
    logistic_interaction,
    ratio_ci_mover,
    ror_delta,
)
from MorphoNAS_DevPriors.task_registry import CARTPOLE_REF, DIFFICULTY_ORDER  # noqa: E402


def counts_from_summary(path: str) -> dict:
    with open(path) as f:
        s = json.load(f)
    mn = s["morphonas"]
    n_rand = s["n_random_valid"]
    return {
        "nonweak": {"x_mn": mn["nonweak_count"], "n_mn": mn["n_total"],
                    "x_rand": s["random_nonweak_count"], "n_rand": n_rand},
        "solved": {"x_mn": mn["solved_count"], "n_mn": mn["n_total"],
                   "x_rand": s["random_solved_count"], "n_rand": n_rand},
    }


def cartpole_counts() -> dict:
    return {bar: {"x_mn": c["mn_count"], "n_mn": c["mn_n"],
                  "x_rand": c["rand_count"], "n_rand": c["rand_n"]}
            for bar, c in CARTPOLE_REF.items()}


def main() -> None:
    p = argparse.ArgumentParser(description="ratio / RoR / ladder analysis")
    p.add_argument("inputs", nargs="*", help="task=summary_stats.json")
    p.add_argument("--no-cartpole", action="store_true", help="omit the pinned CartPole anchor")
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    tasks = {} if args.no_cartpole else {"cartpole": cartpole_counts()}
    for tok in args.inputs:
        if "=" not in tok:
            raise SystemExit(f"expected name=path, got {tok!r}")
        name, path = tok.split("=", 1)
        tasks[name.lower()] = counts_from_summary(path)

    ordered = [t for t in DIFFICULTY_ORDER if t in tasks]
    ordered += [t for t in tasks if t not in ordered]  # any unknown tasks last
    print(f"Tasks (difficulty order): {ordered}\n")

    report = {"order": ordered, "per_task": {}, "pairwise_ror": {}, "trend": {}}

    # ── per-task ratios + within-task Fisher ──
    for t in ordered:
        report["per_task"][t] = {}
        for bar in ("nonweak", "solved"):
            c = tasks[t][bar]
            r = ratio_ci_mover(c["x_mn"], c["n_mn"], c["x_rand"], c["n_rand"])
            fish = fisher_within_task(c["x_mn"], c["n_mn"], c["x_rand"], c["n_rand"])
            report["per_task"][t][bar] = {"ratio": r, "fisher": fish, "counts": c}
        nw = report["per_task"][t]["nonweak"]["ratio"]
        hi = "inf" if nw["hi"] is None else f"{nw['hi']:.1f}"
        pt = "inf" if not np.isfinite(nw["point"]) else f"{nw['point']:.2f}"
        print(f"  {t:13s} non-weak: MN {nw['mn_count']}/{nw['mn_n']} "
              f"vs rand {nw['rand_count']}/{nw['rand_n']} = {pt}x [{nw['lo']:.2f}, {hi}]  "
              f"Fisher p={report['per_task'][t]['nonweak']['fisher'].get('p_one_sided_greater', float('nan')):.2e}")

    # ── pairwise RoR (each harder rung vs CartPole, and vs its predecessor) ──
    print("\nRatio-of-ratios (cross-task interaction):")
    for bar in ("nonweak", "solved"):
        report["pairwise_ror"][bar] = {}
        for i in range(1, len(ordered)):
            harder, easier = ordered[i], ordered[0]  # vs the CartPole/easiest anchor
            ror = ror_delta(tasks[harder][bar], tasks[easier][bar])
            inter = logistic_interaction(tasks[harder][bar], tasks[easier][bar])
            key = f"{harder}_vs_{easier}"
            report["pairwise_ror"][bar][key] = {"ror": ror, "interaction": inter}
            if bar == "nonweak":
                verdict = ("GROWS" if ror["grows"] else
                           "persists" if ror["ror_ci"][1] >= 1 else "killed")
                cc = " (cc=0.5)" if ror["continuity_correction"] else ""
                print(f"  [{bar}] {key}: RoR={ror['ror_point']:.2f} "
                      f"[{ror['ror_ci'][0]:.2f}, {ror['ror_ci'][1]:.2f}]{cc} "
                      f"p={ror['p_two_sided']:.2e} -> {verdict}  "
                      f"| interaction OR={inter['interaction_or']:.2f} p={inter['p_two_sided']:.2e}")

    # ── trend across the ladder ──
    if len(ordered) >= 3:
        from scipy.stats import spearmanr

        ranks = list(range(len(ordered)))
        for bar in ("nonweak", "solved"):
            rrs = []
            for t in ordered:
                c = tasks[t][bar]
                p_mn = c["x_mn"] / c["n_mn"]
                p_rd = (c["x_rand"] + 0.5) / (c["n_rand"] + 1) if c["x_rand"] == 0 else c["x_rand"] / c["n_rand"]
                rrs.append(p_mn / p_rd)
            rho, pval = spearmanr(ranks, rrs)
            entry = {"ratios": rrs, "spearman_rho": float(rho), "spearman_p": float(pval)}
            if len(ordered) >= 4:
                ln = np.log(rrs)
                coef, cov = np.polyfit(ranks, ln, 1, cov=True)
                entry["log_rr_slope"] = float(coef[0])
                entry["log_rr_slope_se"] = float(np.sqrt(cov[0, 0]))
            report["trend"][bar] = entry
            if bar == "nonweak":
                print(f"\nLadder trend [{bar}]: ratios={[f'{r:.1f}' for r in rrs]} "
                      f"Spearman rho={rho:.3f} (p={pval:.3f})"
                      + (f", log-RR slope={entry.get('log_rr_slope', float('nan')):.2f}"
                         f"+-{entry.get('log_rr_slope_se', float('nan')):.2f}"
                         if "log_rr_slope" in entry else ""))
    else:
        print("\n(Trend needs >= 3 tasks; add ladder rungs as they complete.)")

    out = args.out or "experiments/ratio_analysis.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
