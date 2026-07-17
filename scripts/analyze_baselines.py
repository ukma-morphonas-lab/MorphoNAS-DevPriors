#!/usr/bin/env python3
"""
Baseline analysis: the full MN-vs-CPPN/HyperNEAT field + the de-confounder.

Reads every CPPN summary_stats.json (experiments/baselines/cppn_hyperneat/*/...),
prints the per-task R(MN/CPPN) with MOVER CI and Fisher p, tags each cell
structure-gated vs structure-free, and runs the within-task de-confounder as a
ratio-of-ratios: RoR = (MN/CPPN)_gated / (MN/CPPN)_free. If the MN-over-CPPN
advantage is reward-gated (the same causal signature as MN-over-random), RoR > 1
with a CI excluding 1 on the densifiable pairs, and ~1 on the non-densifiable one.
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.append(os.path.abspath("code"))
from MorphoNAS_DevPriors.ratio_stats import fisher_within_task, ratio_ci_mover, ror_delta  # noqa: E402

# structure of the reward at the non-weak bar (from the reach-map de-confounder)
GATED = {"cartpole", "acrobot", "mountaincar", "cartpole_masked", "pendulum_sparse",
         "acrobot_shaped", "vision_digits"}
FREE = {"pendulum", "mountaincar_shaped"}
OTHER = {"lunarlander": "trivial gentle-descent bar (0 landings)",
         "frozenlake_shaped": "navigation progress (structure-free)"}

# densifiable de-confounder pairs: (gated/sparse cell, free/dense cell)
PAIRS = [("pendulum_sparse", "pendulum"), ("mountaincar", "mountaincar_shaped"),
         ("acrobot", "acrobot_shaped")]  # acrobot_shaped is non-densifiable -> expect RoR~1


def load(path):
    s = json.load(open(path))
    mn = s["morphonas"]
    return {
        "task": s["task"], "x_mn": mn["nonweak_count"], "n_mn": mn["n_total"],
        "x_cppn": s["random_nonweak_count"], "n_cppn": s["n_random_valid"],
    }


def main():
    # the real runs: pilot_k5 first, then k50 (which overwrites where a bump exists),
    # then the vision summary. cells[task]=d means the later (higher-k) run wins.
    paths = sorted(glob.glob("experiments/baselines/cppn_hyperneat/*/pilot_k5/summary_stats.json"))
    paths += sorted(glob.glob("experiments/baselines/cppn_hyperneat/*/k50/summary_stats.json"))
    paths += sorted(glob.glob("experiments/baselines/cppn_hyperneat/vision_digits/summary_stats.json"))
    cells = {}
    for p in paths:
        try:
            d = load(p)
        except (KeyError, json.JSONDecodeError):
            continue
        cells[d["task"]] = d

    def tag(t):
        if t in GATED:
            return "gated"
        if t in FREE:
            return "free"
        return OTHER.get(t, "?")

    print("=" * 92)
    print("%-20s %-26s %8s %8s  %-22s %10s" % ("task", "reward-structure", "MN%", "CPPN%", "R(MN/CPPN) [95% CI]", "Fisher p"))
    print("-" * 92)
    rows = sorted(cells.values(), key=lambda d: -(d["x_mn"] / max(d["n_mn"], 1)) / max(d["x_cppn"] / max(d["n_cppn"], 1), 1e-9))
    for d in rows:
        ci = ratio_ci_mover(d["x_mn"], d["n_mn"], d["x_cppn"], d["n_cppn"])
        fp = fisher_within_task(d["x_mn"], d["n_mn"], d["x_cppn"], d["n_cppn"])
        hi = f"{ci['hi']:.1f}" if ci["hi"] else "inf"
        mnp = 100 * d["x_mn"] / d["n_mn"]
        cpp = 100 * d["x_cppn"] / max(d["n_cppn"], 1)
        print("%-20s %-26s %7.2f%% %7.2f%%  %6.2fx [%5.2f,%6s] %10.1e"
              % (d["task"], tag(d["task"]), mnp, cpp, ci["point"], ci["lo"], hi, fp["p_one_sided_greater"]))

    print("\n" + "=" * 92)
    print("De-confounder (ratio-of-ratios): RoR = (MN/CPPN)_gated / (MN/CPPN)_free")
    print("  RoR > 1 (CI excl. 1)  => the MN-over-CPPN advantage is reward-gated (vanishes when free)")
    print("  RoR ~ 1               => non-densifiable cell (advantage intact, as expected)")
    print("-" * 92)
    for gated_t, free_t in PAIRS:
        if gated_t not in cells or free_t not in cells:
            print("  (missing %s or %s)" % (gated_t, free_t))
            continue
        a, b = cells[gated_t], cells[free_t]
        ta = {"x_mn": a["x_mn"], "n_mn": a["n_mn"], "x_rand": a["x_cppn"], "n_rand": a["n_cppn"]}
        tb = {"x_mn": b["x_mn"], "n_mn": b["n_mn"], "x_rand": b["x_cppn"], "n_rand": b["n_cppn"]}
        rr = ror_delta(ta, tb)
        flag = "GATED (excl. 1)" if rr["ci_excludes_1"] and rr["grows"] else ("~parity" if not rr["ci_excludes_1"] else "inverted")
        print("  %-22s vs %-20s  RoR=%6.2f [%.2f, %.2f]  p=%.1e  %s"
              % (gated_t, free_t, rr["ror_point"], rr["ror_ci"][0], rr["ror_ci"][1], rr["p_two_sided"], flag))
        print("        (MN/CPPN: %s=%.2fx  %s=%.2fx)" % (gated_t, rr["rr_a"], free_t, rr["rr_b"]))


if __name__ == "__main__":
    main()
