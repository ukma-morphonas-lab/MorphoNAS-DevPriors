#!/usr/bin/env python3
"""
Baseline figure: the three prior gradients across the task field.

The reach-map measured one gradient -- MorphoNAS vs structureless random wiring.
This overlays a second: the CPPN / HyperNEAT prior (a structured but
NON-developmental indirect encoding), (N,E)-matched. The story in one figure:

  * MorphoNAS sits at the top (the developmental prior),
  * CPPN / HyperNEAT in the middle (a real geometric prior -- above the null),
  * structureless random at the floor,

with the gap between MN and CPPN (panel B) being the recurrence-circularity
closer: where it is large, morphogenesis's wiring is specifically better than the
other indirect encoding's, not just better than noise.

Panel A: competence prevalence (log) per task, three series, tasks ordered by the
reach-map MN/random ratio. Panel B: R(MN/CPPN) and R(CPPN/random) per task, with
MOVER CIs and a parity line.

Reads the CPPN summary_stats.json files (default: every
experiments/baselines/cppn_hyperneat/*/pilot_k5/) for MN + CPPN counts; the
structureless-random rates are the reach-map's published matched-random numbers
(REACHMAP_RANDOM, sourced from reach-map-findings.md). Output:
experiments/baselines/baseline_gradient.png
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath("code"))
from MorphoNAS_DevPriors.ratio_stats import ratio_ci_mover  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reach-map matched-random (structureless) competence: (competent, total).
# Source: reach-map-findings.md sections 1 + 8 (the published matched-random ladder).
REACHMAP_RANDOM = {
    "mountaincar": (3, 48545),
    "acrobot": (40, 24801),
    "cartpole_masked": (235, 25117),
    "cartpole": (65, 11586),
    "lunarlander": (421, 5200),
}

# Pretty labels + the reach-map MN/random ratio used to order the x-axis.
LABEL = {
    "mountaincar": "MountainCar",
    "acrobot": "Acrobot",
    "cartpole_masked": "CartPole\n(masked / memory)",
    "cartpole": "CartPole",
    "lunarlander": "LunarLander",
}


def load_cppn(path: str) -> dict:
    with open(path) as f:
        s = json.load(f)
    mn = s["morphonas"]
    return {
        "task": s["task"],
        "mn_comp": mn["nonweak_count"], "mn_n": mn["n_total"],
        "cppn_comp": s["random_nonweak_count"], "cppn_n": s["n_random_valid"],
    }


def main():
    args = sys.argv[1:]
    paths = args or sorted(glob.glob("experiments/baselines/cppn_hyperneat/*/pilot_k5/summary_stats.json"))
    rows = []
    for p in paths:
        if "=" in p:
            p = p.split("=", 1)[1]
        if not os.path.exists(p):
            continue
        d = load_cppn(p)
        t = d["task"]
        if t not in REACHMAP_RANDOM:  # headline figure = the matched-random ladder + memory only
            continue
        d["rand_comp"], d["rand_n"] = REACHMAP_RANDOM.get(t, (None, None))
        d["mn_rate"] = 100.0 * d["mn_comp"] / d["mn_n"]
        d["cppn_rate"] = 100.0 * d["cppn_comp"] / d["cppn_n"]
        d["rand_rate"] = (100.0 * d["rand_comp"] / d["rand_n"]) if d["rand_comp"] is not None else None
        d["r_mn_cppn"] = ratio_ci_mover(d["mn_comp"], d["mn_n"], d["cppn_comp"], d["cppn_n"])
        d["r_cppn_rand"] = (ratio_ci_mover(d["cppn_comp"], d["cppn_n"], d["rand_comp"], d["rand_n"])
                            if d["rand_comp"] is not None else None)
        rows.append(d)

    if not rows:
        print("no CPPN summaries found yet")
        return

    # order by the reach-map MN/random ratio (descending); unknown-random tasks last
    def order_key(d):
        if d["rand_rate"]:
            return -(d["mn_rate"] / d["rand_rate"])
        return 1e9
    rows.sort(key=order_key)

    tasks = [r["task"] for r in rows]
    xlabels = [LABEL.get(t, t) for t in tasks]
    x = np.arange(len(rows))

    cmn, ccppn, crand = "#1a5276", "#c0392b", "#7f8c8d"
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # ── Panel A: the three gradients (competence prevalence) ──
    mn = [r["mn_rate"] for r in rows]
    cppn = [r["cppn_rate"] for r in rows]
    rand = [r["rand_rate"] if r["rand_rate"] else np.nan for r in rows]
    axA.plot(x, mn, "-o", color=cmn, lw=2.2, ms=6, label="MorphoNAS (developmental)")
    axA.plot(x, cppn, "--s", color=ccppn, lw=2.0, ms=5, label="CPPN / HyperNEAT (geometric)")
    axA.plot(x, rand, ":^", color=crand, lw=1.8, ms=5, label="structureless random (null)")
    axA.set_yscale("log")
    axA.set_xticks(x)
    axA.set_xticklabels(xlabels, fontsize=8)
    axA.set_ylabel("competence prevalence  (% of nets non-weak, no learning)")
    axA.set_title("A. Three priors across the field (tasks ordered by MN/random)")
    axA.legend(fontsize=8, loc="upper right")
    axA.grid(True, which="both", alpha=0.2)
    for xi, r in zip(x, rows):
        axA.annotate(f"{r['mn_rate']:.1f}%", (xi, r["mn_rate"]), textcoords="offset points",
                     xytext=(0, 7), ha="center", fontsize=7, color=cmn)

    # ── Panel B: R(MN/CPPN) and R(CPPN/random) per task ──
    def pts(key):
        p, lo, hi = [], [], []
        for r in rows:
            c = r[key]
            if c is None:
                p.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
            p.append(c["point"])
            lo.append(c["point"] - c["lo"])
            hi.append((c["hi"] - c["point"]) if c["hi"] else np.nan)
        return np.array(p), np.array([lo, hi])
    w = 0.36
    p1, e1 = pts("r_mn_cppn")
    p2, e2 = pts("r_cppn_rand")
    axB.bar(x - w / 2, p1, w, yerr=e1, color=cmn, alpha=0.85, capsize=3, label="R = MN / CPPN")
    axB.bar(x + w / 2, p2, w, yerr=e2, color=ccppn, alpha=0.85, capsize=3, label="R = CPPN / random")
    axB.axhline(1.0, color="grey", ls=":", lw=1)
    axB.text(x[-1], 1.0, " parity", color="grey", va="bottom", ha="right", fontsize=8)
    axB.set_yscale("log")
    axB.set_xticks(x)
    axB.set_xticklabels(xlabels, fontsize=8)
    axB.set_ylabel("competence-prevalence ratio  (log)")
    axB.set_title("B. MN beats the geometric encoding (blue);\nthe geometric encoding beats the null (red)")
    axB.legend(fontsize=8, loc="upper right")
    axB.grid(True, which="both", axis="y", alpha=0.2)

    fig.suptitle("Indirect-encoding baseline: morphogenesis vs CPPN/HyperNEAT vs structureless random "
                 "(un-evolved, (N,E)-matched)", fontsize=11, y=1.00)
    fig.tight_layout()
    os.makedirs("experiments/baselines", exist_ok=True)
    out = "experiments/baselines/baseline_gradient.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out, "| tasks:", ", ".join(tasks))


if __name__ == "__main__":
    main()
