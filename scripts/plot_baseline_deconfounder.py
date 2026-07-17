#!/usr/bin/env python3
"""
Baseline de-confounder figure: the MN-over-CPPN advantage is reward-gated.

For each within-task pair, R(MN/CPPN) on the structure-gated cell vs the
structure-free cell, with MOVER CIs and the ratio-of-ratios RoR = gated/free
annotated. Two densifiable pairs (Pendulum, MountainCar) show the advantage
present-then-erased (RoR >> 1, CI excludes 1); the non-densifiable Acrobot pair
is the negative control (RoR ~ 1, advantage intact in both). This mirrors the
MN-over-random de-confounder -- morphogenesis's edge over the geometric encoding
is specifically about reaching structured target states.

Output: experiments/baselines/baseline_deconfounder.png
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath("code"))
from MorphoNAS_DevPriors.ratio_stats import ratio_ci_mover, ror_delta  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (pair label, gated cell, free cell)
PAIRS = [
    ("Pendulum\n(densifiable)", "pendulum_sparse", "pendulum"),
    ("MountainCar\n(densifiable)", "mountaincar", "mountaincar_shaped"),
    ("Acrobot\n(non-densifiable\ncontrol)", "acrobot", "acrobot_shaped"),
]


def summary(task):
    """Prefer the k50 run, else pilot_k5."""
    for sub in ("k50", "pilot_k5"):
        p = f"experiments/baselines/cppn_hyperneat/{task}/{sub}/summary_stats.json"
        if os.path.exists(p):
            return json.load(open(p))
    raise FileNotFoundError(task)


def counts(task):
    s = summary(task)
    mn = s["morphonas"]
    return mn["nonweak_count"], mn["n_total"], s["random_nonweak_count"], s["n_random_valid"]


def main():
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    cgate, cfree = "#1a5276", "#e08a3c"
    x = np.arange(len(PAIRS))
    w = 0.36
    for i, (label, gt, ft) in enumerate(PAIRS):
        xg, ng, xc, nc = counts(gt)
        rg = ratio_ci_mover(xg, ng, xc, nc)
        xg2, ng2, xc2, nc2 = counts(ft)
        rf = ratio_ci_mover(xg2, ng2, xc2, nc2)
        rr = ror_delta({"x_mn": xg, "n_mn": ng, "x_rand": xc, "n_rand": nc},
                       {"x_mn": xg2, "n_mn": ng2, "x_rand": xc2, "n_rand": nc2})

        def bar(xpos, ci, color, lab):
            pt = ci["point"]
            hi = ci["hi"] if ci["hi"] else pt
            ax.bar(xpos, pt, w, color=color, alpha=0.88,
                   yerr=[[max(pt - ci["lo"], 0)], [max(hi - pt, 0)]], capsize=4,
                   label=lab if i == 0 else None)
            ax.text(xpos, pt, f" {pt:.2f}x", ha="center", va="bottom", fontsize=8)
        bar(x[i] - w / 2, rg, cgate, "structure-gated reward")
        bar(x[i] + w / 2, rf, cfree, "structure-free reward")
        top = max(rg["point"], rf["point"], (rg["hi"] or rg["point"]))
        ax.annotate(f"RoR={rr['ror_point']:.1f}\n[{rr['ror_ci'][0]:.1f},{rr['ror_ci'][1]:.1f}]\np={rr['p_two_sided']:.0e}",
                    (x[i], top * 1.35), ha="center", va="bottom", fontsize=8,
                    color=("#198754" if rr["ci_excludes_1"] and rr["grows"] else "#777"))

    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.text(x[-1] + 0.5, 1.0, "parity", color="grey", va="bottom", ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(0.45, 30)  # headroom so the RoR annotations clear the title
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in PAIRS], fontsize=9)
    ax.set_ylabel("R = MorphoNAS competence / CPNN-HyperNEAT competence  (log)")
    ax.set_title("The MN-over-CPPN advantage is reward-gated\n(within-task de-confounder: present on the structured-target reward, "
                 "erased when the reward is structure-free)", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, which="both", axis="y", alpha=0.2)
    fig.tight_layout()
    out = "experiments/baselines/baseline_deconfounder.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
