#!/usr/bin/env python3
"""
Memory rung figure: the velocity-masked CartPole observation de-confounder.

Two panels:
  A. De-confounder -- R (MN-over-matched-random competence ratio) vs selectivity
     (MN prevalence at the bar), for the masked (memory-required) and unmasked
     (reactive-control) conditions on the SAME genomes and matched-random graphs.
     The masked curve rises with selectivity (4.3x -> 27x); the unmasked curve is
     flat (~5-6x). At matched prevalence the memory condition shows the larger
     advantage: masking velocity amplifies the prior's edge.
  B. The ceiling -- survival (1 - CDF) of the MN pool's mean episode length, masked
     vs unmasked. Both share the do-nothing floor (~9.4); unmasked reaches the full
     500 (reactive balance), masked caps near 50 (memory bottleneck without learning).

Reads the committed result files; no re-evaluation. Output:
experiments/memory_rung_deconfounder.png
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


def mn_rewards(task):
    out = []
    for fp in glob.glob(f"experiments/{task}/pool/networks/*.json"):
        if os.path.basename(fp).startswith("."):
            continue
        d = json.load(open(fp))
        if d.get("valid"):
            out.append(float(d["baseline_reward"]))
    return np.array(out)


def rand_rewards(control_dir):
    out = []
    for line in open(os.path.join(control_dir, "results.jsonl")):
        r = json.loads(line)
        if r.get("valid"):
            out.append(float(r["baseline_reward"]))
    return np.array(out)


def curve(mn, rand, bars):
    """R, CI, and MN-prevalence at each bar (higher reward = better)."""
    nm, nr = len(mn), len(rand)
    prev, R, lo, hi = [], [], [], []
    for b in bars:
        xm = int((mn >= b).sum())
        xr = int((rand >= b).sum())
        if xm == 0:
            continue
        ci = ratio_ci_mover(xm, nm, xr, nr)
        prev.append(xm / nm * 100.0)
        R.append(ci["point"])
        lo.append(ci["lo"])
        hi.append(ci["hi"] if ci["hi"] else np.nan)
    return np.array(prev), np.array(R), np.array(lo), np.array(hi)


def main():
    mn_m = mn_rewards("cartpole_masked")
    rand_m = rand_rewards("experiments/cartpole_masked/matched_random/full_k50")
    mn_u = mn_rewards("cartpole")
    rand_u = rand_rewards("experiments/cartpole/matched_random/full_k50")

    # masked bars span its reachable range; unmasked its competent range (>=200)
    pm, Rm, lom, him = curve(mn_m, rand_m, list(range(13, 46)))
    pu, Ru, lou, hiu = curve(mn_u, rand_u, [200, 250, 300, 350, 400, 450, 475])

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # ── Panel A: de-confounder (R vs selectivity) ──
    cmask, cunmask = "#c0392b", "#2c6fbb"
    axA.fill_between(pm, lom, him, color=cmask, alpha=0.15)
    axA.plot(pm, Rm, "-o", color=cmask, ms=4, lw=2,
             label="masked  (velocity hidden -> memory required)")
    axA.fill_between(pu, lou, hiu, color=cunmask, alpha=0.15)
    axA.plot(pu, Ru, "-s", color=cunmask, ms=4, lw=2,
             label="unmasked (velocity visible -> reactive control)")
    axA.axhline(1.0, color="grey", ls=":", lw=1)
    axA.text(pm.max(), 1.0, " parity", color="grey", va="bottom", ha="right", fontsize=8)
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.invert_xaxis()  # tighter (more selective) bars to the right
    axA.set_xlabel("MN selectivity: % of MorphoNAS nets clearing the bar  (tighter →)")
    axA.set_ylabel("R  =  MN competence rate / matched-random rate")
    axA.set_title("A. Observation de-confounder (same genomes + graphs)")
    axA.legend(fontsize=8, loc="upper right")
    axA.grid(True, which="both", alpha=0.2)
    # annotate the matched-prevalence contrast (~4.8%)
    axA.annotate("at matched ~4.8% selectivity:\nmasked 17.4x  vs  unmasked 6.0x",
                 xy=(4.8, 17.4), xytext=(2.4, 3.4), fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="#555", lw=1))

    # ── Panel B: the ceiling (survival of mean episode length) ──
    grid = np.linspace(8, 500, 600)
    surv_m = np.array([(mn_m >= g).mean() * 100 for g in grid])
    surv_u = np.array([(mn_u >= g).mean() * 100 for g in grid])
    axB.plot(grid, surv_u, color=cunmask, lw=2, label="unmasked (reaches 500)")
    axB.plot(grid, surv_m, color=cmask, lw=2, label="masked (caps ~50)")
    axB.axvline(9.4, color="grey", ls=":", lw=1)
    axB.text(9.8, 60, "do-nothing\nfloor 9.4", color="grey", fontsize=8, va="center")
    axB.axvline(475, color="#888", ls="--", lw=1)
    axB.text(470, 30, "solved 475", color="#888", fontsize=8, ha="right", rotation=90, va="center")
    axB.set_xscale("log")
    axB.set_xlabel("mean episode length over 20 rollouts")
    axB.set_ylabel("% of MorphoNAS pool reaching ≥ x")
    axB.set_title("B. The memory ceiling (MN pool, n=5000 each)")
    axB.legend(fontsize=8, loc="upper right")
    axB.grid(True, which="both", alpha=0.2)

    fig.suptitle("Memory rung: velocity-masked CartPole -- the prior's advantage extends to "
                 "and concentrates on temporal integration", fontsize=11, y=1.00)
    fig.tight_layout()
    out = "experiments/memory_rung_deconfounder.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
