#!/usr/bin/env python3
"""F4 — the (N,E)-clean selectivity dose-response: the Mantel-Haenszel OR as the
competence bar tightens, on the strong instruments. Reuses the weak-matched controls;
computes MH OR at a bar grid per instrument. x = bar strictness (normalized within each
instrument), y = MH OR. Output: experiments/figures_neclean/F4_dose_response.png.
"""
import os
import sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
sys.path.insert(0, "code")
import analyze_bar_policy as abp  # noqa: E402

P = "experiments"
INSTR = [  # (label, task, sub, bar grid loose→strict)
    ("Acrobot", "acrobot", "full_allpool", [-450, -300, -200, -100]),
    ("masked-CartPole", "cartpole_masked", "allpool_weak", [13, 20, 28, 36, 40, 44]),
    ("Pendulum-sparse", "pendulum_sparse", "allpool_weak", [13, 16, 20, 24, 28]),
]
COL = {"Acrobot": "#2c7fb8", "masked-CartPole": "#238b45", "Pendulum-sparse": "#969696"}


def mh_or_at_bar(mn_rw, mn_ne, rows, bar):
    c = defaultdict(lambda: [0, 0, 0, 0])
    for r, (N, E) in zip(mn_rw, mn_ne):
        c[(N, E)][0 if r >= bar else 1] += 1
    for row in rows:
        c[(row[1], row[2])][2 if row[0] >= bar else 3] += 1
    num = den = 0.0
    for a, b, cc, d in c.values():
        n = a + b + cc + d
        if n:
            num += a * d / n
            den += b * cc / n
    return num / den if den > 0 else float("nan")


def main():
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    for label, task, sub, bars in INSTR:
        _, mn_rw, mn_ne, _ = abp.load_mn_pool(task)
        rows, _ = abp.load_random(task, sub)
        mn_rw = np.asarray(mn_rw, float)
        ors = [mh_or_at_bar(mn_rw, mn_ne, rows, b) for b in bars]
        xs = np.linspace(0, 1, len(bars))  # normalized strictness
        ax.plot(xs, ors, "o-", color=COL[label], label=label, lw=2, ms=6)
        for x, o, b in zip(xs, ors, bars):
            ax.annotate(f"{o:.0f}×", (x, o), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=7.5, color=COL[label])
    ax.set_yscale("log")
    ax.axhline(1, ls="--", c="#666", lw=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["loosest bar\n(any structure)", "strictest bar\n(near the ceiling)"])
    ax.set_ylabel("(N,E)-clean Mantel–Haenszel odds ratio")
    ax.set_title("Selectivity dose-response, (N,E)-clean: the advantage concentrates\n"
                 "as the competence bar tightens (monotonic on the strong instruments)", fontsize=10.5)
    ax.legend(fontsize=9)
    fig.tight_layout()
    os.makedirs(f"{P}/figures_neclean", exist_ok=True)
    out = f"{P}/figures_neclean/F4_dose_response.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
