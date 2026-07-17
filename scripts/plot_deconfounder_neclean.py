#!/usr/bin/env python3
"""F2 — the (N,E)-clean de-confounder, both axes (the causal core).

Reward axis: MH OR under a structure-gated vs structure-free reward — the advantage is
erased on the densifiable tasks (MountainCar, Pendulum), intact on Acrobot (tip-height
itself structure-gated). Memory axis: MH OR unmasked vs masked. Reads each cell's
`mantel_haenszel.json`. Output: experiments/figures_neclean/F2_deconfounder.png.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = "experiments"


def orv(path):
    return json.load(open(path))["mantel_haenszel"]["or_mh"]


REWARD = [  # (task, gated, free)
    ("MountainCar", f"{P}/mountaincar/matched_random/allpool_weak/mantel_haenszel.json",
     f"{P}/mountaincar_shaped/matched_random/allpool_weak/mantel_haenszel.json"),
    ("Pendulum", f"{P}/pendulum_sparse/matched_random/allpool_weak/mantel_haenszel.json",
     f"{P}/pendulum/matched_random/allpool_weak/mantel_haenszel.json"),
    ("Acrobot", f"{P}/acrobot/matched_random/full_allpool/mantel_haenszel.json",
     f"{P}/acrobot_shaped/matched_random/allpool_weak/mantel_haenszel.json"),
]
MEMORY = [  # (task, unmasked, masked)
    ("CartPole", f"{P}/cartpole/matched_random/allpool_weak/mantel_haenszel.json",
     f"{P}/cartpole_masked/matched_random/allpool_weak/mantel_haenszel.json"),
    ("Acrobot", f"{P}/acrobot/matched_random/full_allpool/mantel_haenszel.json",
     f"{P}/acrobot_masked/matched_random/allpool_weak/mantel_haenszel.json"),
]


def grouped(ax, items, a_lab, b_lab, a_col, b_col):
    x = np.arange(len(items))
    w = 0.38
    a = [orv(p) for _, p, _ in items]
    b = [orv(q) for _, _, q in items]
    ax.bar(x - w / 2, a, w, label=a_lab, color=a_col)
    ax.bar(x + w / 2, b, w, label=b_lab, color=b_col)
    ax.axhline(1, ls="--", c="#666", lw=1)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([t for t, _, _ in items])
    for xi, (av, bv) in enumerate(zip(a, b)):
        ax.text(xi - w / 2, av * 1.06, f"{av:.1f}×", ha="center", fontsize=8)
        ax.text(xi + w / 2, bv * 1.06, (f"{bv:.2f}×" if bv < 1 else f"{bv:.1f}×"), ha="center", fontsize=8)
    ax.legend(fontsize=8.5)


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.8))
    grouped(axL, REWARD, "structure-gated reward", "structure-free reward", "#2c7fb8", "#fdae6b")
    axL.set_ylabel("(N,E)-clean Mantel–Haenszel odds ratio")
    axL.set_title("Reward axis — advantage erased when the\nreward credits undirected progress", fontsize=10)
    grouped(axR, MEMORY, "unmasked (reactive)", "masked (memory required)", "#74c476", "#238b45")
    axR.set_title("Observation axis — masking amplifies on\nCartPole; ~unchanged on Acrobot", fontsize=10)
    fig.suptitle("The de-confounder, (N,E)-clean: the advantage is reward- and observation-gated", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(f"{P}/figures_neclean", exist_ok=True)
    out = f"{P}/figures_neclean/F2_deconfounder.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
