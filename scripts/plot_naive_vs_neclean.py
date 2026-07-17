#!/usr/bin/env python3
"""F3 — the discipline figure: the naive matched-to-competent ratio vs the (N,E)-clean
Mantel-Haenszel OR, per rung. Quantifies the size/density confound (8–340× collapsing to
2.8–22.6×), and shows the one case where it runs the other way (LunarLander parity → 5.25×,
the confound that masked an advantage). naive = the published headline; clean from
`mantel_haenszel.json`. Output: experiments/figures_neclean/F3_naive_vs_neclean.png.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = "experiments"
ROWS = [  # (label, naive matched-to-competent ratio [published], mantel_haenszel.json)
    ("CartPole", 8.42, f"{P}/cartpole/matched_random/allpool_weak/mantel_haenszel.json"),
    ("Acrobot", 63.6, f"{P}/acrobot/matched_random/full_allpool/mantel_haenszel.json"),
    ("MountainCar", 338.7, f"{P}/mountaincar/matched_random/allpool_weak/mantel_haenszel.json"),
    ("LunarLander", 1.05, f"{P}/lunarlander/matched_random/allpool_weak/mantel_haenszel.json"),
    ("masked-CartPole", 11.0, f"{P}/cartpole_masked/matched_random/allpool_weak/mantel_haenszel.json"),
    ("masked-Acrobot", 97.4, f"{P}/acrobot_masked/matched_random/allpool_weak/mantel_haenszel.json"),
    ("Pendulum-sparse", 2.06, f"{P}/pendulum_sparse/matched_random/allpool_weak/mantel_haenszel.json"),
]


def main():
    naive = [r[1] for r in ROWS]
    clean = [json.load(open(r[2]))["mantel_haenszel"]["or_mh"] for r in ROWS]
    labels = [r[0] for r in ROWS]
    x = np.arange(len(ROWS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.8, 5.0))
    ax.bar(x - w / 2, naive, w, label="naive (matched-to-competent prevalence ratio)", color="#bdbdbd")
    ax.bar(x + w / 2, clean, w, label="(N,E)-clean Mantel–Haenszel OR", color="#2c7fb8")
    ax.set_yscale("log")
    ax.axhline(1, ls="--", c="#666", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylabel("ratio (log scale)")
    ax.set_title("The size/density confound, quantified: the naive ratio inflates the advantage\n"
                 "(LunarLander runs the other way — the confound there masked a real ~5× advantage)",
                 fontsize=10.5)
    ax.legend(fontsize=9, loc="upper right")
    for xi, (n, c) in enumerate(zip(naive, clean)):
        ax.text(xi - w / 2, n * 1.06, f"{n:g}×", ha="center", fontsize=7.5)
        ax.text(xi + w / 2, c * 1.06, f"{c:.1f}×", ha="center", fontsize=7.5)
    ax.set_ylim(0.5, max(naive) * 2.2)
    fig.tight_layout()
    os.makedirs(f"{P}/figures_neclean", exist_ok=True)
    out = f"{P}/figures_neclean/F3_naive_vs_neclean.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
