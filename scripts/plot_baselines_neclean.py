#!/usr/bin/env python3
"""F6 — the (N,E)-clean indirect-encoding baseline: MorphoNAS vs CPPN/HyperNEAT
Mantel-Haenszel OR per task (the Movement-2 analogue of F1). MN > CPPN on the
structured-control + memory rungs, closing toward parity off them. Reads each cell's
`mantel_haenszel.json`. Output: experiments/figures_neclean/F6_baselines.png.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = "experiments/baselines/cppn_hyperneat"
CELLS = [  # (label, family)
    ("acrobot_shaped", "structured"),
    ("acrobot", "structured"),
    ("cartpole_masked", "structured"),
    ("cartpole", "structured"),
    ("pendulum_sparse", "structured"),
    ("mountaincar", "structured"),
    ("pendulum", "structure-free"),
    ("mountaincar_shaped", "structure-free"),
]
COLOR = {"structured": "#6a51a3", "structure-free": "#bcbddc"}


def main():
    rows = []
    for t, fam in CELLS:
        d = json.load(open(f"{P}/{t}/allpool_weak/mantel_haenszel.json"))["mantel_haenszel"]
        rows.append((t, fam, d["or_mh"], d["ci"][0], d["ci"][1]))
    rows.sort(key=lambda r: r[2])
    y = list(range(len(rows)))
    ors = [r[2] for r in rows]
    xerr = [[r[2] - r[3] for r in rows], [r[4] - r[2] for r in rows]]
    colors = [COLOR[r[1]] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(y, ors, xerr=xerr, color=colors, alpha=0.9, error_kw=dict(ecolor="#444", capsize=3, lw=1))
    ax.axvline(1.0, ls="--", c="#666", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("(N,E)-clean Mantel–Haenszel OR — MorphoNAS vs CPPN/HyperNEAT")
    ax.set_title("Morphogenesis vs the leading indirect encoding, size/density-controlled\n"
                 "(MN > CPPN on structured rungs; closes toward parity off them)", fontsize=10.5)
    for i, r in enumerate(rows):
        ax.text(r[2] + 0.15, i, f"{r[2]:.1f}×", va="center", fontsize=8.5)
    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(color=COLOR[k], label=k) for k in COLOR], loc="lower right", fontsize=8.5)
    ax.set_xlim(0, max(ors) * 1.2)
    fig.tight_layout()
    os.makedirs("experiments/figures_neclean", exist_ok=True)
    out = "experiments/figures_neclean/F6_baselines.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
