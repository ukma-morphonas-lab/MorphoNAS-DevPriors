#!/usr/bin/env python3
"""F1 — the (N,E)-clean magnitude gradient: Mantel-Haenszel OR per rung, with CIs.

The headline figure under the locked metric: how the developmental-prior advantage
grades across families once size and edge-density are controlled. Reads each rung's
`mantel_haenszel.json`. Output: experiments/figures_neclean/F1_gradient.png.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

P = "experiments"
RUNGS = [  # (label, family, mantel_haenszel.json)
    ("Acrobot-shaped", "control", f"{P}/acrobot_shaped/matched_random/allpool_weak/mantel_haenszel.json"),
    ("Acrobot", "control", f"{P}/acrobot/matched_random/full_allpool/mantel_haenszel.json"),
    ("masked-Acrobot", "memory", f"{P}/acrobot_masked/matched_random/allpool_weak/mantel_haenszel.json"),
    ("MountainCar", "control", f"{P}/mountaincar/matched_random/allpool_weak/mantel_haenszel.json"),
    ("LunarLander", "control", f"{P}/lunarlander/matched_random/allpool_weak/mantel_haenszel.json"),
    ("masked-CartPole", "memory", f"{P}/cartpole_masked/matched_random/allpool_weak/mantel_haenszel.json"),
    ("Pendulum-sparse", "control", f"{P}/pendulum_sparse/matched_random/allpool_weak/mantel_haenszel.json"),
    ("CartPole", "control", f"{P}/cartpole/matched_random/allpool_weak/mantel_haenszel.json"),
    ("MountainCar-shaped", "inversion", f"{P}/mountaincar_shaped/matched_random/allpool_weak/mantel_haenszel.json"),
    ("Pendulum-dense", "inversion", f"{P}/pendulum/matched_random/allpool_weak/mantel_haenszel.json"),
    ("FrozenLake-shaped", "inversion", f"{P}/frozenlake_shaped/matched_random/allpool_weak/mantel_haenszel.json"),
]
COLOR = {"control": "#2c7fb8", "memory": "#238b45", "inversion": "#d7301f"}
LABEL = {"control": "control (advantage)", "memory": "memory (advantage)", "inversion": "structure-free (inversion)"}


def main():
    rows = []
    for label, fam, path in RUNGS:
        d = json.load(open(path))["mantel_haenszel"]
        rows.append((label, fam, d["or_mh"], d["ci"][0], d["ci"][1]))
    rows.sort(key=lambda r: r[2])
    y = list(range(len(rows)))
    ors = [r[2] for r in rows]
    xerr = [[r[2] - r[3] for r in rows], [r[4] - r[2] for r in rows]]
    colors = [COLOR[r[1]] for r in rows]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.barh(y, ors, xerr=xerr, color=colors, alpha=0.88,
            error_kw=dict(ecolor="#444", capsize=3, lw=1))
    ax.axvline(1.0, ls="--", c="#666", lw=1, zorder=0)
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlabel("(N,E)-clean Mantel–Haenszel odds ratio  (MorphoNAS vs (N,E)-matched random wiring)")
    ax.set_title("Developmental-prior advantage, size/density-controlled\n(advantage on control & memory; inverts where the reward is structure-free)",
                 fontsize=11)
    for i, r in enumerate(rows):
        ax.text(r[2] * 1.04, i, f"{r[2]:.1f}×", va="center", fontsize=8.5)
    ax.legend(handles=[mpatches.Patch(color=COLOR[k], label=LABEL[k]) for k in COLOR],
              loc="lower right", fontsize=8.5, framealpha=0.9)
    ax.set_xlim(0.1, ax.get_xlim()[1] * 1.3)
    fig.tight_layout()
    os.makedirs(f"{P}/figures_neclean", exist_ok=True)
    out = f"{P}/figures_neclean/F1_gradient.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}  ({len(rows)} rungs, OR {min(ors):.2f}–{max(ors):.1f})")


if __name__ == "__main__":
    main()
