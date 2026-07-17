#!/usr/bin/env python3
"""Diagnostic — why a threshold metric is the right tool: the prior's advantage lives in
the structured TAIL, not the bulk. Reward distributions of MorphoNAS vs (N,E)-matched
random wiring. Left: masked-CartPole (both pile at the ~9.4 do-nothing floor; MN has the
structured tail). Right: Acrobot-shaped (bimodal — 'stuck' vs 'swung up'; the trough bar
sits in the empty gap). Output: experiments/figures_neclean/D1_tail.png.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
sys.path.insert(0, "code")
import analyze_bar_policy as abp  # noqa: E402

P = "experiments"


def rewards(task, sub):
    _, mn_rw, _, _ = abp.load_mn_pool(task)
    rows, _ = abp.load_random(task, sub)
    return np.asarray(mn_rw, float), np.asarray([r[0] for r in rows], float)


def panel(ax, task, sub, bar, bar_label, title, xlabel, bins):
    mn, rd = rewards(task, sub)
    ax.hist(rd, bins=bins, color="#d7301f", alpha=0.55, label="random wiring", density=True)
    ax.hist(mn, bins=bins, color="#2c7fb8", alpha=0.55, label="MorphoNAS", density=True)
    ax.axvline(bar, ls="--", c="#000", lw=1.3)
    ax.set_yscale("log")
    ax.set_title(title, fontsize=10.5)
    ax.set_xlabel(xlabel)
    ax.legend(fontsize=8.5, loc="upper center")
    ax.annotate(bar_label, (bar, ax.get_ylim()[1] * 0.5), fontsize=8,
                rotation=90, va="center", ha="right")


def main():
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
    panel(axL, "cartpole_masked", "allpool_weak", 22,
          "behavioral bar\n(beats random policy)",
          "masked-CartPole — the tail phenomenon",
          "mean episode length (steps survived)", np.linspace(8, 50, 43))
    panel(axR, "acrobot_shaped", "allpool_weak", 0,
          "trough bar",
          "Acrobot-shaped — bimodal (stuck vs swung up)",
          "shaped reward", np.linspace(-510, 260, 60))
    fig.suptitle("Why a competence threshold (not a distribution-shift metric): the prior's "
                 "advantage is a structured-tail phenomenon", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(f"{P}/figures_neclean", exist_ok=True)
    out = f"{P}/figures_neclean/D1_tail.png"
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
