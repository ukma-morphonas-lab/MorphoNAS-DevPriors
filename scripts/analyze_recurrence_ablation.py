#!/usr/bin/env python3
"""
Compare recurrence-ablation modes against baseline, paired per network.

For each ablation mode (state_reset, edge_removal), match nets by source_id to the
baseline run and report:
  * competence rate per mode,
  * the 2x2 competence-flip table (baseline vs mode),
  * McNemar's exact (binomial) two-sided p on the discordant flips,
  * reward-delta stats (how many nets' avg_reward changed at all, mean/max |delta|)
    -- the guard against a mode that is silently a no-op.

A non-collapse (mode rate ~= baseline, McNemar n.s., but reward_changed > 0) is the
clean, informative null: the ablation *did* perturb behaviour, competence just did
not depend on it. reward_changed == 0 would instead flag an inert ablation.

Companion to run_recurrence_ablation.py.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from math import comb


def load(task, mode):
    p = f"experiments/{task}/recurrence_ablation/{mode}/results.jsonl"
    if not os.path.exists(p):
        return None
    d = {}
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            if r.get("valid"):
                d[r["source_id"]] = r
    return d


def mcnemar_exact(b, c):
    """Two-sided exact McNemar (binomial) p on discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--modes", default="state_reset,edge_removal")
    args = ap.parse_args()

    base = load(args.task, "baseline")
    if base is None:
        sys.exit(f"no baseline results for {args.task}")
    b_comp = sum(1 for r in base.values() if r["is_competent"])
    out = {"task": args.task, "n": len(base), "baseline_competent": b_comp,
           "baseline_rate": b_comp / len(base), "comparisons": {}}

    for mode in args.modes.split(","):
        m = load(args.task, mode)
        if m is None:
            continue
        ids = set(base) & set(m)
        bb = sum(1 for i in ids if base[i]["is_competent"] and m[i]["is_competent"])
        bo = sum(1 for i in ids if base[i]["is_competent"] and not m[i]["is_competent"])
        mo = sum(1 for i in ids if not base[i]["is_competent"] and m[i]["is_competent"])
        nn = sum(1 for i in ids if not base[i]["is_competent"] and not m[i]["is_competent"])
        dr = [abs(base[i]["baseline_reward"] - m[i]["baseline_reward"]) for i in ids]
        changed = sum(1 for x in dr if x > 1e-9)
        m_comp = sum(1 for i in m if m[i]["is_competent"])
        p = mcnemar_exact(bo, mo)
        comp = {"mode_competent": m_comp, "mode_rate": m_comp / len(m),
                "both_competent": bb, "baseline_only": bo, "mode_only": mo, "neither": nn,
                "mcnemar_exact_p": p, "n_paired": len(ids),
                "reward_changed_nets": changed,
                "mean_abs_reward_delta": statistics.mean(dr) if dr else 0.0,
                "max_abs_reward_delta": max(dr) if dr else 0.0}
        out["comparisons"][mode] = comp
        print(f"{args.task}: baseline {b_comp}/{len(base)} ({out['baseline_rate']*100:.2f}%) "
              f"vs {mode} {m_comp} ({comp['mode_rate']*100:.2f}%) | "
              f"flips comp->weak={bo} weak->comp={mo} | McNemar p={p:.3g} | "
              f"reward changed {changed}/{len(ids)} (meanD={comp['mean_abs_reward_delta']:.4f} "
              f"maxD={comp['max_abs_reward_delta']:.3f})")

    with open(f"experiments/{args.task}/recurrence_ablation/comparison.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
