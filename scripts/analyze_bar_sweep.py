#!/usr/bin/env python3
"""
Bar-sweep analysis: recompute the MorphoNAS-over-matched-random ratio R at a
range of non-weak bars, from the STORED rewards (no re-evaluation). The selectivity
profile / dose-response: how sharply the advantage rises as the structured success
bar tightens (a steeper rise = the advantage is more concentrated on the genuinely
structured outcomes). Nested cuts on one sample are a correlated curve, not
independent tests.

MN rewards: experiments/<task>/pool/networks/*.json (baseline_reward).
Random rewards: the matched-random control results.jsonl (valid rows).

No args -> reproduces the canonical selectivity profiles (Pendulum dense + sparse,
Acrobot, masked CartPole). --task/--control-dir/--bars runs an ad-hoc sweep.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath("code"))
from MorphoNAS_DevPriors.ratio_stats import fisher_within_task, ratio_ci_mover  # noqa: E402


def mn_rewards(task):
    out = []
    for fp in glob.glob(f"experiments/{task}/pool/networks/*.json"):
        if os.path.basename(fp).startswith("."):
            continue
        with open(fp) as f:
            d = json.load(f)
        if d.get("valid"):
            out.append(float(d["baseline_reward"]))
    return out


def rand_rewards(control_dir):
    out = []
    with open(os.path.join(control_dir, "results.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("valid"):
                out.append(float(r["baseline_reward"]))
    return out


def sweep(label, mn, rand, bars, higher_is_better):
    n_mn, n_rand = len(mn), len(rand)
    print(f"\n== {label} ==  MN n={n_mn}  random n={n_rand}  (higher_is_better={higher_is_better})")
    print(f"{'bar':>10s} {'MN>=bar':>14s} {'rand>=bar':>16s} {'R':>9s}  {'95% CI':>18s} {'Fisher p':>10s}")
    for b in bars:
        if higher_is_better:
            xm = sum(1 for v in mn if v >= b); xr = sum(1 for v in rand if v >= b)
        else:
            xm = sum(1 for v in mn if v <= b); xr = sum(1 for v in rand if v <= b)
        ci = ratio_ci_mover(xm, n_mn, xr, n_rand)
        p = fisher_within_task(xm, n_mn, xr, n_rand).get("p_one_sided_greater", float("nan"))
        hi = f"{ci['hi']:.2f}" if ci['hi'] else "inf"
        pt = f"{ci['point']:.2f}" if ci['point'] != float('inf') else "inf"
        print(f"{b:>10.1f} {xm:>6d}({xm/n_mn*100:>5.2f}%) {xr:>7d}({xr/n_rand*100:>6.3f}%) "
              f"{pt:>9s}  [{ci['lo']:.2f}, {hi}] {p:>10.1e}")


def canonical():
    """The committed selectivity profiles, reproducible from stored rewards."""
    # Pendulum dense: higher (less negative) reward is better; top 20/10/5% of MN.
    dmn = mn_rewards("pendulum")
    drand = rand_rewards("experiments/pendulum/matched_random/full")
    dbars = [float(np.percentile(dmn, q)) for q in (80, 90, 95)]
    print("Dense Pendulum bars (top 20/10/5% of MN dense reward):", [round(b, 1) for b in dbars])
    sweep("DENSE pendulum (reward >= bar)", dmn, drand, dbars, higher_is_better=True)

    # Pendulum sparse: upright-step count, higher is better; soft top-decile + stricter.
    smn = mn_rewards("pendulum_sparse")
    srand = rand_rewards("experiments/pendulum_sparse/matched_random/full_k50")
    sweep("SPARSE pendulum (upright-steps >= bar)", smn, srand,
          [13.0, 16.0, 20.0, 24.0, 28.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0, 45.0, 50.0],
          higher_is_better=True)

    # Acrobot: graded structured bar (steps-to-success), the strong-instrument check
    # that the dose-response is not a soft bang-bang artifact. R rises 63.6x -> ~216x.
    amn = mn_rewards("acrobot")
    arand = rand_rewards("experiments/acrobot/matched_random/full_k50")
    sweep("ACROBOT (reward >= bar; -450 non-weak .. -100 solved)", amn, arand,
          [-450.0, -400.0, -350.0, -300.0, -250.0, -200.0, -150.0, -100.0],
          higher_is_better=True)

    # Masked CartPole: the memory rung dose-response (mean episode length).
    mmn = mn_rewards("cartpole_masked")
    mrand = rand_rewards("experiments/cartpole_masked/matched_random/full_k50")
    sweep("CARTPOLE-MASKED (mean episode length >= bar; 28 non-weak)", mmn, mrand,
          [13.0, 16.0, 20.0, 24.0, 28.0, 32.0, 36.0, 40.0, 44.0],
          higher_is_better=True)


def main():
    ap = argparse.ArgumentParser(description="matched-random ratio bar-sweep (selectivity profile)")
    ap.add_argument("--task", help="pool task name; omit to run the canonical sweeps")
    ap.add_argument("--control-dir", help="matched-random control dir holding results.jsonl")
    ap.add_argument("--bars", help="comma-separated bar values")
    ap.add_argument("--lower-is-better", action="store_true",
                    help="count reward <= bar instead of >= bar")
    args = ap.parse_args()

    if args.task:
        if not (args.control_dir and args.bars):
            raise SystemExit("--task requires --control-dir and --bars")
        mn = mn_rewards(args.task)
        rand = rand_rewards(args.control_dir)
        bars = [float(x) for x in args.bars.split(",")]
        rel = "<=" if args.lower_is_better else ">="
        sweep(f"{args.task} (reward {rel} bar)", mn, rand, bars,
              higher_is_better=not args.lower_is_better)
    else:
        canonical()


if __name__ == "__main__":
    main()
