#!/usr/bin/env python3
"""
(N,E)-clean de-confounder: re-express the reward- and observation-axis de-confounders
as ratios of Mantel-Haenszel ODDS RATIOS (the locked primary metric), instead of the
matched-to-competent prevalence ratios.

RoR = OR(structure-gated) / OR(structure-free).  RoR >> 1 with a CI excluding 1 =>
the advantage is reward-gated (erased when the reward stops paying for the structured
target). The non-densifiable Acrobot pair is the negative control (RoR ~ 1 = intact).

For the observation/memory axis, RoR = OR(masked) / OR(unmasked): RoR > 1 => the memory
requirement amplifies the advantage.

Reads each cell's mantel_haenszel.json (or_mh, ln_or, se_ln). Delta-method CI on ln(RoR)
treating the two cells as independent (same as ratio_stats.ror_delta). Read-only.
"""
from __future__ import annotations
import json
import math
import os

P = "experiments"


def mh(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))["mantel_haenszel"]
    return d["ln_or"], d["se_ln"], d["or_mh"]


def ror(name, gated_path, free_path, intact_is_expected=False):
    g = mh(gated_path)
    f = mh(free_path)
    if g is None or f is None:
        miss = gated_path if g is None else free_path
        print(f"  {name:>30}: PENDING ({os.path.relpath(miss)} not yet computed)")
        return
    lg, sg, og = g
    lf, sf, of = f
    ln = lg - lf
    se = math.sqrt(sg * sg + sf * sf)
    point = math.exp(ln)
    lo, hi = math.exp(ln - 1.96 * se), math.exp(ln + 1.96 * se)
    z = ln / se if se else math.inf
    p = math.erfc(abs(z) / math.sqrt(2))
    if intact_is_expected:
        read = "INTACT (not erased)" if hi >= 0.9 else "erased?? check"
    else:
        read = "GATED (erased)" if lo > 1.5 else ("intact" if lo < 1 < hi else "weak")
    print(f"  {name:>30}: OR {og:6.2f} / {of:6.2f} = RoR {point:6.2f}x [{lo:5.1f},{hi:6.1f}] p={p:.1e}  {read}")


def main():
    print("REWARD axis  —  RoR = OR(structure-gated) / OR(structure-free):")
    ror("MountainCar (densifiable)",
        f"{P}/mountaincar/matched_random/allpool_weak/mantel_haenszel.json",
        f"{P}/mountaincar_shaped/matched_random/allpool_weak/mantel_haenszel.json")
    ror("Pendulum (densifiable)",
        f"{P}/pendulum_sparse/matched_random/allpool_weak/mantel_haenszel.json",
        f"{P}/pendulum/matched_random/allpool_weak/mantel_haenszel.json")
    ror("Acrobot (non-densifiable ctrl)",
        f"{P}/acrobot/matched_random/full_allpool/mantel_haenszel.json",
        f"{P}/acrobot_shaped/matched_random/allpool_weak/mantel_haenszel.json",
        intact_is_expected=True)

    print("\nOBSERVATION/MEMORY axis  —  RoR = OR(masked) / OR(unmasked)  (>1 = memory amplifies):")
    ror("CartPole (masked vs unmasked)",
        f"{P}/cartpole_masked/matched_random/allpool_weak/mantel_haenszel.json",
        f"{P}/cartpole/matched_random/allpool_weak/mantel_haenszel.json")
    ror("Acrobot (masked vs unmasked)",
        f"{P}/acrobot_masked/matched_random/allpool_weak/mantel_haenszel.json",
        f"{P}/acrobot/matched_random/full_allpool/mantel_haenszel.json")


if __name__ == "__main__":
    main()
