#!/usr/bin/env python3
"""
Re-aggregate recurrence_structure/per_source.jsonl split by competence.

The all-net median conflates the competent tail (the prior of interest) with the
non-competent majority. Here we compare, per (N,E)-matched source, the grown net's
structure to the MEAN of ITS matched-random graphs (an (N,E)-paired difference),
and report competent vs weak separately:

  d_scc = grown.largest_scc_frac - mean(random.largest_scc_frac at same source)
  d_srb = grown.spectral_radius_bin - mean(random.spectral_radius_bin)

Positive d_scc => the grown net has a LARGER recurrent core than size-matched random.
No re-run: reads the saved per_source.jsonl.
"""
import json
import os
import statistics
import sys


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def summarize(rows, label):
    n = len(rows)
    if n == 0:
        print(f"  {label:10s}: (none)")
        return
    g_scc = [r["grown"]["largest_scc_frac"] for r in rows]
    r_scc = [statistics.mean([x["largest_scc_frac"] for x in r["random"]]) for r in rows if r["random"]]
    d_scc = [r["grown"]["largest_scc_frac"] - statistics.mean([x["largest_scc_frac"] for x in r["random"]])
             for r in rows if r["random"]]
    d_srb = [r["grown"]["spectral_radius_bin"] - statistics.mean([x["spectral_radius_bin"] for x in r["random"]])
             for r in rows if r["random"]]
    g_cyc = sum(r["grown"]["has_cycle"] for r in rows) / n
    frac_grown_more = sum(1 for d in d_scc if d > 0) / len(d_scc) if d_scc else float("nan")
    print(f"  {label:10s} (n={n}): "
          f"SCC-frac grown med={med(g_scc):.3f} vs matched-random med={med(r_scc):.3f} | "
          f"paired d_SCC med={med(d_scc):+.3f} (grown>random in {frac_grown_more*100:.0f}%) | "
          f"paired d_specBin med={med(d_srb):+.3f} | grown has_cycle={g_cyc*100:.1f}%")


def main():
    tasks = sys.argv[1:] or ["acrobot", "cartpole", "cartpole_masked", "acrobot_masked",
                             "mountaincar", "pendulum_sparse", "lunarlander_sparse", "frozenlake_shaped"]
    for task in tasks:
        p = f"experiments/{task}/recurrence_structure/per_source.jsonl"
        if not os.path.exists(p):
            print(f"{task}: no per_source.jsonl"); continue
        rows = [json.loads(l) for l in open(p)]
        rows = [r for r in rows if r.get("valid")]
        comp = [r for r in rows if r.get("is_competent")]
        weak = [r for r in rows if not r.get("is_competent")]
        print(f"\n{task}: {len(rows)} valid ({len(comp)} competent, {len(weak)} weak)")
        summarize(comp, "COMPETENT")
        summarize(weak, "weak")


if __name__ == "__main__":
    main()
