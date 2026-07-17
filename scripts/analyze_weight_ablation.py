#!/usr/bin/env python3
"""
Weight-ablation decision rule: topology vs weights.

Reads the five factorial cells (auto-discovered under --ablation-dir, each a
<cell>/summary_stats.json from run_weight_ablation.py) and applies the
decision rule on the non-weak bar:

  baseline = (mn_regrow, mn_keep)   competent by construction (~reference top)
  floor    = (random,    uniform)   the ~0.12% floor
  DECISIVE = (mn_regrow,  uniform)  MN wiring, bad (uniform) weights

  DECISIVE >> floor (near baseline) -> prior is STRUCTURAL, weight confound REFUTED
  DECISIVE ~= floor                 -> advantage is weight-borne, H1 KILL direction
  (random, mn_empirical) ~= floor   -> marginal MN weights alone do not rescue
                                       random topology (topology is the locus)

"Close to baseline" is quantified by the recovery fraction
  (p_decisive - p_floor) / (p_baseline - p_floor): ~1 = fully structural, ~0 = fully
weight-borne. The ratio decisive/floor (MOVER CI) is reported as the effect size.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.append(os.path.abspath("code"))

from MorphoNAS_DevPriors.ratio_stats import ratio_ci_mover  # noqa: E402

CELLS = ["mn_regrow__mn_keep", "mn_regrow__uniform", "mn_regrow__mn_shuffle",
         "random__mn_empirical", "random__uniform"]


def _aggregate_shards(cell_dir):
    """Sum sharded cell summaries (summary_shard*.json) into one cell summary."""
    shards = sorted(glob.glob(os.path.join(cell_dir, "**", "summary_shard*.json"),
                              recursive=True))
    if not shards:
        return None
    n_valid = comp = solv = 0
    cell = topo = wm = None
    for sp in shards:
        with open(sp) as f:
            s = json.load(f)
        n_valid += s.get("n_valid", 0)
        comp += s.get("competent_count", 0)
        solv += s.get("solved_count", 0)
        cell, topo, wm = s.get("cell"), s.get("topology"), s.get("weight_mode")
    return {"cell": cell or os.path.basename(cell_dir), "topology": topo, "weight_mode": wm,
            "n_valid": n_valid, "competent_count": comp,
            "competent_rate": comp / n_valid if n_valid else 0.0,
            "solved_count": solv, "merged_from_shards": len(shards)}


def load_cells(ablation_dir, explicit):
    cells = {}
    # each cell subdir: prefer the unsharded summary, else aggregate its shards
    for cell_dir in sorted(glob.glob(os.path.join(ablation_dir, "*"))):
        if not os.path.isdir(cell_dir):
            continue
        sp = os.path.join(cell_dir, "summary_stats.json")
        if os.path.exists(sp):
            with open(sp) as f:
                s = json.load(f)
        else:
            s = _aggregate_shards(cell_dir)
        if s is not None:
            cells[s.get("cell", os.path.basename(cell_dir))] = s
    for tok in explicit:
        name, path = tok.split("=", 1)
        with open(path) as f:
            cells[name] = json.load(f)
    return cells


def rate(cell):
    if cell is None:
        return None
    # accept both weight-ablation cell summaries and an injected control summary (the floor)
    n = cell.get("n_valid", cell.get("n_random_valid", 0))
    x = cell.get("competent_count", cell.get("random_nonweak_count", 0))
    p = cell.get("competent_rate", cell.get("random_nonweak_rate", (x / n if n else 0.0)))
    return {"x": x, "n": n, "p": p}


def main() -> None:
    p = argparse.ArgumentParser(description="topology-vs-weights decision rule")
    p.add_argument("--ablation-dir", default="experiments/acrobot/weight_ablation")
    p.add_argument("--cell", action="append", default=[], help="name=summary.json override")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cells = load_cells(args.ablation_dir, args.cell)
    print(f"Found cells: {sorted(cells)}\n")

    baseline = rate(cells.get("mn_regrow__mn_keep"))
    floor = rate(cells.get("random__uniform"))
    decisive = rate(cells.get("mn_regrow__uniform"))
    empirical = rate(cells.get("random__mn_empirical"))
    shuffle = rate(cells.get("mn_regrow__mn_shuffle"))

    for name, r in [("baseline (mn_regrow,mn_keep)", baseline),
                    ("DECISIVE (mn_regrow,uniform)", decisive),
                    ("shuffle  (mn_regrow,mn_shuffle)", shuffle),
                    ("empirical(random,mn_empirical)", empirical),
                    ("floor    (random,uniform)", floor)]:
        if r is not None:
            print(f"  {name:34s} {r['x']}/{r['n']} = {r['p']*100:.3f}%")
        else:
            print(f"  {name:34s} (missing)")

    report = {"cells": {k: rate(cells.get(k)) for k in CELLS}}

    if decisive and floor and baseline:
        dec_vs_floor = ratio_ci_mover(decisive["x"], decisive["n"], floor["x"], floor["n"])
        denom = (baseline["p"] - floor["p"])
        recovery = (decisive["p"] - floor["p"]) / denom if denom > 0 else float("nan")
        report["decisive_vs_floor"] = dec_vs_floor
        report["recovery_fraction"] = recovery
        if recovery >= 0.5:
            verdict = "STRUCTURAL (weight confound refuted)"
        elif recovery <= 0.1:
            verdict = "WEIGHT-BORNE (H1 kill direction)"
        else:
            verdict = "PARTIAL (advantage is jointly topology+weights)"
        report["verdict"] = verdict
        dhi = "inf" if dec_vs_floor["hi"] is None else f"{dec_vs_floor['hi']:.1f}"
        print(f"\n  decisive/floor = {dec_vs_floor['point']:.1f}x "
              f"[{dec_vs_floor['lo']:.1f}, {dhi}]")
        print(f"  recovery fraction (floor->baseline) = {recovery:.2f}")
        print(f"  VERDICT: {verdict}")

    if empirical and floor:
        emp_vs_floor = ratio_ci_mover(empirical["x"], empirical["n"], floor["x"], floor["n"])
        report["empirical_vs_floor"] = emp_vs_floor
        rescued = emp_vs_floor["lo"] > 1.5
        print(f"\n  (random,mn_empirical)/floor = {emp_vs_floor['point']:.2f}x "
              f"[{emp_vs_floor['lo']:.2f}, {emp_vs_floor['hi']}] -> "
              f"{'marginal weights help random topo' if rescued else 'weights alone do NOT rescue random topology'}")

    out = args.out or os.path.join(args.ablation_dir, "weight_ablation_verdict.json")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
