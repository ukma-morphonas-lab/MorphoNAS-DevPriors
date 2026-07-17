#!/usr/bin/env python3
"""
(N, E)-adjusted Mantel-Haenszel odds ratio.

The symmetric-done-right robustness cut. Match random to EVERY MorphoNAS net
(the control run with --min-stratum weak), bin both arms by exact (neurons,
edges), and report the (N, E)-stratified MH odds ratio of competence with the
Robins-Breslow-Greenland variance. This controls for (N, E) directly rather than
via matching, and is non-circular (the MN arm over all 5000 is not selected on
outcome).

Reports alongside it the marginal matched-to-all ratio (MOVER CI), which shows
the headline (matched-to-competent) is the conservative end.

Read: MH OR stays large and significant -> the advantage is not
an (N, E) artifact. Collapses toward 1 -> it was.

Inputs:
  --pool-dir       MorphoNAS arm: every valid pool net's (N, E) + competence
  --control-jsonl  random arm: the --min-stratum weak control results.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

sys.path.append(os.path.abspath("code"))

from MorphoNAS_DevPriors.ratio_stats import Z95, ratio_ci_mover  # noqa: E402


def load_mn_arm(pool_dir: str, cap_source_id=None):
    """(N,E) -> [n_competent, n_total] over all valid MorphoNAS pool nets."""
    nd = os.path.join(pool_dir, "networks")
    cells = defaultdict(lambda: [0, 0])
    total_comp = total = 0
    for fn in os.listdir(nd):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(nd, fn)) as f:
            d = json.load(f)
        if not d.get("valid"):
            continue
        if cap_source_id is not None and int(d.get("network_id", 0)) > cap_source_id:
            continue
        st = d.get("network_stats", {})
        key = (int(st.get("neurons", 0)), int(st.get("connections", 0)))
        comp = 1 if d.get("stratum", "weak") != "weak" else 0
        cells[key][0] += comp
        cells[key][1] += 1
        total_comp += comp
        total += 1
    return cells, total_comp, total


def load_random_arm(jsonl: str, cap_source_id=None):
    """(N,E) -> [n_competent, n_total] over valid matched-random graphs."""
    cells = defaultdict(lambda: [0, 0])
    total_comp = total = 0
    with open(jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("valid"):
                continue
            if cap_source_id is not None and int(r.get("source_id", 0)) > cap_source_id:
                continue
            key = (int(r["num_neurons"]), int(r["num_connections"]))
            comp = 1 if r.get("is_competent") else 0
            cells[key][0] += comp
            cells[key][1] += 1
            total_comp += comp
            total += 1
    return cells, total_comp, total


def mantel_haenszel(mn_cells, rand_cells):
    """MH OR of competence (MN vs random), stratified by (N,E), RBG variance."""
    R = S = 0.0
    vR = vRS = vS = 0.0
    n_strata = used = 0
    for key in set(mn_cells) & set(rand_cells):
        a, n_mn = mn_cells[key]      # MN competent, MN total
        b = n_mn - a                 # MN non-competent
        c, n_rd = rand_cells[key]    # random competent, random total
        d = n_rd - c                 # random non-competent
        n = n_mn + n_rd
        n_strata += 1
        if n == 0:
            continue
        used += 1
        Ri = a * d / n
        Si = b * c / n
        R += Ri
        S += Si
        Pi = (a + d) / n
        Qi = (b + c) / n
        vR += Pi * Ri
        vRS += Pi * Si + Qi * Ri
        vS += Qi * Si
    if R == 0 or S == 0:
        return {"or_mh": None, "note": "degenerate (R or S = 0)", "n_strata": n_strata}
    or_mh = R / S
    var_ln = vR / (2 * R * R) + vRS / (2 * R * S) + vS / (2 * S * S)
    se = math.sqrt(var_ln)
    lo = or_mh * math.exp(-Z95 * se)
    hi = or_mh * math.exp(Z95 * se)
    return {"or_mh": or_mh, "ci": [lo, hi], "se_ln": se,
            "n_strata_shared": used, "ln_or": math.log(or_mh)}


def main() -> None:
    p = argparse.ArgumentParser(description="Mantel-Haenszel (N,E)-adjusted OR")
    p.add_argument("--pool-dir", required=True)
    p.add_argument("--control-jsonl", required=True,
                   help="results.jsonl from the --min-stratum weak control run")
    p.add_argument("--cap-source-id", type=int, default=None,
                   help="restrict both arms to source/network ids <= this value "
                        "(default: use the full pool)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    mn_cells, mn_comp, mn_tot = load_mn_arm(args.pool_dir, args.cap_source_id)
    rd_cells, rd_comp, rd_tot = load_random_arm(args.control_jsonl, args.cap_source_id)

    mh = mantel_haenszel(mn_cells, rd_cells)
    marginal = ratio_ci_mover(mn_comp, mn_tot, rd_comp, rd_tot)

    print(f"MN arm:     {mn_comp}/{mn_tot} competent over {len(mn_cells)} (N,E) cells")
    print(f"Random arm: {rd_comp}/{rd_tot} competent over {len(rd_cells)} (N,E) cells")
    print(f"Shared (N,E) strata: {mh.get('n_strata_shared')}")
    print()
    mpt = "inf" if not math.isfinite(marginal["point"]) else f"{marginal['point']:.2f}"
    mhi = "inf" if marginal["hi"] is None else f"{marginal['hi']:.2f}"
    print(f"Marginal matched-to-all ratio: {mpt}x  95% CI [{marginal['lo']:.2f}, {mhi}]")
    if mh.get("or_mh") is not None:
        print(f"(N,E)-adjusted MH odds ratio:  {mh['or_mh']:.2f}  "
              f"95% CI [{mh['ci'][0]:.2f}, {mh['ci'][1]:.2f}]  "
              f"({mh['n_strata_shared']} strata)")
        verdict = "NOT an (N,E) artifact" if mh["ci"][0] > 1 else "collapses toward 1"
        print(f"-> {verdict}")
    else:
        print(f"MH OR: {mh.get('note')}")

    report = {"mn_arm": {"competent": mn_comp, "total": mn_tot, "n_cells": len(mn_cells)},
              "random_arm": {"competent": rd_comp, "total": rd_tot, "n_cells": len(rd_cells)},
              "marginal_ratio": marginal, "mantel_haenszel": mh}
    if args.cap_source_id is not None:
        report["source_cap"] = args.cap_source_id
    out = args.out or os.path.join(os.path.dirname(args.control_jsonl) or ".",
                                   "mantel_haenszel.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
