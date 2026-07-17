#!/usr/bin/env python3
"""Competence-threshold robustness sweep (offline: no rollouts, no gym).

Answers reviewers R1-W3 / R2-W3: the competence bar is a rare-tail binary, and an
odds ratio on a rare event can move with where the bar sits. Both arms already store
each network's aggregate `baseline_reward`, and competence is simply
`baseline_reward >= cutoff` (the non-weak stratum boundary). So sweeping the cutoff is
a pure re-analysis of stored rewards -- no re-evaluation.

For each task we:
  1. reproduce the PUBLISHED (N,E)-clean MH odds ratio at the calibrated cutoff c0
     (a sanity gate: the recomputed OR must match the committed mantel_haenszel.json);
  2. sweep the cutoff across a band chosen to induce MN competence rates from strict
     (~2%) to lax (~20%), recomputing the (N,E)-stratified MH OR + 95% CI at each;
  3. report whether the OR stays in a band, whether its CI stays on one side of 1, and
     whether the cross-task ordering is preserved over the whole band.

The MH estimator (Robins-Breslow-Greenland variance) is copied verbatim from
`analyze_mantel_haenszel.py` so this script is standalone and cannot drift from a
stale import; the copy is checked against the committed ORs by the sanity gate above.

Run:  .venv/bin/python scripts/threshold_sweep.py
Out:  experiments/_threshold_sweep/summary.json  + a printed table.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, "code")
from MorphoNAS_DevPriors.task_registry import get_task  # noqa: E402

Z95 = 1.959963984540054

# published control OR lives in a specific matched_random subdir per task (the
# --min-stratum weak, all-pool control); discovered in the audit, pinned here.
PUBLISHED_DIR = {
    "acrobot": "full_allpool",
    "cartpole": "allpool_weak",
    "mountaincar": "allpool_weak",
    "pendulum_sparse": "allpool_weak",
    "lunarlander": "allpool_weak",
    "cartpole_masked": "allpool_weak",
    "acrobot_masked": "allpool_weak",
    "frozenlake_shaped": "allpool_weak",
}
TASKS = list(PUBLISHED_DIR)


def mantel_haenszel(mn_cells, rand_cells):
    """MH OR of competence (MN vs random), stratified by (N,E), RBG variance.
    Verbatim from analyze_mantel_haenszel.py (gated by the reproduction check)."""
    R = S = 0.0
    vR = vRS = vS = 0.0
    used = 0
    for key in set(mn_cells) & set(rand_cells):
        a, n_mn = mn_cells[key]
        b = n_mn - a
        c, n_rd = rand_cells[key]
        d = n_rd - c
        n = n_mn + n_rd
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
        return {"or_mh": None, "n_strata_shared": used}
    or_mh = R / S
    var_ln = vR / (2 * R * R) + vRS / (2 * R * S) + vS / (2 * S * S)
    se = math.sqrt(var_ln)
    return {"or_mh": or_mh, "ci": [or_mh * math.exp(-Z95 * se), or_mh * math.exp(Z95 * se)],
            "se_ln": se, "n_strata_shared": used}


def load_mn(task):
    """[(reward, (N,E))] for every valid grown network in the pool."""
    out = []
    for fn in glob.glob(f"experiments/{task}/pool/networks/*.json"):
        d = json.load(open(fn))
        if not d.get("valid"):
            continue
        st = d.get("network_stats", {})
        out.append((float(d["baseline_reward"]),
                    (int(st.get("neurons", 0)), int(st.get("connections", 0)))))
    return out


def load_rand(task):
    """[(reward, (N,E))] for every valid matched-random graph in the control run."""
    jl = f"experiments/{task}/matched_random/{PUBLISHED_DIR[task]}/results.jsonl"
    out = []
    with open(jl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("valid"):
                continue
            out.append((float(r["baseline_reward"]),
                        (int(r["num_neurons"]), int(r["num_connections"]))))
    return out


def cells_at(rows, cutoff):
    """(N,E) -> [competent, total] with competent := reward >= cutoff."""
    cells = defaultdict(lambda: [0, 0])
    comp = 0
    for reward, key in rows:
        c = 1 if reward >= cutoff else 0
        cells[key][0] += c
        cells[key][1] += 1
        comp += c
    return cells, comp, len(rows)


def quantile_cutoff(rewards, target_rate):
    """Smallest reward cutoff whose >= share of MN is closest to target_rate."""
    xs = sorted(rewards, reverse=True)
    k = max(1, min(len(xs), round(target_rate * len(xs))))
    return xs[k - 1]


def published_or(task):
    mh = glob.glob(f"experiments/{task}/matched_random/{PUBLISHED_DIR[task]}/mantel_haenszel.json")
    d = json.load(open(mh[0]))["mantel_haenszel"]
    return d["or_mh"], d.get("n_strata_shared")


def main():
    os.makedirs("experiments/_threshold_sweep", exist_ok=True)
    summary = {}
    RATES = [0.02, 0.03, 0.05, 0.08, 0.12, 0.15, 0.20]
    print(f"{'task':18s} {'cutoff':>9s} {'MNrate':>7s} {'OR':>7s} {'CIlo':>7s} {'CIhi':>7s}  note")
    for task in TASKS:
        mn, rand = load_mn(task), load_rand(task)
        mn_rewards = [r for r, _ in mn]
        c0 = get_task(task).stratum_bounds[0][2]           # non-weak boundary
        pub_or, pub_strata = published_or(task)

        # --- sanity gate: reproduce the published OR at the calibrated cutoff ---
        cells_mn, cmn, nmn = cells_at(mn, c0)
        cells_rd, crd, nrd = cells_at(rand, c0)
        base = mantel_haenszel(cells_mn, cells_rd)
        ok = base["or_mh"] is not None and abs(base["or_mh"] - pub_or) / pub_or < 0.01
        gate = "REPRO-OK" if ok else f"REPRO-FAIL (got {base['or_mh']:.2f} vs {pub_or:.2f})"
        print(f"{task:18s} {c0:9.2f} {cmn/nmn:7.1%} {base['or_mh']:7.2f} "
              f"{base['ci'][0]:7.2f} {base['ci'][1]:7.2f}  c0 [published {pub_or:.2f}] {gate}")

        rows = [{"cutoff": c0, "mn_rate": cmn / nmn, "or": base["or_mh"],
                 "ci": base["ci"], "kind": "published_c0", "repro_ok": ok,
                 "n_strata": base["n_strata_shared"]}]

        # --- sweep: cutoffs targeting a grid of MN competence rates ---
        seen = set()
        for tr in RATES:
            cut = quantile_cutoff(mn_rewards, tr)
            if cut in seen:
                continue
            seen.add(cut)
            cm, a, na = cells_at(mn, cut)
            cr, _, _ = cells_at(rand, cut)
            mh = mantel_haenszel(cm, cr)
            if mh["or_mh"] is None:
                continue
            rows.append({"cutoff": cut, "mn_rate": a / na, "or": mh["or_mh"],
                         "ci": mh["ci"], "kind": f"target_{tr:.0%}",
                         "n_strata": mh["n_strata_shared"]})
            print(f"{'':18s} {cut:9.2f} {a/na:7.1%} {mh['or_mh']:7.2f} "
                  f"{mh['ci'][0]:7.2f} {mh['ci'][1]:7.2f}  {rows[-1]['kind']}")

        ors = [r["or"] for r in rows]
        summary[task] = {"published_or": pub_or, "c0": c0, "repro_ok": ok,
                         "or_min": min(ors), "or_max": max(ors), "sweep": rows}
        print()

    json.dump(summary, open("experiments/_threshold_sweep/summary.json", "w"), indent=2)
    print("wrote experiments/_threshold_sweep/summary.json")


if __name__ == "__main__":
    main()
