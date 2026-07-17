#!/usr/bin/env python3
"""Source-cluster bootstrap CI for the (N,E)-adjusted Mantel-Haenszel OR.

The committed ``mantel_haenszel.json`` reports a Robins-Breslow-Greenland (RBG)
interval that treats every graph row as independent. But up to ``k`` random
controls are nested under each grown source, and all rows share one fixed
20-seed rollout panel, so the RBG interval understates uncertainty. This script
recomputes the same exact-(N,E) MH point estimate and attaches a nonparametric
*source-cluster* bootstrap 95% CI: the resampling unit is the grown source, and
each resampled source carries its grown net together with all its valid random
nulls (so the nesting is respected). The CI remains conditional on the fixed
rollout seed panel; that is a separate, disclosed conditioning.

Competence is read exactly as ``analyze_mantel_haenszel.py`` does: grown =
``stratum != "weak"``; random = ``is_competent``.

    python scripts/analyze_source_cluster_bootstrap.py \
        --pool-dir experiments/acrobot/pool \
        --control-jsonl experiments/acrobot/matched_random/full_allpool/results.jsonl \
        --replicates 20000 --seed 20260717
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
from collections import defaultdict

import numpy as np


def load_sources(pool_dir: str, control_jsonl: str, cap_source_id=None):
    """Per-source records: (N,E), grown competent flag, and its valid nulls' counts."""
    grown: dict[int, tuple] = {}
    nd = os.path.join(pool_dir, "networks")
    for fn in os.listdir(nd):
        if not fn.endswith(".json") or fn.startswith("."):
            continue
        with open(os.path.join(nd, fn)) as f:
            d = json.load(f)
        if not d.get("valid"):
            continue
        sid = int(d.get("network_id", 0))
        if cap_source_id is not None and sid > cap_source_id:
            continue
        st = d.get("network_stats", {})
        ne = (int(st.get("neurons", 0)), int(st.get("connections", 0)))
        comp = 1 if d.get("stratum", "weak") != "weak" else 0
        grown[sid] = (ne, comp)

    # random nulls, aggregated per source (they share the source's (N,E))
    rnd_tot: dict[int, int] = defaultdict(int)
    rnd_comp: dict[int, int] = defaultdict(int)
    with open(control_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not r.get("valid"):
                continue
            sid = int(r.get("source_id", 0))
            if cap_source_id is not None and sid > cap_source_id:
                continue
            rnd_tot[sid] += 1
            rnd_comp[sid] += 1 if r.get("is_competent") else 0
    return grown, rnd_tot, rnd_comp


def mh_or(a, n_mn, c, n_rd):
    """Vectorised MH OR over strata arrays (RBG-style R/S ratio)."""
    n = n_mn + n_rd
    ok = n > 0
    b = n_mn - a
    d = n_rd - c
    R = np.sum(np.where(ok, a * d / np.where(ok, n, 1), 0.0))
    S = np.sum(np.where(ok, b * c / np.where(ok, n, 1), 0.0))
    return R / S if S > 0 else math.nan


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool-dir", required=True)
    p.add_argument("--control-jsonl", required=True)
    p.add_argument("--cap-source-id", type=int, default=None)
    p.add_argument("--replicates", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20260717)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    grown, rnd_tot, rnd_comp = load_sources(args.pool_dir, args.control_jsonl, args.cap_source_id)
    sids = sorted(grown)
    # map (N,E) -> stratum index
    strata = {}
    for sid in sids:
        ne = grown[sid][0]
        strata.setdefault(ne, len(strata))
    S = len(strata)
    src_stratum = np.array([strata[grown[sid][0]] for sid in sids], dtype=np.int64)
    g_comp = np.array([grown[sid][1] for sid in sids], dtype=np.float64)
    r_tot = np.array([rnd_tot.get(sid, 0) for sid in sids], dtype=np.float64)
    r_comp = np.array([rnd_comp.get(sid, 0) for sid in sids], dtype=np.float64)
    N = len(sids)

    def strata_mh(idx):
        a = np.bincount(src_stratum[idx], weights=g_comp[idx], minlength=S)
        n_mn = np.bincount(src_stratum[idx], minlength=S).astype(np.float64)
        c = np.bincount(src_stratum[idx], weights=r_comp[idx], minlength=S)
        n_rd = np.bincount(src_stratum[idx], weights=r_tot[idx], minlength=S)
        return mh_or(a, n_mn, c, n_rd)

    point = strata_mh(np.arange(N))
    rng = np.random.default_rng(args.seed)
    reps = np.empty(args.replicates, dtype=np.float64)
    for b in range(args.replicates):
        reps[b] = strata_mh(rng.integers(0, N, N))
    reps = reps[np.isfinite(reps)]
    lo, hi = np.percentile(reps, [2.5, 97.5])

    grown_comp = int(g_comp.sum())
    result = {
        "method": "nonparametric source-cluster bootstrap (resample grown sources with nested nulls)",
        "conditional_on": "fixed 20-seed rollout panel (not resampled)",
        "n_sources": N,
        "grown_competent": grown_comp,
        "random_total": int(r_tot.sum()),
        "random_competent": int(r_comp.sum()),
        "n_strata_total": S,
        "or_mh_point": point,
        "ci_cluster_bootstrap_95": [float(lo), float(hi)],
        "replicates_used": int(reps.size),
        "seed": args.seed,
        "platform": f"{platform.system()} {platform.machine()}",
    }
    if args.cap_source_id is not None:
        result["source_cap"] = args.cap_source_id

    out = args.out or os.path.join(os.path.dirname(args.control_jsonl) or ".",
                                   "mh_cluster_bootstrap.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"point MH OR       : {point:.3f}  ({grown_comp} grown competent, "
          f"{int(r_comp.sum())}/{int(r_tot.sum())} random)")
    print(f"cluster-boot 95%CI: [{lo:.2f}, {hi:.2f}]  ({reps.size} reps, {N} source clusters)")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
