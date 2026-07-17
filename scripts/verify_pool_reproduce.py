#!/usr/bin/env python3
"""
Reproducibility check for the Acrobot MorphoNAS pool rates that the reach-map uses as its
ratio numerator (non-weak 10.26%, solved 1.74%).

Two checks:
  A) Arithmetic: re-derive each network's stratum from its STORED baseline_reward
     and confirm the totals match pool_metadata.json. (Trusts stored rewards.)
  B) Engine: re-grow each sampled genome and re-evaluate it on the CURRENT
     vendored engine, using the network's own stored eval_seeds, then compare the
     recomputed reward/stratum to what's stored. (Proves the rates reproduce
     end-to-end on this engine, via the same run_rollouts path the control uses.)

Usage:
  .venv/bin/python3 scripts/verify_pool_reproduce.py            # all non-weak + 300 weak
  .venv/bin/python3 scripts/verify_pool_reproduce.py --all      # re-grow all 5000 (slow)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from multiprocessing import Pool, cpu_count

import numpy as np

sys.path.append(os.path.abspath("code"))

from MorphoNAS.grid import Grid  # noqa: E402
from MorphoNAS_DevPriors.experiment_acrobot import (  # noqa: E402
    create_propagator,
    get_stratum,
    load_network_from_file,
    run_rollouts,
)

POOL_DIR = "experiments/acrobot/pool"
STRATA = ["weak", "low_mid", "high_mid", "near_perfect", "perfect"]


def _regrow_and_eval(path: str) -> dict:
    """Re-grow one genome and re-evaluate on the current engine."""
    from MorphoNAS_DevPriors.parallel_utils import configure_worker_threads

    configure_worker_threads()

    genome, meta = load_network_from_file(path)
    with open(path) as f:
        stored = json.load(f)
    stored_reward = float(stored["baseline_reward"])
    stored_stratum = stored["stratum"]
    eval_seeds = stored.get("rollout_data", {}).get("eval_seeds", [])

    grid = Grid(genome)
    grid.run_simulation(verbose=False)
    G = grid.get_graph()
    n, e = G.number_of_nodes(), G.number_of_edges()

    prop = create_propagator(grid)
    res = run_rollouts(
        prop, len(eval_seeds), seeds=eval_seeds,
        reset_plastic_each_episode=True,
    )
    recomputed = float(res["avg_reward"])

    return {
        "network_id": stored.get("network_id"),
        "stored_reward": stored_reward,
        "recomputed_reward": recomputed,
        "abs_err": abs(recomputed - stored_reward),
        "stored_stratum": stored_stratum,
        "recomputed_stratum": get_stratum(recomputed).value,
        "stored_n": stored.get("network_stats", {}).get("neurons"),
        "stored_e": stored.get("network_stats", {}).get("connections"),
        "regrown_n": n,
        "regrown_e": e,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="re-grow all 5000 (slow)")
    ap.add_argument("--sample-weak", type=int, default=300)
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 2))
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(POOL_DIR, "networks", "network_*.json")))

    # ── Check A: arithmetic over ALL stored rewards ──
    print("=" * 64)
    print("CHECK A — re-derive strata from STORED rewards (all 5000)")
    print("=" * 64)
    counts = {s: 0 for s in STRATA}
    stored_meta_mismatch = 0
    for p in files:
        with open(p) as f:
            d = json.load(f)
        rederived = get_stratum(float(d["baseline_reward"])).value
        counts[rederived] += 1
        if rederived != d["stratum"]:
            stored_meta_mismatch += 1
    total = sum(counts.values())
    nonweak = total - counts["weak"]
    print(f"  n={total}  strata={counts}")
    print(f"  stored-stratum field disagreements: {stored_meta_mismatch}")
    print(f"  -> non-weak {nonweak}/{total} = {100*nonweak/total:.3f}%  "
          f"solved {counts['perfect']}/{total} = {100*counts['perfect']/total:.3f}%")

    meta = json.load(open(os.path.join(POOL_DIR, "pool_metadata.json")))
    mc = meta["stratum_counts"]
    print(f"  metadata says: {mc}")
    print(f"  MATCH: {counts == mc}")

    # ── Check B: re-grow + re-eval a sample on the current engine ──
    nonweak_files, weak_files = [], []
    for p in files:
        with open(p) as f:
            d = json.load(f)
        (weak_files if d["stratum"] == "weak" else nonweak_files).append(p)

    rng = np.random.default_rng(args.seed)
    if args.all:
        sample = files
    else:
        wk = list(rng.choice(weak_files, size=min(args.sample_weak, len(weak_files)),
                             replace=False))
        sample = nonweak_files + wk
    print()
    print("=" * 64)
    print(f"CHECK B — re-grow + re-eval {len(sample)} networks on CURRENT engine")
    print(f"          (all {len(nonweak_files)} non-weak + "
          f"{len(sample)-len(nonweak_files)} sampled weak)")
    print("=" * 64)

    with Pool(args.workers) as pool:
        recs = pool.map(_regrow_and_eval, sample)

    errs = np.array([r["abs_err"] for r in recs])
    changed = [r for r in recs if r["stored_stratum"] != r["recomputed_stratum"]]
    graph_changed = [r for r in recs
                     if r["stored_n"] != r["regrown_n"] or r["stored_e"] != r["regrown_e"]]

    print(f"  reward reproduction |recomputed - stored|: "
          f"max={errs.max():.4g}  mean={errs.mean():.4g}  median={np.median(errs):.4g}")
    print(f"  graphs that re-grew to different (N,E): {len(graph_changed)}")
    print(f"  networks that changed stratum: {len(changed)}")
    for r in changed[:20]:
        print(f"    id {r['network_id']}: {r['stored_stratum']} "
              f"({r['stored_reward']:.1f}) -> {r['recomputed_stratum']} "
              f"({r['recomputed_reward']:.1f})")

    # Within the re-grown sample, do the key rates reproduce?
    nonweak_recs = [r for r in recs if r["stored_stratum"] != "weak"]
    nonweak_stay = sum(1 for r in nonweak_recs if r["recomputed_stratum"] != "weak")
    perfect_recs = [r for r in recs if r["stored_stratum"] == "perfect"]
    perfect_stay = sum(1 for r in perfect_recs if r["recomputed_stratum"] == "perfect")
    weak_recs = [r for r in recs if r["stored_stratum"] == "weak"]
    weak_jump = sum(1 for r in weak_recs if r["recomputed_stratum"] != "weak")
    print()
    print(f"  non-weak sources still non-weak: {nonweak_stay}/{len(nonweak_recs)}")
    print(f"  perfect (solved) sources still solved: {perfect_stay}/{len(perfect_recs)}")
    print(f"  weak sources that jumped to non-weak: {weak_jump}/{len(weak_recs)}")

    out = {
        "check_A": {"counts": counts, "metadata": mc, "match": counts == mc,
                    "nonweak_rate": nonweak / total, "solved_rate": counts["perfect"] / total},
        "check_B": {
            "n_resampled": len(sample), "n_nonweak": len(nonweak_recs),
            "reward_abs_err_max": float(errs.max()),
            "reward_abs_err_mean": float(errs.mean()),
            "graphs_changed_NE": len(graph_changed),
            "stratum_changed": len(changed),
            "nonweak_reproduced": [nonweak_stay, len(nonweak_recs)],
            "perfect_reproduced": [perfect_stay, len(perfect_recs)],
            "weak_jumped": [weak_jump, len(weak_recs)],
        },
    }
    os.makedirs("experiments/acrobot/matched_random", exist_ok=True)
    with open("experiments/acrobot/matched_random/pool_reproduce_check.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  wrote experiments/acrobot/matched_random/pool_reproduce_check.json")


if __name__ == "__main__":
    main()
