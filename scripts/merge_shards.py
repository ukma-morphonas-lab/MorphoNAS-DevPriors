#!/usr/bin/env python3
"""
Merge fleet shards back into one result, for the matched-random control and the
pool generator.

  --mode control : concatenate every shard's results.jsonl, recompute the full
    ratio + CI + verdict (the single-shard summary the control would have written
    unsharded), using the pool's MorphoNAS counts as the numerator.

  --mode pool : gather every shard's networks/*.json, renumber network_id
    globally by sorted (seed) so ids are stable and contiguous, write the merged
    networks/ dir, combined pool_metadata.json (summed strata, pooled reward
    summary, gate decision), and network_ids.txt.

Usage:
  merge_shards.py --mode control \
      --shard-glob 'experiments/acrobot/matched_random/full_k50/shard*' \
      --pool-dir experiments/acrobot/pool \
      --out-dir experiments/acrobot/matched_random/full_k50

  merge_shards.py --mode pool \
      --shard-glob 'experiments/lunarlander/pool_shards/shard*' \
      --out-dir experiments/lunarlander/pool
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.append(os.path.abspath("code"))

from MorphoNAS_DevPriors.ratio_stats import grade_vs_reference, ratio_ci_mover  # noqa: E402
from MorphoNAS_DevPriors.task_registry import CARTPOLE_REF, SOLVED_STRATUM, STRATA_ORDER  # noqa: E402


def _shard_dirs(shard_glob: str) -> list[str]:
    dirs = sorted(d for d in glob.glob(shard_glob) if os.path.isdir(d))
    if not dirs:
        raise SystemExit(f"no shard dirs match {shard_glob!r}")
    return dirs


def merge_control(args) -> None:
    dirs = _shard_dirs(args.shard_glob)
    os.makedirs(args.out_dir, exist_ok=True)
    merged_path = os.path.join(args.out_dir, "results.jsonl")
    results = []
    with open(merged_path, "w") as out:
        for d in dirs:
            rp = os.path.join(d, "results.jsonl")
            if not os.path.exists(rp):
                print(f"  WARN: no results.jsonl in {d}, skipping")
                continue
            with open(rp) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    out.write(line + "\n")
                    results.append(json.loads(line))
    print(f"merged {len(results)} graph results from {len(dirs)} shards")

    with open(os.path.join(args.pool_dir, "pool_metadata.json")) as f:
        meta = json.load(f)
    counts = meta["stratum_counts"]
    n_total = sum(counts.values())
    mn = {"n_total": n_total, "stratum_counts": counts,
          "nonweak_count": n_total - counts.get("weak", 0),
          "solved_count": counts.get(SOLVED_STRATUM, 0)}

    valid = [r for r in results if r.get("valid")]
    n_valid = len(valid)
    strata = {s: 0 for s in STRATA_ORDER}
    for r in valid:
        strata[r["stratum"]] += 1
    competent = sum(1 for r in valid if r.get("is_competent"))
    solved = sum(1 for r in valid if r.get("is_solved"))

    acro = {"nonweak": ratio_ci_mover(mn["nonweak_count"], mn["n_total"], competent, n_valid),
            "solved": ratio_ci_mover(mn["solved_count"], mn["n_total"], solved, n_valid)}
    cart = {bar: ratio_ci_mover(c["mn_count"], c["mn_n"], c["rand_count"], c["rand_n"])
            for bar, c in CARTPOLE_REF.items()}
    verdict = {bar: grade_vs_reference(acro[bar]["lo"], acro[bar]["hi"],
                                       cart[bar]["point"], cart[bar]["lo"], cart[bar]["hi"])
               for bar in ("nonweak", "solved")}

    summary = {
        "experiment": "matched_random_control", "merged_from_shards": len(dirs),
        "n_random_total": len(results), "n_random_valid": n_valid,
        "valid_rate": n_valid / len(results) if results else 0.0, "strata": strata,
        "random_nonweak_count": competent, "random_nonweak_rate": competent / n_valid if n_valid else 0.0,
        "random_solved_count": solved, "random_solved_rate": solved / n_valid if n_valid else 0.0,
        "morphonas": mn, "ratio_nonweak": acro["nonweak"], "ratio_solved": acro["solved"],
        "cartpole_reference": cart, "verdict": verdict,
    }
    with open(os.path.join(args.out_dir, "summary_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  non-weak ratio: {acro['nonweak']['point']:.2f}x "
          f"[{acro['nonweak']['lo']:.2f}, {acro['nonweak']['hi']}] "
          f"verdict={verdict['nonweak']['verdict'].upper()}")
    print(f"  -> {args.out_dir}/summary_stats.json")


def merge_pool(args) -> None:
    dirs = _shard_dirs(args.shard_glob)
    out_nd = os.path.join(args.out_dir, "networks")
    os.makedirs(out_nd, exist_ok=True)

    records = []
    for d in dirs:
        for fp in glob.glob(os.path.join(d, "networks", "network_*.json")):
            with open(fp) as f:
                records.append(json.load(f))
    if not records:
        raise SystemExit("no network_*.json found across shards")
    # global ids by sorted seed (stable, contiguous); seeds are disjoint by stride
    records.sort(key=lambda r: r["seed"])
    seen = set()
    counts = {s: 0 for s in STRATA_ORDER}
    rewards = []
    nid = 1
    for r in records:
        if r["seed"] in seen:
            continue
        seen.add(r["seed"])
        r["network_id"] = nid
        counts[r["stratum"]] = counts.get(r["stratum"], 0) + 1
        rewards.append(r["baseline_reward"])
        with open(os.path.join(out_nd, f"network_{nid:06d}.json"), "w") as f:
            json.dump(r, f, indent=2)
        nid += 1
    valid_count = nid - 1
    nonweak = valid_count - counts.get("weak", 0)

    # carry forward env/dim/strata metadata from a shard's metadata if present
    base_meta = {}
    for d in dirs:
        for mp in glob.glob(os.path.join(d, "pool_metadata*.json")):
            with open(mp) as f:
                base_meta = json.load(f)
            break
        if base_meta:
            break

    a = np.array(rewards, dtype=float)
    reward_summary = {"mean": float(a.mean()), "std": float(a.std()),
                      "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max()),
                      "percentiles": {str(q): float(np.percentile(a, q))
                                      for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]}}
    gate_frac = base_meta.get("gate_frac", 0.01)
    metadata = {
        "experiment": "pool", "task": base_meta.get("task"),
        "env_name": base_meta.get("env_name"), "input_dim": base_meta.get("input_dim"),
        "output_dim": base_meta.get("output_dim"), "merged_from_shards": len(dirs),
        "valid_count": valid_count, "stratum_counts": counts,
        "stratum_bounds": base_meta.get("stratum_bounds"),
        "strata_provisional": base_meta.get("strata_provisional"),
        "nonweak_count": nonweak, "competent_fraction": nonweak / valid_count if valid_count else 0.0,
        "gate_frac": gate_frac, "gate_pass": (nonweak / valid_count if valid_count else 0.0) >= gate_frac,
        "reward_summary": reward_summary, "genome_params": base_meta.get("genome_params"),
    }
    with open(os.path.join(args.out_dir, "pool_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(args.out_dir, "network_ids.txt"), "w") as f:
        for i in range(1, valid_count + 1):
            f.write(f"{i}\n")
    print(f"merged pool: {valid_count} valid, nonweak={nonweak} "
          f"({metadata['competent_fraction']*100:.2f}%), "
          f"gate={'PASS' if metadata['gate_pass'] else 'FAIL'}")
    print(f"  strata={counts}")
    print(f"  -> {args.out_dir}/pool_metadata.json (+ {valid_count} networks)")


def main() -> None:
    p = argparse.ArgumentParser(description="Merge fleet shards (control or pool)")
    p.add_argument("--mode", required=True, choices=["control", "pool"])
    p.add_argument("--shard-glob", required=True, help="glob of shard dirs")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pool-dir", default=None, help="control mode: pool for MN numerator")
    args = p.parse_args()
    if args.mode == "control":
        if not args.pool_dir:
            raise SystemExit("--mode control requires --pool-dir")
        merge_control(args)
    else:
        merge_pool(args)


if __name__ == "__main__":
    main()
