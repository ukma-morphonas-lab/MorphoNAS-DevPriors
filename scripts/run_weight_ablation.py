#!/usr/bin/env python3
"""
Topology-vs-weights ablation (topology vs weights). One cell per run.

A 2x2(+1) factorial over the competent sources, crossing
{mn_regrow, random} topology x {mn_keep, uniform, mn_empirical, mn_shuffle}
weights, all evaluated on the same task path. Run each cell as its own
invocation (--topology / --weight-mode), then analyze_weight_ablation.py
applies the decision rule.

Cells:
  --topology mn_regrow --weight-mode mn_keep      reference corner (~stored reward)
  --topology mn_regrow --weight-mode uniform      DECISIVE: MN wiring, bad weights
  --topology mn_regrow --weight-mode mn_shuffle   optional: weight placement matters?
  --topology random    --weight-mode mn_empirical (random wiring, MN-like weights)
  --topology random    --weight-mode uniform      the floor (== the control)

Each source is regrown once (deterministic) and then evaluated over k weight/
topology draws (k=1 for the deterministic mn_keep cell). Weights for the
mn_empirical cell are drawn from the pooled MorphoNAS edge weights produced by
analyze_mn_weight_distribution.py (--empirical-weights).

Sharded by source (--num-shards/--shard-index, strided) + progress file, same as
the control.
"""

from __future__ import annotations

import os as _os

# Cap BLAS/OpenMP threads BEFORE numpy imports so W workers x N_CPU threads can
# never oversubscribe on a big Linux box. SDL dummy keeps headless pygame/Box2D
# from blocking on a display. Set in-process: a non-interactive ssh command does
# not source /etc/profile.d or .bashrc, so a shell export alone would not apply.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")
_os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import argparse
import json
import logging
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np

sys.path.append(os.path.abspath("code"))

from MorphoNAS.genome import Genome  # noqa: E402
from MorphoNAS.grid import Grid  # noqa: E402
from MorphoNAS_DevPriors.logging_config import setup_logging  # noqa: E402
from MorphoNAS_DevPriors.progress import ProgressWriter  # noqa: E402
from MorphoNAS_DevPriors.random_graph import generate_random_rnn  # noqa: E402
from MorphoNAS_DevPriors.task_registry import SOLVED_STRATUM, STRATA_ORDER, get_task, run_rollouts  # noqa: E402

logger = logging.getLogger(__name__)

_W = {}


def _init_worker(task_name, rollouts, weight_lo, weight_hi, max_retries,
                 topology, weight_mode, empirical_path):
    import gymnasium as gym

    from MorphoNAS_DevPriors.parallel_utils import configure_worker_threads

    configure_worker_threads()
    spec = get_task(task_name)
    _W["spec"] = spec
    _W["env"] = gym.make(spec.env_name)
    _W["rollouts"] = int(rollouts)
    _W["wr"] = (float(weight_lo), float(weight_hi))
    _W["max_retries"] = int(max_retries)
    _W["topology"] = topology
    _W["weight_mode"] = weight_mode
    _W["empirical"] = np.load(empirical_path) if empirical_path else None


def _apply_weight_mode(G, rng):
    """Overwrite G's edge weights in place per the cell's weight mode."""
    mode = _W["weight_mode"]
    if mode == "mn_keep":
        return G  # weights as grown
    edges = list(G.edges())
    if mode == "uniform":
        for (i, j) in edges:
            G[i][j]["weight"] = float(rng.uniform(*_W["wr"]))
    elif mode == "mn_shuffle":
        w = np.array([G[i][j]["weight"] for (i, j) in edges], dtype=float)
        rng.shuffle(w)
        for (i, j), wv in zip(edges, w):
            G[i][j]["weight"] = float(wv)
    elif mode == "mn_empirical":
        emp = _W["empirical"]
        picks = rng.integers(0, len(emp), size=len(edges))
        for (i, j), p in zip(edges, picks):
            G[i][j]["weight"] = float(emp[p])
    else:
        raise ValueError(f"unknown weight_mode {mode}")
    return G


def _eval_source(task: dict) -> list[dict]:
    spec = _W["spec"]
    src_id = task["source_id"]
    n, e = task["num_neurons"], task["num_connections"]
    eval_seeds = task["eval_seeds"][: _W["rollouts"]]
    base_seed = task["base_seed"]
    k = task["num_random"]

    out = []
    base = {"source_id": src_id, "source_stratum": task["source_stratum"],
            "num_neurons": n, "num_connections": e,
            "topology": _W["topology"], "weight_mode": _W["weight_mode"]}

    if n < spec.min_neurons:
        return [{**base, "k": 0, "valid": False, "error": "insufficient_neurons"}]

    base_G = None
    if _W["topology"] == "mn_regrow":
        with open(task["genome_path"]) as f:
            genome = Genome.from_dict(json.load(f)["genome"])
        grid = Grid(genome)
        grid.run_simulation(verbose=False)
        base_G = grid.get_graph()

    # mn_keep on a regrown topology is deterministic -> a single draw.
    n_draws = 1 if _W["weight_mode"] == "mn_keep" else k

    for kk in range(n_draws):
        gen_seed = base_seed + src_id * 100 + kk
        rng = np.random.default_rng(gen_seed)
        if _W["topology"] == "mn_regrow":
            G = base_G.copy()
            _apply_weight_mode(G, rng)
        else:  # random topology
            wm = "mn_empirical" if _W["weight_mode"] == "mn_empirical" else "uniform"
            G = generate_random_rnn(n, e, rng, weight_range=_W["wr"],
                                    max_retries=_W["max_retries"], weight_mode=wm,
                                    empirical_weights=_W["empirical"])
            if G is None:
                out.append({**base, "k": kk, "random_seed": gen_seed,
                            "valid": False, "error": "not_weakly_connected"})
                continue
        res = run_rollouts(spec.make_propagator(G), len(eval_seeds), seeds=eval_seeds,
                           reset_plastic_each_episode=True, env=_W["env"])
        reward = float(res.get("avg_reward", 0.0))
        stratum = spec.get_stratum(reward)
        out.append({**base, "k": kk, "random_seed": gen_seed, "valid": True,
                    "baseline_reward": reward, "stratum": stratum,
                    "is_competent": stratum != "weak", "is_solved": stratum == SOLVED_STRATUM})
    return out


def load_sources(pool_dir, min_stratum):
    nd = os.path.join(pool_dir, "networks")
    min_idx = STRATA_ORDER.index(min_stratum)
    srcs = []
    for fn in sorted(os.listdir(nd)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(nd, fn)
        with open(path) as f:
            d = json.load(f)
        if not d.get("valid") or STRATA_ORDER.index(d.get("stratum", "weak")) < min_idx:
            continue
        st = d.get("network_stats", {})
        srcs.append({"network_id": d["network_id"], "num_neurons": int(st.get("neurons", 0)),
                     "num_connections": int(st.get("connections", 0)),
                     "stratum": d["stratum"], "genome_path": path,
                     "eval_seeds": list(d.get("rollout_data", {}).get("eval_seeds", []))})
    return srcs


def main() -> None:
    p = argparse.ArgumentParser(description="weight ablation (one cell per run)")
    p.add_argument("--task", type=str, default="acrobot")
    p.add_argument("--pool-dir", type=str, default=None)
    p.add_argument("--min-stratum", type=str, default="low_mid", choices=STRATA_ORDER)
    p.add_argument("--topology", type=str, required=True, choices=["mn_regrow", "random"])
    p.add_argument("--weight-mode", type=str, required=True,
                   choices=["mn_keep", "uniform", "mn_empirical", "mn_shuffle"])
    p.add_argument("--empirical-weights", type=str, default=None,
                   help=".npy of pooled MN edge weights (analyze_mn_weight_distribution.py)")
    p.add_argument("--num-random", type=int, default=5)
    p.add_argument("--rollouts", type=int, default=20)
    p.add_argument("--weight-lo", type=float, default=0.01)
    p.add_argument("--weight-hi", type=float, default=1.0)
    p.add_argument("--max-retries", type=int, default=100)
    p.add_argument("--base-seed", type=int, default=6_300_000,
                   help="gen seed = base + source_id*100 + k (disjoint from control)")
    p.add_argument("--max-sources", type=int, default=0, help="cap sources (0=all); for slices")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--max-workers", type=int, default=max(1, cpu_count() - 2))
    args = p.parse_args()

    spec = get_task(args.task)
    pool_dir = args.pool_dir or f"experiments/{args.task}/pool"
    cell = f"{args.topology}__{args.weight_mode}"
    out_dir = args.output_dir or f"experiments/{args.task}/weight_ablation/{cell}"
    os.makedirs(out_dir, exist_ok=True)
    setup_logging(log_dir=out_dir, log_file="weight_ablation.log")

    if args.weight_mode == "mn_empirical" and not args.empirical_weights:
        raise SystemExit("--weight-mode mn_empirical requires --empirical-weights")
    if args.topology == "mn_regrow" and args.weight_mode == "mn_empirical":
        raise SystemExit("mn_empirical is a random-topology cell; use mn_keep/uniform/mn_shuffle for mn_regrow")
    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(f"--shard-index must be in [0, {args.num_shards})")

    from MorphoNAS_DevPriors.parallel_utils import configure_worker_threads
    configure_worker_threads()

    sources = load_sources(pool_dir, args.min_stratum)
    if args.num_shards > 1:
        sources = sources[args.shard_index :: args.num_shards]
    if args.max_sources > 0:
        sources = sources[: args.max_sources]
    for s in sources:
        s["base_seed"] = args.base_seed
        s["num_random"] = args.num_random
        s["source_id"] = s["network_id"]
        s["source_stratum"] = s["stratum"]

    suffix = f" [shard {args.shard_index}/{args.num_shards}]" if args.num_shards > 1 else ""
    logger.info(f"weight-ablation cell={cell} task={args.task}{suffix} | sources={len(sources)} "
                f"k={args.num_random} base_seed={args.base_seed}")

    progress = ProgressWriter(os.path.join(out_dir, "progress.json"), total=len(sources),
                              phase=f"weight_ablation:{cell}",
                              meta={"task": args.task, "cell": cell, "min_stratum": args.min_stratum,
                                    "shard_index": args.shard_index, "num_shards": args.num_shards})

    t0 = time.time()
    results: list[dict] = []
    res_path = os.path.join(out_dir, "results.jsonl")
    with open(res_path, "w") as rf, Pool(
        args.max_workers, initializer=_init_worker,
        initargs=(args.task, args.rollouts, args.weight_lo, args.weight_hi, args.max_retries,
                  args.topology, args.weight_mode, args.empirical_weights)) as pool:
        for i, rows in enumerate(pool.imap_unordered(_eval_source, sources, chunksize=2), start=1):
            for r in rows:
                results.append(r)
                rf.write(json.dumps(r) + "\n")
            if i % 10 == 0 or i == len(sources):
                rf.flush()
                os.fsync(rf.fileno())
                valid = [r for r in results if r.get("valid")]
                comp = sum(1 for r in valid if r.get("is_competent"))
                progress.update(i, sources_done=i, graphs=len(results),
                                valid=len(valid), competent=comp)
                rate = (time.time() - t0) / i
                logger.info(f"[{i}/{len(sources)}] graphs={len(results)} competent={comp} "
                            f"| {rate:.2f}s/src ETA {rate*(len(sources)-i)/60:.1f} min")

    valid = [r for r in results if r.get("valid")]
    comp = sum(1 for r in valid if r.get("is_competent"))
    solv = sum(1 for r in valid if r.get("is_solved"))
    strata = {s: 0 for s in STRATA_ORDER}
    for r in valid:
        strata[r["stratum"]] += 1
    summary = {
        "experiment": "weight_ablation", "task": args.task, "cell": cell,
        "topology": args.topology, "weight_mode": args.weight_mode,
        "shard": {"index": args.shard_index, "num_shards": args.num_shards,
                  "is_full_run": args.num_shards == 1},
        "config": {"min_stratum": args.min_stratum, "num_random": args.num_random,
                   "rollouts": args.rollouts, "base_seed": args.base_seed,
                   "n_sources": len(sources)},
        "n_graphs": len(results), "n_valid": len(valid),
        "competent_count": comp, "competent_rate": comp / len(valid) if valid else 0.0,
        "solved_count": solv, "solved_rate": solv / len(valid) if valid else 0.0,
        "strata": strata, "elapsed_sec": time.time() - t0,
    }
    sname = ("summary_stats.json" if args.num_shards == 1
             else f"summary_shard{args.shard_index:03d}.json")
    with open(os.path.join(out_dir, sname), "w") as f:
        json.dump(summary, f, indent=2)
    progress.done(graphs=len(results), valid=len(valid), competent=comp)
    logger.info(f"weight-ablation DONE cell={cell}{suffix}: valid={len(valid)} competent={comp} "
                f"({summary['competent_rate']*100:.3f}%) -> {out_dir}/{sname}")


if __name__ == "__main__":
    main()
