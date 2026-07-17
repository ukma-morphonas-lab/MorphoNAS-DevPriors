#!/usr/bin/env python3
"""
Reach-map MorphoNAS pool generator, task-parameterized + shardable for the fleet.

Grow random-genome networks on a developmental grid, evaluate each on the task
(per-seed rollouts), assign strata, and write one JSON per valid network in the
exact format the matched-random control reads (network_id, seed, genome,
network_stats, rollout_data, stratum). This is the pool-generation step that produces a
new rung's MorphoNAS pool (LunarLander, MountainCar). Acrobot/CartPole pools
already exist and are not regrown.

Pilot vs full: --target-valid 500 is the gate pilot (check competent fraction
>= --gate-frac before committing a full ~5000 pool); --target-valid 5000 is the
full pool.

Sharding (fleet): --num-shards S --shard-index I. Shard I tries the disjoint
strided seed sequence  pool_seed_start + I + j*S  (j = 0,1,2,...) until it
collects ceil(target_valid / S) valid networks. Striding gives non-overlapping
seeds and a balanced size mix per shard. Each shard writes networks with local
ids into its own --output-dir; merge_shards.py --mode pool renumbers globally
and writes the combined pool_metadata.json.

Progress: <output-dir>/progress.json (atomic, fsync'd) tracks valid-collected vs
target with rate/ETA, plus attempts and the live stratum histogram.
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
import math
import os
import sys
from datetime import datetime
from multiprocessing import Pool, cpu_count
from typing import Optional

import numpy as np

sys.path.append(os.path.abspath("code"))

from MorphoNAS.genome import Genome  # noqa: E402
from MorphoNAS.grid import Grid  # noqa: E402
from MorphoNAS.weight_channel import WeightChannel, set_weight_channel  # noqa: E402
from MorphoNAS_DevPriors.wchannel_cli import (  # noqa: E402
    add_wchannel_args,
    build_weight_channel,
    wchannel_to_dict,
)
from MorphoNAS_DevPriors.logging_config import setup_logging  # noqa: E402
from MorphoNAS_DevPriors.progress import ProgressWriter  # noqa: E402
from MorphoNAS_DevPriors.task_registry import STRATA_ORDER, get_task, run_rollouts  # noqa: E402

logger = logging.getLogger(__name__)

_W = {}


def _init_worker(task_name, n_rollouts, min_neurons, min_edges, genome_params, wchannel):
    import gymnasium as gym

    from MorphoNAS_DevPriors.parallel_utils import configure_worker_threads

    configure_worker_threads()
    # Weight channel: make the engine config live in this worker (robust across fork/spawn).
    # `wchannel` is now the full WeightChannel object (MNv2 M5-M9 need >3 knobs), threaded
    # via initargs. Default (saturating) == v1.1 byte-for-byte. (design sec. 4.3-A, sec. 7;
    # mn-v2-sweep-design.md sec. 9.2)
    set_weight_channel(wchannel)
    spec = get_task(task_name)
    _W["spec"] = spec
    _W["env"] = spec.make_env()
    _W["rollouts"] = int(n_rollouts)
    _W["min_neurons"] = int(min_neurons)
    _W["min_edges"] = int(min_edges)
    _W["genome_params"] = genome_params


def _grow_eval(args) -> dict:
    network_idx, seed = args
    spec = _W["spec"]
    gp = _W["genome_params"]

    rng = np.random.default_rng(seed)
    genome = Genome.random(
        rng=rng,
        size_x=gp["size_x"],
        size_y=gp["size_y"],
        max_growth_steps=gp["max_growth_steps"],
        num_morphogens=gp["num_morphogens"],
    )
    grid = Grid(genome)
    grid.run_simulation(verbose=False)
    G = grid.get_graph()
    n, e = int(G.number_of_nodes()), int(G.number_of_edges())

    if n < _W["min_neurons"] or e < _W["min_edges"]:
        return {"network_id": network_idx, "seed": int(seed), "valid": False,
                "error": "insufficient_neurons_or_edges",
                "network_stats": {"neurons": n, "connections": e}}

    eval_seeds = list(range(int(seed), int(seed) + _W["rollouts"]))
    res = run_rollouts(spec.make_propagator(G), len(eval_seeds), seeds=eval_seeds,
                       reset_plastic_each_episode=True, env=_W["env"])
    mean_reward = float(res.get("avg_reward", 0.0))
    stratum = spec.get_stratum(mean_reward)
    return {
        "network_id": network_idx,
        "seed": int(seed),
        "valid": True,
        "baseline_reward": mean_reward,
        "stratum": stratum,
        "genome": genome.to_dict(),
        "network_stats": {"neurons": n, "connections": e},
        "rollout_data": {"rewards": res.get("rewards", []),
                         "lengths": res.get("lengths", []),
                         "eval_seeds": eval_seeds},
    }


def _regrow_eval(path: str) -> dict:
    """v2 regrow mode (design sec. 4.3-A): re-grow ONE stored genome under the
    worker's (v2) weight channel and re-score it on the SAME stored eval_seeds,
    keeping network_id. Topology is invariant (sec. 3), so the genome-paired v2 net
    has the same (N,E) and is scored under the UNCHANGED v1.1 strata -- the
    weight-channel de-confounder ("more nets clear the same bar under the same
    seeds")."""
    spec = _W["spec"]
    with open(path) as f:
        d = json.load(f)
    network_id = d["network_id"]
    seed = d.get("seed")
    genome = Genome.from_dict(d["genome"])

    grid = Grid(genome)
    grid.run_simulation(verbose=False)
    G = grid.get_graph()
    n, e = int(G.number_of_nodes()), int(G.number_of_edges())

    stored_stats = d.get("network_stats", {})
    stored_n = stored_stats.get("neurons")
    stored_e = stored_stats.get("connections")
    topology_match = (stored_n == n and stored_e == e)

    # Reuse the net's stored eval_seeds verbatim (design sec. 4.2 invariant). Fall
    # back to range(seed, seed+rollouts) only if the stored net lacks them (flagged).
    stored_seeds = list(d.get("rollout_data", {}).get("eval_seeds", []))
    used_fallback = not stored_seeds
    eval_seeds = stored_seeds if stored_seeds else list(range(int(seed), int(seed) + _W["rollouts"]))

    if n < _W["min_neurons"] or e < _W["min_edges"]:
        # Should never happen (topology invariant); records a hard red flag if it does.
        return {"network_id": network_id, "seed": seed, "valid": False,
                "error": "insufficient_neurons_or_edges_after_regrow",
                "network_stats": {"neurons": n, "connections": e},
                "topology_match": topology_match,
                "stored_network_stats": {"neurons": stored_n, "connections": stored_e}}

    res = run_rollouts(spec.make_propagator(G), len(eval_seeds), seeds=eval_seeds,
                       reset_plastic_each_episode=True, env=_W["env"])
    mean_reward = float(res.get("avg_reward", 0.0))
    stratum = spec.get_stratum(mean_reward)
    return {
        "network_id": network_id,
        "seed": seed,
        "valid": True,
        "baseline_reward": mean_reward,
        "stratum": stratum,
        "genome": genome.to_dict(),
        "network_stats": {"neurons": n, "connections": e},
        "rollout_data": {"rewards": res.get("rewards", []),
                         "lengths": res.get("lengths", []),
                         "eval_seeds": eval_seeds},
        "topology_match": topology_match,
        "used_fallback_seeds": used_fallback,
        "stored_baseline_reward": d.get("baseline_reward"),
        "stored_stratum": d.get("stratum"),
    }


def run_regrow_mode(args, spec, out_dir, networks_dir, genome_params, wchannel) -> None:
    """v2 weight-channel regrow (design sec. 4.3-A): re-grow + re-score every stored
    genome in args.regrow_from under the v2 weight channel, keeping network_id and
    reusing each net's stored eval_seeds, scored under the UNCHANGED v1.1 strata."""
    src_nd = os.path.join(args.regrow_from, "networks")
    paths = [os.path.join(src_nd, fn) for fn in sorted(os.listdir(src_nd))
             if fn.endswith(".json") and not fn.startswith(".")]
    # only re-grow networks that were VALID in the source pool
    valid_paths = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        if d.get("valid", False):
            valid_paths.append(p)

    logger.info("=" * 60)
    logger.info(f"V2 REGROW -- task={args.task}  source={args.regrow_from}")
    logger.info("=" * 60)
    logger.info(f"Weight channel: {wchannel}")
    logger.info(f"Env: {spec.env_name} ({spec.input_dim} in / {spec.output_dim} out)")
    logger.info(f"Source valid networks: {len(valid_paths)} of {len(paths)} files")
    logger.info(f"Strata HELD FIXED (v1.1 bars); eval_seeds reused per-net; network_id kept.")
    logger.info(f"Workers: {args.max_workers}")
    logger.info("=" * 60)

    progress = ProgressWriter(
        os.path.join(out_dir, "progress.json"),
        total=len(valid_paths),
        phase="v2_regrow",
        meta={"task": args.task, "regrow_from": args.regrow_from,
              "wchannel_mode": wchannel.mode, "weight_channel": wchannel_to_dict(wchannel)},
    )

    stratum_counts = {s: 0 for s in STRATA_ORDER}
    valid_count, invalid_count = 0, 0
    topology_mismatch, fallback_seeds = 0, 0
    stratum_changes_vs_v11 = 0
    all_rewards: list[float] = []
    done = 0

    with Pool(args.max_workers, initializer=_init_worker,
              initargs=(args.task, args.eval_rollouts, spec.min_neurons,
                        args.min_edges, genome_params, wchannel)) as pool:
        for result in pool.imap_unordered(_regrow_eval, valid_paths, chunksize=8):
            done += 1
            if result.get("valid"):
                valid_count += 1
                stratum_counts[result["stratum"]] += 1
                all_rewards.append(result["baseline_reward"])
                if not result.get("topology_match", True):
                    topology_mismatch += 1
                if result.get("used_fallback_seeds"):
                    fallback_seeds += 1
                if result.get("stored_stratum") is not None and result["stored_stratum"] != result["stratum"]:
                    stratum_changes_vs_v11 += 1
                nid = result["network_id"]
                # strip the v1.1-comparison provenance from the stored net (keep schema clean,
                # provenance lives in pool_metadata) but keep topology_match as a per-net flag
                net_out = {k: v for k, v in result.items()
                           if k not in ("stored_baseline_reward", "stored_stratum", "used_fallback_seeds")}
                with open(os.path.join(networks_dir, f"network_{nid:06d}.json"), "w") as f:
                    json.dump(net_out, f, indent=2)
            else:
                invalid_count += 1
                logger.warning(f"REGROW INVALID id={result.get('network_id')} "
                               f"err={result.get('error')} stored={result.get('stored_network_stats')} "
                               f"regrown={result.get('network_stats')}")
            if done % args.log_every == 0 or done == len(valid_paths):
                nonweak = valid_count - stratum_counts["weak"]
                progress.update(done, valid=valid_count, invalid=invalid_count,
                                nonweak=nonweak, strata=dict(stratum_counts),
                                topology_mismatch=topology_mismatch)
                logger.info(f"regrown={done}/{len(valid_paths)} valid={valid_count} "
                            f"nonweak={nonweak} topo_mismatch={topology_mismatch} | strata={stratum_counts}")

    nonweak = valid_count - stratum_counts["weak"]
    competent_frac = nonweak / valid_count if valid_count else 0.0
    reward_summary = {}
    if all_rewards:
        a = np.array(all_rewards)
        reward_summary = {"mean": float(a.mean()), "std": float(a.std()),
                          "median": float(np.median(a)), "min": float(a.min()),
                          "max": float(a.max()),
                          "percentiles": {str(q): float(np.percentile(a, q))
                                          for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]}}

    metadata = {
        "experiment": "pool_v2_regrow",
        "task": args.task,
        "env_name": spec.env_name,
        "input_dim": spec.input_dim,
        "output_dim": spec.output_dim,
        "regrow_from": args.regrow_from,
        "weight_channel": wchannel_to_dict(wchannel),
        "shard": {"index": 0, "num_shards": 1, "is_full_pool": True},
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "topology_mismatch_count": topology_mismatch,
        "fallback_seed_count": fallback_seeds,
        "stratum_changes_vs_v11": stratum_changes_vs_v11,
        "stratum_counts": stratum_counts,
        "stratum_bounds": spec.stratum_bounds_dict(),
        "strata_provisional": spec.provisional,
        "nonweak_count": nonweak,
        "competent_fraction": competent_frac,
        "reward_summary": reward_summary,
        "genome_params": genome_params,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "pool_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    progress.done(valid=valid_count, nonweak=nonweak, topology_mismatch=topology_mismatch)
    logger.info("=" * 60)
    logger.info(f"V2 REGROW DONE: valid={valid_count} nonweak={nonweak} "
                f"({competent_frac*100:.2f}%) topo_mismatch={topology_mismatch} "
                f"fallback_seeds={fallback_seeds} stratum_changes_vs_v11={stratum_changes_vs_v11}")
    if reward_summary:
        logger.info(f"reward: mean={reward_summary['mean']:.2f} median={reward_summary['median']:.2f} "
                    f"range=[{reward_summary['min']:.1f},{reward_summary['max']:.1f}] "
                    f"p90={reward_summary['percentiles']['90']:.2f} p99={reward_summary['percentiles']['99']:.2f}")
    logger.info(f"strata: {stratum_counts}")
    logger.info(f"Output: {out_dir}/pool_metadata.json")


def main() -> None:
    p = argparse.ArgumentParser(description="reach-map pool generator (task + shard)")
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--target-valid", type=int, default=5000)
    p.add_argument("--max-seeds", type=int, default=400000,
                   help="max candidate seeds this shard will try before giving up")
    p.add_argument("--start-seed", type=int, default=None,
                   help="default: task.pool_seed_start")
    p.add_argument("--eval-rollouts", type=int, default=20)
    p.add_argument("--min-edges", type=int, default=5)
    p.add_argument("--size-x", type=int, default=None)
    p.add_argument("--size-y", type=int, default=None)
    p.add_argument("--max-growth-steps", type=int, default=200)
    p.add_argument("--num-morphogens", type=int, default=3)
    p.add_argument("--gate-frac", type=float, default=0.01,
                   help="pilot gate: min competent fraction to deem the rung usable")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--output-dir", type=str, default=None,
                   help="default: experiments/<task>/pool")
    p.add_argument("--max-workers", type=int, default=max(1, cpu_count() - 2))
    p.add_argument("--log-every", type=int, default=100)
    # engine weight channel (design sec. 2, 4.3, 7; MNv2 M5-M9 in mn-v2-sweep-design.md sec. 2).
    # NOT --weight-mode, which is taken elsewhere with unrelated meanings.
    add_wchannel_args(p)
    p.add_argument("--regrow-from", type=str, default=None,
                   help="v2 regrow mode: re-grow + re-score the stored genomes under this "
                        "pool dir's networks/ (keeping network_id + reusing eval_seeds) "
                        "instead of sampling fresh seeds. (design sec. 4.3-A)")
    args = p.parse_args()

    spec = get_task(args.task)
    start_seed = args.start_seed if args.start_seed is not None else spec.pool_seed_start
    gx = args.size_x or spec.grid_size[0]
    gy = args.size_y or spec.grid_size[1]
    out_dir = args.output_dir or f"experiments/{args.task}/pool"
    networks_dir = os.path.join(out_dir, "networks")
    os.makedirs(networks_dir, exist_ok=True)
    setup_logging(log_dir=out_dir, log_file="pool_build.log")

    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(f"--shard-index must be in [0, {args.num_shards})")

    from MorphoNAS_DevPriors.parallel_utils import configure_worker_threads
    configure_worker_threads()

    per_shard_target = math.ceil(args.target_valid / args.num_shards)
    genome_params = {"size_x": gx, "size_y": gy,
                     "max_growth_steps": args.max_growth_steps,
                     "num_morphogens": args.num_morphogens}
    wchannel = build_weight_channel(args)

    # v2 regrow mode (design sec. 4.3-A): re-grow stored genomes under the v2 weight
    # channel instead of sampling fresh seeds. Self-contained path; returns when done.
    if args.regrow_from:
        run_regrow_mode(args, spec, out_dir, networks_dir, genome_params, wchannel)
        return

    suffix = f" [shard {args.shard_index}/{args.num_shards}]" if args.num_shards > 1 else ""
    logger.info("=" * 60)
    logger.info(f"POOL GENERATION -- task={args.task}{suffix}")
    logger.info("=" * 60)
    logger.info(f"Env: {spec.env_name} ({spec.input_dim} in / {spec.output_dim} out) | grid {gx}x{gy}")
    logger.info(f"Target valid (this shard): {per_shard_target} of {args.target_valid} total")
    logger.info(f"Seed stride: {start_seed} + {args.shard_index} + j*{args.num_shards}")
    logger.info(f"Workers: {args.max_workers}")
    if spec.provisional:
        logger.info("NOTE: strata PROVISIONAL -- read reward_summary to calibrate before full pool.")
    logger.info("=" * 60)

    progress = ProgressWriter(
        os.path.join(out_dir, "progress.json"),
        total=per_shard_target,
        phase="pool_grow",
        meta={"task": args.task, "grid": [gx, gy], "target_valid_total": args.target_valid,
              "shard_index": args.shard_index, "num_shards": args.num_shards,
              "provisional_strata": spec.provisional},
    )

    stratum_counts = {s: 0 for s in STRATA_ORDER}
    valid_count, invalid_count, attempts = 0, 0, 0
    all_rewards: list[float] = []
    local_id = 1
    # strided, disjoint seed generator for this shard
    seed_iter = (start_seed + args.shard_index + j * args.num_shards
                 for j in range(args.max_seeds))

    with Pool(args.max_workers, initializer=_init_worker,
              initargs=(args.task, args.eval_rollouts, spec.min_neurons,
                        args.min_edges, genome_params, wchannel)) as pool:
        batch = []
        BATCH = 1000
        exhausted = False
        while valid_count < per_shard_target and not exhausted:
            batch = []
            for _ in range(BATCH):
                try:
                    s = next(seed_iter)
                except StopIteration:
                    exhausted = True
                    break
                batch.append((local_id + len(batch), s))
            if not batch:
                break
            local_id += len(batch)
            for result in pool.imap_unordered(_grow_eval, batch, chunksize=8):
                attempts += 1
                if result.get("valid"):
                    valid_count += 1
                    stratum_counts[result["stratum"]] += 1
                    all_rewards.append(result["baseline_reward"])
                    nid = result["network_id"]
                    with open(os.path.join(networks_dir, f"network_{nid:06d}.json"), "w") as f:
                        json.dump(result, f, indent=2)
                else:
                    invalid_count += 1
                if attempts % args.log_every == 0:
                    nonweak = valid_count - stratum_counts["weak"]
                    progress.update(min(valid_count, per_shard_target), attempts=attempts,
                                    valid=valid_count, invalid=invalid_count,
                                    nonweak=nonweak, strata=dict(stratum_counts))
                    logger.info(f"attempts={attempts} valid={valid_count}/{per_shard_target} "
                                f"nonweak={nonweak} | strata={stratum_counts}")
                if valid_count >= per_shard_target:
                    break

    nonweak = valid_count - stratum_counts["weak"]
    competent_frac = nonweak / valid_count if valid_count else 0.0
    reward_summary = {}
    if all_rewards:
        a = np.array(all_rewards)
        reward_summary = {"mean": float(a.mean()), "std": float(a.std()),
                          "median": float(np.median(a)), "min": float(a.min()),
                          "max": float(a.max()),
                          "percentiles": {str(q): float(np.percentile(a, q))
                                          for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]}}

    metadata = {
        "experiment": "pool",
        "task": args.task,
        "env_name": spec.env_name,
        "input_dim": spec.input_dim,
        "output_dim": spec.output_dim,
        "shard": {"index": args.shard_index, "num_shards": args.num_shards,
                  "is_full_pool": args.num_shards == 1},
        "target_valid_total": args.target_valid,
        "target_valid_this_shard": per_shard_target,
        "start_seed": start_seed,
        "weight_channel": wchannel_to_dict(wchannel),
        "attempts": attempts,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "validity_rate": valid_count / attempts if attempts else 0.0,
        "stratum_counts": stratum_counts,
        "stratum_bounds": spec.stratum_bounds_dict(),
        "strata_provisional": spec.provisional,
        "nonweak_count": nonweak,
        "competent_fraction": competent_frac,
        "gate_frac": args.gate_frac,
        "gate_pass": competent_frac >= args.gate_frac,
        "reward_summary": reward_summary,
        "genome_params": genome_params,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    meta_name = ("pool_metadata.json" if args.num_shards == 1
                 else f"pool_metadata_shard{args.shard_index:03d}.json")
    with open(os.path.join(out_dir, meta_name), "w") as f:
        json.dump(metadata, f, indent=2)

    progress.done(valid=valid_count, attempts=attempts, nonweak=nonweak)
    logger.info("=" * 60)
    logger.info(f"POOL DONE{suffix}: valid={valid_count} nonweak={nonweak} "
                f"competent_frac={competent_frac*100:.2f}% gate({args.gate_frac*100:.1f}%)="
                f"{'PASS' if metadata['gate_pass'] else 'FAIL'}")
    if reward_summary:
        logger.info(f"reward: mean={reward_summary['mean']:.1f} median={reward_summary['median']:.1f} "
                    f"range=[{reward_summary['min']:.0f},{reward_summary['max']:.0f}] "
                    f"p90={reward_summary['percentiles']['90']:.1f} p99={reward_summary['percentiles']['99']:.1f}")
        if spec.provisional:
            logger.info("Calibrate strata from these percentiles before the full pool.")
    logger.info(f"Output: {out_dir}/{meta_name}")


if __name__ == "__main__":
    main()
