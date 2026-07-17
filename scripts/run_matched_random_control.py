#!/usr/bin/env python3
"""
Matched-random control, generalized across the difficulty ladder + shardable
for the fleet. Serves three threads:

  * A (Acrobot divider bump): --task acrobot --min-stratum low_mid --num-random 50
  * D (symmetry / MH input):  --task acrobot --min-stratum weak --num-random 5
  * C (new rungs):            --task lunarlander|mountaincar --min-stratum low_mid

For each MorphoNAS source network at >= --min-stratum, generate k random directed
graphs with the SAME (neuron count, edge count), weights ~U[0.01, 1.0], require
weak connectivity, and evaluate them on the task under the pool's own protocol.
The matched-random competence rate vs the MorphoNAS rate gives that task's
developmental-prior ratio (the Acrobot analogue of CartPole's 8.4x).

This generalizes the locked ``run_acrobot_matched_random.py`` (kept untouched
for provenance). For ``--task acrobot`` the protocol, generator RNG order, and
statistics are identical, so it reproduces the published numbers.

Sharding (fleet): --num-shards S --shard-index I evaluates the strided slice
tasks[I::S], writing results.jsonl + progress.json into --output-dir. Each task's
graph is independent and keyed by a fixed generation seed, so the union of all
shards equals the unsharded run regardless of S. Combine with merge_shards.py.

Progress: writes <output-dir>/progress.json (atomic, fsync'd) and a heartbeat
JSONL every --progress-every graphs, with done/total/rate/ETA.
"""

from __future__ import annotations

import os as _os

# Cap BLAS/OpenMP threads BEFORE numpy imports so W workers x N_CPU threads can
# never oversubscribe on a big Linux box (the "fine on Mac, crawls on Ubuntu"
# trap). SDL dummy keeps headless pygame/Box2D from blocking on a display. These
# must be set in-process: a non-interactive `ssh host 'python ...'` does not
# source /etc/profile.d or .bashrc, so a shell export alone would not apply.
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
import time
from multiprocessing import Pool, cpu_count
from typing import Optional

import numpy as np

sys.path.append(os.path.abspath("code"))

from MorphoNAS_DevPriors.logging_config import setup_logging  # noqa: E402
from MorphoNAS_DevPriors.progress import ProgressWriter  # noqa: E402
from MorphoNAS_DevPriors.random_graph import generate_random_rnn  # noqa: E402
from MorphoNAS_DevPriors.ratio_stats import grade_vs_reference, ratio_ci_mover  # noqa: E402
from MorphoNAS_DevPriors.task_registry import (  # noqa: E402
    CARTPOLE_REF,
    SOLVED_STRATUM,
    STRATA_ORDER,
    get_task,
    run_rollouts,
)

logger = logging.getLogger(__name__)


# ── Worker plumbing ─────────────────────────────────────────────────────────────

_W = {}


def _init_worker(task_name, rollouts, weight_lo, weight_hi, max_retries):
    import gymnasium as gym

    from MorphoNAS_DevPriors.parallel_utils import configure_worker_threads

    configure_worker_threads()
    spec = get_task(task_name)
    _W["spec"] = spec
    _W["env"] = spec.make_env()
    _W["rollouts"] = int(rollouts)
    _W["weight_range"] = (float(weight_lo), float(weight_hi))
    _W["max_retries"] = int(max_retries)


def _eval_one(task: dict) -> dict:
    spec = _W["spec"]
    num_neurons = task["num_neurons"]
    num_edges = task["num_connections"]
    gen_seed = task["random_seed"]
    eval_seeds = task["eval_seeds"]

    base = {
        "source_id": task["source_id"],
        "source_stratum": task["source_stratum"],
        "random_seed": gen_seed,
        "num_neurons": num_neurons,
        "num_connections": num_edges,
    }

    if num_neurons < spec.min_neurons:
        return {**base, "valid": False, "error": "insufficient_neurons"}

    rng = np.random.default_rng(gen_seed)
    G = generate_random_rnn(
        num_neurons,
        num_edges,
        rng,
        weight_range=_W["weight_range"],
        max_retries=_W["max_retries"],
    )
    if G is None:
        return {**base, "valid": False, "error": "not_weakly_connected"}

    propagator = spec.make_propagator(G)
    res = run_rollouts(
        propagator,
        _W["rollouts"],
        seeds=eval_seeds[: _W["rollouts"]],
        reset_plastic_each_episode=True,
        env=_W["env"],
    )
    baseline_reward = float(res.get("avg_reward", 0.0))
    stratum = spec.get_stratum(baseline_reward)

    return {
        **base,
        "valid": True,
        "baseline_reward": baseline_reward,
        "stratum": stratum,
        "is_competent": stratum != "weak",
        "is_solved": stratum == SOLVED_STRATUM,
        "rewards": res.get("rewards", []),
        "eval_seeds": eval_seeds[: _W["rollouts"]],
    }


# ── Source loading ──────────────────────────────────────────────────────────────

def load_sources(pool_dir: str, min_stratum: str) -> list[dict]:
    networks_dir = os.path.join(pool_dir, "networks")
    min_idx = STRATA_ORDER.index(min_stratum)
    sources: list[dict] = []
    for fname in sorted(os.listdir(networks_dir)):
        # skip dotfiles, incl. macOS AppleDouble "._network_*.json" that BSD tar
        # injects (binary resource forks that are not valid JSON)
        if not fname.endswith(".json") or fname.startswith("."):
            continue
        with open(os.path.join(networks_dir, fname)) as f:
            d = json.load(f)
        if not d.get("valid", False):
            continue
        stratum = d.get("stratum", "weak")
        if STRATA_ORDER.index(stratum) < min_idx:
            continue
        stats = d.get("network_stats", {})
        sources.append(
            {
                "network_id": d["network_id"],
                "num_neurons": int(stats.get("neurons", 0)),
                "num_connections": int(stats.get("connections", 0)),
                "stratum": stratum,
                "eval_seeds": list(d.get("rollout_data", {}).get("eval_seeds", [])),
            }
        )
    return sources


def morphonas_rates(pool_dir: str) -> dict:
    with open(os.path.join(pool_dir, "pool_metadata.json")) as f:
        meta = json.load(f)
    counts = meta["stratum_counts"]
    total = sum(counts.values())
    nonweak = total - counts.get("weak", 0)
    solved = counts.get(SOLVED_STRATUM, 0)
    return {
        "n_total": total,
        "stratum_counts": counts,
        "nonweak_count": nonweak,
        "nonweak_rate": nonweak / total if total else 0.0,
        "solved_count": solved,
        "solved_rate": solved / total if total else 0.0,
    }


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="matched-random control (ladder + shard)")
    parser.add_argument("--task", type=str, default="acrobot")
    parser.add_argument("--pool-dir", type=str, default=None,
                        help="default: experiments/<task>/pool")
    parser.add_argument("--min-stratum", type=str, default="low_mid", choices=STRATA_ORDER)
    parser.add_argument("--num-random", type=int, default=5)
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--weight-lo", type=float, default=0.01)
    parser.add_argument("--weight-hi", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=None,
                        help="gen seed = base + source_id*100 + k; default: task.control_base_seed")
    parser.add_argument("--seed-mode", type=str, default="paired", choices=["paired", "fixed"])
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="default: experiments/<task>/matched_random/run")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-workers", type=int, default=max(1, cpu_count() - 2))
    args = parser.parse_args()

    spec = get_task(args.task)
    pool_dir = args.pool_dir or f"experiments/{args.task}/pool"
    out_dir = args.output_dir or f"experiments/{args.task}/matched_random/run"
    base_seed = args.base_seed if args.base_seed is not None else spec.control_base_seed
    os.makedirs(out_dir, exist_ok=True)
    setup_logging(log_dir=out_dir, log_file="matched_random.log")

    from MorphoNAS_DevPriors.parallel_utils import configure_worker_threads
    configure_worker_threads()

    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(f"--shard-index must be in [0, {args.num_shards})")

    mn = morphonas_rates(pool_dir)
    sources = load_sources(pool_dir, args.min_stratum)
    if args.max_sources > 0:
        sources = sources[: args.max_sources]

    if args.seed_mode == "fixed":
        from MorphoNAS_DevPriors.experiment_acrobot import VERIFICATION_SEEDS
        fixed_seeds = list(VERIFICATION_SEEDS)[: args.rollouts]

    tasks: list[dict] = []
    for s in sources:
        eval_seeds = fixed_seeds if args.seed_mode == "fixed" else s["eval_seeds"]
        for k in range(args.num_random):
            tasks.append(
                {
                    "source_id": s["network_id"],
                    "source_stratum": s["stratum"],
                    "num_neurons": s["num_neurons"],
                    "num_connections": s["num_connections"],
                    "random_seed": base_seed + s["network_id"] * 100 + k,
                    "eval_seeds": eval_seeds,
                }
            )

    n_total_all = len(tasks)
    if args.num_shards > 1:
        tasks = tasks[args.shard_index :: args.num_shards]

    suffix = f" [shard {args.shard_index}/{args.num_shards}]" if args.num_shards > 1 else ""
    logger.info("=" * 60)
    logger.info(f"MATCHED-RANDOM CONTROL -- task={args.task}{suffix}")
    logger.info("=" * 60)
    logger.info(f"Env: {spec.env_name} ({spec.input_dim} in / {spec.output_dim} out)")
    logger.info(f"Pool: {pool_dir} | MN n={mn['n_total']} non-weak={mn['nonweak_count']} "
                f"({mn['nonweak_rate']*100:.3f}%) solved={mn['solved_count']} "
                f"({mn['solved_rate']*100:.3f}%)")
    logger.info(f"Sources >= {args.min_stratum}: {len(sources)} | k={args.num_random} | "
                f"base_seed={base_seed} | seed_mode={args.seed_mode}")
    logger.info(f"Graphs this shard: {len(tasks)} of {n_total_all} | workers={args.max_workers}")
    if spec.provisional:
        logger.info("NOTE: task strata are PROVISIONAL (pilot-calibrated).")
    logger.info("=" * 60)

    progress = ProgressWriter(
        os.path.join(out_dir, "progress.json"),
        total=len(tasks),
        phase="control",
        meta={
            "task": args.task, "min_stratum": args.min_stratum,
            "num_random": args.num_random, "seed_mode": args.seed_mode,
            "shard_index": args.shard_index, "num_shards": args.num_shards,
            "n_total_all_shards": n_total_all,
        },
    )

    t0 = time.time()
    results: list[dict] = []
    results_path = os.path.join(out_dir, "results.jsonl")
    with open(results_path, "w") as rf, Pool(
        args.max_workers,
        initializer=_init_worker,
        initargs=(args.task, args.rollouts, args.weight_lo, args.weight_hi, args.max_retries),
    ) as pool:
        for i, r in enumerate(pool.imap_unordered(_eval_one, tasks, chunksize=4), start=1):
            results.append(r)
            rf.write(json.dumps(r) + "\n")
            if i % args.progress_every == 0 or i == len(tasks):
                rf.flush()
                os.fsync(rf.fileno())
                valid = sum(1 for x in results if x.get("valid"))
                comp = sum(1 for x in results if x.get("is_competent"))
                solv = sum(1 for x in results if x.get("is_solved"))
                progress.update(i, valid=valid, competent=comp, solved=solv)
                rate = (time.time() - t0) / i
                logger.info(f"[{i}/{len(tasks)}] valid={valid} competent={comp} solved={solv} "
                            f"| {rate:.3f}s/graph ETA {rate*(len(tasks)-i)/60:.1f} min")

    # ── Aggregate (this shard; full run when num_shards == 1) ──
    valid = [r for r in results if r.get("valid")]
    n_valid = len(valid)
    strata = {s: 0 for s in STRATA_ORDER}
    for r in valid:
        strata[r["stratum"]] += 1
    competent = sum(1 for r in valid if r["is_competent"])
    solved = sum(1 for r in valid if r["is_solved"])

    summary = {
        "experiment": "matched_random_control",
        "task": args.task,
        "env_name": spec.env_name,
        "engine": "vendored code/MorphoNAS (rules == MorphoNAS-work@v1, commit 72902afc)",
        "shard": {"index": args.shard_index, "num_shards": args.num_shards,
                  "is_full_run": args.num_shards == 1},
        "config": {
            "min_stratum": args.min_stratum,
            "num_random_per_source": args.num_random,
            "rollouts": args.rollouts,
            "weight_range": [args.weight_lo, args.weight_hi],
            "max_retries": args.max_retries,
            "seed_mode": args.seed_mode,
            "base_seed": base_seed,
            "n_sources": len(sources),
            "strata_provisional": spec.provisional,
        },
        "n_random_total": len(tasks),
        "n_random_valid": n_valid,
        "valid_rate": n_valid / len(tasks) if tasks else 0.0,
        "strata": strata,
        "random_nonweak_count": competent,
        "random_solved_count": solved,
        "morphonas": mn,
        "elapsed_sec": time.time() - t0,
    }

    # Ratios + verdict only meaningful on a full (single-shard) run; merge does
    # this for multi-shard. Compute here when single shard for back-compat.
    if args.num_shards == 1:
        acro = {
            "nonweak": ratio_ci_mover(mn["nonweak_count"], mn["n_total"], competent, n_valid),
            "solved": ratio_ci_mover(mn["solved_count"], mn["n_total"], solved, n_valid),
        }
        cart = {bar: ratio_ci_mover(c["mn_count"], c["mn_n"], c["rand_count"], c["rand_n"])
                for bar, c in CARTPOLE_REF.items()}
        summary["ratio_nonweak"] = acro["nonweak"]
        summary["ratio_solved"] = acro["solved"]
        summary["cartpole_reference"] = cart
        summary["verdict"] = {
            bar: grade_vs_reference(acro[bar]["lo"], acro[bar]["hi"],
                                    cart[bar]["point"], cart[bar]["lo"], cart[bar]["hi"])
            for bar in ("nonweak", "solved")
        }

    summary_name = ("summary_stats.json" if args.num_shards == 1
                    else f"summary_shard{args.shard_index:03d}.json")
    with open(os.path.join(out_dir, summary_name), "w") as f:
        json.dump(summary, f, indent=2)

    progress.done(valid=n_valid, competent=competent, solved=solved)
    logger.info(f"DONE{suffix}: valid={n_valid}/{len(tasks)} competent={competent} "
                f"solved={solved} | {summary['elapsed_sec']:.1f}s -> {out_dir}/{summary_name}")


if __name__ == "__main__":
    main()
