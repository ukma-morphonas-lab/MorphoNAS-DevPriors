#!/usr/bin/env python3
"""
RWG axis: the architecture-independent task-difficulty classifier.

For each task, sample N random-weight networks of ONE FIXED reference architecture
(independent of MorphoNAS shapes), evaluate them on the task under the shared
protocol (per-episode reset, argmax over outputs, no plasticity), and report the
fraction that clear the non-weak / solved bars. The structure axis is
  r = 1 / non-weak-fraction.

Why a fixed architecture (not MN-matched (N,E) shapes): an MN-matched random net IS
the matched-random denominator by another name, so r would be circular with R. The
fixed architecture breaks that. A residual coupling still survives (R and 1/RWG are
both random-wiring-difficulty measures, so they co-vary across tasks by
construction) -- which is exactly why this axis is treated as a
difficulty-ORDERING robustness check, not as independent proof the prior is special
(see reach-map.md). The robustness check is satisfied by re-running
with a different --hidden (e.g. 4, 16) and --topology (complete vs feedforward) and
checking the cross-task rank-order of r is preserved.

Reference architectures (sized to the task I/O; K = input_dim + output_dim + hidden):
  * complete    -- fully-connected recurrent digraph, all K*(K-1) directed edges,
                   weights ~ U[weight_lo, weight_hi]. The default.
  * feedforward -- a layered DAG input -> hidden -> output, adjacent layers fully
                   connected, weights ~ U[weight_lo, weight_hi]. For the robustness
                   check (feedforward vs recurrent).
NeuralPropagator picks the input_dim lowest-in-degree nodes as inputs and the last
output_dim nodes (by id) as outputs, so both architectures place I/O correctly.

All samples are scored on the SAME fixed set of episode seeds (a benchmark-difficulty
measure: what fraction of random nets clear the bar on the same starts).

Progress: writes progress.json (atomic, fsync'd) + a heartbeat JSONL with
done/total/rate/ETA, and logs a live [i/N] line with competent/solved counts and
s/net + ETA every --progress-every nets, so a watcher can compute ETA and spot a
stalled or too-slow run.
"""

from __future__ import annotations

import os as _os

# Cap BLAS/OpenMP threads BEFORE numpy imports so W workers x N_CPU threads can
# never oversubscribe on a big multi-core box. SDL dummy keeps headless
# pygame/Box2D from blocking on a display. Set in-process (a non-interactive shell
# would not source the profile).
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

import networkx as nx
import numpy as np

sys.path.append(os.path.abspath("code"))

from MorphoNAS_DevPriors.logging_config import setup_logging  # noqa: E402
from MorphoNAS_DevPriors.progress import ProgressWriter  # noqa: E402
from MorphoNAS_DevPriors.random_graph import generate_random_rnn  # noqa: E402
from MorphoNAS_DevPriors.ratio_stats import wilson_ci  # noqa: E402
from MorphoNAS_DevPriors.task_registry import (  # noqa: E402
    SOLVED_STRATUM,
    STRATA_ORDER,
    get_task,
    run_rollouts,
)

logger = logging.getLogger(__name__)

_W = {}


def build_reference_graph(topology, input_dim, output_dim, hidden, rng, weight_range):
    """One fixed reference architecture with random weights for this sample."""
    K = input_dim + output_dim + hidden
    if topology == "complete":
        # complete digraph: pass the full edge count so generate_random_rnn fixes the
        # topology (all K*(K-1) edges) and randomises only the weights.
        return generate_random_rnn(K, K * (K - 1), rng, weight_range=weight_range)
    if topology == "feedforward":
        G = nx.DiGraph()
        G.add_nodes_from(range(K))
        inp = range(0, input_dim)
        hid = range(input_dim, input_dim + hidden)
        out = range(input_dim + hidden, K)
        lo, hi = weight_range
        for layer_from, layer_to in ((inp, hid), (hid, out)):
            for i in layer_from:
                for j in layer_to:
                    G.add_edge(i, j, weight=float(rng.uniform(lo, hi)))
        return G
    raise ValueError(f"unknown topology: {topology}")


def _init_worker(task_name, topology, hidden, rollouts, weight_lo, weight_hi,
                 eval_seeds, max_retries):
    from MorphoNAS_DevPriors.parallel_utils import configure_worker_threads

    configure_worker_threads()
    spec = get_task(task_name)
    _W["spec"] = spec
    _W["env"] = spec.make_env()
    _W["topology"] = topology
    _W["hidden"] = int(hidden)
    _W["rollouts"] = int(rollouts)
    _W["weight_range"] = (float(weight_lo), float(weight_hi))
    _W["eval_seeds"] = list(eval_seeds)
    _W["max_retries"] = int(max_retries)


def _eval_one(args) -> dict:
    sample_idx, gen_seed = args
    spec = _W["spec"]
    rng = np.random.default_rng(gen_seed)
    G = build_reference_graph(_W["topology"], spec.input_dim, spec.output_dim,
                              _W["hidden"], rng, _W["weight_range"])
    if G is None:
        return {"sample_idx": sample_idx, "random_seed": gen_seed, "valid": False,
                "error": "not_weakly_connected"}

    res = run_rollouts(spec.make_propagator(G), _W["rollouts"], seeds=_W["eval_seeds"],
                       reset_plastic_each_episode=True, env=_W["env"])
    mean_reward = float(res.get("avg_reward", 0.0))
    stratum = spec.get_stratum(mean_reward)
    return {
        "sample_idx": sample_idx,
        "random_seed": gen_seed,
        "valid": True,
        "mean_reward": mean_reward,
        "stratum": stratum,
        "is_competent": stratum != "weak",
        "is_solved": stratum == SOLVED_STRATUM,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="RWG axis (fixed-architecture task-difficulty classifier)")
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--num-samples", type=int, default=5000)
    p.add_argument("--hidden", type=int, default=8,
                   help="hidden neurons; K = input_dim + output_dim + hidden")
    p.add_argument("--topology", type=str, default="complete", choices=["complete", "feedforward"])
    p.add_argument("--rollouts", type=int, default=20)
    p.add_argument("--weight-lo", type=float, default=0.01)
    p.add_argument("--weight-hi", type=float, default=1.0)
    p.add_argument("--base-seed", type=int, default=20_000_000,
                   help="per-sample weight seed = base + sample_idx")
    p.add_argument("--eval-seed-base", type=int, default=42,
                   help="fixed episode seeds shared across all samples: range(base, base+rollouts)")
    p.add_argument("--max-retries", type=int, default=100)
    p.add_argument("--output-dir", type=str, default=None,
                   help="default: experiments/reach_map/rwg/<task>__<topology>_h<hidden>")
    p.add_argument("--max-workers", type=int, default=max(1, cpu_count() - 2))
    p.add_argument("--progress-every", type=int, default=200)
    args = p.parse_args()

    spec = get_task(args.task)
    K = spec.input_dim + spec.output_dim + args.hidden
    out_dir = args.output_dir or f"experiments/reach_map/rwg/{args.task}__{args.topology}_h{args.hidden}"
    os.makedirs(out_dir, exist_ok=True)
    setup_logging(log_dir=out_dir, log_file="rwg_axis.log")

    from MorphoNAS_DevPriors.parallel_utils import configure_worker_threads
    configure_worker_threads()

    eval_seeds = list(range(args.eval_seed_base, args.eval_seed_base + args.rollouts))
    tasks = [(i, args.base_seed + i) for i in range(args.num_samples)]

    logger.info("=" * 60)
    logger.info(f"RWG AXIS -- task={args.task}")
    logger.info("=" * 60)
    logger.info(f"Env: {spec.env_name} ({spec.input_dim} in / {spec.output_dim} out)")
    logger.info(f"Reference arch: {args.topology}, K={K} (hidden={args.hidden}), "
                f"weights ~ U[{args.weight_lo}, {args.weight_hi}]")
    logger.info(f"Samples: {args.num_samples} | rollouts/net: {args.rollouts} "
                f"(fixed seeds {eval_seeds[0]}..{eval_seeds[-1]}) | workers: {args.max_workers}")
    if spec.provisional:
        logger.info("NOTE: task strata are PROVISIONAL -- the non-weak bar must be calibrated "
                    "(same bar as the cell's R) before r is comparable.")
    logger.info("=" * 60)

    progress = ProgressWriter(
        os.path.join(out_dir, "progress.json"),
        total=len(tasks),
        phase="rwg_axis",
        meta={"task": args.task, "topology": args.topology, "hidden": args.hidden,
              "K": K, "num_samples": args.num_samples, "rollouts": args.rollouts,
              "weight_range": [args.weight_lo, args.weight_hi],
              "provisional_strata": spec.provisional},
    )

    t0 = time.time()
    results: list[dict] = []
    results_path = os.path.join(out_dir, "results.jsonl")
    with open(results_path, "w") as rf, Pool(
        args.max_workers,
        initializer=_init_worker,
        initargs=(args.task, args.topology, args.hidden, args.rollouts,
                  args.weight_lo, args.weight_hi, eval_seeds, args.max_retries),
    ) as pool:
        for i, r in enumerate(pool.imap_unordered(_eval_one, tasks, chunksize=8), start=1):
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
                            f"| {rate:.4f}s/net ETA {rate*(len(tasks)-i)/60:.1f} min")

    valid = [r for r in results if r.get("valid")]
    n_valid = len(valid)
    strata = {s: 0 for s in STRATA_ORDER}
    for r in valid:
        strata[r["stratum"]] += 1
    competent = sum(1 for r in valid if r["is_competent"])
    solved = sum(1 for r in valid if r["is_solved"])
    rewards = np.array([r["mean_reward"] for r in valid]) if valid else np.array([])

    nonweak_rate = competent / n_valid if n_valid else 0.0
    solved_rate = solved / n_valid if n_valid else 0.0
    # r = 1 / non-weak-fraction (the structure axis); inf if no random net clears it.
    r_axis = (1.0 / nonweak_rate) if nonweak_rate > 0 else float("inf")

    summary = {
        "experiment": "rwg_axis",
        "task": args.task,
        "env_name": spec.env_name,
        "reference_arch": {"topology": args.topology, "hidden": args.hidden, "K": K,
                           "weight_range": [args.weight_lo, args.weight_hi]},
        "num_samples": args.num_samples,
        "n_valid": n_valid,
        "rollouts": args.rollouts,
        "eval_seeds": eval_seeds,
        "base_seed": args.base_seed,
        "strata": strata,
        "stratum_bounds": spec.stratum_bounds_dict(),
        "strata_provisional": spec.provisional,
        "nonweak_count": competent,
        "nonweak_rate": nonweak_rate,
        "nonweak_ci": list(wilson_ci(competent, n_valid)) if n_valid else None,
        "solved_count": solved,
        "solved_rate": solved_rate,
        "r_axis_inv_nonweak": r_axis,
        "reward_summary": ({
            "mean": float(rewards.mean()), "std": float(rewards.std()),
            "median": float(np.median(rewards)), "min": float(rewards.min()),
            "max": float(rewards.max()),
            "percentiles": {str(q): float(np.percentile(rewards, q))
                            for q in [1, 5, 10, 25, 50, 75, 90, 95, 99]},
        } if valid else {}),
        "elapsed_sec": time.time() - t0,
    }
    with open(os.path.join(out_dir, "summary_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)

    progress.done(valid=n_valid, competent=competent, solved=solved)
    logger.info("=" * 60)
    logger.info(f"RWG DONE: task={args.task} valid={n_valid} "
                f"non-weak={competent} ({nonweak_rate*100:.3f}%) "
                f"solved={solved} ({solved_rate*100:.3f}%) | r=1/nonweak={r_axis:.1f}")
    if valid:
        rs = summary["reward_summary"]
        logger.info(f"reward: mean={rs['mean']:.2f} median={rs['median']:.2f} "
                    f"range=[{rs['min']:.1f},{rs['max']:.1f}] "
                    f"p90={rs['percentiles']['90']:.2f} p99={rs['percentiles']['99']:.2f}")
    logger.info(f"Output: {out_dir}/summary_stats.json")


if __name__ == "__main__":
    main()
