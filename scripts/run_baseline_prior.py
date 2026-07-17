#!/usr/bin/env python3
"""
Indirect-encoding baseline prior: the un-evolved generative-process comparison
(notes/baselines-design.md).

This is the matched-random control (run_matched_random_control.py) with the
graph GENERATOR swapped for an indirect-encoding baseline sampler. For each
competent MorphoNAS source network it generates k baseline graphs at the SAME
(neuron, edge) counts, evaluates them on the task under the pool's own protocol
(per-episode reset, argmax, no learning), and reports the competence-prevalence
ratio R = MN-rate / baseline-rate. The matched-random arm answers "better than
structureless wiring?"; this answers "better than the other leading INDIRECT
encoding (CPPN / HyperNEAT) at the same (N,E) budget?" -- the recurrence-
circularity closer (both arms are recurrence-capable, so the comparison is
generative-process vs generative-process).

Encodings (--encoding):
  * cppn_hyperneat -- a random CPPN over a coordinate substrate, top-E edges by
    |weight| (weakly-connected), realised weights ~ U[0.01,1]; (N,E)-matched per
    source. The primary arm.
  * neat_init      -- the minimal initial NEAT genome (a perceptron); a fixed,
    tiny architecture, NOT (N,E)-matched -- the feedforward-floor reference. Run
    it with --max-sources 1 --num-random N for N weight-randomised samples.

OUTPUT SCHEMA IS IDENTICAL to the matched-random control: results.jsonl
(num_neurons, num_connections, stratum, is_competent, ...) and summary_stats.json
(morphonas{...}, n_random_valid, random_nonweak_count, random_solved_count, ...),
so analyze_ratios.py and analyze_mantel_haenszel.py ingest the baseline arm with
NO changes (the baseline is structurally "another control arm").

Sharding (fleet): --num-shards S --shard-index I evaluates tasks[I::S]; each
graph is keyed by a fixed generation seed, so the union of shards equals the
unsharded run. Combine with merge_shards.py (same as the matched-random control).
"""

from __future__ import annotations

import os as _os

# Cap BLAS/OpenMP threads BEFORE numpy imports (W workers x N threads would
# oversubscribe a big Linux box). SDL dummy keeps headless pygame/Box2D off a
# display. Set in-process: a non-interactive ssh does not source the profile.
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

from MorphoNAS_DevPriors.baseline_encoders import (  # noqa: E402
    DEFAULT_ACTIVATIONS,
    sample_cppn_hyperneat,
    sample_neat_initial,
)
from MorphoNAS_DevPriors.logging_config import setup_logging  # noqa: E402
from MorphoNAS_DevPriors.progress import ProgressWriter  # noqa: E402
from MorphoNAS_DevPriors.ratio_stats import ratio_ci_mover  # noqa: E402
from MorphoNAS_DevPriors.task_registry import (  # noqa: E402
    SOLVED_STRATUM,
    STRATA_ORDER,
    get_task,
    run_rollouts,
)

logger = logging.getLogger(__name__)

ENCODINGS = ("cppn_hyperneat", "neat_init")

# ── Worker plumbing ─────────────────────────────────────────────────────────────

_W = {}


def _init_worker(task_name, rollouts, encoding, cppn_hidden, connectivity, weight_mode):
    from MorphoNAS_DevPriors.parallel_utils import configure_worker_threads

    configure_worker_threads()
    spec = get_task(task_name)
    _W["spec"] = spec
    _W["env"] = spec.make_env()
    _W["rollouts"] = int(rollouts)
    _W["encoding"] = encoding
    _W["cppn_hidden"] = int(cppn_hidden)
    _W["connectivity"] = connectivity
    _W["weight_mode"] = weight_mode


def _build_graph(spec, num_neurons, num_edges, rng):
    """Dispatch to the selected baseline sampler. Returns nx.DiGraph or None."""
    enc = _W["encoding"]
    if enc == "cppn_hyperneat":
        return sample_cppn_hyperneat(
            num_neurons, num_edges, spec.input_dim, spec.output_dim, rng,
            cppn_hidden=_W["cppn_hidden"], connectivity=_W["connectivity"],
            weight_mode=_W["weight_mode"],
        )
    if enc == "neat_init":
        return sample_neat_initial(spec.input_dim, spec.output_dim, rng)
    raise ValueError(f"unknown encoding: {enc}")


def _eval_one(task: dict) -> dict:
    spec = _W["spec"]
    req_neurons = task["num_neurons"]
    req_edges = task["num_connections"]
    gen_seed = task["random_seed"]
    eval_seeds = task["eval_seeds"]

    base = {
        "source_id": task["source_id"],
        "source_stratum": task["source_stratum"],
        "random_seed": gen_seed,
    }

    if req_neurons < spec.min_neurons:
        return {**base, "num_neurons": req_neurons, "num_connections": req_edges,
                "valid": False, "error": "insufficient_neurons"}

    rng = np.random.default_rng(gen_seed)
    G = _build_graph(spec, req_neurons, req_edges, rng)
    if G is None:
        return {**base, "num_neurons": req_neurons, "num_connections": req_edges,
                "valid": False, "error": "sampler_returned_none"}

    # Report the REALISED (N,E): equals requested for cppn_hyperneat by
    # construction, but is the perceptron's fixed (N,E) for neat_init. MH bins
    # on these, so they must be the actual graph stats.
    num_neurons = int(G.number_of_nodes())
    num_edges = int(G.number_of_edges())

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
        "num_neurons": num_neurons,
        "num_connections": num_edges,
        "valid": True,
        "baseline_reward": baseline_reward,
        "stratum": stratum,
        "is_competent": stratum != "weak",
        "is_solved": stratum == SOLVED_STRATUM,
        "rewards": res.get("rewards", []),
        "eval_seeds": eval_seeds[: _W["rollouts"]],
    }


# ── Source loading (identical to the matched-random control) ─────────────────────

def load_sources(pool_dir: str, min_stratum: str) -> list[dict]:
    networks_dir = os.path.join(pool_dir, "networks")
    min_idx = STRATA_ORDER.index(min_stratum)
    sources: list[dict] = []
    for fname in sorted(os.listdir(networks_dir)):
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
    parser = argparse.ArgumentParser(description="indirect-encoding baseline prior (matched (N,E), shardable)")
    parser.add_argument("--task", type=str, default="acrobot")
    parser.add_argument("--encoding", type=str, default="cppn_hyperneat", choices=ENCODINGS)
    parser.add_argument("--cppn-hidden", type=int, default=4, help="CPPN hidden nodes (frozen a priori)")
    parser.add_argument("--connectivity", type=str, default="recurrent", choices=["recurrent", "layered"])
    parser.add_argument("--weight-mode", type=str, default="uniform", choices=["uniform", "cppn"],
                        help="uniform = iid U[0.01,1] realised weights (topology-only contrast); "
                             "cppn = CPPN sets magnitudes (the signed-weight robustness arm)")
    parser.add_argument("--pool-dir", type=str, default=None, help="default: experiments/<task>/pool")
    parser.add_argument("--min-stratum", type=str, default="low_mid", choices=STRATA_ORDER)
    parser.add_argument("--num-random", type=int, default=5, help="baseline graphs per source (k)")
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=None,
                        help="gen seed = base + source_id*100 + k; default: task.control_base_seed + 900_000_000")
    parser.add_argument("--max-sources", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="default: experiments/baselines/<encoding>/<task>/run")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--max-workers", type=int, default=max(1, cpu_count() - 2))
    args = parser.parse_args()

    spec = get_task(args.task)
    pool_dir = args.pool_dir or f"experiments/{args.task}/pool"
    out_dir = args.output_dir or f"experiments/baselines/{args.encoding}/{args.task}/run"
    # Offset the baseline seed namespace well clear of the matched-random control's
    # (base + source*100 + k) so a baseline graph never collides with a control graph.
    base_seed = args.base_seed if args.base_seed is not None else spec.control_base_seed + 900_000_000
    os.makedirs(out_dir, exist_ok=True)
    setup_logging(log_dir=out_dir, log_file="baseline_prior.log")

    from MorphoNAS_DevPriors.parallel_utils import configure_worker_threads
    configure_worker_threads()

    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(f"--shard-index must be in [0, {args.num_shards})")

    mn = morphonas_rates(pool_dir)
    sources = load_sources(pool_dir, args.min_stratum)
    if args.max_sources > 0:
        sources = sources[: args.max_sources]

    tasks: list[dict] = []
    for s in sources:
        for k in range(args.num_random):
            tasks.append(
                {
                    "source_id": s["network_id"],
                    "source_stratum": s["stratum"],
                    "num_neurons": s["num_neurons"],
                    "num_connections": s["num_connections"],
                    "random_seed": base_seed + s["network_id"] * 100 + k,
                    "eval_seeds": s["eval_seeds"],
                }
            )

    n_total_all = len(tasks)
    if args.num_shards > 1:
        tasks = tasks[args.shard_index :: args.num_shards]

    suffix = f" [shard {args.shard_index}/{args.num_shards}]" if args.num_shards > 1 else ""
    logger.info("=" * 60)
    logger.info(f"BASELINE PRIOR ({args.encoding}) -- task={args.task}{suffix}")
    logger.info("=" * 60)
    logger.info(f"Env: {spec.env_name} ({spec.input_dim} in / {spec.output_dim} out)")
    logger.info(f"Pool: {pool_dir} | MN n={mn['n_total']} non-weak={mn['nonweak_count']} "
                f"({mn['nonweak_rate']*100:.3f}%) solved={mn['solved_count']} "
                f"({mn['solved_rate']*100:.3f}%)")
    if args.encoding == "cppn_hyperneat":
        logger.info(f"CPPN: hidden={args.cppn_hidden} acts={DEFAULT_ACTIVATIONS} "
                    f"connectivity={args.connectivity} | (N,E)-matched per source")
    logger.info(f"Sources >= {args.min_stratum}: {len(sources)} | k={args.num_random} | base_seed={base_seed}")
    logger.info(f"Graphs this shard: {len(tasks)} of {n_total_all} | workers={args.max_workers}")
    if spec.provisional:
        logger.info("NOTE: task strata are PROVISIONAL (pilot-calibrated).")
    logger.info("=" * 60)

    progress = ProgressWriter(
        os.path.join(out_dir, "progress.json"),
        total=len(tasks),
        phase="baseline_prior",
        meta={
            "task": args.task, "encoding": args.encoding, "min_stratum": args.min_stratum,
            "num_random": args.num_random, "cppn_hidden": args.cppn_hidden,
            "connectivity": args.connectivity, "shard_index": args.shard_index,
            "num_shards": args.num_shards, "n_total_all_shards": n_total_all,
        },
    )

    t0 = time.time()
    results: list[dict] = []
    results_path = os.path.join(out_dir, "results.jsonl")
    with open(results_path, "w") as rf, Pool(
        args.max_workers,
        initializer=_init_worker,
        initargs=(args.task, args.rollouts, args.encoding, args.cppn_hidden, args.connectivity, args.weight_mode),
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
        "experiment": "baseline_prior",
        "arm": args.encoding,
        "task": args.task,
        "env_name": spec.env_name,
        "engine": "vendored code/MorphoNAS (canonical ruleset + per-episode reset + W[post,pre] fix d378b8e06, no plasticity)",
        "encoding_config": {
            "encoding": args.encoding,
            "cppn_hidden": args.cppn_hidden,
            "activations": list(DEFAULT_ACTIVATIONS),
            "connectivity": args.connectivity,
            "weight_mode": args.weight_mode,
            "ne_matched": args.encoding == "cppn_hyperneat",
            "weight_range": [0.01, 1.0],
        },
        "shard": {"index": args.shard_index, "num_shards": args.num_shards,
                  "is_full_run": args.num_shards == 1},
        "config": {
            "min_stratum": args.min_stratum,
            "num_random_per_source": args.num_random,
            "rollouts": args.rollouts,
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

    # Ratio MN-vs-baseline only meaningful on a full (single-shard) run; merge
    # handles multi-shard (same as the matched-random control).
    if args.num_shards == 1:
        summary["ratio_nonweak"] = ratio_ci_mover(mn["nonweak_count"], mn["n_total"], competent, n_valid)
        summary["ratio_solved"] = ratio_ci_mover(mn["solved_count"], mn["n_total"], solved, n_valid)

    summary_name = ("summary_stats.json" if args.num_shards == 1
                    else f"summary_shard{args.shard_index:03d}.json")
    with open(os.path.join(out_dir, summary_name), "w") as f:
        json.dump(summary, f, indent=2)

    progress.done(valid=n_valid, competent=competent, solved=solved)
    r_pt = summary.get("ratio_nonweak", {}).get("point")
    r_str = f" | R_nonweak={r_pt:.2f}x" if r_pt is not None else ""
    logger.info(f"DONE{suffix}: valid={n_valid}/{len(tasks)} competent={competent} "
                f"solved={solved}{r_str} | {summary['elapsed_sec']:.1f}s -> {out_dir}/{summary_name}")


if __name__ == "__main__":
    main()
