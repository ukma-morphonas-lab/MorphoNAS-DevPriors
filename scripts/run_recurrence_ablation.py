#!/usr/bin/env python3
"""
Recurrence ablation for the memory (masked-POMDP) rungs.

Question: is the grown network's masked-task competence carried by cross-step
recurrence? Hold the genome fixed (regrow the exact grown wiring), remove the
network's memory in one of two ways, and re-evaluate with the SAME eval seeds and
the SAME (masked) env the pool used:

  --recurrence-mode baseline      grown net, unchanged. MUST reproduce the pool's
                                  competence rate byte-for-byte (the faithfulness
                                  check for this self-contained rollout loop).
  --recurrence-mode state_reset   reset the propagator's hidden state BEFORE every
                                  env step, so each action is a memoryless
                                  obs -> action map (the two within-step thinking
                                  sub-steps remain; nothing integrates across env
                                  steps). The decisive memory ablation.
  --recurrence-mode edge_removal  greedily break directed cycles (feedback-arc set)
                                  so the wiring is a DAG, then evaluate normally.
                                  A structural confirm; secondary to state_reset.

If masked competence collapses toward the feedforward floor under state_reset, the
masked advantage is recurrence-borne -- it closes the memory-mediation gap directly
(the fixed feedforward reference is a between-architecture proxy; this is a
within-net ablation on the exact grown wiring).

Mirrors run_weight_ablation.py (source loading, sharding, progress, Pool). The
rollout loop is a verbatim copy of experiment_acrobot.run_rollouts with a single
added, flag-gated per-step reset, so the shared/locked eval file is left untouched
and `baseline` mode reproduces the committed masked numbers exactly.
"""

from __future__ import annotations

import os as _os

# Cap BLAS/OpenMP threads BEFORE numpy imports so W workers x N_CPU threads never
# oversubscribe on a big Linux box; SDL dummy keeps headless pygame from blocking.
# Set in-process: a non-interactive ssh command does not source profile scripts.
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
from MorphoNAS_DevPriors.task_registry import SOLVED_STRATUM, STRATA_ORDER, get_task  # noqa: E402

logger = logging.getLogger(__name__)

_W = {}


def _remove_recurrent_edges(G):
    """Break every directed cycle by removing DFS back-edges under a fixed node
    order -- one pass, O(V+E), deterministic. Result is a DAG (feedforward wiring)
    with the same nodes; a valid feedback-arc set is removed. Nets are <=400 nodes
    so DFS depth is bounded well within the raised recursion limit."""
    import sys as _sys

    H = G.copy()
    _sys.setrecursionlimit(max(10000, 8 * H.number_of_nodes()))
    color = {n: 0 for n in H.nodes()}   # 0 unvisited, 1 on-stack, 2 done
    back = []

    def dfs(u):
        color[u] = 1
        for v in H.successors(u):
            if color[v] == 0:
                dfs(v)
            elif color[v] == 1:
                back.append((u, v))   # back-edge -> closes a cycle
        color[u] = 2

    for s in list(H.nodes()):
        if color[s] == 0:
            dfs(s)
    H.remove_edges_from(back)
    return H


def _run_rollouts_recur(propagator, num_rollouts, seeds, env, reset_state_each_step):
    """Verbatim copy of experiment_acrobot.run_rollouts' episode/step loop, with a
    single flag-gated per-step reset added. Env is passed in already wrapped."""
    rewards: list[float] = []
    for i in range(int(num_rollouts)):
        propagator.reset()  # canonical per-episode reset (unchanged)
        if seeds is not None and i < len(seeds):
            observation, _ = env.reset(seed=int(seeds[i]))
        else:
            observation, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            obs = np.array(observation).flatten()
            if reset_state_each_step:
                propagator.reset()  # remove cross-step memory (the ablation)
            propagator.propagate(obs)
            action = int(propagator.get_output().argmax().item())
            observation, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)
        rewards.append(float(total_reward))
    return {"avg_reward": float(np.mean(rewards)) if rewards else 0.0}


def _init_worker(task_name, rollouts, mode):
    from MorphoNAS_DevPriors.parallel_utils import configure_worker_threads

    configure_worker_threads()
    spec = get_task(task_name)
    _W["spec"] = spec
    # make_env() applies the task's env_wrapper (the velocity/angular-velocity mask
    # for the masked POMDP rungs) -- a bare gym.make(env_name) would silently
    # evaluate the UNMASKED task and produce wrong numbers.
    _W["env"] = spec.make_env()
    _W["rollouts"] = int(rollouts)
    _W["mode"] = mode


def _eval_source(task: dict) -> dict:
    spec = _W["spec"]
    mode = _W["mode"]
    n, e = task["num_neurons"], task["num_connections"]
    base = {"source_id": task["source_id"], "source_stratum": task["source_stratum"],
            "num_neurons": n, "num_connections": e, "recurrence_mode": mode}
    if n < spec.min_neurons:
        return {**base, "valid": False, "error": "insufficient_neurons"}

    with open(task["genome_path"]) as f:
        genome = Genome.from_dict(json.load(f)["genome"])
    grid = Grid(genome)
    grid.run_simulation(verbose=False)  # deterministic regrow of the exact wiring
    base_G = grid.get_graph()
    G = base_G

    if mode == "edge_removal":
        G = _remove_recurrent_edges(base_G)
    elif mode == "random_edge_removal":
        # capacity control: drop the SAME NUMBER of edges edge_removal drops, but
        # chosen at random -- isolates "lost recurrence" from "lost edges".
        f_removed = base_G.number_of_edges() - _remove_recurrent_edges(base_G).number_of_edges()
        G = base_G.copy()
        if f_removed > 0 and base_G.number_of_edges() > 0:
            edges = list(base_G.edges())
            rng = np.random.default_rng(30_000_000 + int(task["source_id"]))
            drop = rng.choice(len(edges), size=min(f_removed, len(edges)), replace=False)
            G.remove_edges_from([edges[i] for i in drop])
    reset_each_step = (mode == "state_reset")

    eval_seeds = task["eval_seeds"][: _W["rollouts"]]
    res = _run_rollouts_recur(spec.make_propagator(G), len(eval_seeds), eval_seeds,
                              _W["env"], reset_each_step)
    reward = float(res["avg_reward"])
    stratum = spec.get_stratum(reward)
    return {**base, "valid": True, "baseline_reward": reward, "stratum": stratum,
            "is_competent": stratum != "weak", "is_solved": stratum == SOLVED_STRATUM}


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
        srcs.append({"source_id": d["network_id"], "source_stratum": d["stratum"],
                     "num_neurons": int(st.get("neurons", 0)),
                     "num_connections": int(st.get("connections", 0)),
                     "genome_path": path,
                     "eval_seeds": list(d.get("rollout_data", {}).get("eval_seeds", []))})
    return srcs


def main() -> None:
    p = argparse.ArgumentParser(description="recurrence ablation (one mode per run)")
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--pool-dir", type=str, default=None)
    p.add_argument("--min-stratum", type=str, default="weak", choices=STRATA_ORDER,
                   help="weak = the full pool (the paper's competence denominator)")
    p.add_argument("--recurrence-mode", type=str, required=True,
                   choices=["baseline", "state_reset", "edge_removal", "random_edge_removal"])
    p.add_argument("--rollouts", type=int, default=20)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--max-workers", type=int, default=max(1, cpu_count() - 2))
    args = p.parse_args()

    spec = get_task(args.task)
    pool_dir = args.pool_dir or f"experiments/{args.task}/pool"
    out_dir = args.output_dir or f"experiments/{args.task}/recurrence_ablation/{args.recurrence_mode}"
    os.makedirs(out_dir, exist_ok=True)
    setup_logging(log_dir=out_dir, log_file="recurrence_ablation.log")

    if not (0 <= args.shard_index < args.num_shards):
        raise SystemExit(f"--shard-index must be in [0, {args.num_shards})")

    from MorphoNAS_DevPriors.parallel_utils import configure_worker_threads
    configure_worker_threads()

    sources = load_sources(pool_dir, args.min_stratum)
    if args.num_shards > 1:
        sources = sources[args.shard_index :: args.num_shards]
    suffix = f" [shard {args.shard_index}/{args.num_shards}]" if args.num_shards > 1 else ""
    logger.info(f"recurrence-ablation mode={args.recurrence_mode} task={args.task}{suffix} | "
                f"sources={len(sources)} rollouts={args.rollouts}")

    progress = ProgressWriter(os.path.join(out_dir, "progress.json"), total=len(sources),
                              phase=f"recurrence_ablation:{args.recurrence_mode}",
                              meta={"task": args.task, "mode": args.recurrence_mode,
                                    "shard_index": args.shard_index, "num_shards": args.num_shards})

    t0 = time.time()
    results: list[dict] = []
    res_path = os.path.join(out_dir, "results.jsonl")
    with open(res_path, "w") as rf, Pool(
        args.max_workers, initializer=_init_worker,
        initargs=(args.task, args.rollouts, args.recurrence_mode)) as pool:
        for i, r in enumerate(pool.imap_unordered(_eval_source, sources, chunksize=4), start=1):
            results.append(r)
            rf.write(json.dumps(r) + "\n")
            if i % 50 == 0 or i == len(sources):
                rf.flush()
                os.fsync(rf.fileno())
                valid = [x for x in results if x.get("valid")]
                comp = sum(1 for x in valid if x.get("is_competent"))
                progress.update(i, valid=len(valid), competent=comp)
                rate = (time.time() - t0) / i
                logger.info(f"[{i}/{len(sources)}] valid={len(valid)} competent={comp} "
                            f"({100 * comp / len(valid) if valid else 0:.2f}%) | {rate:.2f}s/src "
                            f"ETA {rate * (len(sources) - i) / 60:.1f} min")

    valid = [x for x in results if x.get("valid")]
    comp = sum(1 for x in valid if x.get("is_competent"))
    solv = sum(1 for x in valid if x.get("is_solved"))
    summary = {
        "experiment": "recurrence_ablation", "task": args.task,
        "recurrence_mode": args.recurrence_mode,
        "shard": {"index": args.shard_index, "num_shards": args.num_shards,
                  "is_full_run": args.num_shards == 1},
        "config": {"min_stratum": args.min_stratum, "rollouts": args.rollouts,
                   "n_sources": len(sources)},
        "n_valid": len(valid), "competent_count": comp,
        "competent_rate": comp / len(valid) if valid else 0.0,
        "solved_count": solv, "elapsed_sec": time.time() - t0,
    }
    sname = ("summary_stats.json" if args.num_shards == 1
             else f"summary_shard{args.shard_index:03d}.json")
    with open(os.path.join(out_dir, sname), "w") as f:
        json.dump(summary, f, indent=2)
    progress.done(valid=len(valid), competent=comp)
    logger.info(f"recurrence-ablation DONE mode={args.recurrence_mode} task={args.task}{suffix}: "
                f"valid={len(valid)} competent={comp} "
                f"({summary['competent_rate'] * 100:.3f}%) -> {out_dir}/{sname}")


if __name__ == "__main__":
    main()
