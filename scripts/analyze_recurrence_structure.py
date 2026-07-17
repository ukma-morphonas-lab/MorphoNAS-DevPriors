#!/usr/bin/env python3
"""
Structural recurrence measure (offline: no gym, no rollouts).

Is grown wiring more recurrent than (N,E)-matched random, before any behaviour?
This turns the paper's "recurrent-dynamics prior" from an outcome-inferred label
into a directly measured topological fact. For each grown net (regrow the genome ->
graph) and, at the SAME (neurons, edges), m freshly sampled matched-random graphs
(the identical generator the competence control uses), compute topology-only
recurrence statistics:

  * largest_scc_frac  -- size of the largest strongly-connected component / N. A
                         value > 1/N means a directed cycle exists; the fraction is
                         the share of neurons inside one mutually-reachable core.
  * has_cycle         -- largest SCC size > 1 (the network contains recurrence).
  * spectral_radius   -- max |eigenvalue| of the adjacency. Reported on the BINARY
                         adjacency (pure topological loop gain, the fair grown-vs-
                         random measure since grown weights ~0.125 are smaller than
                         random U[0.01,1]) and on the weighted adjacency (as-evaluated).

Headline: the (neurons, edges)-stratified Mantel-Haenszel OR of has_cycle
(grown vs random) -- the structural analogue of the competence odds ratio -- plus
median largest_scc_frac and spectral radius for each arm.

One task per invocation; matched-random is generated per grown source, so the
comparison is (N,E)-controlled by construction and needs no results.jsonl.
"""

from __future__ import annotations

import os as _os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

import argparse
import json
import logging
import os
import statistics
import sys
import time
from multiprocessing import Pool, cpu_count

import networkx as nx
import numpy as np

sys.path.append(os.path.abspath("code"))

from MorphoNAS.genome import Genome  # noqa: E402
from MorphoNAS.grid import Grid  # noqa: E402
from MorphoNAS_DevPriors.logging_config import setup_logging  # noqa: E402
from MorphoNAS_DevPriors.progress import ProgressWriter  # noqa: E402
from MorphoNAS_DevPriors.random_graph import generate_random_rnn  # noqa: E402
from MorphoNAS_DevPriors.task_registry import STRATA_ORDER, get_task  # noqa: E402

logger = logging.getLogger(__name__)

_W = {}


def _recurrence_stats(G) -> dict:
    n = G.number_of_nodes()
    if n == 0:
        return {"n": 0, "largest_scc": 0, "largest_scc_frac": 0.0,
                "frac_in_cycle": 0.0, "has_cycle": False,
                "spectral_radius_bin": 0.0, "spectral_radius_w": 0.0}
    scc_sizes = [len(c) for c in nx.strongly_connected_components(G)]
    largest = max(scc_sizes) if scc_sizes else 0
    in_cycle = sum(s for s in scc_sizes if s > 1)
    Ab = nx.to_numpy_array(G, weight=None)
    Aw = nx.to_numpy_array(G, weight="weight")
    sr_bin = float(np.max(np.abs(np.linalg.eigvals(Ab)))) if Ab.size else 0.0
    sr_w = float(np.max(np.abs(np.linalg.eigvals(Aw)))) if Aw.size else 0.0
    return {"n": int(n), "largest_scc": int(largest),
            "largest_scc_frac": largest / n, "frac_in_cycle": in_cycle / n,
            "has_cycle": bool(largest > 1),
            "spectral_radius_bin": sr_bin, "spectral_radius_w": sr_w}


def _init_worker(task_name, m, weight_lo, weight_hi, max_retries, gen_base):
    from MorphoNAS_DevPriors.parallel_utils import configure_worker_threads

    configure_worker_threads()
    _W["spec"] = get_task(task_name)
    _W["m"] = int(m)
    _W["wr"] = (float(weight_lo), float(weight_hi))
    _W["max_retries"] = int(max_retries)
    _W["gen_base"] = int(gen_base)


def _process_source(task: dict) -> dict:
    spec = _W["spec"]
    n, e = task["num_neurons"], task["num_connections"]
    out = {"source_id": task["source_id"], "num_neurons": n, "num_connections": e,
           "stratum": task["stratum"], "is_competent": task["stratum"] != "weak"}
    if n < spec.min_neurons:
        return {**out, "valid": False}

    with open(task["genome_path"]) as f:
        genome = Genome.from_dict(json.load(f)["genome"])
    grid = Grid(genome)
    grid.run_simulation(verbose=False)
    grown = _recurrence_stats(grid.get_graph())

    rand_stats = []
    for j in range(_W["m"]):
        rng = np.random.default_rng(_W["gen_base"] + task["source_id"] * 100 + j)
        Gr = generate_random_rnn(n, e, rng, weight_range=_W["wr"],
                                 max_retries=_W["max_retries"])
        if Gr is not None:
            rand_stats.append(_recurrence_stats(Gr))
    return {**out, "valid": True, "grown": grown, "random": rand_stats}


def load_sources(pool_dir):
    nd = os.path.join(pool_dir, "networks")
    srcs = []
    for fn in sorted(os.listdir(nd)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(nd, fn)
        with open(path) as f:
            d = json.load(f)
        if not d.get("valid"):
            continue
        st = d.get("network_stats", {})
        srcs.append({"source_id": d["network_id"], "stratum": d.get("stratum", "weak"),
                     "num_neurons": int(st.get("neurons", 0)),
                     "num_connections": int(st.get("connections", 0)),
                     "genome_path": path})
    return srcs


def _mh_or(strata: dict) -> float:
    """Mantel-Haenszel OR of has_cycle, grown vs random, over (n,e) strata.
    strata[(n,e)] = [a, b, c, d] = grown-cycle, grown-nocycle, rand-cycle, rand-nocycle."""
    num = den = 0.0
    for a, b, c, d in strata.values():
        nk = a + b + c + d
        if nk == 0:
            continue
        num += a * d / nk
        den += b * c / nk
    if den == 0:
        return float("inf") if num > 0 else float("nan")
    return num / den


def _med(xs):
    """Median, or None for an empty sample (never 0.0 -- that reads as a real value)."""
    return float(statistics.median(xs)) if xs else None


def _rate(num, den):
    """Proportion, or None when the denominator is empty (0/0 must not render as 0.0)."""
    return num / den if den else None


def _fmt(x, nd=3):
    """None-safe float formatting for the log line."""
    return "n/a" if x is None else f"{x:.{nd}f}"


def _aggregate(rows: list[dict], task: str, num_random: int, elapsed: float) -> dict:
    """Build summary_stats from per-source rows (shared by the live run and --from-per-source).

    The competence cut (structure of the COMPETENT grown tail) is emitted here rather
    than hand-computed from per_source.jsonl: it is the campaign's headline.
    """
    valid = [r for r in rows if r.get("valid")]
    strata: dict = {}                      # (n,e) -> [a,b,c,d] for has_cycle MH
    grown_scc_frac, grown_srb, grown_srw = [], [], []
    rand_scc_frac, rand_srb, rand_srw = [], [], []
    grown_comp_scc, grown_comp_srb = [], []
    n_grown = n_rand = 0
    grown_cycle = rand_cycle = 0
    grown_comp_cycle = grown_comp = 0

    for r in valid:
        n, e = r["num_neurons"], r["num_connections"]
        key = (n, e)
        a, b, c, d = strata.get(key, [0, 0, 0, 0])
        gc = r["grown"]["has_cycle"]
        a += int(gc)
        b += int(not gc)
        n_grown += 1
        grown_cycle += int(gc)
        grown_scc_frac.append(r["grown"]["largest_scc_frac"])
        grown_srb.append(r["grown"]["spectral_radius_bin"])
        grown_srw.append(r["grown"]["spectral_radius_w"])
        if r["is_competent"]:
            grown_comp += 1
            grown_comp_cycle += int(gc)
            grown_comp_scc.append(r["grown"]["largest_scc_frac"])
            grown_comp_srb.append(r["grown"]["spectral_radius_bin"])
        for rs in r.get("random") or []:
            rc = rs["has_cycle"]
            c += int(rc)
            d += int(not rc)
            n_rand += 1
            rand_cycle += int(rc)
            rand_scc_frac.append(rs["largest_scc_frac"])
            rand_srb.append(rs["spectral_radius_bin"])
            rand_srw.append(rs["spectral_radius_w"])
        strata[key] = [a, b, c, d]

    return {
        "experiment": "recurrence_structure", "task": task,
        "n_grown": n_grown, "n_random": n_rand, "n_strata": len(strata),
        "n_grown_competent": grown_comp,
        "num_random_per_source": num_random,
        "has_cycle": {
            "grown_rate": _rate(grown_cycle, n_grown),
            "random_rate": _rate(rand_cycle, n_rand),
            "mh_or_grown_vs_random": _mh_or(strata),
            # null (NOT 0.0) when the task has no competent grown nets -- e.g. the
            # lunarlander_sparse landing frontier null. 0/0 must never read as "0%".
            "grown_competent_rate": _rate(grown_comp_cycle, grown_comp),
        },
        "largest_scc_frac": {"grown_median": _med(grown_scc_frac),
                             "random_median": _med(rand_scc_frac),
                             "grown_competent_median": _med(grown_comp_scc)},
        "spectral_radius_binary": {"grown_median": _med(grown_srb),
                                   "random_median": _med(rand_srb),
                                   "grown_competent_median": _med(grown_comp_srb)},
        "spectral_radius_weighted": {"grown_median": _med(grown_srw),
                                     "random_median": _med(rand_srw)},
        "elapsed_sec": elapsed,
    }


def _write_summary(summary: dict, out_dir: str) -> None:
    with open(os.path.join(out_dir, "summary_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)
    hc, scc = summary["has_cycle"], summary["largest_scc_frac"]
    logger.info(
        f"recurrence-structure DONE task={summary['task']}: has_cycle grown "
        f"{_fmt(hc['grown_rate'])} vs random {_fmt(hc['random_rate'])} "
        f"| MH OR {_fmt(hc['mh_or_grown_vs_random'], 2)} | scc_frac median grown "
        f"{_fmt(scc['grown_median'])} vs random {_fmt(scc['random_median'])} "
        f"| COMPETENT (n={summary['n_grown_competent']}): scc "
        f"{_fmt(scc['grown_competent_median'])}, has_cycle "
        f"{_fmt(hc['grown_competent_rate'])} -> {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="structural recurrence: grown vs (N,E)-matched random")
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--pool-dir", type=str, default=None)
    p.add_argument("--num-random", type=int, default=4, help="matched-random graphs per grown source")
    p.add_argument("--weight-lo", type=float, default=0.01)
    p.add_argument("--weight-hi", type=float, default=1.0)
    p.add_argument("--max-retries", type=int, default=100)
    p.add_argument("--gen-base", type=int, default=20_000_000, help="disjoint from control/ablation seed bases")
    p.add_argument("--max-sources", type=int, default=0, help="cap sources (0=all)")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--max-workers", type=int, default=max(1, cpu_count() - 2))
    p.add_argument("--from-per-source", action="store_true",
                   help="rebuild summary_stats.json from the saved per_source.jsonl (no regrow)")
    args = p.parse_args()

    pool_dir = args.pool_dir or f"experiments/{args.task}/pool"
    out_dir = args.output_dir or f"experiments/{args.task}/recurrence_structure"
    os.makedirs(out_dir, exist_ok=True)
    setup_logging(log_dir=out_dir, log_file="recurrence_structure.log")

    if args.from_per_source:
        res_path = os.path.join(out_dir, "per_source.jsonl")
        with open(res_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        # preserve the original run's provenance rather than fabricating new values
        num_random, elapsed = args.num_random, 0.0
        prev = os.path.join(out_dir, "summary_stats.json")
        if os.path.exists(prev):
            with open(prev) as f:
                old = json.load(f)
            num_random = old.get("num_random_per_source", num_random)
            elapsed = old.get("elapsed_sec", 0.0)
        logger.info(f"re-aggregating task={args.task} from {res_path} "
                    f"({len(rows)} rows, no regrow)")
        _write_summary(_aggregate(rows, args.task, num_random, elapsed), out_dir)
        return

    sources = load_sources(pool_dir)
    if args.max_sources > 0:
        sources = sources[: args.max_sources]
    logger.info(f"recurrence-structure task={args.task} | grown sources={len(sources)} "
                f"m={args.num_random}")

    progress = ProgressWriter(os.path.join(out_dir, "progress.json"), total=len(sources),
                              phase="recurrence_structure", meta={"task": args.task})

    t0 = time.time()
    rows: list[dict] = []
    res_path = os.path.join(out_dir, "per_source.jsonl")
    with open(res_path, "w") as rf, Pool(
        args.max_workers, initializer=_init_worker,
        initargs=(args.task, args.num_random, args.weight_lo, args.weight_hi,
                  args.max_retries, args.gen_base)) as pool:
        for i, r in enumerate(pool.imap_unordered(_process_source, sources, chunksize=4), start=1):
            rows.append(r)
            rf.write(json.dumps(r) + "\n")
            if i % 100 == 0 or i == len(sources):
                rf.flush()
                os.fsync(rf.fileno())
                rate = (time.time() - t0) / i
                progress.update(i)
                logger.info(f"[{i}/{len(sources)}] | {rate:.3f}s/src "
                            f"ETA {rate * (len(sources) - i) / 60:.1f} min")

    # ── aggregate ────────────────────────────────────────────────────────────
    _write_summary(_aggregate(rows, args.task, args.num_random, time.time() - t0), out_dir)
    progress.done()


if __name__ == "__main__":
    main()
