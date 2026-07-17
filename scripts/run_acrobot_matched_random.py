#!/usr/bin/env python3
"""
Acrobot matched-random control (developmental-prior strength on a harder task).

Ports the CartPole B2 matched-random protocol to Acrobot. For each competent
MorphoNAS Acrobot network in the existing pool, we generate k random directed
graphs with the SAME (neuron count, edge count), assign weights ~Uniform[0.01, 1.0],
require weak connectivity, and evaluate them on Acrobot-v1 under the SAME protocol
the pool used. The output is the matched-random competence rate, which we compare to
the MorphoNAS rate to get the Acrobot analogue of CartPole's 8.4x ratio.

Two protocols are held fixed so the cross-task comparison is valid:
  * Graph generation matches CartPole B2 exactly
    (../MorphoNAS-PL/code/MorphoNAS_PL/experimentB2_random_rnn.py::generate_random_rnn):
    sample num_edges distinct ordered pairs uniformly without replacement, weights
    ~U[0.01, 1.0], retry up to 100x until weakly connected.
  * Evaluation reuses the Acrobot pool's own functions (run_rollouts, get_stratum,
    NeuralPropagator config: 6 in / 3 out, tanh, extra_thinking_time=2,
    additive_update=False), baseline only (no plasticity, eta=0).

Seed handling: the Acrobot pool used per-seed evaluation (each network on
range(gen_seed, gen_seed+20)), unlike the CartPole pool/B2 which used fixed seeds.
To keep the within-task MorphoNAS-vs-random comparison fair, each random graph is
evaluated on its SOURCE network's stored eval_seeds (paired by seeds). Use
--seed-mode fixed to instead reproduce the CartPole B2 style (VERIFICATION_SEEDS).

Usage:
  # Minimal verification slice (build the pipeline, measure timing):
  .venv/bin/python3 scripts/run_acrobot_matched_random.py \
      --max-sources 10 --num-random 2 --output-dir experiments/acrobot/matched_random/verify

  # Full run (matches B2: competent sources, k=5):
  .venv/bin/python3 scripts/run_acrobot_matched_random.py \
      --min-stratum low_mid --num-random 5 --output-dir experiments/acrobot/matched_random/full
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from typing import Optional

import networkx as nx
import numpy as np

sys.path.append(os.path.abspath("code"))

from MorphoNAS.neural_propagation import NeuralPropagator  # noqa: E402
from MorphoNAS_DevPriors.experiment_acrobot import (  # noqa: E402
    ENV_NAME,
    INPUT_DIM,
    MIN_NEURONS,
    OUTPUT_DIM,
    Stratum,
    get_stratum,
    run_rollouts,
)
from MorphoNAS_DevPriors.logging_config import setup_logging  # noqa: E402

logger = logging.getLogger(__name__)

STRATA_ORDER = ["weak", "low_mid", "high_mid", "near_perfect", "perfect"]
SOLVED_STRATUM = "perfect"  # Acrobot reward >= -100 (Gymnasium solved threshold)

# CartPole B2 reference counts (the published 8.4x), for the cross-task comparison.
# MorphoNAS over the full 50k B0.5 pool; random over the matched-to-competent B2 pool.
#   non-weak (reward >= 200): MN 2362/50000 = 4.724%, random 65/11586 = 0.561% -> 8.42x
#   solved   (reward >= 475): MN  769/50000 = 1.538%, random 26/11586 = 0.224% -> 6.86x
CARTPOLE_REF = {
    "nonweak": {"mn_count": 2362, "mn_n": 50000, "rand_count": 65, "rand_n": 11586},
    "solved": {"mn_count": 769, "mn_n": 50000, "rand_count": 26, "rand_n": 11586},
}

Z95 = 1.959963984540054  # standard normal 0.975 quantile


# ── Statistics: Wilson proportion CI + MOVER ratio CI ───────────────────────────

def wilson_ci(x: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion. Lower bound is exactly 0
    when x == 0, which is what lets the ratio CI stay well-defined with no
    random successes."""
    if n == 0:
        return (0.0, 1.0)
    p = x / n
    d = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = (z / d) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def ratio_ci_mover(x1: int, n1: int, x2: int, n2: int, z: float = Z95) -> dict:
    """95% CI for the ratio (x1/n1) / (x2/n2) of two INDEPENDENT binomial
    proportions, via MOVER combined with Wilson intervals (Donner & Zou 2012).

    Chosen over the Katz log method because it stays finite and correct when the
    denominator arm has zero successes (x2 == 0): the lower bound is then
    sqrt(l1*(2*p1-l1))/u2 and the upper bound is +inf. Validated against Katz-log
    and a 40k bootstrap on the CartPole counts (agreement within ~2%).
    """
    p1 = x1 / n1 if n1 else 0.0
    p2 = x2 / n2 if n2 else 0.0
    l1, u1 = wilson_ci(x1, n1, z)
    l2, u2 = wilson_ci(x2, n2, z)

    point = (p1 / p2) if p2 > 0 else math.inf

    # Lower limit
    if p2 == 0:
        lo = math.sqrt(max(l1 * (2 * p1 - l1), 0.0)) / u2 if u2 > 0 else math.inf
    else:
        A = u2 * (2 * p2 - u2)
        disc = max(p1 * p1 * p2 * p2 - l1 * (2 * p1 - l1) * u2 * (2 * p2 - u2), 0.0)
        lo = (p1 * p2 - math.sqrt(disc)) / A

    # Upper limit
    Au = l2 * (2 * p2 - l2)
    if Au <= 0:  # zero random successes -> unbounded above
        hi = math.inf
    else:
        disc_u = max(p1 * p1 * p2 * p2 - u1 * (2 * p1 - u1) * l2 * (2 * p2 - l2), 0.0)
        hi = (p1 * p2 + math.sqrt(disc_u)) / Au

    return {
        "point": point,
        "lo": lo,
        "hi": hi if math.isfinite(hi) else None,  # JSON-safe (null = unbounded)
        "mn_rate": p1,
        "mn_ci": list(wilson_ci(x1, n1, z)),
        "rand_rate": p2,
        "rand_ci": list(wilson_ci(x2, n2, z)),
        "mn_count": x1,
        "mn_n": n1,
        "rand_count": x2,
        "rand_n": n2,
    }


def grade_vs_reference(acro_lo: float, acro_hi: Optional[float],
                       ref_point: float, ref_lo: float, ref_hi: float) -> dict:
    """Graded comparison. Primary test (user-specified): Acrobot CI lower
    bound vs CartPole's point ratio. Secondary, stricter test: do the two CIs
    overlap."""
    hi = acro_hi if acro_hi is not None else math.inf
    if acro_lo > ref_point:
        primary = "grows"  # the cross-task ratio increases (RoR CI excludes 1)
    elif hi >= ref_point:
        primary = "persists"  # CI straddles the reference; does not collapse
    else:
        primary = "shrinks"  # strong claim killed
    cis_overlap = not (acro_lo > ref_hi or hi < ref_lo)
    return {
        "verdict": primary,
        "acrobot_ci_lo_vs_ref_point": acro_lo - ref_point,
        "cis_overlap": cis_overlap,
        "reference_point": ref_point,
        "reference_ci": [ref_lo, ref_hi],
    }


# ── Matched-random graph generation (ported verbatim from CartPole B2) ──────────

def generate_random_rnn(
    num_nodes: int,
    num_edges: int,
    rng: np.random.Generator,
    weight_range: tuple[float, float] = (0.01, 1.0),
    max_retries: int = 100,
) -> Optional[nx.DiGraph]:
    """Random weakly-connected directed graph with given node/edge counts.

    Identical method to B2: sample num_edges distinct ordered pairs uniformly
    without replacement, weights ~U[weight_range], retry until weakly connected.
    """
    max_possible = num_nodes * (num_nodes - 1)
    if num_edges > max_possible:
        num_edges = max_possible

    all_pairs = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]

    for _attempt in range(max_retries):
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))

        chosen = rng.choice(len(all_pairs), size=num_edges, replace=False)
        for idx in chosen:
            i, j = all_pairs[idx]
            w = float(rng.uniform(weight_range[0], weight_range[1]))
            G.add_edge(i, j, weight=w)

        if nx.is_weakly_connected(G):
            return G

    return None


def evaluate_random_rnn(
    G: nx.DiGraph,
    *,
    num_rollouts: int,
    seeds: Optional[list[int]],
    env=None,
) -> dict:
    """Baseline Acrobot evaluation of a random graph (no plasticity).

    Mirrors B2.evaluate_random_rnn but with Acrobot I/O dims and the pool's
    run_rollouts. NeuralPropagator config is identical to the pool's
    (_create_propagator in run_acrobot_pool.py): tanh, extra_thinking_time=2,
    additive_update=False, graph_diameter computed internally.
    """
    propagator = NeuralPropagator(
        G=G,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        activation_function=NeuralPropagator.tanh_activation,
        extra_thinking_time=2,
        additive_update=False,
    )
    return run_rollouts(
        propagator,
        num_rollouts,
        seeds=seeds,
        reset_plastic_each_episode=True,
        env=env,
    )


# ── Worker plumbing ─────────────────────────────────────────────────────────────

_worker_env = None
_worker_rollouts = 20
_worker_weight_range = (0.01, 1.0)
_worker_max_retries = 100


def _init_worker(env_name, rollouts, weight_lo, weight_hi, max_retries):
    global _worker_env, _worker_rollouts, _worker_weight_range, _worker_max_retries
    import gymnasium as gym

    from MorphoNAS_DevPriors.parallel_utils import configure_worker_threads

    configure_worker_threads()
    _worker_env = gym.make(env_name)
    _worker_rollouts = int(rollouts)
    _worker_weight_range = (float(weight_lo), float(weight_hi))
    _worker_max_retries = int(max_retries)


def _eval_one(task: dict) -> dict:
    src_id = task["source_id"]
    num_neurons = task["num_neurons"]
    num_edges = task["num_connections"]
    gen_seed = task["random_seed"]
    eval_seeds = task["eval_seeds"]

    base = {
        "source_id": src_id,
        "source_stratum": task["source_stratum"],
        "random_seed": gen_seed,
        "num_neurons": num_neurons,
        "num_connections": num_edges,
    }

    if num_neurons < MIN_NEURONS:
        return {**base, "valid": False, "error": "insufficient_neurons"}

    rng = np.random.default_rng(gen_seed)
    G = generate_random_rnn(
        num_neurons,
        num_edges,
        rng,
        weight_range=_worker_weight_range,
        max_retries=_worker_max_retries,
    )
    if G is None:
        return {**base, "valid": False, "error": "not_weakly_connected"}

    res = evaluate_random_rnn(
        G,
        num_rollouts=_worker_rollouts,
        seeds=eval_seeds[: _worker_rollouts],
        env=_worker_env,
    )
    baseline_reward = float(res.get("avg_reward", 0.0))
    stratum = get_stratum(baseline_reward)

    return {
        **base,
        "valid": True,
        "baseline_reward": baseline_reward,
        "stratum": stratum.value,
        "is_competent": stratum.value != "weak",
        "is_solved": stratum.value == SOLVED_STRATUM,
        "rewards": res.get("rewards", []),
        "eval_seeds": eval_seeds[: _worker_rollouts],
    }


# ── Source loading ──────────────────────────────────────────────────────────────

def load_sources(pool_dir: str, min_stratum: str) -> list[dict]:
    """Read pool network JSONs, keep sources at >= min_stratum, return their stats."""
    networks_dir = os.path.join(pool_dir, "networks")
    min_idx = STRATA_ORDER.index(min_stratum)
    sources: list[dict] = []
    for fname in sorted(os.listdir(networks_dir)):
        if not fname.endswith(".json"):
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
                "baseline_reward": d.get("baseline_reward"),
                "eval_seeds": list(d.get("rollout_data", {}).get("eval_seeds", [])),
            }
        )
    return sources


def morphonas_rates(pool_dir: str) -> dict:
    """MorphoNAS competence rates over the full pool (the comparison denominators)."""
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
    parser = argparse.ArgumentParser(description="Acrobot matched-random control")
    parser.add_argument("--pool-dir", type=str, default="experiments/acrobot/pool")
    parser.add_argument(
        "--min-stratum", type=str, default="low_mid", choices=STRATA_ORDER,
        help="Match to sources at >= this stratum (low_mid = competent, B2-exact; "
             "weak = all networks)",
    )
    parser.add_argument("--num-random", type=int, default=5,
                        help="Random graphs per source network (B2 used 5)")
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--weight-lo", type=float, default=0.01)
    parser.add_argument("--weight-hi", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=6000000,
                        help="Graph-gen seed = base_seed + source_id*100 + k")
    parser.add_argument(
        "--seed-mode", type=str, default="paired", choices=["paired", "fixed"],
        help="paired: eval each random graph on its source's eval_seeds (matches "
             "the Acrobot pool); fixed: use VERIFICATION_SEEDS (CartPole B2 style)",
    )
    parser.add_argument("--max-sources", type=int, default=0,
                        help="Cap number of sources (0 = all); for verification slices")
    parser.add_argument("--output-dir", type=str,
                        default="experiments/acrobot/matched_random")
    parser.add_argument("--max-workers", type=int, default=max(1, cpu_count() - 2))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(log_dir=args.output_dir, log_file="matched_random.log")

    from MorphoNAS_DevPriors.parallel_utils import configure_worker_threads
    configure_worker_threads()

    mn = morphonas_rates(args.pool_dir)
    sources = load_sources(args.pool_dir, args.min_stratum)
    if args.max_sources > 0:
        sources = sources[: args.max_sources]

    if args.seed_mode == "fixed":
        from MorphoNAS_DevPriors.experiment_acrobot import VERIFICATION_SEEDS
        fixed_seeds = VERIFICATION_SEEDS[: args.rollouts]

    tasks: list[dict] = []
    for s in sources:
        eval_seeds = (
            fixed_seeds if args.seed_mode == "fixed" else s["eval_seeds"]
        )
        for k in range(args.num_random):
            tasks.append(
                {
                    "source_id": s["network_id"],
                    "source_stratum": s["stratum"],
                    "num_neurons": s["num_neurons"],
                    "num_connections": s["num_connections"],
                    "random_seed": args.base_seed + s["network_id"] * 100 + k,
                    "eval_seeds": eval_seeds,
                }
            )

    logger.info("=" * 60)
    logger.info("ACROBOT MATCHED-RANDOM CONTROL")
    logger.info("=" * 60)
    logger.info(f"Environment: {ENV_NAME}  ({INPUT_DIM} in / {OUTPUT_DIM} out)")
    logger.info(f"Pool dir: {args.pool_dir}")
    logger.info(f"MorphoNAS pool: n={mn['n_total']}, non-weak={mn['nonweak_count']} "
                f"({mn['nonweak_rate']*100:.2f}%), solved={mn['solved_count']} "
                f"({mn['solved_rate']*100:.2f}%)")
    logger.info(f"Sources (>= {args.min_stratum}): {len(sources)}")
    logger.info(f"Random per source (k): {args.num_random}  -> total graphs: {len(tasks)}")
    logger.info(f"Weights ~ U[{args.weight_lo}, {args.weight_hi}], "
                f"weakly-connected, max_retries={args.max_retries}")
    logger.info(f"Eval: {args.rollouts} rollouts, seed-mode={args.seed_mode}, "
                f"workers={args.max_workers}")
    logger.info("=" * 60)

    t0 = time.time()
    results: list[dict] = []
    results_path = os.path.join(args.output_dir, "results.jsonl")
    with open(results_path, "w") as rf, Pool(
        args.max_workers,
        initializer=_init_worker,
        initargs=(ENV_NAME, args.rollouts, args.weight_lo, args.weight_hi,
                  args.max_retries),
    ) as pool:
        for i, r in enumerate(
            pool.imap_unordered(_eval_one, tasks, chunksize=4), start=1
        ):
            results.append(r)
            rf.write(json.dumps(r) + "\n")
            if i % 50 == 0 or i == len(tasks):
                valid = sum(1 for x in results if x.get("valid"))
                comp = sum(1 for x in results if x.get("is_competent"))
                solv = sum(1 for x in results if x.get("is_solved"))
                rate = (time.time() - t0) / i
                logger.info(
                    f"[{i}/{len(tasks)}] valid={valid} competent={comp} solved={solv} "
                    f"| {rate:.2f}s/graph, ETA {rate*(len(tasks)-i)/60:.1f} min"
                )

    # ── Aggregate ──
    valid = [r for r in results if r.get("valid")]
    n_valid = len(valid)
    strata = {s: 0 for s in STRATA_ORDER}
    for r in valid:
        strata[r["stratum"]] += 1
    competent = sum(1 for r in valid if r["is_competent"])
    solved = sum(1 for r in valid if r["is_solved"])

    # ── Ratios with CIs (asymmetric denominators: MN over full pool, random over
    #    the matched-to-competent graphs), plus the CartPole reference and verdict ──
    acro = {
        "nonweak": ratio_ci_mover(mn["nonweak_count"], mn["n_total"], competent, n_valid),
        "solved": ratio_ci_mover(mn["solved_count"], mn["n_total"], solved, n_valid),
    }
    cart = {
        bar: ratio_ci_mover(c["mn_count"], c["mn_n"], c["rand_count"], c["rand_n"])
        for bar, c in CARTPOLE_REF.items()
    }
    verdict = {
        bar: grade_vs_reference(
            acro[bar]["lo"], acro[bar]["hi"],
            cart[bar]["point"], cart[bar]["lo"], cart[bar]["hi"],
        )
        for bar in ("nonweak", "solved")
    }

    summary = {
        "experiment": "acrobot_matched_random",
        "env_name": ENV_NAME,
        "engine": "vendored code/MorphoNAS (byte-identical to MorphoNAS-PL public; "
                  "rules == MorphoNAS-work@v1, tag v1->5b3e8a23->commit 72902afc)",
        "config": {
            "min_stratum": args.min_stratum,
            "num_random_per_source": args.num_random,
            "rollouts": args.rollouts,
            "weight_range": [args.weight_lo, args.weight_hi],
            "max_retries": args.max_retries,
            "seed_mode": args.seed_mode,
            "base_seed": args.base_seed,
            "n_sources": len(sources),
        },
        "n_random_total": len(tasks),
        "n_random_valid": n_valid,
        "valid_rate": n_valid / len(tasks) if tasks else 0.0,
        "strata": strata,
        "random_nonweak_count": competent,
        "random_nonweak_rate": competent / n_valid if n_valid else 0.0,
        "random_solved_count": solved,
        "random_solved_rate": solved / n_valid if n_valid else 0.0,
        "morphonas": mn,
        # primary = non-weak (the bar the 8.4x is built on); solved = robustness
        "ratio_nonweak": acro["nonweak"],
        "ratio_solved": acro["solved"],
        "cartpole_reference": cart,
        "verdict": verdict,
        "elapsed_sec": time.time() - t0,
    }

    with open(os.path.join(args.output_dir, "summary_stats.json"), "w") as f:
        json.dump(summary, f, indent=2)

    def fmt_ci(d):
        hi = "inf" if d["hi"] is None else f"{d['hi']:.2f}"
        pt = "inf" if not math.isfinite(d["point"]) else f"{d['point']:.2f}"
        return f"{pt}x  95% CI [{d['lo']:.2f}, {hi}]"

    logger.info("=" * 60)
    logger.info("COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Random graphs: {len(tasks)} total, {n_valid} valid "
                f"({summary['valid_rate']*100:.1f}%)")
    logger.info(f"Strata: {strata}")
    logger.info(f"Matched-random competence: non-weak {competent}/{n_valid} "
                f"({summary['random_nonweak_rate']*100:.3f}%), solved {solved}/{n_valid} "
                f"({summary['random_solved_rate']*100:.3f}%)")
    logger.info(f"MorphoNAS: non-weak {mn['nonweak_count']}/{mn['n_total']} "
                f"({mn['nonweak_rate']*100:.3f}%), solved {mn['solved_count']}/{mn['n_total']} "
                f"({mn['solved_rate']*100:.3f}%)")
    logger.info("-" * 60)
    logger.info(f"PRIMARY  non-weak ratio: {fmt_ci(acro['nonweak'])}")
    logger.info(f"         CartPole ref:   {fmt_ci(cart['nonweak'])}")
    logger.info(f"         VERDICT: {verdict['nonweak']['verdict'].upper()} "
                f"(Acrobot CI-lo {acro['nonweak']['lo']:.2f} vs ref point "
                f"{cart['nonweak']['point']:.2f}; CIs overlap="
                f"{verdict['nonweak']['cis_overlap']})")
    logger.info(f"ROBUST   solved   ratio: {fmt_ci(acro['solved'])}")
    logger.info(f"         CartPole ref:   {fmt_ci(cart['solved'])}")
    logger.info(f"         VERDICT: {verdict['solved']['verdict'].upper()} "
                f"(CIs overlap={verdict['solved']['cis_overlap']})")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
