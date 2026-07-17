#!/usr/bin/env python3
"""Smoke test for the Pendulum cells + RWG axis (fast, no fleet).

Checks: (1) the new TaskSpecs build env + propagator; (2) a MorphoNAS genome grows
and evaluates under both the dense and sparse cells; (3) the de-confounder invariant
-- dense and sparse re-score BYTE-IDENTICAL trajectories (same actions, same obs);
(4) the fixed RWG reference architecture builds and evaluates.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.append(os.path.abspath("code"))

import numpy as np

from MorphoNAS.genome import Genome
from MorphoNAS.grid import Grid
from MorphoNAS_DevPriors.task_registry import get_task, run_rollouts
from MorphoNAS_DevPriors.random_graph import generate_random_rnn


def grow(seed, spec):
    rng = np.random.default_rng(seed)
    g = Genome.random(rng=rng, size_x=spec.grid_size[0], size_y=spec.grid_size[1],
                      max_growth_steps=200, num_morphogens=3)
    grid = Grid(g)
    grid.run_simulation(verbose=False)
    return grid.get_graph()


def rollout_record(propagator, env, seed):
    """One episode, mirroring run_rollouts, recording the action + obs sequence."""
    propagator.reset()
    obs, _ = env.reset(seed=seed)
    actions, obs_sig, total = [], [], 0.0
    done = False
    while not done:
        propagator.propagate(np.array(obs).flatten())
        a = int(propagator.get_output().argmax().item())
        actions.append(a)
        obs, r, term, trunc, _ = env.step(a)
        obs_sig.append(round(float(np.array(obs).sum()), 8))
        total += float(r)
        done = bool(term or trunc)
    return actions, obs_sig, total


def main():
    dense = get_task("pendulum")
    sparse = get_task("pendulum_sparse")
    print(f"pendulum:        env={dense.env_name} dims={dense.input_dim}/{dense.output_dim} grid={dense.grid_size}")
    print(f"pendulum_sparse: env={sparse.env_name} dims={sparse.input_dim}/{sparse.output_dim} "
          f"seed_start={sparse.pool_seed_start} ctrl_seed={sparse.control_base_seed}")
    assert dense.pool_seed_start == sparse.pool_seed_start, "genomes must be shared"
    assert dense.control_base_seed == sparse.control_base_seed, "random graphs must be paired"

    # Env wrappers sane?
    de, se = dense.make_env(), sparse.make_env()
    print(f"dense action_space={de.action_space} | sparse action_space={se.action_space}")
    assert de.action_space.n == 3 and se.action_space.n == 3

    # Grow a few genomes (shared seeds), evaluate dense + sparse, verify invariant.
    print("\n-- grow + eval + de-confounder invariant --")
    mism = 0
    for k in range(6):
        seed = dense.pool_seed_start + k
        G = grow(seed, dense)
        n, e = G.number_of_nodes(), G.number_of_edges()
        if n < dense.min_neurons:
            print(f"  seed {seed}: n={n} < min, skip")
            continue
        eval_seed = seed  # one episode
        a_d, o_d, r_d = rollout_record(dense.make_propagator(G), dense.make_env(), eval_seed)
        a_s, o_s, r_s = rollout_record(sparse.make_propagator(G), sparse.make_env(), eval_seed)
        same_a = a_d == a_s
        same_o = o_d == o_s
        if not (same_a and same_o):
            mism += 1
        # full 20-rollout means via the real harness
        es = list(range(seed, seed + 20))
        md = run_rollouts(dense.make_propagator(G), 20, seeds=es, env=dense.make_env())["avg_reward"]
        ms = run_rollouts(sparse.make_propagator(G), 20, seeds=es, env=sparse.make_env())["avg_reward"]
        print(f"  seed {seed}: n={n} e={e} | actions_identical={same_a} obs_identical={same_o} "
              f"| dense1={r_d:.1f} sparse1={r_s:.0f} | dense20={md:.1f} sparse20(upright-steps)={ms:.1f}")
    assert mism == 0, f"{mism} trajectory mismatches -- de-confounder invariant BROKEN"
    print("  -> invariant OK: dense and sparse re-score identical trajectories")

    # RWG reference architecture builds + evaluates.
    print("\n-- RWG reference architecture --")
    for topo in ("complete",):
        K = dense.input_dim + dense.output_dim + 8
        rng = np.random.default_rng(20_000_000)
        Grwg = generate_random_rnn(K, K * (K - 1), rng, weight_range=(0.01, 1.0))
        res = run_rollouts(dense.make_propagator(Grwg), 5,
                           seeds=list(range(42, 47)), env=dense.make_env())
        print(f"  {topo} K={K} edges={Grwg.number_of_edges()} | dense avg over 5 = {res['avg_reward']:.1f}")

    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
