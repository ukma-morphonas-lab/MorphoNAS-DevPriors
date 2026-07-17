"""
Matched-random directed-graph generator, ported verbatim from CartPole B2
(../MorphoNAS-PL/code/MorphoNAS_PL/experimentB2_random_rnn.py::generate_random_rnn)
and shared by the matched-random control and the weight ablation so both
arms use one generator.

For ``weight_mode="uniform"`` the RNG call order is byte-identical to B2 and to
the locked ``run_acrobot_matched_random.py`` (one ``rng.choice`` for the edge
set, then one ``rng.uniform`` per edge), so a given seed reproduces the exact
same graph the published control used. The other weight modes exist only for
the weight-ablation factorial and never touch the uniform reproduction path.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np


def generate_random_rnn(
    num_nodes: int,
    num_edges: int,
    rng: np.random.Generator,
    weight_range: tuple[float, float] = (0.01, 1.0),
    max_retries: int = 100,
    *,
    weight_mode: str = "uniform",
    empirical_weights: Optional[np.ndarray] = None,
) -> Optional[nx.DiGraph]:
    """Random weakly-connected directed graph with given node/edge counts.

    Topology: sample ``num_edges`` distinct ordered pairs (i != j) uniformly
    without replacement, retry up to ``max_retries`` until weakly connected.

    Weight modes:
      * ``uniform``       -- w ~ U[weight_range], the B2 control (default).
      * ``mn_empirical``  -- w drawn iid with replacement from
                             ``empirical_weights`` (the pooled MorphoNAS edge
                             weights), for the (random-topo, MN-weights) cell.

    Returns None if no weakly-connected graph is found within ``max_retries``.
    """
    max_possible = num_nodes * (num_nodes - 1)
    if num_edges > max_possible:
        num_edges = max_possible

    all_pairs = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]

    for _attempt in range(max_retries):
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))

        chosen = rng.choice(len(all_pairs), size=num_edges, replace=False)

        if weight_mode == "uniform":
            for idx in chosen:
                i, j = all_pairs[idx]
                w = float(rng.uniform(weight_range[0], weight_range[1]))
                G.add_edge(i, j, weight=w)
        elif weight_mode == "mn_empirical":
            if empirical_weights is None or len(empirical_weights) == 0:
                raise ValueError("mn_empirical weight_mode requires empirical_weights")
            picks = rng.integers(0, len(empirical_weights), size=len(chosen))
            for idx, p in zip(chosen, picks):
                i, j = all_pairs[idx]
                G.add_edge(i, j, weight=float(empirical_weights[p]))
        else:
            raise ValueError(f"unknown weight_mode: {weight_mode}")

        if nx.is_weakly_connected(G):
            return G

    return None
