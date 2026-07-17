"""
Indirect-encoding baseline samplers for the un-evolved generative-process
comparison (see notes/baselines-design.md).

Each sampler emits a weighted ``nx.DiGraph`` in the SAME representation the
matched-random control (``random_graph.generate_random_rnn``) and the MorphoNAS
pools use, so it plugs straight into ``task_registry.TaskSpec.make_propagator(G)
-> run_rollouts`` with no change to the evaluation path. The point of the
comparison is to ask whether morphogenesis organises useful wiring better than
the other leading *indirect* encoding (CPPN / HyperNEAT) at the same size
budget, which the matched-random (structureless) null cannot answer.

Design (baselines-design.md §2-3): all three arms are matched at identical
(N, E) and all draw realised edge weights iid ~ U[0.01, 1.0] (the random arm's
weight process). The CPPN's output magnitude is used ONLY to *select* which E
edges exist (the topology); the realised weights carry no CPPN information. So
``sample_cppn_hyperneat`` vs ``generate_random_rnn`` is a pure topology contrast
-- exactly where the weight-ablation localised the prior (topological in the
large-net regime; weight pattern is shuffle-invariant).

Seed-exactness (the repo's byte-exact convention, as in random_graph.py): a
given ``np.random.Generator`` reproduces the exact same graph. RNG call order
for ``sample_cppn_hyperneat``:
  1. ``W1``        -- CPPN input->hidden weights, rng.normal
  2. ``act_idx``   -- per-hidden-node activation choice, rng.integers
  3. ``W2``        -- CPPN hidden->output weights, rng.normal
  4. (topology is computed deterministically from the CPPN; no RNG)
  5. ``weights``   -- realised edge weights, rng.uniform(0.01, 1.0, E)
The I/O-node convention NeuralPropagator imposes (inputs = lowest in-degree,
ties by insertion order; outputs = highest node labels) is satisfied by giving
input nodes labels 0..input_dim-1 with NO incoming edges and output nodes the
highest labels; nodes are inserted in label order so ties resolve to inputs.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np

# ── CPPN activation set (the canonical compositional-pattern set) ────────────────
# Each takes and returns a numpy array. Fixed a priori (not tuned).
_ACTIVATIONS = {
    "identity": lambda x: x,
    "sin": lambda x: np.sin(x),
    "gaussian": lambda x: np.exp(-(x * x)),
    "tanh": lambda x: np.tanh(x),
    "abs": lambda x: np.abs(x),
}
DEFAULT_ACTIVATIONS = ("identity", "sin", "gaussian", "tanh", "abs")

WEIGHT_RANGE = (0.01, 1.0)  # realised-weight range, matching the random arm


def _substrate_coords(input_dim: int, output_dim: int, num_hidden: int) -> np.ndarray:
    """(N, 2) coordinates (layer_x in {-1, 0, +1}, y in linspace[-1,1]) for a
    3-layer substrate. Node labels: inputs 0..input_dim-1, hidden next, outputs
    last output_dim. linspace handles the 1- and 0-node layers cleanly."""
    n = input_dim + num_hidden + output_dim
    coords = np.zeros((n, 2), dtype=np.float64)
    coords[:input_dim, 0] = -1.0
    coords[:input_dim, 1] = np.linspace(-1.0, 1.0, input_dim)
    if num_hidden > 0:
        hid = slice(input_dim, input_dim + num_hidden)
        coords[hid, 0] = 0.0
        coords[hid, 1] = np.linspace(-1.0, 1.0, num_hidden)
    out = slice(input_dim + num_hidden, n)
    coords[out, 0] = 1.0
    coords[out, 1] = np.linspace(-1.0, 1.0, output_dim)
    return coords


def _candidate_pairs(n: int, input_dim: int, connectivity: str) -> tuple[np.ndarray, np.ndarray]:
    """Ordered (source, target) candidate edges. Targets exclude input nodes
    (inputs never receive edges -> in-degree 0 -> selected as inputs). 'recurrent'
    (default) lets any non-input be a target from any other node; 'layered'
    restricts to input->hidden->output."""
    targets = np.arange(input_dim, n)  # all non-input nodes
    if connectivity == "recurrent":
        sources = np.arange(n)
        S, T = np.meshgrid(sources, targets, indexing="ij")
        src, tgt = S.ravel(), T.ravel()
        mask = src != tgt
        return src[mask], tgt[mask]
    if connectivity == "layered":
        raise NotImplementedError("layered substrate is the robustness arm; add when needed")
    raise ValueError(f"unknown connectivity: {connectivity}")


def _cppn_query(coords, src, tgt, rng, cppn_hidden, activations):
    """Vectorised random-CPPN forward over every candidate pair. Returns |w| per
    pair. Consumes RNG in the documented order (W1, act_idx, W2)."""
    x1, y1 = coords[src, 0], coords[src, 1]
    x2, y2 = coords[tgt, 0], coords[tgt, 1]
    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    bias = np.ones_like(x1)
    feats = np.stack([x1, y1, x2, y2, bias, dist], axis=1)  # (P, 6)

    n_feat = feats.shape[1]
    W1 = rng.normal(0.0, 1.0, size=(n_feat, cppn_hidden))
    act_idx = rng.integers(0, len(activations), size=cppn_hidden)
    W2 = rng.normal(0.0, 1.0, size=cppn_hidden)

    H = feats @ W1  # (P, cppn_hidden)
    for j in range(cppn_hidden):
        H[:, j] = _ACTIVATIONS[activations[act_idx[j]]](H[:, j])
    out = H @ W2  # (P,)
    return np.abs(out)


class _UnionFind:
    __slots__ = ("parent", "rank", "ncomp")

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.ncomp = n

    def find(self, a: int) -> int:
        p = self.parent
        while p[a] != a:
            p[a] = p[p[a]]
            a = p[a]
        return a

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.ncomp -= 1
        return True


def sample_cppn_hyperneat(
    num_nodes: int,
    num_edges: int,
    input_dim: int,
    output_dim: int,
    rng: np.random.Generator,
    *,
    cppn_hidden: int = 4,
    activations: tuple = DEFAULT_ACTIVATIONS,
    connectivity: str = "recurrent",
    weight_range: tuple[float, float] = WEIGHT_RANGE,
    weight_mode: str = "uniform",
) -> Optional[nx.DiGraph]:
    """A random CPPN queried over a coordinate substrate, keeping the top-E edges
    by |CPPN weight| (adjusted minimally to guarantee weak connectivity), with
    realised weights ~ U[weight_range]. Matches (num_nodes, num_edges) by
    construction. Returns None if a weakly-connected graph at that (N, E) is
    impossible (E < N-1, or N too small for the substrate) -- the same
    None-on-failure contract as generate_random_rnn.
    """
    n = int(num_nodes)
    e = int(num_edges)
    num_hidden = n - input_dim - output_dim
    if num_hidden < 0 or e < n - 1:
        # cannot place the substrate, or cannot be weakly connected at this E
        return None

    coords = _substrate_coords(input_dim, output_dim, num_hidden)
    src, tgt = _candidate_pairs(n, input_dim, connectivity)
    if e > len(src):
        return None

    absw = _cppn_query(coords, src, tgt, rng, cppn_hidden, activations)
    order = np.argsort(-absw, kind="stable")  # candidates by |w| desc, ties by index

    # Pass 1: a weakly-connected spanning structure from the highest-|w| edges
    # that extend connectivity (undirected union-find -> weak connectivity).
    uf = _UnionFind(n)
    chosen: list[int] = []
    in_chosen = np.zeros(len(src), dtype=bool)
    for idx in order:
        if uf.ncomp == 1:
            break
        if uf.union(int(src[idx]), int(tgt[idx])):
            chosen.append(int(idx))
            in_chosen[idx] = True
    if uf.ncomp != 1:
        return None  # candidate set could not connect all nodes (should not happen here)

    # Pass 2: fill to E with the next highest-|w| edges not yet chosen.
    for idx in order:
        if len(chosen) >= e:
            break
        if not in_chosen[idx]:
            chosen.append(int(idx))
            in_chosen[idx] = True

    # weights: 'uniform' = iid U[range] (the random arm's process; topology-only
    # contrast); 'cppn' = the CPPN's own |output| on the kept edges, min-max
    # rescaled into [range] (the signed-weight robustness arm -- the CPPN also
    # sets magnitudes). Same seed => same topology in both modes (the weight draw
    # is the only difference), so it isolates the weight channel.
    if weight_mode == "uniform":
        weights = rng.uniform(weight_range[0], weight_range[1], size=len(chosen))
    elif weight_mode == "cppn":
        wk = absw[np.asarray(chosen)]
        lo, hi = float(wk.min()), float(wk.max())
        span = hi - lo
        if span <= 0:
            weights = np.full(len(chosen), 0.5 * (weight_range[0] + weight_range[1]))
        else:
            weights = weight_range[0] + (wk - lo) / span * (weight_range[1] - weight_range[0])
    else:
        raise ValueError(f"unknown weight_mode: {weight_mode}")
    G = nx.DiGraph()
    G.add_nodes_from(range(n))  # insertion order = label order (tie-break favours inputs)
    for k, idx in enumerate(chosen):
        G.add_edge(int(src[idx]), int(tgt[idx]), weight=float(weights[k]))
    return G


def sample_neat_initial(
    input_dim: int,
    output_dim: int,
    rng: np.random.Generator,
    *,
    weight_range: tuple[float, float] = WEIGHT_RANGE,
) -> nx.DiGraph:
    """The minimal initial NEAT genome: a perceptron, input_dim inputs fully
    connected to output_dim outputs, no hidden, weights ~ U[weight_range]. Its
    (N, E) is fixed and small, so it is reported as its own row (not (N,E)-matched)
    -- the feedforward-floor reference (cf. the RWG feedforward axis = 0%)."""
    n = input_dim + output_dim
    G = nx.DiGraph()
    G.add_nodes_from(range(n))  # inputs 0..input_dim-1, outputs last output_dim
    for i in range(input_dim):
        for o in range(input_dim, n):
            G.add_edge(i, o, weight=float(rng.uniform(weight_range[0], weight_range[1])))
    return G


# ── self-test: byte-stability, weak connectivity, I/O placement ──────────────────
if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.abspath("code"))
    from MorphoNAS.neural_propagation import NeuralPropagator

    IN, OUT, N, E = 6, 3, 400, 1027  # Acrobot dims, median MN (N,E)

    g1 = sample_cppn_hyperneat(N, E, IN, OUT, np.random.default_rng(123), cppn_hidden=4)
    g2 = sample_cppn_hyperneat(N, E, IN, OUT, np.random.default_rng(123), cppn_hidden=4)
    assert g1 is not None and g2 is not None
    assert g1.number_of_nodes() == N and g1.number_of_edges() == E, (g1.number_of_nodes(), g1.number_of_edges())
    e1 = sorted((u, v, round(d["weight"], 12)) for u, v, d in g1.edges(data=True))
    e2 = sorted((u, v, round(d["weight"], 12)) for u, v, d in g2.edges(data=True))
    assert e1 == e2, "NOT seed-exact"
    assert nx.is_weakly_connected(g1), "not weakly connected"

    # inputs (labels 0..IN-1) must have in-degree 0 and be selected as the inputs
    assert all(g1.in_degree(i) == 0 for i in range(IN)), "input nodes have incoming edges"
    prop = NeuralPropagator(g1, input_dim=IN, output_dim=OUT,
                            activation_function=NeuralPropagator.tanh_activation,
                            extra_thinking_time=2, additive_update=False)
    info = prop.get_input_nodes_info()
    assert sorted(info["selected_nodes"]) == list(range(IN)), info["selected_nodes"]

    # a different seed gives a different topology
    g3 = sample_cppn_hyperneat(N, E, IN, OUT, np.random.default_rng(124), cppn_hidden=4)
    e3 = sorted((u, v) for u, v, _ in g3.edges(data=True))
    assert e3 != [(u, v) for u, v, _ in e1], "topology did not change with seed"

    # NEAT-initial
    gn = sample_neat_initial(IN, OUT, np.random.default_rng(1))
    assert gn.number_of_nodes() == IN + OUT and gn.number_of_edges() == IN * OUT
    assert nx.is_weakly_connected(gn)

    print(f"OK  cppn_hyperneat: N={g1.number_of_nodes()} E={g1.number_of_edges()} "
          f"weakly_connected=True seed_exact=True inputs={info['selected_nodes']}")
    print(f"OK  neat_initial:   N={gn.number_of_nodes()} E={gn.number_of_edges()}")
