"""
Tier-1 decoders for the phase-2 evolved-competitive comparison
(notes/phase2-design.md sec. 2).

Three genotype->phenotype maps, each taking a box-constrained vector x in
[0,1]^d and returning a weighted nx.DiGraph in the SAME representation the
reach-map and phase 1 use, so each plugs straight into
``TaskSpec.make_propagator(G) -> run_rollouts``. Only ``decode`` differs across
arms; the optimizer (CMA-ES) and the evaluation are shared (the design's
"encoding-isolated" tier 1, the clean genotype->phenotype attribution).

  * morphonas -- x = ``Genome.flatten()`` (d = n^2 + 11n + 12 = 54 at n=3);
                 decode = ``Genome.from_flattened(x) -> Grid -> get_graph``. This
                 is byte-faithful to the vendored ``CMAESOptimizer`` path (the
                 same flatten round-trip, the same clip-to-[0,1]).
  * cppn      -- x = CPPN weights: W1 (n_feat x h), per-hidden activation-select
                 genes (h), W2 (h); d = (n_feat+2)*h = 8h at n_feat=6 (32 at h=4).
                 The CPPN is queried over the SAME coordinate substrate phase 1
                 used (``baseline_encoders._substrate_coords`` / ``_candidate_pairs``),
                 the top-E edges by |output| are kept (weakly connected, the same
                 union-find), and -- the HyperNEAT-faithful primary -- the CPPN's
                 signed output sets each connection weight (``weight_mode="cppn"``).
                 This GENERALIZES ``baseline_encoders.sample_cppn_hyperneat`` to
                 take EXPLICIT CPPN weights instead of sampling them.
  * direct    -- x = edge weights of the fixed "complete_h8" recurrent reference
                 net the reach-map RWG axis used
                 (``run_rwg_axis.build_reference_graph("complete", ...)``);
                 d = K*(K-1), K = in+out+hidden. No compression: one weight per
                 edge. The no-compression floor of the ladder.

Weight conventions (frozen a priori, phase2-design.md sec. 6/10):
  * CPPN internal weights (W1, W2) and direct edge weights map x in [0,1]
    affinely to the signed box [-PARAM_SCALE, +PARAM_SCALE] (PARAM_SCALE=3.0), so
    CMA's [0,1] bounds cover a symmetric signed range (the N(0,1) prior phase 1's
    random CPPN drew from sits well inside +-3).
  * The HyperNEAT connection weight (``weight_mode="cppn"``) is the CPPN's signed
    output clipped to [-WEIGHT_MAX, +WEIGHT_MAX] (WEIGHT_MAX=3.0) -- the standard
    HyperNEAT weight cap. ``weight_mode="uniform"`` is the topology-only
    robustness arm (every kept edge weight 1.0), carrying phase 1's isolation
    forward. Signed weights are encoding-native here (a CPPN/weight-search
    naturally produces them); MN's weights stay whatever development produces.
    The genotype->phenotype map -- including its weight semantics -- IS the
    treatment, not a confound (design sec. 2).

All three are deterministic functions of x (seed-exact; matmul-free except the
CPPN's single numpy matmul, which is bit-exact same-host only -- immaterial to
seed aggregates, the same caveat as phase 1). A decode that cannot produce a
valid graph returns None (the driver scores None as worst-fitness). MorphoNAS is
the one arm whose graph need not be weakly connected (the reach-map never
required it of MN), so ``decode_morphonas`` returns whatever development grows.
"""

from __future__ import annotations

import signal
import threading
from typing import Optional

import networkx as nx
import numpy as np

from MorphoNAS.genome import Genome
from MorphoNAS.genome_strategies import DefaultMetaParametersStrategy
from MorphoNAS.grid import Grid
from MorphoNAS_DevPriors.baseline_encoders import (  # reuse the frozen phase-1 helpers
    DEFAULT_ACTIVATIONS,
    _ACTIVATIONS,
    _UnionFind,
    _candidate_pairs,
    _substrate_coords,
)

# ── Frozen knobs (phase2-design.md sec. 6) ───────────────────────────────────────
PARAM_SCALE = 3.0          # x in [0,1] -> [-3,3] for CPPN W1/W2 and direct edge weights
WEIGHT_MAX = 3.0           # HyperNEAT connection-weight magnitude ceiling (|w| <= 3)
WEIGHT_LO = 0.05           # HyperNEAT connection-weight magnitude floor (no near-zero edges)
N_FEAT = 6                 # CPPN substrate features: x1, y1, x2, y2, bias, dist
SAFETY_MAX_NODES = 1600    # 40x40 grid cap; a decode larger than this is a wall-clock fail

# Shared phase-2 constants (used by both the tier-1 and tier-2 drivers).
EVAL_SEED_BASE = 42        # fixed eval seeds per task: range(42, 42+R) (the reach-map base)

# Phase-2a frozen knobs (phase2a-design.md sec. 2, 3, FROZEN VALUES).
MN_DECODE_TIME_CAP_S = 15.0  # wall-time cap on a single MorphoNAS growth simulation -> invalid;
                             # rare off-manifold genomes can grow for minutes (design sec. 2a/3).

# Frozen CPPN substrate (N, E) = median competent-MN (N,E) from each task's phase-1
# pool (measured 2026-06-01 over experiments/<task>/pool; design sec. 6). MN and CPPN
# stay size-comparable per task; cartpole_masked shares cartpole's genomes/pool. Used
# by the tier-1 CPPN decoder and the tier-2 HyperNEAT substrate.
FROZEN_SUBSTRATE_NE = {
    "cartpole": (100, 283),
    "acrobot": (400, 1089),
    "lunarlander": (400, 898),
    "mountaincar": (400, 1061),
    "pendulum": (400, 1106),
    "cartpole_masked": (100, 275),
    # acrobot_masked added 2026-06-06 (sweep-design §13 A7, promote to full cppn/direct
    # baselines): median (N,E) over the masked-competent set of the 500-net masked
    # saturating(=v1.1) pilot experiments/acrobot_masked/pool_mnv2_saturating, using the
    # FLOOR-ESCAPE non-weak bar (reward > -500; 27 of 500 = the 5.4% reach-map rate) that
    # task_registry's 2026-06-06 calibration sets (floor-limited; cartpole_masked's p90 rule
    # is inapplicable here). Same recipe as the others -- it reproduces cartpole_masked (100,275).
    "acrobot_masked": (400, 1063),
}


def _to_signed(x: np.ndarray, scale: float = PARAM_SCALE) -> np.ndarray:
    """Map x in [0,1] affinely to the signed box [-scale, +scale]."""
    return (2.0 * np.asarray(x, dtype=np.float64) - 1.0) * scale


def assemble_substrate_graph(
    out_signed: np.ndarray,
    src: np.ndarray,
    tgt: np.ndarray,
    num_nodes: int,
    num_edges: int,
    *,
    weight_mode: str = "cppn",
    weight_max: float = WEIGHT_MAX,
):
    """Build the weighted DiGraph from a per-candidate-pair signed output (the CPPN
    output, whether from an explicit-weight tier-1 CPPN or a NEAT-evolved tier-2
    CPPN). Keeps the top-``num_edges`` by |output| with a weak-connectivity guarantee
    (union-find), then sets weights: ``cppn`` = signed, magnitude min-max rescaled
    across the kept edges into [WEIGHT_LO, weight_max] (HyperNEAT-faithful, graded);
    ``uniform`` = 1.0 (topology-only). Returns None if (N,E) can't be weakly
    connected. Shared by the tier-1 CPPN decoder and the tier-2 HyperNEAT arm so
    both use one substrate-assembly path."""
    n, e = int(num_nodes), int(num_edges)
    absw = np.abs(out_signed)
    order = np.argsort(-absw, kind="stable")
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
        return None
    for idx in order:
        if len(chosen) >= e:
            break
        if not in_chosen[idx]:
            chosen.append(int(idx))
            in_chosen[idx] = True

    ci = np.asarray(chosen)
    if weight_mode == "cppn":
        ok = out_signed[ci]
        absk = np.abs(ok)
        lo, hi = float(absk.min()), float(absk.max())
        span = hi - lo
        if span <= 0:
            mag = np.full(len(ci), 0.5 * (WEIGHT_LO + weight_max))
        else:
            mag = WEIGHT_LO + (absk - lo) / span * (weight_max - WEIGHT_LO)
        w = np.where(ok >= 0, mag, -mag)
    elif weight_mode == "uniform":
        w = np.ones(len(ci), dtype=np.float64)
    else:
        raise ValueError(f"unknown weight_mode: {weight_mode}")

    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    for k, idx in enumerate(chosen):
        G.add_edge(int(src[idx]), int(tgt[idx]), weight=float(w[k]))
    return G


# ── MorphoNAS decoder ─────────────────────────────────────────────────────────────

def morphonas_dim(num_morphogens: int = 3) -> int:
    """d = n^2 + 11n + 12 (= 54 at n=3); the length the flattened genome must have
    for ``Genome.from_flattened`` to recover this morphogen count."""
    n = int(num_morphogens)
    return n * n + 11 * n + 12


def morphonas_x0(rng, *, num_morphogens: int = 3, size_x: int = 20, size_y: int = 20,
                 max_growth_steps: int = 200) -> np.ndarray:
    """Initial CMA mean for the MN arm: a flattened random genome. Defaults match
    the reach-map pool regime (per-task grid, 200 growth steps), so generation 0
    of the MN arm is a draw from the SAME distribution as that task's phase-1
    pool -- the design's "each curve starts at its phase-1 prevalence"."""
    g = Genome.random(rng, size_x=size_x, size_y=size_y,
                      num_morphogens=num_morphogens, max_growth_steps=max_growth_steps)
    return g.flatten()


class _DecodeTimeout(BaseException):
    """Raised by the SIGALRM handler when a growth simulation exceeds its wall-time
    cap. Subclasses BaseException (not Exception) so a bare ``except Exception`` in
    the vendored engine cannot swallow it -- the same convention KeyboardInterrupt
    uses."""


def _alarm_handler(signum, frame):  # pragma: no cover - signal-driven
    raise _DecodeTimeout()


def _can_use_alarm() -> bool:
    """SIGALRM-based caps work only on the main thread of a process. Pool workers
    run the cell in their main thread (OK); guard so any non-main-thread caller
    (or a non-POSIX platform) falls back to an uncapped decode rather than crashing."""
    return (hasattr(signal, "SIGALRM") and hasattr(signal, "setitimer")
            and threading.current_thread() is threading.main_thread())


def _grow_capped(genome, time_cap_s: Optional[float]) -> Optional[nx.DiGraph]:
    """``Grid(genome).run_simulation() -> get_graph()`` under an optional SIGALRM
    wall-time cap. Returns None on a degenerate grow (raises, empty graph, runaway
    above the 40x40 grid cap) or a cap timeout. Shared by ``decode_morphonas`` (the
    tier-1 flattened-vector path) and ``grow_genome`` (the tier-2 GA's genome-native
    path) so both honour the same cap."""
    armed = False
    old_handler = None
    try:
        grid = Grid(genome)
        if time_cap_s and time_cap_s > 0 and _can_use_alarm():
            old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, float(time_cap_s))
            armed = True
        grid.run_simulation(verbose=False)
        G = grid.get_graph()
    except _DecodeTimeout:
        return None
    except Exception:
        return None
    finally:
        if armed:
            signal.setitimer(signal.ITIMER_REAL, 0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)
    if G is None or G.number_of_nodes() == 0:
        return None
    if G.number_of_nodes() > SAFETY_MAX_NODES:
        return None
    return G


def decode_morphonas(x, time_cap_s: Optional[float] = None) -> Optional[nx.DiGraph]:
    """x (in [0,1]^54) -> from_flattened -> Grid.run_simulation -> get_graph. Clips
    to [0,1] exactly as the vendored optimizer does. Returns None on a degenerate
    decode (raises, empty graph, or a runaway above the 40x40 grid cap). When
    ``time_cap_s`` is set, a single growth simulation exceeding the cap is treated as
    invalid (returns None) via a SIGALRM watchdog -- rare off-manifold genomes can
    grow for minutes (phase2a-design.md sec. 2a/3)."""
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    try:
        genome = Genome.from_flattened(x)
    except Exception:
        return None
    return _grow_capped(genome, time_cap_s)


def grow_genome(genome, time_cap_s: Optional[float] = None) -> Optional[nx.DiGraph]:
    """Tier-2 MorphoNAS-GA growth: a ``Genome`` object (NOT a flattened vector) grown
    to its graph under the same capped ``Grid.run_simulation -> get_graph`` as
    ``decode_morphonas``. Used so the GA scores genomes on their native development
    (no flatten round-trip, which would re-normalize integer genes)."""
    return _grow_capped(genome, time_cap_s)


# ── Phase-2a shared validity + morphogen-count + manifold-bounds helpers ──────────

def is_valid_graph(G, spec) -> bool:
    """Frozen phase-2a validity bar (phase2a-design.md sec. 2), uniform across arms:
    (1) decode succeeded -- ``G is not None`` with ``0 < N <= SAFETY_MAX_NODES``
        (the decode wall-time cap, applied in ``decode_morphonas``, yields None ->
        caught here);
    (2) ``N >= spec.min_neurons`` (= input_dim + output_dim, the seed-paper floor);
    (3) ``nx.is_weakly_connected(G)`` -- closes the asymmetry vs the baselines and
        filters the large-but-disconnected dead nets bar (2) admits.

    A no-op for the always-valid baselines (CPPN/direct decode to weakly-connected
    graphs at fixed N >= floor by construction); it binds only on MorphoNAS."""
    if G is None:
        return False
    n = G.number_of_nodes()
    if n <= 0 or n > SAFETY_MAX_NODES:
        return False
    if n < spec.min_neurons:
        return False
    return nx.is_weakly_connected(G)


class FixedMorphogenMetaStrategy(DefaultMetaParametersStrategy):
    """Phase-2a fixed-n meta-parameters (phase2a-design.md sec. 5b): identical to the
    vendored default except ``MUTATION_PROB_MORPHOGEN = 0``, so the GA's morphogen-
    count mutation never fires and n stays at its init value. Combined with the
    default crossover (which keeps n equal for equal-n parents) this fixes n across
    a whole MorphoNAS-GA run, isolating the optimizer from morphogen-count search."""

    def get_parameters(self) -> dict:
        p = super().get_parameters()
        p["MUTATION_PROB_MORPHOGEN"] = 0.0
        return p


def manifold_bounds(num_morphogens: int = 3) -> tuple:
    """Robustness-arm only (phase2a-design.md sec. 5/9): per-gene ``[lo, hi]`` bounds
    in flattened ``[0,1]^d`` coordinates that hard-restrict CMA to the natural
    ``Genome.random`` manifold, instead of the primary's full ``[0,1]`` box. Each
    gene's ``RANDOM_*_RANGE`` is mapped through ``flatten``'s normalization: genes
    already in ``[0,1]`` keep their raw sampling range; integer genes (growth steps,
    grid size, max axon length) map through ``Genome._normalize_integer`` and are
    clipped to ``[0,1]`` (a few ``RANDOM_*`` integer maxima exceed the normalization
    ceiling). Returns ``(lo_list, hi_list)`` for ``cma`` ``bounds=[lo, hi]``.

    NOT exercised by the phase-2a primary run (full ``[0,1]`` + viable init); provided
    so ``run_evo_tier1.py --manifold-bounds`` can run the cheaper, more constrained
    robustness search and show the result is not an artifact of the bound choice."""
    n = int(num_morphogens)
    d = morphonas_dim(n)
    mp = DefaultMetaParametersStrategy().get_parameters()

    def _int_bounds(rng_range, mid_args):
        lo_v, hi_v = rng_range
        a = Genome._normalize_integer(lo_v, *mid_args)
        b = Genome._normalize_integer(hi_v, *mid_args)
        lo_n, hi_n = min(a, b), max(a, b)
        return float(np.clip(lo_n, 0.0, 1.0)), float(np.clip(hi_n, 0.0, 1.0))

    lo = np.zeros(d)
    hi = np.ones(d)
    # Scalar genes, in flatten() order.
    scalar = [
        _int_bounds(mp["RANDOM_GROWTH_STEPS_RANGE"], (0, 200, 500)),  # max_growth_steps
        _int_bounds(mp["RANDOM_GRID_SIZE_RANGE"], (10, 40, 200)),     # size_x
        _int_bounds(mp["RANDOM_GRID_SIZE_RANGE"], (10, 40, 200)),     # size_y
        tuple(mp["RANDOM_DIFFUSION_RATE_RANGE"]),                     # diffusion_rate
        tuple(mp["RANDOM_DIVISION_THRESHOLD_RANGE"]),                 # division_threshold
        tuple(mp["RANDOM_DIFFERENTIATION_THRESHOLD_RANGE"]),          # cell_differentiation_threshold
        tuple(mp["RANDOM_AXON_GROWTH_THRESHOLD_RANGE"]),              # axon_growth_threshold
        _int_bounds(mp["RANDOM_AXON_LENGTH_RANGE"], (1, 3, 10)),      # max_axon_length
        tuple(mp["RANDOM_AXON_CONNECT_THRESHOLD_RANGE"]),             # axon_connect_threshold
        tuple(mp["RANDOM_SELF_CONNECT_RANGE"]),                       # self_connect_isolated_neurons_fraction
        tuple(mp["RANDOM_WEIGHT_TARGET_RANGE"]),                      # weight_adjustment_target
        tuple(mp["RANDOM_WEIGHT_RATE_RANGE"]),                        # weight_adjustment_rate
    ]
    for i, (a, b) in enumerate(scalar):
        lo[i], hi[i] = a, b
    idx = len(scalar)
    sec = tuple(mp["RANDOM_SECRETION_RANGE"])
    inh = tuple(mp["RANDOM_INHIBITION_RANGE"])
    dif = tuple(mp["RANDOM_DIFFUSION_PATTERN_RANGE"])
    for _ in range(2 * n):              # progenitor + neuron secretion rates
        lo[idx], hi[idx] = sec; idx += 1
    for _ in range(n * n):              # inhibition matrix
        lo[idx], hi[idx] = inh; idx += 1
    for _ in range(9 * n):              # diffusion patterns (renormalized on decode)
        lo[idx], hi[idx] = dif; idx += 1
    assert idx == d, (idx, d)
    lo = np.clip(lo, 0.0, 1.0)
    hi = np.clip(hi, 0.0, 1.0)
    return lo.tolist(), hi.tolist()


# ── CPPN / HyperNEAT decoder ──────────────────────────────────────────────────────

def cppn_dim(cppn_hidden: int = 4, n_feat: int = N_FEAT) -> int:
    """d = (n_feat + 2) * h: W1 (n_feat*h) + activation-select genes (h) + W2 (h)."""
    return (int(n_feat) + 2) * int(cppn_hidden)


def _cppn_forward(coords, src, tgt, W1, act_idx, W2, activations) -> np.ndarray:
    """Vectorised explicit-weight CPPN forward over every candidate pair. Mirrors
    ``baseline_encoders._cppn_query`` but takes W1/act_idx/W2 explicitly (no RNG)
    and returns the SIGNED output (the caller uses |.| for topology and the sign
    for the HyperNEAT weight)."""
    x1, y1 = coords[src, 0], coords[src, 1]
    x2, y2 = coords[tgt, 0], coords[tgt, 1]
    dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    bias = np.ones_like(x1)
    feats = np.stack([x1, y1, x2, y2, bias, dist], axis=1)  # (P, N_FEAT)
    H = feats @ W1                                          # (P, h)
    for j in range(W1.shape[1]):
        H[:, j] = _ACTIVATIONS[activations[act_idx[j]]](H[:, j])
    return H @ W2                                          # (P,)


def decode_cppn(
    x,
    num_nodes: int,
    num_edges: int,
    input_dim: int,
    output_dim: int,
    *,
    cppn_hidden: int = 4,
    activations: tuple = DEFAULT_ACTIVATIONS,
    connectivity: str = "recurrent",
    weight_mode: str = "cppn",
    param_scale: float = PARAM_SCALE,
    weight_max: float = WEIGHT_MAX,
) -> Optional[nx.DiGraph]:
    """Decode an explicit-weight CPPN over the (input_dim, output_dim, N=num_nodes)
    substrate, keeping the top-``num_edges`` connections by |CPPN output| (weakly
    connected, union-find). ``weight_mode="cppn"`` sets each weight to the CPPN's
    signed output (clipped to +-weight_max), HyperNEAT-faithful;
    ``weight_mode="uniform"`` sets every kept edge to 1.0 (topology-only).
    Matches (num_nodes, num_edges) by construction. Returns None if a
    weakly-connected graph at that (N,E) is impossible."""
    n = int(num_nodes)
    e = int(num_edges)
    h = int(cppn_hidden)
    num_hidden = n - input_dim - output_dim
    if num_hidden < 0 or e < n - 1:
        return None

    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    d = cppn_dim(h)
    if x.size != d:
        raise ValueError(f"cppn decode expects dim {d} (cppn_hidden={h}), got {x.size}")

    W1 = _to_signed(x[:N_FEAT * h], param_scale).reshape(N_FEAT, h)
    act_g = x[N_FEAT * h:N_FEAT * h + h]
    act_idx = np.minimum((act_g * len(activations)).astype(int), len(activations) - 1)
    W2 = _to_signed(x[N_FEAT * h + h:N_FEAT * h + 2 * h], param_scale)

    coords = _substrate_coords(input_dim, output_dim, num_hidden)
    src, tgt = _candidate_pairs(n, input_dim, connectivity)
    if e > len(src):
        return None

    out = _cppn_forward(coords, src, tgt, W1, act_idx, W2, activations)
    # Top-E weak-connected assembly + weight semantics are shared with the tier-2
    # HyperNEAT arm (NEAT-evolved CPPN over the same substrate).
    return assemble_substrate_graph(out, src, tgt, n, e,
                                    weight_mode=weight_mode, weight_max=weight_max)


# ── direct decoder (RWG "complete_h8" reference net) ───────────────────────────────

def direct_K(input_dim: int, output_dim: int, hidden: int = 8) -> int:
    return int(input_dim) + int(output_dim) + int(hidden)


def direct_dim(input_dim: int, output_dim: int, hidden: int = 8) -> int:
    """d = K*(K-1): one weight per directed edge of the complete recurrent net."""
    K = direct_K(input_dim, output_dim, hidden)
    return K * (K - 1)


def _direct_pairs(K: int) -> list:
    """Canonical edge order: lexicographic (i, j), i != j -- identical to
    ``random_graph.generate_random_rnn``'s ``all_pairs``, so the topology and the
    node/edge order match the RWG axis's complete net exactly."""
    return [(i, j) for i in range(K) for j in range(K) if i != j]


def decode_direct(
    x,
    input_dim: int,
    output_dim: int,
    *,
    hidden: int = 8,
    param_scale: float = PARAM_SCALE,
) -> nx.DiGraph:
    """x (in [0,1]^{K*(K-1)}) -> signed edge weights on the fixed complete_h8
    recurrent net. Topology is constant (every directed pair); only the weights
    vary, so this is "evolve the weights of the RWG reference net". Inputs land on
    labels 0..in-1 and outputs on the top labels (NeuralPropagator's stable
    in-degree tie-break), deterministic across all evals."""
    K = direct_K(input_dim, output_dim, hidden)
    pairs = _direct_pairs(K)
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    if x.size != len(pairs):
        raise ValueError(f"direct decode expects dim {len(pairs)} (K={K}), got {x.size}")
    w = _to_signed(x, param_scale)
    G = nx.DiGraph()
    G.add_nodes_from(range(K))
    for k, (i, j) in enumerate(pairs):
        G.add_edge(i, j, weight=float(w[k]))
    return G


# ── self-test: dims, determinism, weak connectivity, I/O placement, end-to-end ────
if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.abspath("code"))
    from MorphoNAS.neural_propagation import NeuralPropagator
    from MorphoNAS_DevPriors.task_registry import get_task, run_rollouts

    IN, OUT = 6, 3  # Acrobot dims
    N, E = 400, 1089  # acrobot median competent-MN (N,E) from the phase-1 pool
    rng = np.random.default_rng(20260601)

    def _prop_io(G):
        p = NeuralPropagator(G, input_dim=IN, output_dim=OUT,
                             activation_function=NeuralPropagator.tanh_activation,
                             extra_thinking_time=2, additive_update=False)
        return p, p.get_input_nodes_info()["selected_nodes"]

    # MorphoNAS. Most random genomes grow degenerate nets (1 node, 0 edges, or a
    # NaN decode -> None); the pool filters to the valid ones, and so must the
    # fitness wrapper. Draw until a pool-valid net (N >= IN+OUT, E >= 5) for the
    # end-to-end check, and confirm degenerate decodes return None or a small graph
    # (never raise).
    assert morphonas_dim(3) == 54
    x_mn = morphonas_x0(rng)
    assert x_mn.size == 54, x_mn.size
    g_mn = None
    for _ in range(60):
        xx = morphonas_x0(rng)
        gg = decode_morphonas(xx)  # must never raise
        if gg is not None and gg.number_of_nodes() >= IN + OUT and gg.number_of_edges() >= 5:
            g_mn, x_mn = gg, xx
            break
    assert g_mn is not None, "no valid MN net in 60 draws (unexpected)"
    g_mn2 = decode_morphonas(x_mn)
    assert sorted(g_mn.edges()) == sorted(g_mn2.edges()), "MN decode not deterministic"

    # CPPN
    assert cppn_dim(4) == 32
    x_cp = rng.uniform(0, 1, cppn_dim(4))
    g_cp = decode_cppn(x_cp, N, E, IN, OUT, cppn_hidden=4, weight_mode="cppn")
    assert g_cp is not None
    assert g_cp.number_of_nodes() == N and g_cp.number_of_edges() == E, \
        (g_cp.number_of_nodes(), g_cp.number_of_edges())
    assert nx.is_weakly_connected(g_cp), "cppn graph not weakly connected"
    g_cp2 = decode_cppn(x_cp, N, E, IN, OUT, cppn_hidden=4, weight_mode="cppn")
    e1 = sorted((u, v, round(d["weight"], 12)) for u, v, d in g_cp.edges(data=True))
    e2 = sorted((u, v, round(d["weight"], 12)) for u, v, d in g_cp2.edges(data=True))
    assert e1 == e2, "cppn decode not deterministic"
    _, io_cp = _prop_io(g_cp)
    assert sorted(io_cp) == list(range(IN)), io_cp
    # HyperNEAT weights are signed + graded (not all saturated to one value); check
    # the mechanism is signed-capable and graded across a few independent draws.
    neg_seen = pos_seen = graded_seen = False
    for s in range(8):
        xs = np.random.default_rng(700 + s).uniform(0, 1, cppn_dim(4))
        gs = decode_cppn(xs, N, E, IN, OUT, cppn_hidden=4, weight_mode="cppn")
        wss = np.array([d["weight"] for _, _, d in gs.edges(data=True)])
        neg_seen |= bool((wss < 0).any())
        pos_seen |= bool((wss > 0).any())
        graded_seen |= bool(np.unique(np.round(np.abs(wss), 6)).size > 5)
        assert np.all(np.abs(wss) <= WEIGHT_MAX + 1e-9)
    assert neg_seen and pos_seen, "cppn weights never signed across draws"
    assert graded_seen, "cppn weight magnitudes never graded across draws"
    # uniform mode: all weights 1.0, same topology
    g_cu = decode_cppn(x_cp, N, E, IN, OUT, cppn_hidden=4, weight_mode="uniform")
    assert all(d["weight"] == 1.0 for _, _, d in g_cu.edges(data=True))
    assert sorted(g_cu.edges()) == sorted(g_cp.edges()), "uniform changed topology"

    # direct
    K = direct_K(IN, OUT, 8)
    assert direct_dim(IN, OUT, 8) == K * (K - 1)
    x_di = rng.uniform(0, 1, direct_dim(IN, OUT, 8))
    g_di = decode_direct(x_di, IN, OUT, hidden=8)
    assert g_di.number_of_nodes() == K and g_di.number_of_edges() == K * (K - 1)
    assert nx.is_weakly_connected(g_di)
    p_di, io_di = _prop_io(g_di)
    assert sorted(io_di) == list(range(IN)), io_di
    g_di2 = decode_direct(x_di, IN, OUT, hidden=8)
    assert sorted((u, v, round(d["weight"], 12)) for u, v, d in g_di.edges(data=True)) == \
        sorted((u, v, round(d["weight"], 12)) for u, v, d in g_di2.edges(data=True))

    # end-to-end: each decoder's graph runs on Acrobot and yields a finite reward
    spec = get_task("acrobot")
    for name, G in (("morphonas", g_mn), ("cppn", g_cp), ("direct", g_di)):
        res = run_rollouts(spec.make_propagator(G), 2, seeds=[42, 43], env=spec.make_env())
        assert np.isfinite(res["avg_reward"]), (name, res)
        print(f"OK  {name:10s} N={G.number_of_nodes():4d} E={G.number_of_edges():5d} "
              f"avg_reward(2 eps)={res['avg_reward']:.1f}")

    # ── phase-2a additions ─────────────────────────────────────────────────────
    # is_valid_graph: the structural bar (decode + N>=min + weakly-connected),
    # uniform across arms; a no-op for the always-valid baselines.
    assert is_valid_graph(g_cp, spec)            # cppn: weakly connected at N>=floor
    assert is_valid_graph(g_di, spec)            # direct: complete recurrent net
    assert is_valid_graph(g_mn, spec)            # the pool-valid MN net found above
    assert not is_valid_graph(None, spec)        # failed decode -> invalid
    # N >= min but disconnected -> rejected (the NEW MN bar; closes the asymmetry)
    g_disc = nx.DiGraph()
    g_disc.add_nodes_from(range(spec.min_neurons + 5))
    g_disc.add_edge(0, 1)                         # one tiny component, the rest isolated
    assert not is_valid_graph(g_disc, spec), "disconnected graph passed validity"
    # below the N floor -> rejected
    g_small = nx.DiGraph(); g_small.add_edge(0, 1)
    assert not is_valid_graph(g_small, spec), "sub-floor graph passed validity"

    # decode wall-time cap: a generous cap must not change a normal decode.
    g_capped = decode_morphonas(x_mn, time_cap_s=30.0)
    assert g_capped is not None and sorted(g_capped.edges()) == sorted(g_mn.edges()), \
        "generous wall-time cap altered a normal decode"
    # the SIGALRM watchdog itself: a 0.05s cap on a 5s busy-wait must raise _DecodeTimeout.
    if _can_use_alarm():
        import time as _t
        fired = False
        oldh = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, 0.05)
        try:
            t_end = _t.time() + 5.0
            while _t.time() < t_end:
                pass
        except _DecodeTimeout:
            fired = True
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, oldh)
        assert fired, "SIGALRM watchdog did not fire on a 5s busy-wait with a 0.05s cap"
        print("OK  decode wall-time cap: SIGALRM watchdog fires (_DecodeTimeout)")

    # fixed-n: a MorphoNAS-GA genome built with FixedMorphogenMetaStrategy keeps
    # num_morphogens == 3 through many mutate/crossover events (the strategy
    # propagates through from_dict in both operators).
    meta = FixedMorphogenMetaStrategy()
    assert meta.get_parameters()["MUTATION_PROB_MORPHOGEN"] == 0.0
    grng = np.random.default_rng(12345)
    fam = [Genome.random(grng, size_x=20, size_y=20, num_morphogens=3, max_growth_steps=200,
                         meta_parameters_strategy=meta) for _ in range(6)]
    for _ in range(200):
        if grng.random() < 0.5:
            a, b = grng.choice(len(fam), size=2, replace=False)
            child = Genome.crossover(fam[int(a)], fam[int(b)], grng)
        else:
            child = fam[int(grng.integers(len(fam)))].mutate(grng)
        assert child.num_morphogens == 3, f"morphogen count drifted to {child.num_morphogens}"
        fam[int(grng.integers(len(fam)))] = child
    print("OK  fixed-n: num_morphogens stayed 3 across 200 mutate/crossover events")

    # manifold bounds (robustness arm, not the primary): a valid [0,1] box of the
    # right length with lo <= hi everywhere.
    mlo, mhi = manifold_bounds(3)
    assert len(mlo) == len(mhi) == morphonas_dim(3) == 54
    mlo_a, mhi_a = np.array(mlo), np.array(mhi)
    assert np.all(mlo_a >= 0) and np.all(mhi_a <= 1) and np.all(mlo_a <= mhi_a)
    assert (mlo[3], mhi[3]) == (0.05, 0.2), (mlo[3], mhi[3])  # diffusion_rate raw range
    print(f"OK  manifold_bounds: 54-gene [0,1] box (diffusion_rate gene "
          f"[{mlo[3]:.3f},{mhi[3]:.3f}])")

    print(f"OK  dims: morphonas={morphonas_dim(3)} cppn={cppn_dim(4)} "
          f"direct(acrobot)={direct_dim(IN, OUT, 8)}")
    print("evo_encoders self-test PASSED")
