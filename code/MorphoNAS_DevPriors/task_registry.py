"""
Task registry for the reach-map difficulty ladder.

One source of truth for per-task config (env, I/O dims, strata, grid, seeds), so
the pool generator, matched-random control, and weight ablation share identical
evaluation semantics across the ladder: CartPole < Acrobot < LunarLander <
MountainCar-sparse.

Evaluation reuses the proven Acrobot rollout loop
(``experiment_acrobot.run_rollouts``), which is env-agnostic -- observation ->
propagate -> argmax over output nodes -> env.step -- as long as the env is passed
explicitly. Only the propagator I/O dims and the stratum thresholds differ per
task. Acrobot routed through this registry reproduces ``experiment_acrobot``
exactly (same dims, same bounds, same propagator settings), which the smoke test
verifies against the existing pool.

STRATA CALIBRATION: Acrobot and CartPole bounds are locked (they reproduce the
published 8.42x and 63.6x). LunarLander and MountainCar bounds are marked
``provisional=True`` -- they are first guesses to be finalized from the pilot's
observed reward distribution before the full pool is grown (same pilot-based
procedure the Acrobot bounds came from). The ``perfect`` cut is pinned to the
Gymnasium "solved" threshold for each task and does not move.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from MorphoNAS.neural_propagation import NeuralPropagator

# Re-export the proven, env-agnostic rollout loop so every ladder script uses one
# evaluation code path. Importing it here (rather than copying) guarantees the
# new tasks evaluate exactly like the locked Acrobot result.
from MorphoNAS_DevPriors.experiment_acrobot import run_rollouts  # noqa: F401

STRATA_ORDER = ["weak", "low_mid", "high_mid", "near_perfect", "perfect"]
SOLVED_STRATUM = "perfect"

INF = math.inf


@dataclass(frozen=True)
class TaskSpec:
    name: str
    env_name: str
    input_dim: int
    output_dim: int
    # ordered ((name, low, high), ...); weak low may be -inf, perfect high +inf
    stratum_bounds: tuple
    grid_size: tuple = (20, 20)
    max_growth_steps: int = 200
    num_morphogens: int = 3
    pool_seed_start: int = 5_000_001
    control_base_seed: int = 6_000_000
    solved_note: str = ""
    provisional: bool = False
    env_wrapper: Optional[Callable] = None  # env -> wrapped env (reward-structure variants)
    env_kwargs: dict = field(default_factory=dict)  # extra gym.make kwargs (e.g. is_slippery)

    @property
    def min_neurons(self) -> int:
        return self.input_dim + self.output_dim

    def get_stratum(self, reward: float) -> str:
        for name, low, high in self.stratum_bounds:
            if low <= reward < high:
                return name
        return SOLVED_STRATUM  # reward >= top bound

    def stratum_bounds_dict(self) -> dict:
        return {name: [low, high] for name, low, high in self.stratum_bounds}

    def make_propagator(self, G, *, graph_diameter=None, edge_hook=None) -> NeuralPropagator:
        """NeuralPropagator with this task's dims and the locked settings
        (tanh, extra_thinking_time=2, additive_update=False). Identical to
        experiment_acrobot.create_propagator when dims are 6/3."""
        if edge_hook is not None:
            raise ValueError("plasticity is not supported in this harness")
        return NeuralPropagator(
            G=G,
            input_dim=self.input_dim,
            output_dim=self.output_dim,
            activation_function=NeuralPropagator.tanh_activation,
            extra_thinking_time=2,
            additive_update=False,
        )

    def make_env(self):
        """gym.make(env_name) + the optional reward-structure wrapper. Workers call
        this instead of gym.make so the 2x2 de-confounding variants get wrapped."""
        import gymnasium as gym

        env = gym.make(self.env_name, **self.env_kwargs)
        return self.env_wrapper(env) if self.env_wrapper is not None else env


# ── Locked rungs ─────────────────────────────────────────────────────────────

# CartPole: reference-only (we do not re-grow it; the 8.42x is pinned from B2
# counts). Included so the ladder analysis knows the easy anchor's difficulty.
CARTPOLE = TaskSpec(
    name="cartpole",
    env_name="CartPole-v1",
    input_dim=4,
    output_dim=2,
    stratum_bounds=(
        ("weak", -INF, 200.0),
        ("low_mid", 200.0, 300.0),
        ("high_mid", 300.0, 400.0),
        ("near_perfect", 400.0, 475.0),
        ("perfect", 475.0, INF),
    ),
    grid_size=(10, 10),
    pool_seed_start=7_000_001,
    control_base_seed=7_500_000,
    solved_note="Gymnasium solved: reward >= 475 (perfect); non-weak >= 200.",
)

# Acrobot: locked. Bounds reproduce the published strata exactly.
ACROBOT = TaskSpec(
    name="acrobot",
    env_name="Acrobot-v1",
    input_dim=6,
    output_dim=3,
    stratum_bounds=(
        ("weak", -500.0, -450.0),
        ("low_mid", -450.0, -300.0),
        ("high_mid", -300.0, -200.0),
        ("near_perfect", -200.0, -100.0),
        ("perfect", -100.0, INF),
    ),
    grid_size=(20, 20),
    pool_seed_start=5_000_001,
    control_base_seed=6_000_000,
    solved_note="Gymnasium solved: reward >= -100 (perfect); non-weak >= -450.",
)

# ── New rungs (provisional bounds; pilot calibrates) ─────────────────────────

# LunarLander-v3: 8 obs / 4 actions, Box2D. Reward is unbounded below (crash and
# fuel costs), so unlike Acrobot there is no clean dead-floor; these bounds are
# first guesses. The pilot reports the reward distribution and we finalize the
# non-perfect cuts before the full pool. Gymnasium solved = 200 (pinned).
LUNARLANDER = TaskSpec(
    name="lunarlander",
    env_name="LunarLander-v3",
    input_dim=8,
    output_dim=4,
    stratum_bounds=(
        ("weak", -INF, -121.0),
        ("low_mid", -121.0, -110.0),
        ("high_mid", -110.0, -95.0),
        ("near_perfect", -95.0, 200.0),
        ("perfect", 200.0, INF),
    ),
    grid_size=(20, 20),
    pool_seed_start=8_000_001,
    control_base_seed=8_200_000,
    solved_note=("Gymnasium solved: reward >= 200 (perfect; never reached by random "
                 "genomes). Non-weak cut -121 = top decile of the 504-net pilot "
                 "(10.3%, matches Acrobot's 10.26% selectivity). 2026-05-30 calibration."),
    provisional=False,
)

# MountainCar-v0: 2 obs / 3 actions, sparse. Reward is -1/step, capped at 200, so
# a car that never reaches the flag scores exactly -200 (a clean dead-floor, like
# Acrobot). Anything above -200 means it reached the flag in some episodes.
# Gymnasium solved = -110 (pinned). RISK: MorphoNAS competent fraction may be ~0;
# the pilot's >=1% gate decides whether this rung is usable.
MOUNTAINCAR = TaskSpec(
    name="mountaincar",
    env_name="MountainCar-v0",
    input_dim=2,
    output_dim=3,
    stratum_bounds=(
        ("weak", -INF, -198.0),
        ("low_mid", -198.0, -160.0),
        ("high_mid", -160.0, -130.0),
        ("near_perfect", -130.0, -110.0),
        ("perfect", -110.0, INF),
    ),
    grid_size=(20, 20),
    pool_seed_start=9_000_001,
    control_base_seed=9_200_000,
    solved_note="Gymnasium solved: reward >= -110 (perfect). Non-weak >= -198 = flag-escape behavioral anchor (-200 = never-reached dead floor), locked; intermediate strata bounds provisional.",
    provisional=True,
)

# ── De-confounding 2x2 variants ──────────────────────────────────────────────
# Same genomes as their counterparts (identical pool_seed_start); only the reward
# metric differs. Because evaluation is behavior-fixed (no learning), these re-score
# identical trajectories, isolating "what success requires" from the env's other
# properties. Strata PROVISIONAL until the pilot calibrates.

from MorphoNAS_DevPriors.reward_wrappers import (  # noqa: E402
    DiscretizeTorque,
    ShapedAcrobot,
    ShapedFrozenLake,
    ShapedMountainCar,
    SparseLunarLander,
    SparsePendulum,
)
from MorphoNAS_DevPriors.observation_wrappers import (  # noqa: E402
    MaskObservationChannels,
    MaskVelocityCartPole,
    OneHotDiscreteObservation,
)


def _wrap_sparse_lunar(env):
    return SparseLunarLander(env)


def _wrap_shaped_mc(env):
    return ShapedMountainCar(env, scale=100.0)


def _wrap_shaped_acrobot(env):
    return ShapedAcrobot(env, scale=100.0)


def _wrap_pendulum_discrete(env):
    return DiscretizeTorque(env)


def _wrap_pendulum_sparse(env):
    # SparsePendulum OUTSIDE DiscretizeTorque: identical action mapping to the dense
    # cell, only the reward metric differs (the de-confounder invariant).
    return SparsePendulum(DiscretizeTorque(env))


LUNARLANDER_SPARSE = TaskSpec(
    name="lunarlander_sparse",
    env_name="LunarLander-v3",
    input_dim=8,
    output_dim=4,
    stratum_bounds=(
        ("weak", -INF, 1.0),
        ("low_mid", 1.0, 25.0),
        ("high_mid", 25.0, 55.0),
        ("near_perfect", 55.0, 95.0),
        ("perfect", 95.0, INF),
    ),
    grid_size=(20, 20),
    pool_seed_start=8_000_001,     # SAME genomes as lunarlander (dense)
    control_base_seed=8_400_000,   # disjoint from the dense lunar control (8_200_000)
    solved_note=("Sparse: terminal-only reward (+100 land / -100 crash / 0 timeout). "
                 "Non-weak = lands net-positive. PROVISIONAL, pilot-calibrated."),
    provisional=True,
    env_wrapper=_wrap_sparse_lunar,
)

MOUNTAINCAR_SHAPED = TaskSpec(
    name="mountaincar_shaped",
    env_name="MountainCar-v0",
    input_dim=2,
    output_dim=3,
    stratum_bounds=(
        ("weak", -INF, -200.0),
        ("low_mid", -200.0, -160.0),
        ("high_mid", -160.0, -100.0),
        ("near_perfect", -100.0, -50.0),
        ("perfect", -50.0, INF),
    ),
    grid_size=(20, 20),
    pool_seed_start=9_000_001,     # SAME genomes as mountaincar (sparse)
    control_base_seed=9_200_000,   # SAME random graphs as sparse mountaincar (paired de-confound)
    solved_note=("Shaped: -1/step + 100*(position progress); dense partial-progress "
                 "credit. Non-weak -200 = top decile of the 500-net pilot (9.6%). "
                 "2026-05-31 calibration."),
    provisional=False,
    env_wrapper=_wrap_shaped_mc,
)

ACROBOT_SHAPED = TaskSpec(
    name="acrobot_shaped",
    env_name="Acrobot-v1",
    input_dim=6,
    output_dim=3,
    stratum_bounds=(
        ("weak", -INF, -400.0),
        ("low_mid", -400.0, -250.0),
        ("high_mid", -250.0, 0.0),
        ("near_perfect", 0.0, 130.0),
        ("perfect", 130.0, INF),
    ),
    grid_size=(20, 20),
    pool_seed_start=5_000_001,     # SAME genomes as acrobot
    control_base_seed=6_000_000,   # SAME random graphs as sparse acrobot (paired de-confound)
    solved_note=("Shaped: -1/step + 100*(tip-height gain); dense partial-progress credit. "
                 "Non-weak -400 = top decile of the 500-net pilot (10.2%). Distribution is "
                 "bimodal (stuck ~-497 vs swing-up +130..+237). 2026-05-31 calibration."),
    provisional=False,
    env_wrapper=_wrap_shaped_acrobot,
)


# ── Pendulum cells (de-confounder pair) ───────────
# Pendulum-v1: obs [cos t, sin t, t_dot] (3), continuous torque discretised to 3
# bang-bang actions {-2, 0, +2}. The dense cell keeps the native quadratic reward
# r = -(angle_normalize(t)^2 + 0.1*t_dot^2 + 0.001*u^2) over a fixed 200-step
# horizon; the sparse cell replaces it with an upright-hold count (the SparsePendulum
# wrapper). Both share pool_seed_start (identical genomes) and control_base_seed
# (identical random graphs), exactly like mountaincar / mountaincar_shaped, so the
# de-confounder re-scores identical trajectories. Strata PROVISIONAL: the pilot's
# reward distribution calibrates the non-weak (top-decile) cut before the full pool.

PENDULUM = TaskSpec(
    name="pendulum",
    env_name="Pendulum-v1",
    input_dim=3,
    output_dim=3,
    stratum_bounds=(
        ("weak", -INF, -1322.0),         # non-weak = top decile of the 500-net pilot (p90)
        ("low_mid", -1322.0, -1171.0),   # p90..p99
        ("high_mid", -1171.0, -967.0),   # p99..max
        ("near_perfect", -967.0, -300.0),
        ("perfect", -300.0, INF),        # sustained upright; not reached by these nets
    ),
    grid_size=(20, 20),
    pool_seed_start=10_000_001,
    control_base_seed=10_200_000,
    solved_note=("Dense native reward -(theta^2 + 0.1*theta_dot^2 + 0.001*u^2), 200-step "
                 "horizon, no termination. 500-net pilot: smooth unimodal blob (NO structured "
                 "mode), mean -1444, p90 -1322, max -967. Non-weak -1322 = top decile (the "
                 "LunarLander/MountainCar selectivity rule). No net reaches sustained upright; "
                 "R here is bar-sensitive (companion to the bar-immune sparse cell). 2026-05-31."),
    provisional=False,
    env_wrapper=_wrap_pendulum_discrete,
)

PENDULUM_SPARSE = TaskSpec(
    name="pendulum_sparse",
    env_name="Pendulum-v1",
    input_dim=3,
    output_dim=3,
    stratum_bounds=(
        ("weak", -INF, 13.0),            # non-weak = top decile of the 500-net pilot (p90 ~13)
        ("low_mid", 13.0, 17.0),         # p90..p99
        ("high_mid", 17.0, 28.0),        # p99..max
        ("near_perfect", 28.0, 100.0),
        ("perfect", 100.0, INF),         # sustained upright (>=50% of episode); not reached
    ),
    grid_size=(20, 20),
    pool_seed_start=10_000_001,     # SAME genomes as pendulum (dense)
    control_base_seed=10_200_000,   # SAME random graphs as pendulum (paired de-confound)
    solved_note=("Sparse: +1 per step while |theta| < 0.2 rad (upright band), else 0; episode "
                 "score = upright-step count in [0, 200]. 500-net pilot: median 6.6, p90 13.0, "
                 "max 28.2. Non-weak 13 = top decile. Bar-immune semantic cell (spinner -> 0). "
                 "No net sustains upright; structured behaviour reached is partial/transient "
                 "near-upright occupancy. 2026-05-31 calibration."),
    provisional=False,
    env_wrapper=_wrap_pendulum_sparse,
)


# ── Memory / partial-observability rung (velocity-masked CartPole) ───────────
# The reach-map's "clean entry point" for the memory axis: take the mapped, fully
# observed CartPole and zero its two velocity channels, turning it into a POMDP
# where balancing requires integrating the position history (a recurrent/memory
# motif), with everything else held fixed. Because genome growth is env-independent,
# pool_seed_start 7_000_001 grows genomes byte-identical to unmasked CartPole, and
# control_base_seed 7_500_000 (the never-run CartPole control band) yields random
# graphs that a fresh unmasked control can share -- so masked-vs-unmasked is a
# genome+graph-paired observation de-confounder for the memory axis. Strata
# PROVISIONAL: the pilot's reward distribution calibrates the non-weak (top-decile)
# cut before the full pool.


def _wrap_cartpole_masked(env):
    return MaskVelocityCartPole(env)


CARTPOLE_MASKED = TaskSpec(
    name="cartpole_masked",
    env_name="CartPole-v1",
    input_dim=4,
    output_dim=2,
    stratum_bounds=(
        ("weak", -INF, 28.0),            # non-weak = top decile of the 500-net pilot (p90 28.1)
        ("low_mid", 28.0, 38.0),         # p90..p95
        ("high_mid", 38.0, 45.0),        # p95..p99
        ("near_perfect", 45.0, 475.0),   # p99..max (47); top ~1%
        ("perfect", 475.0, INF),         # Gymnasium solved (pinned); unreachable without velocity
    ),
    grid_size=(10, 10),
    pool_seed_start=7_000_001,     # SAME genomes as cartpole (growth is env-independent)
    control_base_seed=7_500_000,   # SAME random graphs as a fresh unmasked cartpole control (paired de-confound)
    solved_note=("Velocity-masked CartPole (POMDP): cart_velocity (idx 1) and "
                 "pole_angular_velocity (idx 3) zeroed, so balancing requires integrating "
                 "the position history (memory). Reward = mean episode length over 20 rollouts, "
                 "max 500; Gymnasium solved = 475 (pinned, perfect; unreachable here). "
                 "Genome-paired to unmasked cartpole (pool_seed_start 7_000_001) and "
                 "graph-paired to a fresh unmasked control (control_base_seed 7_500_000): "
                 "masked-vs-unmasked is the observation de-confounder for the memory axis. "
                 "500-net pilot: tight floor at 9.4 (do-nothing fall), p90 28.1, p95 38.7, "
                 "max 47. Non-weak 28 = top decile (the LunarLander/MountainCar/Pendulum "
                 "selectivity rule); soft-bar instrument, the bar-sweep carries the dose-response. "
                 "2026-06-01 calibration."),
    provisional=False,
    env_wrapper=_wrap_cartpole_masked,
)


# Second memory task: angular-velocity-masked Acrobot. The Acrobot observation
# [cos t1, sin t1, cos t2, sin t2, w1, w2] has its two angular-velocity channels
# (idx 4, 5) zeroed, hiding exactly the rates an energy-pumping swing-up controller
# needs, so a competent controller must integrate the angle history. Genome-paired
# to acrobot (pool_seed_start 5_000_001) and graph-paired to the acrobot control
# (control_base_seed 6_000_000), so masked-vs-unmasked is the observation
# de-confounder on the strongest control rung (unmasked 63.6x). Strata CALIBRATED
# 2026-06-06 (floor-limited: 94.6% at the -500 floor, only 5.4% escape; non-weak = the
# floor-escape bar reward > -500, see solved_note); usable as a floor-limited instrument.


def _wrap_acrobot_masked(env):
    return MaskObservationChannels(env, (4, 5))


ACROBOT_MASKED = TaskSpec(
    name="acrobot_masked",
    env_name="Acrobot-v1",
    input_dim=6,
    output_dim=3,
    stratum_bounds=(
        ("weak", -INF, -495.0),          # the -500 do-nothing floor (94.6% of the pilot)
        ("low_mid", -495.0, -450.0),     # barely escaped the floor (most of the 5.4% competent)
        ("high_mid", -450.0, -300.0),    # mid escaped tail
        ("near_perfect", -300.0, -100.0),# strong escaped tail (pilot max -263.2)
        ("perfect", -100.0, INF),        # Gymnasium solved (pinned; unreachable under masking)
    ),
    grid_size=(20, 20),
    pool_seed_start=5_000_001,     # SAME genomes as acrobot (growth is env-independent)
    control_base_seed=6_000_000,   # SAME random graphs as acrobot (paired observation de-confound)
    solved_note=("Angular-velocity-masked Acrobot (POMDP): w1 (idx 4), w2 (idx 5) zeroed, "
                 "so swing-up requires integrating the angle history (memory). Reward = "
                 "-1/step to success, -500 floor; Gymnasium solved = -100 (pinned). "
                 "Genome-paired to acrobot (pool_seed_start 5_000_001) and graph-paired "
                 "(control_base_seed 6_000_000): masked-vs-unmasked is the observation "
                 "de-confounder on the strongest control rung (unmasked 63.6x). FLOOR-LIMITED "
                 "(MountainCar-like): the 500-net masked saturating(=v1.1) pilot has 94.6% at the "
                 "-500 do-nothing floor and only 5.4% escaping (reward > -500), matching the "
                 "reach-map masked-Acrobot rate. cartpole_masked's p90 top-decile rule is "
                 "inapplicable here (p90 = the -500 floor), so non-weak is the FLOOR-ESCAPE bar "
                 "(reward > -500; clean gap: floor -500.0, first escaped -494.6, cut at -495). "
                 "Escaped tail graded -495/-450/-300; perfect -100 (Gymnasium solved, pinned, "
                 "unreachable under masking; pilot max -263.2). 2026-06-06 calibration."),
    provisional=False,
    env_wrapper=_wrap_acrobot_masked,
)


# ── Navigation rung (spatial path) ───────────────────────────────────────────
# FrozenLake-4x4 deterministic: the agent sees its grid cell (one-hot, 16) and
# argmaxes over 4 moves; the structure under test is a SPATIAL PATH to the goal.
# Fully observed and Markovian, so the optimal policy is reactive (no memory needed)
# -- this is the family where a recurrence-supplying prior has the least obvious
# purchase, so the matched-random ratio sharpens (advantage absent) or widens
# (advantage present) the prior's identity. Deterministic + per-episode reset makes
# each net's reward 0 or 1 (its fixed policy either traces a path to the goal or
# not), a clean binary competence with no stochastic confound. Strata PROVISIONAL.


def _wrap_frozenlake(env):
    return OneHotDiscreteObservation(env, 16)


FROZENLAKE = TaskSpec(
    name="frozenlake",
    env_name="FrozenLake-v1",
    input_dim=16,
    output_dim=4,
    stratum_bounds=(
        ("weak", -INF, 0.5),             # does not reach the goal
        ("low_mid", 0.5, 0.9),           # (graded band; only populated under is_slippery=True)
        ("high_mid", 0.9, 0.99),
        ("near_perfect", 0.99, 1.0),
        ("perfect", 1.0, INF),           # reaches the goal every episode (deterministic: reward 1.0)
    ),
    grid_size=(20, 20),
    pool_seed_start=11_000_001,
    control_base_seed=11_200_000,
    env_kwargs={"is_slippery": False, "map_name": "4x4"},
    solved_note=("Deterministic FrozenLake-4x4 (is_slippery=False): one-hot state (16) -> "
                 "argmax over 4 actions. Reward = goal-reach fraction over 20 rollouts; "
                 "deterministic + per-episode reset so per-net reward is 0 or 1 (the net's "
                 "fixed policy either traces a path to the goal or not). Non-weak = reaches "
                 "goal. Structure = a spatial path; fully observed MDP (optimal policy is "
                 "reactive), so this probes whether the recurrence prior helps navigation. "
                 "PROVISIONAL pilot calibration."),
    provisional=True,
    env_wrapper=_wrap_frozenlake,
)


# Densified navigation-progress companion to the frozenlake frontier null: re-scores
# the SAME genomes (pool_seed_start 11_000_001) and matched-random graphs
# (control_base_seed 11_200_000) by how close the policy gets to the goal (negative
# closest Manhattan distance), not whether it arrives. The discriminating test: does
# the prior produce policies that make navigation PROGRESS more than random wiring
# (advantage -> widens identity), or not (parity -> sharpens it)? Strata PROVISIONAL.


def _wrap_frozenlake_shaped(env):
    return OneHotDiscreteObservation(ShapedFrozenLake(env), 16)


FROZENLAKE_SHAPED = TaskSpec(
    name="frozenlake_shaped",
    env_name="FrozenLake-v1",
    input_dim=16,
    output_dim=4,
    stratum_bounds=(
        ("weak", -INF, -4.0),            # provisional; dist > 4 from goal (little progress)
        ("low_mid", -4.0, -3.0),
        ("high_mid", -3.0, -2.0),
        ("near_perfect", -2.0, -0.5),
        ("perfect", -0.5, INF),          # reward 0 = reached goal (dist 0); empty (frontier)
    ),
    grid_size=(20, 20),
    pool_seed_start=11_000_001,    # SAME genomes as frozenlake
    control_base_seed=11_200_000,  # SAME random graphs as frozenlake (paired)
    env_kwargs={"is_slippery": False, "map_name": "4x4"},
    solved_note=("Shaped FrozenLake-4x4: reward = -(closest Manhattan distance to the goal "
                 "reached), the densified navigation-progress companion to the sparse "
                 "goal-reach frontier null (0 reach the goal). Non-weak = made progress toward "
                 "the goal. Genome- and graph-paired to frozenlake. PROVISIONAL pilot calibration."),
    provisional=True,
    env_wrapper=_wrap_frozenlake_shaped,
)


# Published CartPole B2 counts (the 8.42x easy anchor), reused by the control
# verdict and the ladder analysis. MN over the 50k B0.5 pool; random over the
# matched-to-competent B2 pool. Pinned, not recomputed.
CARTPOLE_REF = {
    "nonweak": {"mn_count": 2362, "mn_n": 50000, "rand_count": 65, "rand_n": 11586},
    "solved": {"mn_count": 769, "mn_n": 50000, "rand_count": 26, "rand_n": 11586},
}


_REGISTRY = {
    t.name: t for t in (CARTPOLE, ACROBOT, LUNARLANDER, MOUNTAINCAR,
                        LUNARLANDER_SPARSE, MOUNTAINCAR_SHAPED, ACROBOT_SHAPED,
                        PENDULUM, PENDULUM_SPARSE, CARTPOLE_MASKED, ACROBOT_MASKED,
                        FROZENLAKE, FROZENLAKE_SHAPED)
}

# Difficulty order for the ladder, fixed before running
# and justified by task properties (state dim, reward sparsity, horizon), not by
# our own measurement.
DIFFICULTY_ORDER = ["cartpole", "acrobot", "lunarlander", "mountaincar"]


def get_task(name: str) -> TaskSpec:
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise KeyError(f"unknown task '{name}'; known: {sorted(_REGISTRY)}")
    return _REGISTRY[key]


def all_tasks() -> list[str]:
    return list(_REGISTRY.keys())
