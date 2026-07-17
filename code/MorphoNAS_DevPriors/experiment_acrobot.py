"""
Acrobot evaluation harness (developmental priors, baseline / no plasticity).

Wired to the canonical MorphoNAS engine plus the two approved fixes (per-episode
network reset and the W[post,pre] propagation-direction orientation). The plasticity
machinery from the MorphoNAS-PL plasticity paper is intentionally absent here: evaluation
evaluates competence of the *developed* network only. `create_propagator` and
`run_rollouts` therefore build a bare propagator and reset its state between episodes,
with no edge_hook / weight learning. An explicit guard rejects any non-None edge_hook
so plasticity can never run silently in this repo.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Optional

import gymnasium as gym
import numpy as np

from MorphoNAS.genome import Genome
from MorphoNAS.grid import Grid
from MorphoNAS.neural_propagation import NeuralPropagator

_shutdown_event = None

ENV_NAME = "Acrobot-v1"
INPUT_DIM = 6
OUTPUT_DIM = 3
MIN_NEURONS = INPUT_DIM + OUTPUT_DIM  # 9


def set_shutdown_event(event: Any) -> None:
    global _shutdown_event
    _shutdown_event = event


# ── Stratum definitions ──────────────────────────────────────────────
# Acrobot rewards are negative: -500 (never solves) to ~-80 (solves fast).

class Stratum(str, Enum):
    WEAK = "weak"
    LOW_MID = "low_mid"
    HIGH_MID = "high_mid"
    NEAR_PERFECT = "near_perfect"
    PERFECT = "perfect"


STRATUM_BOUNDS = {
    Stratum.WEAK: (-500, -450),
    Stratum.LOW_MID: (-450, -300),
    Stratum.HIGH_MID: (-300, -200),
    Stratum.NEAR_PERFECT: (-200, -100),
    Stratum.PERFECT: (-100, float("inf")),
}


def get_stratum(reward: float) -> Stratum:
    """Assign stratum based on mean baseline reward."""
    for stratum, (low, high) in STRATUM_BOUNDS.items():
        if low <= reward < high:
            return stratum
    # Rewards >= -100 count as perfect (Gymnasium solved threshold)
    return Stratum.PERFECT


def get_stratum_label(stratum: Stratum) -> str:
    labels = {
        Stratum.WEAK: "Weak [-500, -450)",
        Stratum.LOW_MID: "Low-mid [-450, -300)",
        Stratum.HIGH_MID: "High-mid [-300, -200)",
        Stratum.NEAR_PERFECT: "Near-perfect [-200, -100)",
        Stratum.PERFECT: "Perfect [>=-100]",
    }
    return labels.get(stratum, str(stratum))


VERIFICATION_SEEDS = list(range(42, 62))  # 20 fixed seeds


# ── Network loading and creation ─────────────────────────────────────

def load_network_from_file(filepath: str) -> tuple[Genome, dict]:
    with open(filepath, "r") as f:
        data = json.load(f)

    genome = Genome.from_dict(data["genome"])
    metadata = {
        "network_id": data.get("network_id"),
        "seed": data.get("seed"),
        "stratum": data.get("stratum"),
        "baseline_reward": data.get("baseline_reward"),
        "baseline_fitness": data.get("baseline_fitness"),
        "network_stats": data.get("network_stats", {}),
    }
    return genome, metadata


def create_propagator(
    grid: Grid,
    *,
    edge_hook=None,
    graph_diameter: Optional[int] = None,
) -> NeuralPropagator:
    if edge_hook is not None:
        raise ValueError("plasticity is not supported in this (developmental-priors) harness")
    G = grid.get_graph()
    return NeuralPropagator(
        G=G,
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        activation_function=NeuralPropagator.tanh_activation,
        extra_thinking_time=2,
        additive_update=False,
    )


# ── Evaluation ───────────────────────────────────────────────────────

def run_rollouts(
    propagator: NeuralPropagator,
    num_rollouts: int,
    *,
    seeds: Optional[list[int]] = None,
    reset_plastic_each_episode: bool = True,
    record_plasticity: bool = False,
    record_step_trace: bool = False,
    env: Optional[gym.Env] = None,
) -> dict:
    """Run multiple Acrobot episodes, resetting network state each episode (no plasticity)."""
    close_env = False
    if env is None:
        env = gym.make(ENV_NAME, render_mode=None)
        close_env = True

    rewards: list[float] = []
    lengths: list[int] = []

    for i in range(int(num_rollouts)):
        if _shutdown_event is not None and _shutdown_event.is_set():
            break

        propagator.reset()  # reset fix: clear network state between episodes

        if seeds is not None and i < len(seeds):
            observation, _ = env.reset(seed=int(seeds[i]))
        else:
            observation, _ = env.reset()

        total_reward = 0.0
        done = False
        steps = 0

        while not done:
            obs = np.array(observation).flatten()
            propagator.propagate(obs)
            output_values = propagator.get_output()
            action = int(output_values.argmax().item())

            observation, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)

            total_reward += float(reward)
            steps += 1

        rewards.append(float(total_reward))
        lengths.append(int(steps))

    if close_env:
        env.close()

    return {
        "rewards": rewards,
        "lengths": lengths,
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards)) if rewards else 0.0,
        "min_reward": float(np.min(rewards)) if rewards else 0.0,
        "max_reward": float(np.max(rewards)) if rewards else 0.0,
    }
