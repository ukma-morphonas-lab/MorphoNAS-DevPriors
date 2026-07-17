"""
Observation-structure wrappers for the memory / partial-observability rung.

The reward wrappers (reward_wrappers.py) hold the success METRIC fixed and vary
what behaviour earns it. These wrappers instead vary what the controller can
OBSERVE, holding the task, reward, and action space fixed -- isolating the memory
axis. Hiding the velocity channels of a fully-observed control task turns it into
a POMDP: the instantaneous observation no longer determines the optimal action,
so a competent controller must INTEGRATE the position history over time (a
recurrent/memory motif) to recover the hidden velocity. A purely reactive
(memoryless) policy cannot, even in principle, balance from positions alone.

This is the clean entry point for the memory rung: take a mapped task (CartPole,
fully observed, advantage 8.42x) and mask its velocity channels, holding genome
growth, dims, reward, and actions identical -- so the masked and unmasked pools
are genome-paired by seed (growth is env-independent). The MN-vs-random advantage
UNDER masking, compared to the unmasked baseline, is the causal signature for
whether the developmental prior supplies useful MEMORY dynamics (temporal
integration) and not only the reactive control motif. A higher ratio under
masking widens the prior's identity to persistence; parity sharpens it to
reactive control.

  MaskVelocityCartPole -- CartPole-v1 observation is
    [cart_position, cart_velocity, pole_angle, pole_angular_velocity]; this zeros
    the two velocity channels (indices 1, 3), leaving [position, 0, angle, 0]. The
    dimension is kept at 4 (zeroed, not dropped) so the grown architecture and the
    genome population are byte-identical to unmasked CartPole -- the ONLY change is
    the information content of two input channels, exactly the de-confounder
    discipline (hold everything fixed, vary one thing). This is the classic
    non-Markovian / velocity-masked CartPole POMDP.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np


class MaskVelocityCartPole(gym.ObservationWrapper):
    """Zero CartPole's velocity channels (cart_velocity idx 1, pole_angular_velocity
    idx 3), keeping the 4-dim observation shape. Forces temporal integration: the
    optimal action depends on velocity, which is no longer observed and must be
    inferred from the position history, so only a controller with internal memory
    can balance. ObservationWrapper.observation applies the mask to both the reset
    and the step observations, transparently to the rollout loop; the reward and
    action space are untouched."""

    VELOCITY_INDICES = (1, 3)

    def observation(self, observation):
        masked = np.array(observation, dtype=np.float32).copy()
        masked[list(self.VELOCITY_INDICES)] = 0.0
        return masked


class MaskObservationChannels(gym.ObservationWrapper):
    """Generic memory-rung masker: zero a fixed set of observation channels while
    keeping the dimension fixed, so the masked and unmasked pools stay genome-paired
    (growth is env-independent) and the only change is the information content. The
    controller must reconstruct the masked channels from the history of the visible
    ones (temporal integration). Used for the second memory task, velocity-masked
    Acrobot: the observation is [cos t1, sin t1, cos t2, sin t2, w1, w2], and zeroing
    the two angular-velocity channels (idx 4, 5) hides exactly the rates an energy-
    pumping swing-up controller needs, forcing it to integrate the angle history.
    (MaskVelocityCartPole is the equivalent idx {1,3} case kept as its own class for
    the committed CartPole rung.)"""

    def __init__(self, env, indices):
        super().__init__(env)
        self.mask_indices = list(indices)

    def observation(self, observation):
        masked = np.array(observation, dtype=np.float32).copy()
        masked[self.mask_indices] = 0.0
        return masked


class OneHotDiscreteObservation(gym.ObservationWrapper):
    """Encode a Discrete(n) integer observation as a one-hot n-vector, so the
    propagator (which consumes a fixed-width float vector and argmaxes over output
    nodes) can drive a discrete-state task like FrozenLake. input_dim must equal n.
    This is the navigation rung's encoder: the agent sees which grid cell it is in
    (one-hot), and the structure under test is whether the developmental prior biases
    a random-genome net toward a state->action map that traces a path to the goal."""

    def __init__(self, env, n):
        super().__init__(env)
        self.n = int(n)
        self.observation_space = gym.spaces.Box(0.0, 1.0, (self.n,), dtype=np.float32)

    def observation(self, observation):
        v = np.zeros(self.n, dtype=np.float32)
        v[int(observation)] = 1.0
        return v
