"""
Reward-structure wrappers for the de-confounding 2x2 (task x reward-structure).

Evaluation is WITHOUT learning (baseline, eta=0), so a network's behavior
is fixed: a given (genome, seed) produces the same trajectory whatever the reward.
These wrappers therefore do not change behavior -- they change the SUCCESS METRIC
applied to identical trajectories. That is the de-confounder: same env (physics,
dims, actions, behavior), only the competence criterion changes, isolating "what
success requires" from every other property of the task.

  SparseLunarLander -- strip the dense shaping, keep only the terminal outcome
    (+100 land / -100 crash), 0 otherwise. Tests whether MorphoNAS nets actually
    LAND more than random, even though the dense "controlled hover" bar showed no
    edge.

  ShapedMountainCar -- add dense potential-based shaping (progress up the hill) to
    the -1/step. Tests whether handing random wiring partial-progress credit
    collapses the (otherwise infinite) MorphoNAS advantage.

The Pendulum cells add (a discrete-action wrapper plus a sparse/dense de-confounder pair):

  DiscretizeTorque -- turn Pendulum-v1's continuous torque into a Discrete(3)
    bang-bang action {-2, 0, +2}, so the argmax-over-output rollout loop can drive
    it. This is the SAME innermost wrapper under both Pendulum cells, so the dense
    and sparse cells re-score identical trajectories (the de-confounder invariant).

  SparsePendulum -- replace the native quadratic cost with a sparse upright-hold
    reward (+1 per step inside the upright band |theta| < band, else 0; episode
    score = upright-step count). The bar-immune semantic cell: a spinner cannot
    earn it, only sustained near-upright can.
"""

from __future__ import annotations

import math

import gymnasium as gym
import numpy as np


class SparseLunarLander(gym.Wrapper):
    """LunarLander with shaping removed: reward is only the terminal +100 (came to
    rest) / -100 (crash or out-of-bounds), and 0 on every non-terminal step and on
    truncation. Gymnasium's LunarLander assigns reward = +/-100 on the terminating
    step (overwriting the shaping that step), so passing the terminal reward
    through and zeroing the rest yields a pure landed/crashed/timeout signal."""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        sparse = float(reward) if terminated else 0.0
        return obs, sparse, terminated, truncated, info


class ShapedAcrobot(gym.Wrapper):
    """Acrobot with dense potential-based shaping on tip height added to -1/step:
    reward += scale * (H(s') - H(s)), where H = -cos(t1) - cos(t1 + t2) is the tip
    height used in the swing-up success condition (success at H > 1). Gaining height
    earns dense credit, so a net that swings partway up scores above the -500 floor
    without reaching the goal. H is reconstructed from the observation
    [cos t1, sin t1, cos t2, sin t2, w1, w2]."""

    def __init__(self, env, scale: float = 100.0):
        super().__init__(env)
        self.scale = float(scale)
        self._prev_h = None

    @staticmethod
    def _height(obs) -> float:
        c1, s1, c2, s2 = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
        cos_12 = c1 * c2 - s1 * s2  # cos(t1 + t2)
        return -c1 - cos_12

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_h = self._height(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        h = self._height(obs)
        shaped = float(reward) + self.scale * (h - self._prev_h)
        self._prev_h = h
        return obs, shaped, terminated, truncated, info


class ShapedMountainCar(gym.Wrapper):
    """MountainCar with dense potential-based shaping added to the -1/step:
    reward += scale * (position' - position). Rightward progress earns dense
    credit, so a car that oscillates partway up the hill scores above the -200
    dead floor without reaching the flag. Potential-based (telescoping) so the
    shaping is a clean function of progress, not a re-weighting of the task."""

    def __init__(self, env, scale: float = 100.0):
        super().__init__(env)
        self.scale = float(scale)
        self._prev_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_pos = float(obs[0])
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        pos = float(obs[0])
        shaped = float(reward) + self.scale * (pos - self._prev_pos)
        self._prev_pos = pos
        return obs, shaped, terminated, truncated, info


class DiscretizeTorque(gym.Wrapper):
    """Pendulum-v1 (continuous torque in [-2, 2]) -> Discrete(3) bang-bang control:
    action 0 -> -2.0, 1 -> 0.0, 2 -> +2.0. The rollout loop argmaxes over
    output nodes and passes the integer action to env.step, so a discrete action
    space is required; this is the minimal 3-torque bang-bang mapping the design
    pins. The mapping is fixed and is used as the SAME innermost wrapper under both
    the dense (native reward) and the sparse Pendulum cells, so the two re-score
    identical trajectories (the de-confounder invariant)."""

    TORQUES = (-2.0, 0.0, 2.0)

    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Discrete(len(self.TORQUES))

    def step(self, action):
        torque = self.TORQUES[int(action)]
        return self.env.step(np.array([torque], dtype=np.float32))


class ShapedFrozenLake(gym.Wrapper):
    """Dense navigation-progress reward for FrozenLake: replaces the sparse +1-at-goal
    reward with -(closest Manhattan distance to the goal reached during the episode),
    returned at episode end (0 on non-terminal steps). A policy that moves toward the
    goal scores higher even if it never arrives, so this measures navigation PROGRESS
    (skill, not the luck a stochastic env would add) -- the discriminating test of
    whether the developmental prior helps navigation, the densified companion to the
    sparse goal-reach cell (which is a frontier null: 0 reach the goal). Reads the raw
    integer state, so wrap it INSIDE the one-hot observation encoder."""

    def __init__(self, env, ncols: int = 4, goal_state: int = 15):
        super().__init__(env)
        self.ncols = int(ncols)
        self.goal_r, self.goal_c = goal_state // ncols, goal_state % ncols
        self._min_dist = None

    def _dist(self, state: int) -> int:
        return abs(self.goal_r - state // self.ncols) + abs(self.goal_c - state % self.ncols)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._min_dist = self._dist(int(obs))
        return obs, info

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        self._min_dist = min(self._min_dist, self._dist(int(obs)))
        shaped = -float(self._min_dist) if (terminated or truncated) else 0.0
        return obs, shaped, terminated, truncated, info


class SparsePendulum(gym.Wrapper):
    """Sparse upright-hold reward for Pendulum, replacing the native quadratic cost:
    +1 on every step the pole is inside the upright band |theta| < band (default
    0.2 rad), 0 otherwise; episode score = upright-step count. theta is recovered
    from the observation [cos theta, sin theta, theta_dot] as atan2(sin, cos), which
    is exactly angle_normalize(theta) in [-pi, pi] (theta = 0 is upright). This is
    the bar-immune semantic cell of the de-confounder. Wrap it OUTSIDE
    DiscretizeTorque so the action mapping (and therefore the trajectory) is
    identical to the dense cell; only the reward metric changes."""

    def __init__(self, env, band: float = 0.2):
        super().__init__(env)
        self.band = float(band)

    def step(self, action):
        obs, _reward, terminated, truncated, info = self.env.step(action)
        theta = math.atan2(float(obs[1]), float(obs[0]))
        upright = 1.0 if abs(theta) < self.band else 0.0
        return obs, upright, terminated, truncated, info
