"""The task the fleet performs, and the expert that can relabel it.

Kinematic, as everywhere else in this stack: joints integrate commanded deltas
subject to limits and a rate cap. No contacts, no dynamics, no friction. Success
rates from it are harness results, not robot results.

Two things about it are load-bearing for the fleet loop.

**Nodes are not identical.** Each has an :class:`NodeEnvironment` — a systematic
joint bias standing in for a miscalibrated arm, an actuation noise level, and a
goal distribution. A policy trained on one node's conditions is not automatically
good on another's, and that gap is the entire reason a fleet loop exists rather
than a single training run. A loop demonstrated on identical nodes demonstrates
nothing.

**Success requires holding the goal, not touching it.** A policy that flails
through the tolerance ball has not reached the goal, and a benchmark that says it
has will rank a random policy well above zero. This is the mistake the evaluation
harness upstream found the hard way, and it is not repeated here: success means
`settle_steps` consecutive steps inside tolerance.

Goals are sampled in joint space and success is measured in joint space. Sampling
in one space and scoring in another is how a do-nothing policy ends up scoring
38%, which is the other bug that harness found.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from edge_runtime.policy import default_obs_columns

N_JOINTS = 7
JOINT_LIMIT = 2.6
RATE_CAP = 0.15
TOLERANCE = 0.10  # rad, in joint space
SETTLE_STEPS = 5
MAX_STEPS = 120


@dataclass
class NodeEnvironment:
    """The conditions one physical node operates under.

    `joint_bias` is the interesting one: a constant offset added to every
    commanded delta, as a miscalibrated or worn arm would produce. It is
    observable only through its effect, exactly as it would be on hardware —
    the policy is not told which node it is on.
    """

    name: str
    goal_scale: float = 1.0
    start_spread: float = 0.35
    joint_bias: np.ndarray | None = None
    actuation_noise: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.joint_bias is None:
            self.joint_bias = np.zeros(N_JOINTS, dtype=np.float32)
        self.joint_bias = np.asarray(self.joint_bias, dtype=np.float32)


def nominal(seed: int = 0) -> NodeEnvironment:
    """A well-calibrated arm working close in — the conditions v1 was trained for."""
    return NodeEnvironment("nominal", goal_scale=0.55, start_spread=0.25, seed=seed)


def miscalibrated(seed: int = 1, magnitude: float = 0.012) -> NodeEnvironment:
    """A wider workspace, on an arm whose joints drift a little on every command.

    The bias is a twelfth of the rate cap and constant: small enough that the arm
    looks fine to anyone watching it, large enough that it shifts which states
    the arm actually visits.
    """
    rng = np.random.default_rng(seed)
    bias = rng.normal(0, magnitude, N_JOINTS).astype(np.float32)
    return NodeEnvironment(
        "miscalibrated", goal_scale=1.0, start_spread=0.5, joint_bias=bias, seed=seed
    )


def long_reach(seed: int = 2) -> NodeEnvironment:
    """A node working well outside the region v1's training data covered.

    This is the node the fleet loop exists for. Nothing is broken here — the arm
    is fine and the task is solvable, as the expert's 100% shows. The deployed
    policy simply never saw these states, so it extrapolates, and a policy
    extrapolating is a policy guessing. No amount of examining v1's training
    metrics would have revealed it; only running it here does.
    """
    return NodeEnvironment(
        "long-reach", goal_scale=1.6, start_spread=0.85, actuation_noise=0.008, seed=seed
    )


@dataclass
class ReachEnv:
    """Episodic reach task producing the observation contract shared with the device."""

    env: NodeEnvironment = field(default_factory=nominal)
    max_steps: int = MAX_STEPS
    obs_columns: list[str] = field(default_factory=default_obs_columns)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.env.seed)
        self.reset()

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.joints = self._rng.normal(0, self.env.start_spread, N_JOINTS).astype(np.float32)
        self.goal = (self._rng.uniform(-0.8, 0.8, N_JOINTS) * self.env.goal_scale).astype(
            np.float32
        )
        self.gripper = 0.0
        self.t = 0
        self.last_action_norm = 0.0
        self._settled = 0
        self.commanded = np.zeros(N_JOINTS + 1, dtype=np.float32)
        return self.observe()

    def observe(self) -> np.ndarray:
        error = self.goal - self.joints
        return np.array(
            [
                *self.joints,
                *error,
                self.gripper,
                self.t / self.max_steps,
                float(np.linalg.norm(error)),
                self.last_action_norm,
            ],
            dtype=np.float32,
        )

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.goal - self.joints))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, bool]:
        """Apply an action. Returns (observation, done, success).

        :attr:`commanded` is set to what the actuator actually accepted — the
        policy's output after the rate cap and the gripper clamp, and before the
        node's own bias and noise, which are properties of the arm and not of the
        command. That is the action to log: training on a raw output of 5.0 rad
        that the arm executed as 0.15 teaches the policy to reproduce a number
        the robot has never once carried out.
        """
        action = np.asarray(action, dtype=np.float32).ravel()
        commanded = np.clip(action[:N_JOINTS], -RATE_CAP, RATE_CAP)
        if action.size > N_JOINTS:
            self.gripper = float(np.clip(action[N_JOINTS], 0.0, 1.0))
        self.commanded = np.concatenate([commanded, [self.gripper]]).astype(np.float32)

        delta = commanded + self.env.joint_bias
        if self.env.actuation_noise:
            delta = delta + self._rng.normal(0, self.env.actuation_noise, N_JOINTS)

        self.joints = np.clip(self.joints + delta, -JOINT_LIMIT, JOINT_LIMIT).astype(np.float32)
        self.last_action_norm = float(np.linalg.norm(delta))
        self.t += 1

        self._settled = self._settled + 1 if self.distance < TOLERANCE else 0
        success = self._settled >= SETTLE_STEPS
        done = success or self.t >= self.max_steps
        return self.observe(), done, success


# -- the expert --------------------------------------------------------------


def expert_action(observation: np.ndarray, gain: float = 0.6) -> np.ndarray:
    """Proportional control on the goal error, with the gripper closing on arrival.

    This is the stand-in for a human teleoperator. In a real fleet loop, the
    states a policy struggles in are the states you pay someone to demonstrate,
    and what comes back is a corrected action for each one. Here the correction
    is free and exact, which makes the relabelling step *easier* than reality by
    a wide margin — the loop's mechanics are the claim, not its sample
    efficiency.

    It is closed-loop, which is why it copes with a joint bias the policy that
    generated the failing states could not: it observes the error the bias
    creates and keeps correcting.
    """
    observation = np.asarray(observation, dtype=np.float32)
    error = observation[N_JOINTS : 2 * N_JOINTS]
    delta = np.clip(error * gain, -RATE_CAP, RATE_CAP)
    close = float(np.linalg.norm(error) < TOLERANCE)
    return np.concatenate([delta, [close]]).astype(np.float32)


def relabel(observations: np.ndarray) -> np.ndarray:
    """Expert actions for a batch of visited states — the DAgger step."""
    return np.stack([expert_action(obs) for obs in np.atleast_2d(observations)])


class ExpertPolicy:
    """The expert as a policy, for measuring the ceiling of a node's conditions."""

    name = "expert"
    obs_columns = default_obs_columns()

    def forward(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.ndim == 1:
            return expert_action(observation)
        return relabel(observation)


class ZeroPolicy:
    """Does nothing. The anchor that says whether the task is a task.

    A suite on which a do-nothing policy scores well is measuring the
    environment rather than the policy, and every number it produces is
    worthless. Asserted in the tests, not assumed.
    """

    name = "zero"
    obs_columns = default_obs_columns()

    def forward(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        shape = (8,) if observation.ndim == 1 else (len(observation), 8)
        return np.zeros(shape, dtype=np.float32)


def success_rate(policy, environment: NodeEnvironment, episodes: int = 60, seed: int = 0) -> float:
    """Closed-loop success rate of a policy under one node's conditions.

    Used by the hub to sanity-check a candidate before it costs any fleet
    exposure. It is a *simulator* number and it is not what promotion rests on —
    the whole point of the canary is that the hub's model of a node is not the
    node.
    """
    env = ReachEnv(env=environment)
    successes = 0
    for episode in range(episodes):
        obs = env.reset(seed=seed * 10_000 + episode)
        while True:
            obs, done, ok = env.step(policy.forward(obs))
            if done:
                successes += ok
                break
    return successes / episodes
